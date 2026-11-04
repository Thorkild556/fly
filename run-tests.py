"""run-tests.py

Runs Fugal and the kissing-fugal variants on the small real-noise datasets
and reports time, peak memory and Hungarian accuracy for every
(algorithm, dataset, noise level) combination.

    .venv/bin/python run-tests.py                       # 5 algorithms x 3 small datasets
    .venv/bin/python run-tests.py --datasets ACM-DBLP   # the optional large one
    .venv/bin/python run-tests.py --algos fugal kissing-fast --datasets voles --targets 95
    .venv/bin/python run-tests.py --max-iter 50         # quick smoke test
    .venv/bin/python run-tests.py --datasets fw-small --gpus 0 1   # generated datasets, 2 GPUs
    .venv/bin/python run-tests.py --settings tuned      # per-dataset Result-all settings

By default every run uses the benchmark protocol: the same settings rules
for every dataset, depending only on its size class (see
protocol_hyperparameters), one run per method with seed 0.  For
hyperparameter sweeps see sweep-hyperparameters.py.

Datasets: the ones listed below, plus fw-<name> for every dataset made by
generate-framework-datasets.py (targets 5..25 = % noise); fw-small,
fw-medium and fw-large select all of one size.  --gpus runs one job per
GPU in parallel (each run sees only its own GPU); runs on the CPU (Fugal)
then share the CPU cores, so for clean timings use one GPU at a time.

Every run happens in its own subprocess so the peak memory of one run does
not leak into the next.  Rows are appended to results/results.csv as they
finish; the full output of each run is kept in results/logs/.  Runs that
already have a row with the same settings are skipped, so after a crash or
Ctrl+C the same command continues where it stopped (--rerun runs them again).

Reported numbers
  time_s       wall time of the algorithm call (feature extraction + training;
               for Fugal also its final Hungarian step).  Data loading and the
               accuracy evaluation are not included.
  peak_rss_mb  peak resident memory of the process when the algorithm returns.
  peak_gpu_mb  peak GPU memory (CUDA: max reserved, MPS: sampled driver
               memory).  Empty on CPU.

The kissing variants are imported from this folder and pick CUDA, then MPS, then
CPU.  fugal_original.py is a verbatim copy of helpers/pred.py from
https://github.com/idea-iitd/Fugal and always runs on the CPU (float64).
"""

import argparse
import csv
import functools
import importlib.util
import inspect
import json
import os
import queue
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FLY_DIR = ROOT   # the algorithm scripts are next to this file
DATA_DIR = FLY_DIR / "data" / "real_noise"
RESULT_PREFIX = "@@RESULT@@ "
LOG_EVERY = 100   # loss logging interval for scripts whose train_with_adam takes log_every


# ============================================================================
# Datasets.  Ground truth is the identity mapping unless "ground_truth" is set,
# either one file for all targets or {target: file}.  .npy or text, one
# "query target" pair per row.
# ============================================================================

DATASETS = {
    "MultiMagna": {
        "dir": "MultiMagna",
        "query": "yeast0_Y2H1.txt",
        "targets": {t: f"yeast{t}_Y2H1.txt" for t in ("5", "10", "15", "20", "25")},
        "mu": 0.5,
    },
    "highschool": {
        "dir": "contacts-prox-high-school-2013",
        "query": "contacts-prox-high-school-2013_100.txt",
        "targets": {t: f"contacts-prox-high-school-2013_{t}.txt" for t in ("99", "95", "90", "80")},
        "mu": 0.5,
    },
    "voles": {
        "dir": "mamalia-voles-plj-trapping",
        "query": "mammalia-voles-plj-trapping_100.txt",
        "targets": {t: f"mammalia-voles-plj-trapping_{t}.txt" for t in ("99", "95", "90", "80")},
        "mu": 0.5,
    },
    "ACM": {
        "dir": "ACM-DBLP",
        "query": "ACM.txt",
        "targets": {t: f"ACM{t}.txt" for t in ("95", "90", "85", "80")},
        "mu": 0.1,
    },
    "ACM-DBLP": {
        "dir": "ACM-DBLP",
        "query": "ACM.txt",
        "targets": {"DBLP": "DBLP.txt"},
        "mu": 0.1,
        "ground_truth": "pos_pairs.npy",
    },
}
# Made from data/raw with make-noisy-graphs.py (ca-CondMat is the same
# graph as condmat-100.txt, with its own noise).  mu=0.1 as in Result-all
# for the large graphs.
for _name in ("ca-CondMat", "email-enron", "soc-brightkite"):
    DATASETS[_name] = {
        "dir": _name,
        "query": f"{_name}-100.txt",
        "targets": {t: f"{_name}-{t}.txt" for t in ("95", "90", "85", "80")},
        "mu": 0.1,
    }

# Datasets made by generate-framework-datasets.py (Framework_GraphAlignment's
# default noise, 5-25% of the edges removed), as fw-<name>.  Targets are the
# noise percentages.  Graphs that also have a dataset above use its mu and
# hyperparameters ("hp"); the others mu=0.1 as in Result-all and the
# settings for their size class.  "group" is the folder the generator put
# them in, for --datasets fw-small / fw-medium / fw-large.
FRAMEWORK_DIR = FLY_DIR / "data" / "framework_noise"
_FW_HP = {"highschool": "highschool", "voles": "voles", "MultiMagna": "MultiMagna", "ACM": "ACM"}
_fw_index = FRAMEWORK_DIR / "datasets.json"
for _name, _info in (json.loads(_fw_index.read_text()) if _fw_index.exists() else {}).items():
    _hp = _FW_HP.get(_name)
    DATASETS[f"fw-{_name}"] = {
        "dir": str(FRAMEWORK_DIR / _info["dir"]),
        "query": f"{_name}-0.txt",
        "targets": {str(p): f"{_name}-{p}.txt" for p in _info["noise"]},
        "mu": DATASETS[_hp]["mu"] if _hp else 0.1,
        "group": _info["size"],
        **({"hp": _hp} if _hp else {}),
    }
DATASET_GROUPS = {
    f"fw-{size}": [k for k, d in DATASETS.items() if k.startswith("fw-") and d["group"] == size]
    for size in ("small", "medium", "large")
}
DEFAULT_DATASETS = ["MultiMagna", "highschool", "voles"]


# ============================================================================
# Hyperparameters.  COMMON follows the README / Result-all.  For chunked and
# chunked-fast, embed_dim, chunk and max_iter are the values of the reference
# run of each pair in Result-all ("dataset" keys apply to every noise level,
# "dataset/target" keys to one pair).  dense-LA and kissing-fast have no
# reference runs on these datasets: dense-LA uses the defaults from its own
# __main__, kissing-fast the same settings as chunked-fast.
# embed_dim / chunk = None means "auto-selected by the algorithm".
# "medium" / "large" keys apply to datasets of that size class that have no
# entry of their own: ACM's settings for medium, and for large the settings
# of the Result-all runs on slashdot (n=82k) and ippds (n=108k).
# ============================================================================

COMMON = dict(beta=10, row_penalty=10, col_penalty=200, learning_rate=1e-2)

FAST_MAX_ITER = {
    "MultiMagna/25": dict(max_iter=3500),
    "highschool": dict(max_iter=700),
    "highschool/99": dict(max_iter=500),
    "highschool/80": dict(max_iter=1300),
    "voles": dict(max_iter=1000),
    "voles/80": dict(max_iter=4000),
    "ACM": dict(max_iter=10000),
    "ACM-DBLP": dict(max_iter=10000),
}

ALGORITHMS = {
    "fugal": {
        "default": dict(niter=15),
    },
    "dense-LA": {
        "script": "kissingfugal-dense-LA.py",
        "default": dict(embed_dim=500, learning_rate=1e-3, max_iter=30000,
                        kind="relu_squared", favor_features=200),
    },
    "dense-chunked": {
        "script": "kissingfugal-dense-chunked.py",
        "default": dict(embed_dim=100, chunk=64, max_iter=2000),
        "MultiMagna/20": dict(max_iter=3500),
        "MultiMagna/25": dict(max_iter=3500),
        # Result-all lists max_iter=10000 for highschool/95, but its time
        # (33.4s) matches the 1000-iteration runs, so 1000 is used.
        "highschool": dict(embed_dim=20, max_iter=1000),
        "highschool/80": dict(embed_dim=32, max_iter=2000),
        "voles": dict(max_iter=1000),
        "voles/80": dict(max_iter=4000),
        "ACM": dict(embed_dim=1000, chunk=2048, max_iter=10000),
        "ACM-DBLP": dict(embed_dim=1000, chunk=2048, max_iter=10000),
        "medium": dict(embed_dim=1000, chunk=2048, max_iter=10000),
        "large": dict(embed_dim=1000, chunk=2048, max_iter=3000),
    },
    "dense-chunked-fast": {
        "script": "kissingfugal-dense-chunked-fast.py",
        "default": dict(embed_dim=None, chunk=None, max_iter=2000),
        "medium": dict(max_iter=10000),
        "large": dict(beta=25, embed_dim=256, max_iter=3000),
        **FAST_MAX_ITER,
    },
    "kissing-fast": {
        "script": "kissing-fast.py",
        "default": dict(embed_dim=None, chunk=None, max_iter=2000),
        "medium": dict(max_iter=10000),
        "large": dict(beta=25, embed_dim=1000, max_iter=2000),
        **FAST_MAX_ITER,
    },
}

# Fugal and dense-LA keep dense n x n matrices (Fugal several in float64),
# so they are skipped above this many nodes unless --no-size-limit is given.
MAX_N = {"fugal": 30_000, "dense-LA": 30_000}

# Above this many nodes the dense P and the Hungarian assignment do not fit;
# the kissing variants are then scored by top-1 accuracy (argmax of each
# row of P, computed in chunks), like the large runs in Result-all.
HUNGARIAN_MAX_N = 30_000


# ============================================================================
# The benchmark protocol (--settings protocol, the default of run-tests.py):
# one run per method, dataset and noise level, with settings that depend
# only on the size class of the graph, never on the dataset itself.
#   mu        0.5 small, 0.1 medium / large (Result-all's convention), the
#             same for Fugal and the kissing variants
#   kissing   beta=10 (25 above 30k nodes, as in all large Result-all runs),
#             row_penalty=10, col_penalty=200, learning_rate=1e-2 (README)
#             for every variant; embed_dim from recommend_embed_dim (the
#             authors' rule), except dense-LA, whose kernel approximation
#             needs its default 500; chunk automatic (dense-chunked: its
#             default 1024; chunk changes memory and speed, not the result)
#   steps     exactly PROTOCOL_STEPS[size] for every kissing variant, no
#             early stopping.  Whether a run had converged is measured
#             afterwards from its loss history (converged_at, final_gain)
#   seed      0 for the kissing variants (Fugal is deterministic)
# The "tuned" settings (the tables above, per dataset from Result-all) are
# still used by sweep-hyperparameters.py and with --settings tuned.
# ============================================================================

PROTOCOL_MU = {"small": 0.5, "medium": 0.1, "large": 0.1}
PROTOCOL_STEPS = {"small": 10_000, "medium": 10_000, "large": 2_000}
PROTOCOL_SEED = 0
CONVERGENCE_WINDOW = 500   # steps
CONVERGENCE_TOL = 1e-4     # relative to the final best loss


def recommend_embed_dim(n):
    """recommend_embed_dim of kissing-fast.py / dense-chunked-fast.py (it
    only uses n), which picks embed_dim for those two when it is None."""
    return 32 if n <= 500 else 100 if n <= 5000 else 200 if n <= 50000 else 300


def protocol_hyperparameters(algo, dataset, target):
    n = num_nodes(dataset, target)
    size = size_class(n)
    params = dict(mu=PROTOCOL_MU[size])
    if algo == "fugal":
        return {**params, **ALGORITHMS["fugal"]["default"]}
    params.update(COMMON)
    if size == "large":
        params["beta"] = 25
    if algo == "dense-LA":
        params.update(embed_dim=500, kind="relu_squared", favor_features=200, eval_every=None)
    elif algo == "dense-chunked":
        params.update(embed_dim=recommend_embed_dim(n), chunk=1024)
    else:
        params.update(embed_dim=None, chunk=None)   # picked by recommend_embed_dim / auto_chunk_size
    params["max_iter"] = PROTOCOL_STEPS[size]
    return params


def convergence(history, window=CONVERGENCE_WINDOW, tol=CONVERGENCE_TOL):
    """(converged_at, final_gain) of a loss history.  converged_at is the
    first multiple of `window` steps from which the best loss stays within
    tol (relative) of the final best loss; final_gain is the relative
    improvement of the best loss over the last `window` steps.  A run whose
    converged_at is close to its number of steps, or whose final_gain is
    above tol, was still learning when it stopped."""
    import numpy as np
    best = np.minimum.accumulate(np.asarray(history, dtype=float))
    scale = max(abs(best[-1]), 1.0)
    converged_at = next((k for k in range(0, len(best), window)
                         if best[k] - best[-1] <= tol * scale), len(best))
    final_gain = (best[-window - 1] - best[-1]) / scale if len(best) > window else float("nan")
    return converged_at, final_gain


def hyperparameters(algo, dataset, target, max_iter=None, overrides=None, settings="tuned"):
    if settings == "protocol":
        params = protocol_hyperparameters(algo, dataset, target)
        if max_iter is not None and "max_iter" in params:
            params["max_iter"] = max_iter
        params.update(overrides or {})
        return params
    spec = ALGORITHMS[algo]
    ds = DATASETS[dataset]
    hp = ds.get("hp", dataset)
    params = dict(mu=ds["mu"])
    if algo != "fugal":
        params.update(COMMON)
    params.update(spec["default"])
    if hp not in spec:
        params.update(spec.get(size_class(num_nodes(dataset, target)), {}))
    params.update(spec.get(hp, {}))
    params.update(spec.get(f"{hp}/{target}", {}))
    if max_iter is not None and "max_iter" in params:
        params["max_iter"] = max_iter
    params.update(overrides or {})
    return params


# ============================================================================
# Worker: one (algorithm, dataset, target) run in this process
# ============================================================================

def load_script(filename):
    """Import one of the algorithm scripts (their names contain '-')."""
    name = filename[:-3].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, FLY_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_paths(dataset, target):
    ds = DATASETS[dataset]
    folder = DATA_DIR / ds["dir"]   # an absolute "dir" replaces DATA_DIR
    return folder / ds["query"], folder / ds["targets"][target]


def load_graphs(dataset, target):
    """Gq, Gt, n, ids: n is counted from the files, ids[i] is the id node i
    has in the files (helpers/graph_io.py)."""
    from helpers.graph_io import read_graph_pair
    return read_graph_pair(*pair_paths(dataset, target))


@functools.lru_cache(maxsize=None)
def num_nodes(dataset, target):
    from helpers.graph_io import count_nodes
    return count_nodes(*pair_paths(dataset, target))


def size_class(n):
    return "small" if n <= 10_000 else "medium" if n <= 30_000 else "large"


def peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux
    return peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024


class MpsMemorySampler:
    """MPS has no peak-memory counter, so poll the driver allocation."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import torch
        while not self._stop.is_set():
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def top1_assignment(V, W, beta, col_bias=None, chunk=1024):
    """Argmax of every row of P = softmax(beta * normalize(V) normalize(W)^T
    + col_bias) without building P (the softmax does not change the argmax)."""
    import numpy as np
    import torch

    with torch.no_grad():
        Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cols = []
        for start in range(0, V.shape[0], chunk):
            scores = beta * (Vn[start:start + chunk] @ Wn.T)
            if col_bias is not None:
                scores = scores + col_bias
            cols.append(scores.argmax(dim=1).cpu())
    return np.arange(V.shape[0]), torch.cat(cols).numpy()


# Position of the loss history in what each train_with_adam returns.
HISTORY_INDEX = {"dense-LA": 3, "dense-chunked": 2, "dense-chunked-fast": 2, "kissing-fast": 3}


def run_algorithm(algo, Gq, Gt, params, use_gpu=True, restarts=1):
    """Returns (row_ind, col_ind, device, time_s, peak_rss_mb, eval, training).

    Time and memory are taken when the algorithm returns; for the kissing
    variants the assignment below is evaluation and not measured.  eval is
    "hungarian" (one-to-one assignment) or "top1" (row argmax, large n);
    training has the number of steps of the kept run (Fugal: niter) and
    its convergence (see convergence()).
    With restarts > 1 a kissing variant is trained that many times from new
    random starts and the run with the lowest loss is kept; the time
    includes all of them.
    """
    import torch

    if algo == "fugal":
        import fugal_original
        start = time.perf_counter()
        pairs = fugal_original.fugal(Gq, Gt, params["mu"], params["niter"])
        elapsed = time.perf_counter() - start
        rss = peak_rss_mb()
        row_ind, col_ind = zip(*pairs)
        training = dict(steps=params["niter"], converged_at="", final_gain="")
        return row_ind, col_ind, "cpu", elapsed, rss, "hungarian", training

    module = load_script(ALGORITHMS[algo]["script"])
    # Only affects what is printed, so it is not part of params (and of the
    # settings that identify a run in the sweep results).
    extra = {}
    if "log_every" in inspect.signature(module.train_with_adam).parameters:
        extra["log_every"] = LOG_EVERY
    start = time.perf_counter()
    out, best_loss = None, float("inf")
    for restart in range(restarts):
        candidate = module.train_with_adam(Gq, Gt, use_GPU=use_gpu, **params, **extra)
        loss = min(candidate[HISTORY_INDEX[algo]])
        if restarts > 1:
            print(f"[run-tests] restart {restart + 1}/{restarts}: lowest loss {loss:.4f}", flush=True)
        if out is None or loss < best_loss:
            out, best_loss = candidate, loss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    rss = peak_rss_mb()
    history = out[HISTORY_INDEX[algo]]
    converged_at, final_gain = convergence(history)
    training = dict(steps=len(history), converged_at=converged_at,
                    final_gain=f"{final_gain:.2e}")

    beta = params["beta"]
    if algo == "dense-LA":
        P = out[0]
    else:
        if algo == "kissing-fast":
            V, W, col_bias, _, chunk = out
        else:
            (V, W, _), col_bias, chunk = out, None, params["chunk"] or 1024
        if V.shape[0] > HUNGARIAN_MAX_N:
            row_ind, col_ind = top1_assignment(V, W, beta, col_bias, chunk)
            return row_ind, col_ind, V.device.type, elapsed, rss, "top1", training
        P = (module.reconstruct_P(V, W, beta=beta, chunk=chunk, col_bias=col_bias)
             if col_bias is not None else module.reconstruct_P(V, W, beta=beta, chunk=chunk))

    import scipy.optimize
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(P.cpu().numpy(), maximize=True)
    return row_ind, col_ind, P.device.type, elapsed, rss, "hungarian", training


def accuracy(row_ind, col_ind, dataset, target, n, ids):
    import numpy as np

    row_ind, col_ind = np.asarray(row_ind), np.asarray(col_ind)
    gt_file = DATASETS[dataset].get("ground_truth")
    if isinstance(gt_file, dict):
        gt_file = gt_file.get(target)
    if gt_file is None:
        return float(np.sum(row_ind == col_ind) / n)
    gt_path = DATA_DIR / DATASETS[dataset]["dir"] / gt_file
    pairs = (np.load(gt_path) if gt_path.suffix == ".npy" else np.loadtxt(gt_path)).astype(np.int64)
    index = {v: i for i, v in enumerate(ids)}   # ground truth uses the ids of the files
    predicted = dict(zip(row_ind.tolist(), col_ind.tolist()))
    correct = sum(predicted.get(index.get(int(q))) == index.get(int(t)) for q, t in pairs)
    return correct / len(pairs)


def worker(algo, dataset, target, max_iter, overrides, cpu=False, seed=None, restarts=1,
           settings="tuned"):
    sys.path.insert(0, str(FLY_DIR))  # for `from helpers...` inside the scripts
    import random
    import numpy as np
    import torch

    if seed is not None:   # the kissing variants start from random V, W
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    params = hyperparameters(algo, dataset, target, max_iter, overrides, settings)
    Gq, Gt, n, ids = load_graphs(dataset, target)
    print(f"[run-tests] {algo} on {dataset}/{target}  n={n}  "
          f"|Eq|={Gq.number_of_edges()} |Et|={Gt.number_of_edges()}  {params}", flush=True)

    use_mps = not cpu and not torch.cuda.is_available() and torch.backends.mps.is_available()
    if use_mps:
        with MpsMemorySampler() as sampler:
            row_ind, col_ind, device, elapsed, rss, method, training = run_algorithm(
                algo, Gq, Gt, params, restarts=restarts)
        peak_gpu = sampler.peak
    else:
        row_ind, col_ind, device, elapsed, rss, method, training = run_algorithm(
            algo, Gq, Gt, params, use_gpu=not cpu, restarts=restarts)
        peak_gpu = torch.cuda.max_memory_reserved() if device == "cuda" else 0

    result = {
        "algorithm": algo,
        "dataset": dataset,
        "target": target,
        "n": n,
        "device": device,
        "time_s": round(elapsed, 2),
        "peak_rss_mb": round(rss, 1),
        "peak_gpu_mb": round(peak_gpu / (1024 ** 2), 1) if device != "cpu" else "",
        "accuracy": round(accuracy(row_ind, col_ind, dataset, target, n, ids), 4),
        "params": json.dumps(params),
        "eval": method,
        "seed": "" if seed is None else seed,
        "restarts": restarts,
        **training,
        "settings": settings,
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)


# ============================================================================
# Driver
# ============================================================================

FIELDS = ["algorithm", "dataset", "target", "n", "device", "time_s",
          "peak_rss_mb", "peak_gpu_mb", "accuracy", "params", "eval", "seed", "restarts",
          "steps", "converged_at", "final_gain", "settings"]


def open_csv(path, fields):
    """Open a results CSV for appending.  A file written by an older version
    with fewer columns is first rewritten with the new header."""
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows, old_fields = list(reader), reader.fieldnames or []
        if old_fields and old_fields != fields:
            if not set(old_fields) <= set(fields):
                sys.exit(f"{path} has columns {old_fields}, expected {fields}")
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(path)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if f.tell() == 0:
        writer.writeheader()
    return f, writer


def run_key(algo, dataset, target, seed, max_iter, settings, restarts):
    """What identifies a run in results.csv: the combination and every
    setting it ran with, so a run with changed settings is not skipped."""
    params = hyperparameters(algo, dataset, target, max_iter, None, settings)
    return (algo, dataset, target, "" if seed is None else str(seed), str(restarts), settings,
            json.dumps(params, sort_keys=True))


def finished_runs(csv_path):
    """run_key of every row already in csv_path."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(r["algorithm"], r["dataset"], r["target"], r.get("seed") or "",
                 r.get("restarts") or "1", r.get("settings") or "tuned",
                 json.dumps(json.loads(r["params"]), sort_keys=True))
                for r in csv.DictReader(f)}


def too_large(algo, dataset, target):
    return num_nodes(dataset, target) > MAX_N.get(algo, float("inf"))


def run_one(algo, dataset, target, max_iter, log_dir, overrides=None, gpu=None, cpu=False,
            seed=None, restarts=1, settings="tuned"):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", algo, dataset, target]
    if max_iter is not None:
        cmd += ["--max-iter", str(max_iter)]
    name = f"{algo}__{dataset}__{target}"
    if overrides:
        cmd += ["--overrides", json.dumps(overrides)]
        name += "__" + "_".join(f"{k}={v}" for k, v in overrides.items())
    if cpu:
        cmd.append("--cpu")
        name += "__cpu"
    if seed is not None:
        cmd += ["--seed", str(seed)]
        name += f"__seed{seed}"
    if restarts > 1:
        cmd += ["--restarts", str(restarts)]
        name += f"__restarts{restarts}"
    if settings != "tuned":
        cmd += ["--settings", settings]
        name += f"__{settings}"
    env, prefix = None, "    "
    if gpu is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        prefix = f"    [gpu {gpu}] "
    log_path = log_dir / f"{name}.log"
    result = None
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            log.write(line)
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line[len(RESULT_PREFIX):])
            else:
                print(prefix + line, end="", flush=True)
        proc.wait()
    if proc.returncode != 0 or result is None:
        print(f"  FAILED (exit code {proc.returncode}), see {log_path}")
        return None
    return result


def run_jobs(jobs, gpus, run):
    """Calls run(job, gpu) for every job and yields (job, result) as runs
    finish.  With several GPUs, one run per GPU at a time, in parallel."""
    if not gpus or len(gpus) == 1:
        gpu = gpus[0] if gpus else None
        for job in jobs:
            yield job, run(job, gpu)
        return

    free = queue.Queue()
    for gpu in gpus:
        free.put(gpu)

    def task(job):
        gpu = free.get()
        try:
            return run(job, gpu)
        finally:
            free.put(gpu)

    pool = ThreadPoolExecutor(len(gpus))
    try:
        futures = {pool.submit(task, job): job for job in jobs}
        for future in as_completed(futures):
            yield futures[future], future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def expand_datasets(names):
    """Dataset names with the fw-small / fw-medium / fw-large groups expanded."""
    return [d for name in names for d in DATASET_GROUPS.get(name, [name])]


def print_summary(results, header=("algorithm", "dataset", "target", "device", "time_s",
                                    "peak_rss_mb", "peak_gpu_mb", "accuracy", "steps",
                                    "converged_at")):
    rows = [[str(r[h]) for h in header] for r in results]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def print_seed_summary(results):
    """Mean, spread and worst accuracy over the seeds of each setting."""
    import statistics
    groups = {}
    for r in results:
        groups.setdefault((r["algorithm"], r["dataset"], r["target"]), []).append(r)
    rows = []
    for (algo, dataset, target), runs in groups.items():
        accs = [float(r["accuracy"]) for r in runs]
        rows.append({"algorithm": algo, "dataset": dataset, "target": target, "runs": len(runs),
                     "acc_mean": f"{statistics.mean(accs):.4f}",
                     "acc_std": f"{statistics.pstdev(accs):.4f}",
                     "acc_min": f"{min(accs):.4f}",
                     "time_mean": f"{statistics.mean(float(r['time_s']) for r in runs):.2f}"})
    print("\nOver the seeds:")
    print_summary(rows, header=tuple(rows[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--algos", nargs="+", choices=list(ALGORITHMS), default=list(ALGORITHMS))
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, *DATASET_GROUPS],
                        default=DEFAULT_DATASETS,
                        help="dataset names, or fw-small / fw-medium / fw-large for all "
                             "generated datasets of that size")
    parser.add_argument("--targets", nargs="+",
                        help="noise levels to run, e.g. 5 10 or 99 95 (default: all)")
    parser.add_argument("--max-iter", type=int,
                        help="override max_iter of the kissing variants (for smoke tests)")
    parser.add_argument("--gpus", nargs="+", type=int, metavar="ID",
                        help="CUDA devices to use, one run per device in parallel (e.g. --gpus 0 1)")
    parser.add_argument("--cpu", action="store_true", help="run the kissing variants on the CPU")
    parser.add_argument("--no-size-limit", action="store_true",
                        help=f"also run Fugal / dense-LA on graphs above {MAX_N}")
    parser.add_argument("--seeds", nargs="+", type=int, metavar="SEED",
                        help="run every kissing setting once per seed (e.g. --seeds 0 1 2 3 4); "
                             "Fugal is deterministic and runs once")
    parser.add_argument("--restarts", type=int, default=1,
                        help="train the kissing variants this many times per run and keep "
                             "the lowest loss (time includes all restarts)")
    parser.add_argument("--settings", choices=["protocol", "tuned"],
                        help="protocol (default): the same rules for every dataset, see "
                             "protocol_hyperparameters; tuned: the per-dataset Result-all "
                             "settings")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--rerun", action="store_true",
                        help="also run the combinations that already have a row in "
                             "<out>/results.csv (by default they are skipped)")
    parser.add_argument("--worker", nargs=3, metavar=("ALGO", "DATASET", "TARGET"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--overrides", type=json.loads, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(*args.worker, args.max_iter, args.overrides, args.cpu, args.seed, args.restarts,
               args.settings or "tuned")
        return
    settings = args.settings or "protocol"
    seeds = args.seeds or ([PROTOCOL_SEED] if settings == "protocol" else [None])

    log_dir = args.out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "results.csv"

    jobs = []
    for dataset in expand_datasets(args.datasets):
        for target in DATASETS[dataset]["targets"]:
            if args.targets is not None and target not in args.targets:
                continue
            for algo in args.algos:
                if too_large(algo, dataset, target) and not args.no_size_limit:
                    print(f"Skipping {algo} on {dataset}/{target}: n={num_nodes(dataset, target)} "
                          f"> {MAX_N[algo]} (see --no-size-limit)")
                    continue
                for seed in ([None] if algo == "fugal" else seeds):
                    jobs.append((algo, dataset, target, seed))

    # Resume: skip every run that already has a row with the same settings.
    if not args.rerun:
        done = finished_runs(csv_path)
        todo = [job for job in jobs if run_key(*job, args.max_iter, settings,
                                               1 if job[0] == "fugal" else args.restarts) not in done]
        if len(todo) < len(jobs):
            print(f"Skipping {len(jobs) - len(todo)} of {len(jobs)} runs already in {csv_path} "
                  f"(see --rerun)")
        jobs = todo

    def label(job):
        algo, dataset, target, seed = job
        return f"{algo} on {dataset}/{target}" + (f" seed {seed}" if seed is not None else "")

    def run(job, gpu):
        algo, dataset, target, seed = job
        print(f"[{jobs.index(job) + 1}/{len(jobs)}] {label(job)}"
              + (f"  (gpu {gpu})" if gpu is not None else ""), flush=True)
        return run_one(algo, dataset, target, args.max_iter, log_dir, gpu=gpu, cpu=args.cpu,
                       seed=seed, restarts=1 if algo == "fugal" else args.restarts,
                       settings=settings)

    results, failed = [], []
    f, writer = open_csv(csv_path, FIELDS)
    with f:
        for job, result in run_jobs(jobs, args.gpus, run):
            if result is None:
                failed.append(label(job))
                continue
            writer.writerow(result)
            f.flush()
            results.append(result)
            print(f"  done: {label(job)}  time={result['time_s']}s  peak_rss={result['peak_rss_mb']}MB  "
                  f"peak_gpu={result['peak_gpu_mb'] or 'n/a'}MB  acc={result['accuracy']} "
                  f"({result['eval']})  steps={result['steps']}  converged_at={result['converged_at']}")

    if results:
        if args.seeds:
            print_summary(results, header=("algorithm", "dataset", "target", "seed", "device",
                                           "time_s", "peak_rss_mb", "peak_gpu_mb", "accuracy",
                                           "steps", "converged_at"))
            print_seed_summary(results)
        else:
            print_summary(results)
        print(f"\nResults appended to {csv_path}")
    if failed:
        print("\nFailed runs:")
        for line in failed:
            print("  " + line)
        sys.exit(1)


if __name__ == "__main__":
    main()
