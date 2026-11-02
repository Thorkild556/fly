"""sweep-hyperparameters.py

Hyperparameter sweep on top of run-tests.py: runs a few hyperparameter
settings per algorithm and ranks them by accuracy, so you can see which
settings matter and how that changes with the size of the graph.  Each run is measured exactly like in
run-tests.py (own subprocess, time, peak memory, Hungarian accuracy).

    .venv/bin/python sweep-hyperparameters.py                    # ACM 95 (n=9872), then high-school 95 (n=327)
    .venv/bin/python sweep-hyperparameters.py --datasets highschool   # small dataset only
    .venv/bin/python sweep-hyperparameters.py --algos dense-chunked-fast --datasets voles --targets 90
    .venv/bin/python sweep-hyperparameters.py --algos kissing-fast --grid beta=10,25 embed_dim=32,100,200

The default settings are in SMALL_SWEEPS and LARGE_SWEEPS below: the same
one-change-at-a-time combinations on both sizes (embed_dim scaled with n),
so the results can be compared.  The large dataset runs first, so a run
that does not fit in RAM fails before any time is spent on the small one.
--grid replaces them with every combination of the values listed, for all
selected algorithms; keys an algorithm does not take are dropped for it
(e.g. beta for Fugal).  Settings that are not swept keep the per-pair values
from run-tests.py.  A value of None means
"auto" for embed_dim / chunk.

All sweeps are collected in results/sweep.csv.  A run that is already in
that file with exactly the same settings is skipped, so after stopping a
sweep (Ctrl+C) the same command continues where it left off; the run that
was interrupted starts over.  Delete sweep.csv to start from scratch.  The
ranking at the end includes the earlier runs.  The output of each run is
kept in results/logs/.
"""

import argparse
import ast
import csv
import importlib.util
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_tests", ROOT / "run-tests.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


# Small datasets: one change at a time from the run-tests.py settings.
# Each run takes seconds on high-school 95 (dense-LA ~50 s).
_KISSING_SMALL = [
    dict(beta=10, embed_dim=32),
    dict(beta=25, embed_dim=32),
    dict(beta=10, embed_dim=128),
]
SMALL_SWEEPS = {
    "fugal": [dict(mu=0.1), dict(mu=1.0)],
    "dense-LA": [dict(learning_rate=1e-3), dict(learning_rate=1e-2)],
    "dense-chunked": _KISSING_SMALL,
    "dense-chunked-fast": _KISSING_SMALL,
    "kissing-fast": _KISSING_SMALL,
}

# Large datasets: the same combinations with embed_dim scaled up.
# Times measured on ACM 95 with MPS: Fugal ~50 min per run (peak RAM ~8 GB),
# kissing variants ~1.4 s/iteration at embed_dim=200 and ~2 s at 1000, so
# max_iter is cut to 500: accuracies are for comparing settings, not final.
# dense-chunked is left out: it has the same loss as dense-chunked-fast and
# is the slowest of the variants.
_KISSING_LARGE = [
    dict(beta=10, embed_dim=200, max_iter=500),
    dict(beta=25, embed_dim=200, max_iter=500),
    dict(beta=10, embed_dim=1000, max_iter=500),
]
# Follow-up for kissing-fast, the best kissing variant on ACM: the 500-step
# runs were still improving, so these check how much longer training, a
# larger embed_dim and beta 25 help.  ~2 s/step at embed_dim=1000 on MPS,
# so ~1.7 h each (embed_dim=2000 not timed, probably 2.5-3 h).
_KISSING_LARGE_LONG = [
    dict(beta=10, embed_dim=1000, max_iter=3000),
    dict(beta=10, embed_dim=2000, max_iter=3000),
    dict(beta=25, embed_dim=1000, max_iter=3000),
]
LARGE_SWEEPS = {
    "fugal": [dict(mu=0.1), dict(mu=1.0)],
    "dense-LA": [dict(learning_rate=1e-3, max_iter=2000), dict(learning_rate=1e-2, max_iter=2000)],
    "dense-chunked-fast": _KISSING_LARGE,
    "kissing-fast": _KISSING_LARGE + _KISSING_LARGE_LONG,
}
LARGE_DATASETS = {"ACM", "ACM-DBLP", "ca-CondMat", "email-enron", "soc-brightkite"}

DEFAULT_DATASETS = ["ACM", "highschool"]     # large first
DEFAULT_TARGETS = {"ACM": ["95"], "highschool": ["95"]}


def accepted_params(algo):
    """Names of the hyperparameters an algorithm takes."""
    if algo == "fugal":
        return {"mu", "niter"}
    # Read train_with_adam's signature from the source without importing torch.
    source = (rt.FLY_DIR / rt.ALGORITHMS[algo]["script"]).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "train_with_adam":
            return {a.arg for a in node.args.args + node.args.kwonlyargs}
    raise ValueError(f"no train_with_adam in {rt.ALGORITHMS[algo]['script']}")


def combinations(algo, grid):
    """All override dicts for one algorithm; [{}] means one run with defaults.

    `grid` is either {key: [values]} (every combination) or a list of
    override dicts (run as given).
    """
    accepted = accepted_params(algo)
    if isinstance(grid, list):
        combos = [{k: v for k, v in c.items() if k in accepted} for c in grid]
        return [dict(c) for c in {json.dumps(c): c for c in combos}.values()]
    grid = {k: v for k, v in grid.items() if k in accepted}
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def grid_for(algo, dataset, target, custom_grid):
    if custom_grid is not None:
        return custom_grid
    if dataset in LARGE_DATASETS or rt.size_class(rt.num_nodes(dataset, target)) != "small":
        return LARGE_SWEEPS.get(algo)       # None: algorithm skipped on large data
    return SMALL_SWEEPS[algo]


def parse_grid(items):
    """['beta=5,10', 'kind=elu'] -> {'beta': [5, 10], 'kind': ['elu']}"""
    def value(text):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    grid = {}
    for item in items:
        key, _, values = item.partition("=")
        if not values:
            raise SystemExit(f"--grid expects KEY=V1,V2,..., got {item!r}")
        grid[key] = [value(v) for v in values.split(",")]
    return grid


# Hyperparameters that get their own column in sweep.csv (all of them are
# also in the params column).
PARAM_COLUMNS = ["mu", "beta", "embed_dim", "learning_rate", "max_iter", "niter",
                 "kind", "chunk"]
CSV_FIELDS = rt.FIELDS + PARAM_COLUMNS


def run_key(algo, dataset, target, params, seed=None, restarts=1):
    """What identifies a run: rows written before seeds / restarts existed
    have them empty and count as unseeded with one start."""
    seed = "" if seed in (None, "") else str(seed)
    return (algo, dataset, str(target), json.dumps(params, sort_keys=True), seed, str(restarts or 1))


def load_sweep_csv(csv_path):
    """Rows of sweep.csv keyed by run_key.  On first use, earlier
    sweep-<time>.csv files are imported so their runs are not repeated."""
    rows = {}
    sources = [csv_path] if csv_path.exists() else sorted(csv_path.parent.glob("sweep-*.csv"))
    for path in sources:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                params = json.loads(row["params"])
                rows[run_key(row["algorithm"], row["dataset"], row["target"], params,
                             row.get("seed"), row.get("restarts"))] = row
    if not csv_path.exists() and rows:
        print(f"Imported {len(rows)} earlier runs from {', '.join(p.name for p in sources)}")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows.values():
                writer.writerow(with_param_columns(row))
    return rows


def with_param_columns(row):
    params = json.loads(row["params"])
    return {**row, **{k: params.get(k, "") for k in PARAM_COLUMNS}}


def print_ranking(results, keys):
    """One table per (algorithm, dataset, target), best accuracy first."""
    header = ["algorithm", "dataset", "target", *keys, "accuracy", "time_s", "peak_rss_mb"]
    groups = {}
    for r in results:
        groups.setdefault((r["algorithm"], r["dataset"], r["target"]), []).append(r)
    for group in groups.values():
        rt.print_summary(sorted(group, key=lambda r: -float(r["accuracy"])), header)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--algos", nargs="+", choices=list(rt.ALGORITHMS), default=list(rt.ALGORITHMS))
    parser.add_argument("--datasets", nargs="+", choices=[*rt.DATASETS, *rt.DATASET_GROUPS],
                        default=DEFAULT_DATASETS,
                        help=f"default: {' '.join(DEFAULT_DATASETS)} (in this order); "
                             "fw-small / fw-medium / fw-large for all generated datasets of that size")
    parser.add_argument("--targets", nargs="+",
                        help="noise levels to run (default: 95 for ACM and high-school, else all)")
    parser.add_argument("--grid", nargs="+", metavar="KEY=V1,V2",
                        help="custom grid (every combination), replaces the default settings")
    parser.add_argument("--max-iter", type=int,
                        help="override max_iter of the kissing variants (for smoke tests)")
    parser.add_argument("--gpus", nargs="+", type=int, metavar="ID",
                        help="CUDA devices to use, one run per device in parallel (e.g. --gpus 0 1)")
    parser.add_argument("--no-size-limit", action="store_true",
                        help=f"also run Fugal / dense-LA on graphs above {rt.MAX_N}")
    parser.add_argument("--seeds", nargs="+", type=int, metavar="SEED",
                        help="run every kissing setting once per seed; Fugal runs once")
    parser.add_argument("--restarts", type=int, default=1,
                        help="train the kissing variants this many times per run, keep the lowest loss")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    custom_grid = parse_grid(args.grid) if args.grid else None

    jobs = []
    for dataset in rt.expand_datasets(args.datasets):
        targets = args.targets or DEFAULT_TARGETS.get(dataset) or list(rt.DATASETS[dataset]["targets"])
        for target in rt.DATASETS[dataset]["targets"]:
            if target not in targets:
                continue
            for algo in args.algos:
                if rt.too_large(algo, dataset, target) and not args.no_size_limit:
                    print(f"Skipping {algo} on {dataset}/{target}: "
                          f"n={rt.num_nodes(dataset, target)} > {rt.MAX_N[algo]}")
                    continue
                grid = grid_for(algo, dataset, target, custom_grid)
                if grid is None:
                    print(f"Skipping {algo} on {dataset}: no entry in LARGE_SWEEPS")
                    continue
                seeds = [None] if algo == "fugal" else args.seeds or [None]
                jobs += [(algo, dataset, target, o, seed)
                         for o in combinations(algo, grid) for seed in seeds]
    keys = sorted({k for *_, overrides, _ in jobs for k in overrides}) + (["seed"] if args.seeds else [])

    log_dir = args.out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "sweep.csv"
    done = load_sweep_csv(csv_path)

    # Previous results for these runs, and the runs still to do.
    results, todo = [], []
    for algo, dataset, target, overrides, seed in jobs:
        params = rt.hyperparameters(algo, dataset, target, args.max_iter, overrides)
        restarts = 1 if algo == "fugal" else args.restarts
        key = run_key(algo, dataset, target, params, seed, restarts)
        if key in done:
            results.append({**done[key], **{k: params.get(k, "") for k in keys if k != "seed"}})
        else:
            todo.append((algo, dataset, target, overrides, seed))
    print(f"Sweeping {', '.join(keys) or 'nothing'}: {len(jobs)} runs, "
          f"{len(jobs) - len(todo)} already in {csv_path.name}, {len(todo)} to do")

    def label(job):
        algo, dataset, target, overrides, seed = job
        return (f"{algo} on {dataset}/{target}  " + " ".join(f"{k}={v}" for k, v in overrides.items())
                + (f"  seed={seed}" if seed is not None else ""))

    def run(job, gpu):
        algo, dataset, target, overrides, seed = job
        print(f"[{todo.index(job) + 1}/{len(todo)}] {label(job)}"
              + (f"  (gpu {gpu})" if gpu is not None else ""), flush=True)
        return rt.run_one(algo, dataset, target, args.max_iter, log_dir, overrides, gpu=gpu,
                          seed=seed, restarts=1 if algo == "fugal" else args.restarts)

    failed = []
    f, writer = rt.open_csv(csv_path, CSV_FIELDS)
    with f:
        try:
            for job, result in rt.run_jobs(todo, args.gpus, run):
                if result is None:
                    failed.append(label(job))
                    continue
                writer.writerow(with_param_columns(result))
                f.flush()
                params = json.loads(result["params"])
                results.append({**result, **{k: params.get(k, "") for k in keys if k != "seed"}})
                print(f"  done: {label(job)}  time={result['time_s']}s  "
                      f"peak_rss={result['peak_rss_mb']}MB  acc={result['accuracy']} ({result['eval']})")
        except KeyboardInterrupt:
            print(f"\nStopped.  Finished runs are saved in {csv_path}; "
                  f"run the same command again to continue.")
            sys.exit(130)

    if results:
        print_ranking(results, keys)
        print(f"\nAll runs are in {csv_path}")
    if failed:
        print("\nFailed runs:")
        for line in failed:
            print("  " + line)
        sys.exit(1)


if __name__ == "__main__":
    main()
