"""generate-framework-datasets.py

Builds noisy versions of every base graph in data with the default noise
of Framework_GraphAlignment: remove_e(edges, p, no_disc=True), which removes
each edge with probability p unless that would leave one of its nodes
without edges.  p = 5, 10, 15, 20, 25 %, node ids are not shuffled (the
ground truth is the identity, as in the other datasets).

    data/framework_noise/<size>/<name>/<name>-0.txt     the base graph
    data/framework_noise/<size>/<name>/<name>-5.txt     5% noise
    ...
    data/framework_noise/datasets.json                  n, m, size, files

size is small (n <= 10,000), medium (10,000 < n <= 30,000) or large
(n > 30,000).  The base graph is the 0%-noise file of each dataset, made
undirected without self-loops or duplicate edges, with the nodes that have
edges renumbered 0..n-1 in the order of their ids.  Each (dataset, p) has
its own fixed seed, so rerunning gives the same files.

    .venv/bin/python generate-framework-datasets.py              # all datasets
    .venv/bin/python generate-framework-datasets.py highschool voles
"""

import argparse
import importlib.util
import json
import sys
import zlib
from pathlib import Path

import networkx as nx
import numpy as np

import framework_generate as fg

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "framework_noise"
NOISE = [5, 10, 15, 20, 25]

# name -> 0%-noise file.  ca-CondMat is left out: it is the same graph as
# condmat-100.txt.  idsenfr (knowledge graphs) and ACM-DBLP's DBLP are two
# different graphs with an alignment, not a base graph, so DBLP is only used
# as a graph of its own here.
SOURCES = {
    "ca-netscience": "real/ca-netscience/ca-netscience.txt",
    "bio-celegans": "real/bio-celegans/bio-celegans.txt",
    "inf-euroroad": "real/inf-euroroad/inf-euroroad.txt",
    "arenas": "real/arenas/in-arenas.txt",
    "highschool": "real_noise/contacts-prox-high-school-2013/contacts-prox-high-school-2013_100.txt",
    "voles": "real_noise/mamalia-voles-plj-trapping/mammalia-voles-plj-trapping_100.txt",
    "MultiMagna": "real_noise/MultiMagna/yeast0_Y2H1.txt",
    "ACM": "real_noise/ACM-DBLP/ACM.txt",
    "DBLP": "real_noise/ACM-DBLP/DBLP.txt",
    "2cubes": "real_noise/2cubes_sphere/2cubes-100.txt",
    "condmat": "real_noise/condmat/condmat-100.txt",
    "deezer": "real_noise/deezer_europe/deezer-100.txt",
    "email-enron": "real_noise/email-enron/email-enron-100.txt",
    "soc-brightkite": "real_noise/soc-brightkite/soc-brightkite-100.txt",
    "slashdot": "real_noise/slashdot/slashdot-100.txt",
    "buzznet": "real_noise/buzznet/buzznet-100.txt",
    "ippds": "real_noise/ippds/ippds-100.txt",
    "dblp110k": "real_noise/livemocha/dblp110k/dblp-n110000-100.txt",
    "soc_sign": "real_noise/soc_sign/soc_sign-100.txt",
    "LiveMocha": "real_noise/livemocha/soc-LiveMocha-100.txt",
    "amazon": "real_noise/amazon/amazon-100.txt",
}

_spec = importlib.util.spec_from_file_location("make_noisy_graphs", ROOT / "make-noisy-graphs.py")
mng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mng)


def size_class(n):
    return "small" if n <= 10_000 else "medium" if n <= 30_000 else "large"


def base_graph(path):
    G = nx.Graph(mng.read_raw(path))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    ids = {v: i for i, v in enumerate(sorted(G.nodes()))}
    return np.array(sorted(tuple(sorted((ids[u], ids[v]))) for u, v in G.edges()), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("names", nargs="*", metavar="NAME",
                        help=f"datasets to build (default: all): {' '.join(SOURCES)}")
    parser.add_argument("--out", type=Path, default=OUT, help=f"output folder (default: {OUT})")
    args = parser.parse_args()
    out = args.out
    unknown = set(args.names) - set(SOURCES)
    if unknown:
        parser.error(f"unknown dataset(s): {' '.join(sorted(unknown))}")

    index_path = out / "datasets.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for name in args.names or SOURCES:
        edges = base_graph(DATA / SOURCES[name])
        n, m = int(edges.max()) + 1, len(edges)
        size = size_class(n)
        folder = out / size / name
        folder.mkdir(parents=True, exist_ok=True)
        mng.write_edges(folder / f"{name}-0.txt", edges)
        print(f"{name}: n={n}, m={m} -> {size}/", flush=True)

        for p in NOISE:
            np.random.seed(zlib.crc32(f"{name}-{p}".encode()))
            noisy = fg.remove_e(edges, p / 100, no_disc=True)
            mng.write_edges(folder / f"{name}-{p}.txt", noisy)
            print(f"  {name}-{p}.txt: {len(noisy)} edges ({1 - len(noisy) / m:.2%} removed)", flush=True)

        index[name] = {"size": size, "dir": f"{size}/{name}", "n": n, "m": m,
                       "source": SOURCES[name], "noise": NOISE}
        index_path.write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
