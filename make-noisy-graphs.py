"""make-noisy-graphs.py

Turns a raw graph (.mtx from networkrepository.com, or a SNAP / plain edge
list, optionally .gz) into a dataset in the format of data/real_noise:

    <out>/<name>/<name>-100.txt    the cleaned graph, "u v" per line, ids 0..n-1
    <out>/<name>/<name>-95.txt     the same graph with 5% of the edges removed
    ...                            (one file per --keep level)

Cleaning is what was done for condmat (checked against condmat.raw.txt.gz):
edges are made undirected, self-loops and duplicates are dropped, only the
largest connected component is kept and its nodes are renumbered 0..n-1 in
the order of their original ids.

Noise (--method):
  uniform    (default) remove exactly floor(m * (1 - keep)) edges at random,
             nodes may become isolated.  This is what the large graphs in
             this repo show (deezer, condmat, slashdot, buzznet, ippds).
  exact      remove exactly that many edges at random, but never the last
             edge of a node, keeping the line order.  This is what the small
             high-school and voles files show.
  framework  Framework_GraphAlignment's remove_e: every edge is removed with
             probability 1 - keep unless it is the last edge of a node, so
             the number removed varies a little.
--refill adds as many random new edges as were removed (the framework's
noise type 2).  --permute shuffles the target's node ids like the framework
does and writes <name>-95-gt.txt ("source target" per line); without it the
ground truth is the identity, as in all the provided datasets.

    .venv/bin/python make-noisy-graphs.py data/raw/ca-CondMat/ca-CondMat.mtx
    .venv/bin/python make-noisy-graphs.py data/raw/email-enron/email-enron.mtx --keep 95 90
    .venv/bin/python make-noisy-graphs.py data.txt --name test --permute
"""

import argparse
import gzip
from pathlib import Path

import networkx as nx
import numpy as np

import framework_generate as fg

ROOT = Path(__file__).resolve().parent


def read_raw(path):
    """Edge list of a .mtx / SNAP / whitespace-separated file."""
    opener = gzip.open if path.suffix == ".gz" else open
    edges, header_skipped = [], path.suffixes[:1] != [".mtx"]
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(("%", "#")) or not line.strip():
                continue
            if not header_skipped:          # .mtx size line: rows cols entries
                header_skipped = True
                continue
            u, v = line.split()[:2]
            edges.append((int(u), int(v)))
    return edges


def clean(edges):
    G = nx.Graph(edges)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G = G.subgraph(max(nx.connected_components(G), key=len))
    ids = {v: i for i, v in enumerate(sorted(G.nodes()))}
    return np.array(sorted((ids[u], ids[v]) for u, v in G.edges()), dtype=np.int64)


def remove_edges(edges, keep, method, rng):
    m = len(edges)
    n_remove = m - int(m * keep)
    if method == "framework":
        return fg.remove_e(edges, 1 - keep, no_disc=True)
    order = rng.permutation(m)
    if method == "uniform":
        return edges[np.sort(order[n_remove:])]
    degree = np.bincount(edges.ravel())
    removed = np.zeros(m, dtype=bool)
    count = 0
    for i in order:
        if count == n_remove:
            break
        u, v = edges[i]
        if degree[u] > 1 and degree[v] > 1:
            degree[u] -= 1
            degree[v] -= 1
            removed[i] = True
            count += 1
    if count < n_remove:
        print(f"  only {count} of {n_remove} edges could be removed without isolating a node")
    return edges[~removed]


def write_edges(path, edges):
    np.savetxt(path, edges, fmt="%d", delimiter=" ")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("input", type=Path)
    parser.add_argument("--name", help="dataset name (default: input file name)")
    parser.add_argument("--keep", nargs="+", type=int, default=[95, 90, 85, 80],
                        help="percentage of edges to keep, one target file each (default: 95 90 85 80)")
    parser.add_argument("--method", choices=["uniform", "exact", "framework"], default="uniform")
    parser.add_argument("--refill", action="store_true", help="add as many random edges as were removed")
    parser.add_argument("--permute", action="store_true", help="shuffle target node ids, write a ground-truth file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "real_noise")
    args = parser.parse_args()

    name = args.name or args.input.name.split(".")[0]
    folder = args.out / name
    folder.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)                 # used by framework_generate
    rng = np.random.default_rng(args.seed)

    edges = clean(read_raw(args.input))
    n = int(edges.max()) + 1
    write_edges(folder / f"{name}-100.txt", edges)
    print(f"{name}: n={n}, m={len(edges)} -> {folder / f'{name}-100.txt'}")

    for keep in args.keep:
        target = remove_edges(edges, keep / 100, args.method, rng)
        n_removed = len(edges) - len(target)
        if args.refill:
            target = fg.refill_e(target, n, n_removed)
        path = folder / f"{name}-{keep}.txt"
        if args.permute:
            target, gt = fg.permute_target(target, n)
            write_edges(folder / f"{name}-{keep}-gt.txt", np.stack([np.arange(n), gt[0]], axis=1))
        write_edges(path, target)
        isolated = n - len(np.unique(target))
        print(f"  {path.name}: {len(target)} edges ({n_removed} removed"
              f"{', refilled' if args.refill else ''}), {isolated} isolated nodes")


if __name__ == "__main__":
    main()
