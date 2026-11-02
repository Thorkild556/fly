"""Reading a query / target graph pair with the node count taken from the
files, so n is never hard-coded.  Every read_file() and run-tests.py use
this.

n is the number of distinct node ids in the two edge lists.  If the ids are
not exactly 0..n-1 (ids skipped in the files), the nodes are renumbered in
the order of their ids, with the same mapping in both graphs, so node i of
the query still corresponds to node i of the target.  Otherwise every unused
id would become an extra isolated node in both graphs: the training would
work on a different problem and the accuracy would be divided by the wrong n.
"""

import networkx as nx


def read_edges(path, keep_self_loops=True):
    """(u, v) pairs of a whitespace-separated edge list; lines starting
    with # or % and lines with fewer than two columns are skipped."""
    edges = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or parts[0].startswith(("#", "%")):
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v or keep_self_loops:
                edges.append((u, v))
    return edges


def count_nodes(*paths, keep_self_loops=True):
    """The n read_graph_pair gives for these files, without building graphs."""
    return len({u for path in paths for edge in read_edges(path, keep_self_loops) for u in edge})


def read_graph_pair(query_path, target_path, keep_self_loops=True):
    """Returns Gq, Gt, n, ids: both graphs on nodes 0..n-1, and ids[i] =
    the id node i has in the files."""
    edges_q = read_edges(query_path, keep_self_loops)
    edges_t = read_edges(target_path, keep_self_loops)
    ids = sorted({u for edge in edges_q + edges_t for u in edge})
    n = len(ids)
    if ids[0] != 0 or ids[-1] != n - 1:
        index = {v: i for i, v in enumerate(ids)}
        edges_q = [(index[u], index[v]) for u, v in edges_q]
        edges_t = [(index[u], index[v]) for u, v in edges_t]
        print(f"[I/O] node ids run from {ids[0]} to {ids[-1]} but only {n} are used: "
              f"renumbered to 0..{n - 1}")

    Gq, Gt = nx.Graph(), nx.Graph()
    Gq.add_nodes_from(range(n))
    Gt.add_nodes_from(range(n))
    Gq.add_edges_from(edges_q)
    Gt.add_edges_from(edges_t)
    print(f"[I/O] {query_path} / {target_path}: n={n}, "
          f"|Eq|={Gq.number_of_edges()}, |Et|={Gt.number_of_edges()}")
    return Gq, Gt, n, ids
