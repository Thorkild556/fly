"""check-mps-vs-cpu.py

Checks that running the kissing variants on MPS (Apple GPU) computes the
same thing as on the CPU.  For each variant it evaluates the loss and its
gradients for the *same* V, W on both devices, and compares both with a
plain dense implementation of the same loss (no chunking, no sparse tricks):

    P      = softmax(beta * normalize(V) @ normalize(W).T  [+ col_bias])
    loss   = -sum((A @ P) * (P @ B)) + mu * sum(P * D)
             + row_penalty * sum((rowsum(P) - 1)^2)
             + col_penalty * sum((colsum(P) - 1)^2)

dense-LA approximates this softmax, so it is only compared CPU vs MPS.

    .venv/bin/python check-mps-vs-cpu.py                       # ACM 95
    .venv/bin/python check-mps-vs-cpu.py --dataset highschool --target 95
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_tests", ROOT / "run-tests.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)
sys.path.insert(0, str(rt.FLY_DIR))

import torch  # noqa: E402

HP = dict(beta=10.0, mu=0.1, row_penalty=10.0, col_penalty=200.0)


def dense_reference(Gq, Gt, V, W, col_bias=None):
    """The loss above, written out densely in float64 on the CPU."""
    import networkx as nx
    from helpers.pred import feature_extraction

    n = V.shape[0]
    A = torch.tensor(nx.to_numpy_array(Gq, nodelist=range(n)), dtype=torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt, nodelist=range(n)), dtype=torch.float64)
    F1 = torch.tensor(feature_extraction(Gq), dtype=torch.float64)
    F2 = torch.tensor(feature_extraction(Gt), dtype=torch.float64)
    D = torch.cdist(F1, F2)

    V = V.double().clone().requires_grad_(True)
    W = W.double().clone().requires_grad_(True)
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scores = HP["beta"] * Vn @ Wn.T
    if col_bias is not None:
        scores = scores + col_bias.double()
    P = torch.softmax(scores, dim=1)
    loss = (-((A @ P) * (P @ B)).sum() + HP["mu"] * (P * D).sum()
            + HP["row_penalty"] * ((P.sum(1) - 1) ** 2).sum()
            + HP["col_penalty"] * ((P.sum(0) - 1) ** 2).sum())
    loss.backward()
    return loss.item(), V.grad, W.grad


def variant_loss(algo, module, Gq, Gt, V0, W0, col_bias0, device, chunk):
    """Loss and gradients of one variant's own loss function on `device`."""
    V = V0.to(device).clone().requires_grad_(True)
    W = W0.to(device).clone().requires_grad_(True)
    hp = dict(beta=HP["beta"], mu=HP["mu"],
              row_penalty=HP["row_penalty"], col_penalty=HP["col_penalty"])

    if algo == "dense-chunked":
        A_sp, _, BT_sp, F1, F2 = module._build_inputs(Gq, Gt, device)
        blocks, n_rows = module.build_sparse_blocks(A_sp, chunk, device=device)
        loss = module.chunked_fugal_loss(V, W, blocks, BT_sp, F1, F2, **hp,
                                         chunk=chunk, n_chunks_rows=n_rows)[0]
    elif algo == "dense-chunked-fast":
        A_sp, B_sp, F1, F2 = module._build_inputs(Gq, Gt, device)
        blocks, n_rows = module.build_combined_sparse_blocks(A_sp, chunk, device=device)
        loss = module.chunked_fugal_loss_fast(V, W, blocks, B_sp, F1, F2, **hp,
                                              chunk=chunk, n_chunks_rows=n_rows)[0]
    elif algo == "kissing-fast":
        col_bias = col_bias0.to(device).clone().requires_grad_(True)
        A_sp, B_sp, F1, F2 = module._build_inputs(Gq, Gt, device)
        blocks, n_rows = module.build_combined_sparse_blocks(A_sp, chunk, device=device)
        loss = module.chunked_fugal_loss_fast(V, W, col_bias, blocks, B_sp, F1, F2,
                                              HP["beta"], HP["mu"],
                                              HP["row_penalty"], HP["col_penalty"],
                                              chunk, n_rows)[0]
    elif algo == "dense-LA":
        A, B, D = (t.to(device) for t in module.build_inputs(Gq, Gt))
        loss = module.linear_fugal_loss_terms(A, B, D, V, W, **hp, kind="relu_squared")[0]
    loss.backward()
    return float(loss), V.grad.cpu().double(), W.grad.cpu().double()


def rel(a, b):
    """Relative difference of two numbers or tensors."""
    if isinstance(a, float):
        return abs(a - b) / max(abs(b), 1e-12)
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ACM", choices=list(rt.DATASETS))
    parser.add_argument("--target", default="95")
    parser.add_argument("--embed-dim", type=int, default=200)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--algos", nargs="+",
                        default=["dense-chunked", "dense-chunked-fast", "kissing-fast", "dense-LA"])
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available on this machine")

    Gq, Gt, n, _ = rt.load_graphs(args.dataset, args.target)
    print(f"{args.dataset}/{args.target}: n={n}, embed_dim={args.embed_dim}, chunk={args.chunk}, {HP}\n")

    gen = torch.Generator().manual_seed(0)
    V0 = torch.rand(n, args.embed_dim, generator=gen)
    W0 = torch.rand(n, args.embed_dim, generator=gen)
    col_bias0 = 0.1 * torch.randn(n, generator=gen)   # non-zero, so it is tested too

    print("dense float64 reference ...", flush=True)
    ref = dense_reference(Gq, Gt, V0, W0)
    ref_bias = dense_reference(Gq, Gt, V0, W0, col_bias0)

    print(f"\n{'variant':<20}{'device':<8}{'loss':>16}{'loss vs ref':>14}"
          f"{'grad V vs ref':>15}{'grad W vs ref':>15}{'grad V cpu/mps':>16}")
    for algo in args.algos:
        module = rt.load_script(rt.ALGORITHMS[algo]["script"])
        out = {d: variant_loss(algo, module, Gq, Gt, V0, W0, col_bias0, d, args.chunk)
               for d in ("cpu", "mps")}
        r = ref_bias if algo == "kissing-fast" else ref
        for d, (loss, gV, gW) in out.items():
            vs_ref = (f"{rel(loss, r[0]):>14.2e}{rel(gV, r[1]):>15.2e}{rel(gW, r[2]):>15.2e}"
                      if algo != "dense-LA" else f"{'(approx.)':>14}{'':>15}{'':>15}")
            cross = f"{rel(out['mps'][1], out['cpu'][1]):>16.2e}" if d == "mps" else ""
            print(f"{algo:<20}{d:<8}{loss:>16.4f}{vs_ref}{cross}", flush=True)
    print("\nRelative differences around 1e-5 or smaller are float32 rounding; "
          "anything near 1e-2 or above would mean the computation differs.")


if __name__ == "__main__":
    main()
