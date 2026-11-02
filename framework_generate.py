"""Noise generation of Framework_GraphAlignment (the benchmark framework the
Fugal README points to), used straight from the git submodule
Framework_GraphAlignment/ (generation/generate.py):

    remove_e(edges, noise, no_disc=True)   remove each edge with probability
                                           `noise`, unless it is the last
                                           edge of one of its nodes
    refill_e(edges, n, amount)             add `amount` random new edges
    noise_types(noise_level, noise_type)   the framework's three noise types
    generate_graphs(G, source_noise, target_noise, refill)

generate.py starts with `from . import ex`: the framework's sacred
experiment, which pulls in the whole framework and is only used for the
@ex.capture decorator.  It is loaded here with a stand-in `ex` whose
capture leaves the functions unchanged, so the framework code runs as is.
The functions print some progress lines (e.g. "##<1>##").

If the folder is empty, fetch the submodule:  git submodule update --init
"""

import importlib.util
import sys
import types
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).resolve().parent / "Framework_GraphAlignment"


def _load_generate():
    path = FRAMEWORK_DIR / "generation" / "generate.py"
    if not path.exists():
        raise ImportError(f"{path} not found: run `git submodule update --init`")
    package = types.ModuleType("framework_generation")
    package.__path__ = [str(path.parent)]
    package.ex = types.SimpleNamespace(capture=lambda function: function)
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.generate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = _load_generate()
remove_e = generate.remove_e
refill_e = generate.refill_e
noise_types = generate.noise_types
generate_graphs = generate.generate_graphs


def permute_target(Tar_e, n):
    """The node-id shuffle of generate_graphs on its own: relabel the
    target's nodes with a random permutation.  Returns the relabelled edges
    and the ground truth Gt, where source node i corresponds to target node
    Gt[0][i]."""
    import numpy as np

    gt_e = np.array((
        np.arange(n),
        np.random.permutation(n)
    ))
    Gt = (
        gt_e[:, gt_e[1].argsort()][0],
        gt_e[:, gt_e[0].argsort()][1]
    )
    return Gt[0][Tar_e], Gt
