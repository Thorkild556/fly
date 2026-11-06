#!/usr/bin/env bash
# setup-remote.sh: set up this repo on a Linux server with NVIDIA GPUs.
#
#     git clone --recurse-submodules -b tests/thorkild <this repo> fly && cd fly && bash setup-remote.sh
#
# It creates .venv (the path all commands in the scripts use), installs
# requirements.txt (the CUDA 12 package set from the fly repo, torch
# 2.11), checks that PyTorch sees the GPUs and runs a short smoke test of
# every algorithm on each GPU.  Needs an NVIDIA driver that supports
# CUDA 12 (nvidia-smi shows "CUDA Version: 12.x" or higher); it never
# touches the driver or anything outside this folder.
#
# If conda is available (as on the AU servers, whose rules ask for conda
# or docker), .venv is a conda environment with Python 3.12 created inside
# this folder (conda create -p), so nothing is installed system-wide and
# deleting the folder removes everything.  Otherwise a plain venv from
# $PYTHON (default python3, 3.10+) is used.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    if command -v conda >/dev/null 2>&1; then
        conda create -y -p "$PWD/.venv" python=3.12 pip
    else
        "${PYTHON:-python3}" -m venv .venv
    fi
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python - <<'EOF'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch does not see a GPU: check nvidia-smi and the driver version")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory / 1024**3:.0f} GB")
EOF

# 20 training steps per run: checks that everything runs on CUDA (sparse
# matrices included), the accuracies are meaningless.
GPUS=$(.venv/bin/python -c "import torch; print(*range(torch.cuda.device_count()))")
.venv/bin/python run-tests.py --datasets fw-highschool --targets 5 --max-iter 20 \
    --gpus $GPUS --out results-smoke
echo
echo "Setup done.  The smoke-test results are in results-smoke/ (safe to delete)."
