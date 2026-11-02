#!/usr/bin/env bash
# setup-remote.sh: set up this repo on a Linux server with NVIDIA GPUs.
#
#     git clone --recurse-submodules <this repo> fly && cd fly && bash setup-remote.sh
#
# It creates .venv (the path all commands in the scripts use), installs
# requirements.txt (the CUDA 12 package set from the fly repo, torch
# 2.11), checks that PyTorch sees the GPUs and runs a short smoke test of
# every algorithm on each GPU.  Needs Python 3.10+ and an NVIDIA driver
# that supports CUDA 12.6 (driver 560 or newer).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
    "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

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
