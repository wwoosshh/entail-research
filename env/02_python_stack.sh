#!/usr/bin/env bash
# Python venv + PyTorch (cu130) + Triton + research tooling.
# Run as the normal user inside the distro:  bash 02_python_stack.sh [torch-version]
set -euo pipefail
TORCH_VER="${1:-2.14.0}"
VENV="${HOME}/venvs/gpu"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

echo "== [1/4] venv ${VENV}"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging ninja

echo "== [2/4] PyTorch ${TORCH_VER}+cu130 (pulls its pinned Triton on Linux)"
pip install "torch==${TORCH_VER}+cu130" --extra-index-url "${TORCH_INDEX}"

echo "== [3/4] research tooling"
pip install numpy pytest matplotlib pandas rich tabulate
pip install cuda-python cuda-core || echo "WARN: cuda-python/cuda-core failed (optional; used for launching raw PTX)"
pip install nvidia-cutlass-dsl || echo "WARN: nvidia-cutlass-dsl (CuTe DSL) failed (optional)"

echo "== [4/4] flash-attn prebuilt wheel (best effort; PyPI is source-only so this usually skips)"
pip install flash-attn --no-deps --only-binary=:all: || echo "INFO: no prebuilt flash-attn wheel; PyTorch SDPA flash backend is the FA2 baseline on Ada"

grep -q 'venvs/gpu/bin/activate' "${HOME}/.bashrc" || echo 'source ~/venvs/gpu/bin/activate' >> "${HOME}/.bashrc"

python - <<'PY'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda, "| cuda available", torch.cuda.is_available())
import triton; print("triton", triton.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "| cc", f"{p.major}.{p.minor}", "| mem GiB", round(p.total_memory/2**30,1), "| SMs", p.multi_processor_count)
PY
echo "DONE 02 python stack"
