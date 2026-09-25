#!/usr/bin/env bash
# 서빙 엔진용 별도 가상환경을 만든다.
# vLLM 0.30.0과 SGLang 0.5.20은 torch 2.13.0을 고정한다. 기본 gpu 환경은 torch 2.14라서 섞지 않는다.
# 사용: bash 05_serving_engines.sh [vllm|sglang|all]
set -euo pipefail
which_engine="${1:-all}"

mk() {
  local name="$1"; shift
  local venv="$HOME/venvs/$name"
  if [ ! -x "$venv/bin/python" ]; then python3 -m venv "$venv"; fi
  "$venv/bin/pip" install --upgrade pip wheel
  "$venv/bin/pip" install "$@"
  "$venv/bin/python" - <<'EOF'
import importlib
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for m in ("vllm", "sglang", "flashinfer", "transformers"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "not importable:", type(e).__name__, str(e)[:160])
EOF
}

case "$which_engine" in
  vllm) mk vllm "vllm==0.30.0" ;;
  sglang) mk sglang "sglang==0.5.20" ;;
  all) mk vllm "vllm==0.30.0"; mk sglang "sglang==0.5.20" ;;
  *) echo "unknown: $which_engine"; exit 2 ;;
esac
