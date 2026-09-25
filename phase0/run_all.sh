#!/usr/bin/env bash
# Phase 0 week-1 run: install deps, run both workloads, build the report.
# Usage inside WSL:  bash ~/ai_compiler/phase0/run_all.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/venvs/gpu/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$HERE/results" "$HERE/logs"

echo "== deps"
pip install -q diffusers transformers accelerate safetensors huggingface_hub torchao 2>&1 | tail -3
python -c "import diffusers, transformers, torchao; print('diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| torchao', torchao.__version__)"

echo "== workload A: SDXL UNet step"
python "$HERE/bench_sdxl_unet.py" 2>&1 | tee "$HERE/logs/sdxl_unet.log" | grep -vE "^\s*$" | tail -40

echo "== workload B: LLM decode step"
python "$HERE/bench_llm_decode.py" --int4 2>&1 | tee "$HERE/logs/llm_decode.log" | grep -vE "^\s*$" | tail -60

echo "== report"
python "$HERE/make_report.py"
echo "DONE phase0 run_all"
