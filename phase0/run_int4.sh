#!/usr/bin/env bash
# int4-only rerun of workload B (torchao Int4WeightOnlyConfig, tile_packed_to_4d) + report rebuild.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/venvs/gpu/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HERE/results" "$HERE/logs"
python "$HERE/bench_llm_decode.py" --int4 --skip-bf16 --out "$HERE/results/llm_decode_int4.json" 2>&1 | tee "$HERE/logs/llm_decode_int4.log"
python "$HERE/make_report.py"
echo "DONE int4 rerun"
