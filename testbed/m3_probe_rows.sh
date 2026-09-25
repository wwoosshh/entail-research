#!/bin/bash
# M3.1: check capability-table rows with `entail probe` on Gemma 2 2B (the rows the M3 test problems rely on).
# Each engine runs in its own venv; entail comes from this checkout. Results: testbed/results/m3_probe/*.json
# Run in WSL:  bash ~/ai_compiler/testbed/m3_probe_rows.sh
set -u
ROOT=~/ai_compiler
OUT=$ROOT/testbed/results/m3_probe
MODEL=~/models/gemma-2-2b-it
mkdir -p "$OUT"
export PYTHONPATH=$ROOT/entail
run() {  # venv engine consumer fact
  local py=~/venvs/$1/bin/python name="$2_$3_$4"
  echo "== $name"
  "$py" -m entail probe --engine "$2" --consumer "$3" --fact "$4" --model "$MODEL" --out "$OUT/$name.json" \
    > "$OUT/$name.log" 2>&1
  tail -n 2 "$OUT/$name.log"
}
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
run gpu transformers eager ModelProps.softcap
run gpu transformers sdpa ModelProps.softcap
run gpu transformers paged\|sdpa ModelProps.softcap
run gpu transformers sdpa ModelProps.sliding_window
run sglang sglang triton ModelProps.softcap
run sglang sglang torch_native ModelProps.softcap
run vllm vllm FLASH_ATTN ModelProps.softcap
