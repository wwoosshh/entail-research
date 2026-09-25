#!/bin/bash
# Drive the vLLM sweep: one process per (fact, backend) pair. The backend is chosen by environment variable.
set -u
source ~/venvs/vllm/bin/activate
export VLLM_LOGGING_LEVEL=ERROR
cd <workspace>
MODEL=${1:-~/models/gemma-2-2b-it}
for fact in attn_logit_softcapping final_logit_softcapping sliding_window rope_theta; do
  for backend in FLASH_ATTN TRITON_ATTN FLEX_ATTENTION; do
    VLLM_ATTENTION_BACKEND=$backend python sweep/run_vllm.py "$MODEL" "$fact" 2>&1 \
      | grep -E "^RESULT|^SKIP" || echo "RESULT $fact $backend killed"
  done
done
