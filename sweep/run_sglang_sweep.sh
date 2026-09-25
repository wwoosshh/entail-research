#!/bin/bash
# Drive the SGLang sweep: one process per (fact, backend) pair, because a failed engine kills its group.
set -u
source ~/venvs/sglang/bin/activate
cd <workspace>
MODEL=${1:-~/models/gemma-2-2b-it}
for fact in attn_logit_softcapping final_logit_softcapping sliding_window rope_theta; do
  for backend in triton flashinfer torch_native flex_attention; do
    python sweep/run_sglang.py "$MODEL" "$fact" "$backend" 2>&1 | grep -E "^RESULT|^SKIP" || echo "RESULT $fact $backend killed"
  done
done
