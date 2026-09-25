#!/bin/bash
# Re-measure with the log-probability comparator the pairs that token identity could not judge.
set -u
source ~/venvs/sglang/bin/activate
cd <workspace>
for backend in triton flashinfer torch_native flex_attention; do
  python sweep/run_sglang.py ~/models/gemma-2-2b-it final_logit_softcapping $backend --logprob 2>&1 \
    | grep -E "^RESULT" || echo "RESULT final $backend killed"
done
