#!/bin/bash
# Does the softened binding value still bind? Check it on a backend that honoured the fact at the harder value.
set -u
source ~/venvs/sglang/bin/activate
cd <workspace>
for backend in torch_native flex_attention; do
  python sweep/run_sglang.py ~/models/gemma-2-2b-it final_logit_softcapping $backend 2>&1 | grep -E "^RESULT" \
    || echo "RESULT final $backend killed"
done
