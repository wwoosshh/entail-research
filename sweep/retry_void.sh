#!/bin/bash
# PROTOCOL 5.1: re-measure the pairs whose control run did not reproduce, sending one prompt per request.
set -u
source ~/venvs/sglang/bin/activate
cd <workspace>
MODEL=~/models/gemma-2-2b-it
python sweep/run_sglang.py $MODEL attn_logit_softcapping triton --one-at-a-time 2>&1 | grep -E "^RESULT" || echo "RESULT softcap triton killed"
python sweep/run_sglang.py $MODEL final_logit_softcapping triton --one-at-a-time 2>&1 | grep -E "^RESULT" || echo "RESULT final triton killed"
python sweep/run_sglang.py $MODEL final_logit_softcapping flashinfer --one-at-a-time 2>&1 | grep -E "^RESULT" || echo "RESULT final flashinfer killed"
