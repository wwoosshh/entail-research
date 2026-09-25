#!/bin/bash
# M3.1 rerun: SGLang probes with the prefix cache off (the first torch_native probe did not reproduce its control).
set -u
ROOT=~/ai_compiler
OUT=$ROOT/testbed/results/m3_probe
MODEL=~/models/gemma-2-2b-it
export PYTHONPATH=$ROOT/entail
for c in torch_native triton; do
  name="sglang_${c}_ModelProps.softcap"
  [ -f "$OUT/$name.json" ] && mv "$OUT/$name.json" "$OUT/${name}_radix_on.json" && mv "$OUT/$name.log" "$OUT/${name}_radix_on.log"
  ~/venvs/sglang/bin/python -m entail probe --engine sglang --consumer $c --fact ModelProps.softcap --model $MODEL \
    --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  tail -n 2 "$OUT/$name.log"
done
