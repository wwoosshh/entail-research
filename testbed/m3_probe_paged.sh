#!/bin/bash
# M3.1: the paged consumer, named in its own group (transformers.paged_attention.sdpa) after the table change.
# The first run used the old name (transformers.attention.paged|sdpa); its files are kept with an _oldname suffix.
set -u
ROOT=~/ai_compiler
OUT=$ROOT/testbed/results/m3_probe
export PYTHONPATH=$ROOT/entail
cd "$OUT"
for f in transformers_paged*; do
  case "$f" in *'|'*) mv -- "$f" "${f//|/_}"; mv -- "${f//|/_}" "${f//|/_}.oldname" ;; esac
done
~/venvs/gpu/bin/python -m entail probe --engine transformers --consumer "paged|sdpa" --fact ModelProps.softcap \
  --model ~/models/gemma-2-2b-it --out "$OUT/transformers_paged_sdpa_ModelProps.softcap.json" \
  > "$OUT/transformers_paged_sdpa_ModelProps.softcap.log" 2>&1
tail -n 2 "$OUT/transformers_paged_sdpa_ModelProps.softcap.log"
ls "$OUT"
