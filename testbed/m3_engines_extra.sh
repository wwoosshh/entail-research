#!/bin/bash
# M3.5, runs added after the first pass (m3_engines.sh):
#   - the paged run again: its first attempt was stopped by a false config-key refusal (attn_implementation, a key
#     PreTrainedConfig reads itself), fixed in data/config_keys.json; the first result is kept as *_first
#   - a deliberate RoPE override that differs from the files must not be refused (the user's declaration has to reach
#     vLLM's engine core, which gets the config pickled)
set -u
ROOT=~/ai_compiler
OUT=$ROOT/testbed/results/m3
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
cd "$OUT"
for ext in json log record.jsonl; do
  [ -f "fd_softcap_transformers_paged.$ext" ] && mv "fd_softcap_transformers_paged.$ext" "fd_softcap_transformers_paged_first.$ext"
done
PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD="$OUT/fd_softcap_transformers_paged.record.jsonl" ~/venvs/gpu/bin/python \
  "$ROOT/testbed/m3_run_engine.py" transformers ~/models/gemma-2-2b-it "$OUT/fd_softcap_transformers_paged.json" sdpa \
  --paged > "$OUT/fd_softcap_transformers_paged.log" 2>&1
grep -h "^RESULT" "$OUT/fd_softcap_transformers_paged.log" | cut -c1-400
rm -f "$OUT/rope_user.record.jsonl"
PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD="$OUT/rope_user.record.jsonl" ~/venvs/vllm/bin/python \
  "$ROOT/testbed/m3_rope_user.py" "$OUT/rope_user.json" > "$OUT/rope_user.log" 2>&1
grep -h "^RESULT" "$OUT/rope_user.log" | cut -c1-400
echo "extra done"
