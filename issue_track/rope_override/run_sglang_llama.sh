#!/bin/bash
# The SGLang Llama arm, one process per configuration. Usage: bash run_sglang_llama.sh [names...]
set -u
source ~/venvs/sglang/bin/activate
cd <workspace>/issue_track/rope_override
mkdir -p results
NAMES=${@:-SA SA2 SC SE SC_rolecheck}
for n in $NAMES; do
  if [[ "$n" == *_pip ]]; then
    ENTAIL=load ENTAIL_ONLY=rope_alias python sglang_llama.py "$n" > "results/$n.log" 2>&1
  elif [[ "$n" == *_entail || "$n" == *_rolecheck ]]; then
    PYTHONPATH=<workspace>/entail:<workspace>/entail/entail/adapters/autoinstall \
      ENTAIL=load ENTAIL_ONLY=rope_alias python sglang_llama.py "$n" > "results/$n.log" 2>&1
  else
    python sglang_llama.py "$n" > "results/$n.log" 2>&1
  fi
  echo "$n exit=$? $(grep -h '^DONE' results/$n.log)"
done
