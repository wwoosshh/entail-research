#!/bin/bash
# Every configuration of PROTOCOL.md, one process each. Usage: bash run_all.sh [names...]  (default A B C D E)
set -u
source ~/venvs/vllm/bin/activate
cd <workspace>/issue_track/rope_override
mkdir -p results
NAMES=${@:-A B C D E}
for n in $NAMES; do
  if [[ "$n" == *_entail || "$n" == *_rolecheck ]]; then
    PYTHONPATH=<workspace>/entail:<workspace>/entail/entail/adapters/autoinstall \
      ENTAIL=load ENTAIL_ONLY=rope_alias python rope_override.py "$n" > "results/$n.log" 2>&1
  else
    python rope_override.py "$n" > "results/$n.log" 2>&1
  fi
  echo "$n exit=$? $(grep -h '^DONE' results/$n.log)"
done
