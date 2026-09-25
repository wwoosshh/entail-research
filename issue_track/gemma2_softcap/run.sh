#!/usr/bin/env bash
# Runs E0..E3 for 2B, then for 9B. Each size waits until its download is recorded in env/logs/06_fetch_gemma2.log.
set -uo pipefail
source ~/venvs/gpu/bin/activate
cd <workspace>/issue_track/gemma2_softcap
FETCH_LOG=<workspace>/env/logs/06_fetch_gemma2.log
mkdir -p logs
sizes="${1:-2b 9b}"
exps="${2:-e0 e1 e2 e3}"
for size in $sizes; do
  until grep -q "unsloth/gemma-2-${size}-it " "$FETCH_LOG"; do sleep 30; done
  for exp in $exps; do
    python g2softcap.py "$exp" --size "$size" > "logs/${exp}_${size}.log" 2>&1
    echo "$(date +%H:%M:%S) $exp $size exit=$?" >> logs/run.log
  done
done
echo "$(date +%H:%M:%S) all done" >> logs/run.log
