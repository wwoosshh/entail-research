#!/usr/bin/env bash
# E1b (fp32 attention, PROTOCOL revision 1). Waits until the main chain has finished 9B E3, then runs 2B and 9B.
set -uo pipefail
source ~/venvs/gpu/bin/activate
cd <workspace>/issue_track/gemma2_softcap
until grep -q "e3 9b exit=" logs/run.log 2>/dev/null; do sleep 60; done
for size in 2b 9b; do
  python g2softcap.py e1b --size "$size" > "logs/e1b_${size}.log" 2>&1
  echo "$(date +%H:%M:%S) e1b $size exit=$?" >> logs/run.log
done
