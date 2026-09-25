#!/usr/bin/env bash
# GPU rolebench cases that must wait until the Gemma 2 measurement chain and the engine smoke tests are done.
# Runs every GPU case (reproduction + detection checks), then rebuilds both tables.
set -uo pipefail
RUNLOG=<workspace>/issue_track/gemma2_softcap/logs/run.log
until grep -q "smoke sglang exit=" "$RUNLOG" 2>/dev/null; do sleep 60; done
source ~/venvs/gpu/bin/activate
cd <workspace>/rolebench
python run_all.py 03 05 06 08 09 10 11 17 > results/run_gpu_queue.log 2>&1
echo "$(date +%H:%M:%S) rolebench gpu cases exit=$?" >> "$RUNLOG"
python detection_table.py >> results/run_gpu_queue.log 2>&1
