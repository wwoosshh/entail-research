#!/usr/bin/env bash
# After the vLLM smoke retry: rerun case 17 (the ResourceWarning in the harness is fixed) and rebuild the tables.
set -uo pipefail
RUNLOG=<workspace>/issue_track/gemma2_softcap/logs/run.log
until grep -q "smoke vllm retry exit=" "$RUNLOG" 2>/dev/null; do sleep 60; done
source ~/venvs/gpu/bin/activate
cd <workspace>/rolebench
python run_case.py cases/17_sglang_torch_native_softcap > results/run_final_queue.log 2>&1
echo "$(date +%H:%M:%S) rolebench 17 rerun exit=$?" >> "$RUNLOG"
python run_all.py --none >> results/run_final_queue.log 2>&1
python detection_table.py >> results/run_final_queue.log 2>&1
python g1_check.py >> results/run_final_queue.log 2>&1
echo "$(date +%H:%M:%S) rolebench tables rebuilt" >> "$RUNLOG"
