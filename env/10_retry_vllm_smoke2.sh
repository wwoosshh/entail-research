#!/usr/bin/env bash
# Second retry of the vLLM smoke test (utilization 0.88), after the paged-attention probe frees the GPU.
set -uo pipefail
RUNLOG=<workspace>/issue_track/gemma2_softcap/logs/run.log
until grep -q "audit tf_paged rerun exit=" "$RUNLOG" 2>/dev/null; do sleep 60; done
cd <workspace>/env
~/venvs/vllm/bin/python 08_engine_smoke.py vllm > logs/08_vllm_retry2.log 2>&1
echo "$(date +%H:%M:%S) smoke vllm retry2 exit=$?" >> "$RUNLOG"
