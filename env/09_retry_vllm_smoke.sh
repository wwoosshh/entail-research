#!/usr/bin/env bash
# Retry the vLLM smoke test with a smaller context after the W-arm probes free the GPU.
set -uo pipefail
RUNLOG=<workspace>/issue_track/gemma2_softcap/logs/run.log
until grep -q "audit sglang_survey exit=" "$RUNLOG" 2>/dev/null; do sleep 60; done
cd <workspace>/env
~/venvs/vllm/bin/python 08_engine_smoke.py vllm > logs/08_vllm_retry.log 2>&1
echo "$(date +%H:%M:%S) smoke vllm retry exit=$?" >> "$RUNLOG"
