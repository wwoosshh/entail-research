#!/usr/bin/env bash
# After E1b 9B finishes (GPU free): E4 (vLLM reference for 2B), then vLLM and SGLang smoke tests with Qwen3-4B.
set -uo pipefail
G=<workspace>/issue_track/gemma2_softcap
E=<workspace>/env
until grep -q "e1b 9b exit=" "$G/logs/run.log" 2>/dev/null; do sleep 60; done
cd "$G"
~/venvs/vllm/bin/python e4_vllm.py --size 2b > logs/e4_2b.log 2>&1
echo "$(date +%H:%M:%S) e4 2b exit=$?" >> logs/run.log
cd "$E"
~/venvs/vllm/bin/python 08_engine_smoke.py vllm > logs/08_vllm.log 2>&1
echo "$(date +%H:%M:%S) smoke vllm exit=$?" >> "$G/logs/run.log"
~/venvs/sglang/bin/python 08_engine_smoke.py sglang > logs/08_sglang.log 2>&1
echo "$(date +%H:%M:%S) smoke sglang exit=$?" >> "$G/logs/run.log"
