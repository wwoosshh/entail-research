#!/usr/bin/env bash
# Week 3 rerun with the v2 harness (manual CUDA graph per variant + eager-reference checks).
# Waits for the v1 chain to exit first so the two never share the GPU.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/venvs/gpu/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$HERE"
while pgrep -f "run_week3.sh" >/dev/null || pgrep -f "batch_invariance_cost.py" >/dev/null; do sleep 5; done
echo "== ab_attention_v2 start $(date +%T)"
python ab_attention_v2.py 2>&1 | tee logs/ab_attention_v2.log
echo "== bi_speed_v2 start $(date +%T)"
python bi_speed_v2.py 2>&1 | tee logs/bi_speed_v2.log
echo "DONE week3 v2 $(date +%T)"
