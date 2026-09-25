#!/usr/bin/env bash
# Week 3 chain: attention A/B (bf16 + int4, random weights) then batch-invariance cost (real weights).
# Run sequentially so the two never compete for the GPU.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/venvs/gpu/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HERE/results" "$HERE/logs"
cd "$HERE"
echo "== ab_attention start $(date +%T)"
python ab_attention.py 2>&1 | tee logs/ab_attention.log
echo "== batch_invariance_cost start $(date +%T)"
python batch_invariance_cost.py 2>&1 | tee logs/batch_invariance_cost.log
echo "DONE week3 $(date +%T)"
