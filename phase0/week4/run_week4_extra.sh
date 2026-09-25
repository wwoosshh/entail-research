#!/usr/bin/env bash
# Week 4 follow-ups, after run_week4.sh: hf_flex diagnosis, in-graph rediscovery variants, longer masks,
# E2 with the shared-batch declaration, one spec / two backends.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/venvs/gpu/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$HERE"
echo "== diag start $(date +%T)"
python diag_hf_flex_offset.py > logs/diag_hf_flex.log 2>&1
grep -E "^[ABC]_|Traceback|Error:" logs/diag_hf_flex.log | cut -c1-600
echo "== e6 start $(date +%T)"
python e6_one_spec.py > logs/e6.log 2>&1
grep -E "^(triton|flex|same)|Traceback|Error:" logs/e6.log | cut -c1-400
echo "== e3 extra start $(date +%T)"
python e3_rediscovery.py --only flex_declared flex_rediscover flex_rediscover_compiled flex_rediscover_host \
  --out results/e3_rediscovery_extra.json > logs/e3_extra.log 2>&1
grep -E "^  B=|pairwise|correctness|metadata cost|median|Traceback|Error:" logs/e3_extra.log | cut -c1-400
echo "== e3b v2 start $(date +%T)"
python e3b_mask_scaling.py --lengths 4096 16384 32768 65536 131072 \
  --out results/e3b_mask_scaling_v2.json > logs/e3b_v2.log 2>&1
grep -E "^\{|Traceback|Error:" logs/e3b_v2.log | cut -c1-600
echo "== e2 unbacked_shared start $(date +%T)"
python e2_shape_guessing.py --approach unbacked_shared > logs/e2_unbacked_shared.log 2>&1
grep -E "^unbacked_shared|Traceback" logs/e2_unbacked_shared.log | cut -c1-300
echo "DONE extra $(date +%T)"
