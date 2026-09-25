#!/usr/bin/env bash
# Week 4 chain: E3, E3b, E4, then E2 (5 fresh processes). E1 and E5 were run separately.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/venvs/gpu/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$HERE"
mkdir -p logs results
echo "== e3 start $(date +%T)"
python e3_rediscovery.py > logs/e3.log 2>&1
grep -E "^  B=|correctness|metadata cost|median|interleaved|Traceback|Error" logs/e3.log | cut -c1-300
echo "== e3b start $(date +%T)"
python e3b_mask_scaling.py > logs/e3b.log 2>&1
grep -E "^\{|Traceback|Error" logs/e3b.log | cut -c1-400
echo "== e4 start $(date +%T)"
python e4_fact_ablation.py > logs/e4.log 2>&1
grep -E "eager vs eager|correct.: False|median|kernels |position counter|Traceback|Error" logs/e4.log | cut -c1-300
for a in auto dynamic_flag mark_dynamic unbacked static; do
  echo "== e2 $a start $(date +%T)"
  python e2_shape_guessing.py --approach "$a" > "logs/e2_$a.log" 2>&1
  grep -E "^(auto|dynamic_flag|mark_dynamic|unbacked|static)|Traceback" "logs/e2_$a.log" | cut -c1-300
done
echo "DONE week4 $(date +%T)"
