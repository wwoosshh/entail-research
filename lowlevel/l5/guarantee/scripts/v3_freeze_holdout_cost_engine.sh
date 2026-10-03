source ~/venvs/vllm/bin/activate; export PATH=/usr/local/cuda/bin:$PATH
cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
R=$HOME/ai_compiler/lowlevel/l5/guarantee/results
F=$R/freeze_v3.json
echo "v3 start $(date)"
git log --oneline -1; git status --porcelain --untracked-files=no | head
python eval/block_fp8_guarantee/env_record.py $R/env_freeze_v3.json 2>&1 | tail -1
python eval/block_fp8_guarantee/freeze.py $R/calibration_v1.json $R/env_freeze_v3.json $F --cost-budget 20 \
  --engine-budget 1.10 2>&1 | tail -1
python -c "
import json; m=json.load(open('$F')); print(m['entail_git'], m['plan']['check'], m['plan']['integrity'], m['structure_plan']['check'], m['structure_plan']['integrity'], m['plan']['c_acc'], m['oracle']['c'], m['cost_budget'], m['cases'])"
echo "kernel_ir counterexamples (CPU) $(date)"
python tests/test_kernel_ir.py > $R/kernel_ir_v3.txt 2>&1; tail -3 $R/kernel_ir_v3.txt
echo "holdout v3 start $(date)"
python eval/block_fp8_guarantee/run_suite.py holdout $R/holdout_v3 --freeze $F 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-220
python eval/block_fp8_guarantee/aggregate.py $R/holdout_v3 2>&1 | grep -v "WARNING\|INFO" | cut -c1-900
echo "cost v3 start $(date)"
BFG_FREEZE=$F python eval/block_fp8_guarantee/cost.py $R/cost_v3 5 2>&1 | grep --line-buffered -v "WARNING\|INFO"
echo "engine v3 start $(date)"
BFG_ENGINE_CONFIGS=eager,graphs,default BFG_FREEZE=$F python eval/block_fp8_guarantee/engine_smoke.py $R/engine_v3 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-300
echo "ALL V3 DONE $(date)"
