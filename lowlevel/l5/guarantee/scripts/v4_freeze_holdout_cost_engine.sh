source ~/venvs/vllm/bin/activate; export PATH=/usr/local/cuda/bin:$PATH
cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
R=$HOME/ai_compiler/lowlevel/l5/guarantee/results
F=$R/freeze_v4.json
echo "v4 start $(date)"
git log --oneline -1; git status --porcelain --untracked-files=no | head
python eval/block_fp8_guarantee/env_record.py $R/env_freeze_v4.json 2>&1 | tail -1
python eval/block_fp8_guarantee/freeze.py $R/calibration_v1.json $R/env_freeze_v4.json $F --cost-budget 20 \
  --engine-budget 1.10 --engine-gpu-mem 0.80 --engine-rounds graphs=3 2>&1 | tail -1
python -c "
import json; m = json.load(open('$F'))
print(m['entail_git'], m['plan']['check'], m['plan']['integrity'], m['structure_plan']['check'], m['structure_plan']['integrity'], m['plan']['c_acc'], m['oracle']['c'])
print(m['cost_budget']); print(m['engine']); print(m['guarantee_floor']); print(m['cases'])"
echo "kernel_ir tests and the harness self-test (CPU) $(date)"
python tests/test_kernel_ir.py > $R/kernel_ir_v4.txt 2>&1; tail -3 $R/kernel_ir_v4.txt
(cd eval/block_fp8_guarantee && python selftest.py) > $R/selftest_v4.txt 2>&1; tail -2 $R/selftest_v4.txt
echo "holdout v4 start $(date)"
python eval/block_fp8_guarantee/run_suite.py holdout $R/holdout_v4 --freeze $F 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-220
python eval/block_fp8_guarantee/aggregate.py $R/holdout_v4 2>&1 | grep -v "WARNING\|INFO" | cut -c1-900
echo "cost v4 start $(date)"
BFG_FREEZE=$F python eval/block_fp8_guarantee/cost.py $R/cost_v4 5 2>&1 | grep --line-buffered -v "WARNING\|INFO"
echo "engine v4 start $(date)"
BFG_FREEZE=$F python eval/block_fp8_guarantee/engine_smoke.py $R/engine_v4 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-300
echo "ALL V4 DONE $(date)"
