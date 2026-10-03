source ~/venvs/vllm/bin/activate
cd ~/ai_compiler/entail
export PYTHONPATH=$HOME/ai_compiler/entail
F=$HOME/ai_compiler/lowlevel/l5/guarantee/results/freeze_v1.json
O=$HOME/ai_compiler/lowlevel/l5/guarantee/results
echo "env $(date)"
python eval/block_fp8_guarantee/env_record.py $O/env_work_v1.json 2>&1 | tail -1
echo "cost start $(date)"
BFG_FREEZE=$F python eval/block_fp8_guarantee/cost.py $O/cost_v1 5 2>&1 | grep --line-buffered -v "WARNING\|INFO"
echo "engine start $(date)"
BFG_FREEZE=$F python eval/block_fp8_guarantee/engine_smoke.py $O/engine_v1 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-600
echo "ALL DONE $(date)"
