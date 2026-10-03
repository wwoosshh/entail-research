source ~/venvs/vllm/bin/activate; export PATH=/usr/local/cuda/bin:$PATH
cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
F=$HOME/ai_compiler/lowlevel/l5/guarantee/results/freeze_v2.json
O=$HOME/ai_compiler/lowlevel/l5/guarantee/results
echo "holdout v2 start $(date)"
python eval/block_fp8_guarantee/run_suite.py holdout $O/holdout_v2 --freeze $F 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-200
python eval/block_fp8_guarantee/aggregate.py $O/holdout_v2 2>&1 | grep -v "WARNING\|INFO"
echo "cost v2 start $(date)"
BFG_FREEZE=$F python eval/block_fp8_guarantee/cost.py $O/cost_v2 5 2>&1 | grep --line-buffered -v "WARNING\|INFO"
echo "engine v2 start $(date)"
BFG_ENGINE_CONFIGS=eager,graphs,default BFG_FREEZE=$F python eval/block_fp8_guarantee/engine_smoke.py $O/engine_v2 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-300
echo "ALL V2 DONE $(date)"
