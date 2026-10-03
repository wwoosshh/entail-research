source ~/venvs/vllm/bin/activate; export PATH=/usr/local/cuda/bin:$PATH
cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
S=$HOME/ai_compiler/lowlevel/l5/guarantee/scratch
echo "dev4_full start $(date)"
python eval/block_fp8_guarantee/run_suite.py dev $S/dev4_full --freeze $S/freeze_v3_draft.json 2>&1 \
  | grep --line-buffered -v "WARNING\|INFO" | cut -c1-260
python eval/block_fp8_guarantee/aggregate.py $S/dev4_full 2>&1 | grep -v "WARNING\|INFO" | cut -c1-900
echo "dev4_full done $(date)"
