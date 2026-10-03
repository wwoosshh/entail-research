source ~/venvs/vllm/bin/activate; export PATH=/usr/local/cuda/bin:$PATH
cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
R=$HOME/ai_compiler/lowlevel/l5/guarantee/results
S=$HOME/ai_compiler/lowlevel/l5/guarantee/scratch
echo "dev4 start $(date)"
python eval/block_fp8_guarantee/freeze.py $R/calibration_v1.json $R/env_freeze_v2.json $S/freeze_v3_draft.json \
  --cost-budget 20 --engine-budget 1.10 --what "DRAFT for dev4 (not a freeze): v3 plans" 2>&1 | grep -v "WARNING\|INFO"
python eval/block_fp8_guarantee/run_suite.py dev $S/dev4 --freeze $S/freeze_v3_draft.json \
  --only "${ONLY:-I1-integrity-eager,I2-integrity-graph,N8-graph,D-neighbor-T3,D-rows-T1}" 2>&1 \
  | grep --line-buffered -v "WARNING\|INFO" | cut -c1-260
python eval/block_fp8_guarantee/aggregate.py $S/dev4 2>&1 | grep -v "WARNING\|INFO" | cut -c1-700
echo "dev4 done $(date)"
