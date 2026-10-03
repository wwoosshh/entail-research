source ~/venvs/vllm/bin/activate; export PATH=/usr/local/cuda/bin:$PATH
cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
R=$HOME/ai_compiler/lowlevel/l5/guarantee/results
S=$HOME/ai_compiler/lowlevel/l5/guarantee/scratch
echo "dev5 start $(date)"; git log --oneline -1; git status --porcelain --untracked-files=no | head -3
python eval/block_fp8_guarantee/env_record.py $S/env_v4_draft.json 2>&1 | tail -1
python eval/block_fp8_guarantee/freeze.py $R/calibration_v1.json $S/env_v4_draft.json $S/freeze_v4_draft.json \
  --cost-budget 20 --engine-budget 1.10 --engine-gpu-mem 0.80 --engine-rounds graphs=3 \
  --what "DRAFT for dev5 (not a freeze): v4 plans and settings" 2>&1 | tail -1
echo "-- the holdout against the v3 manifest must not start (code and commit differ):"
python eval/block_fp8_guarantee/run_suite.py holdout $S/dev5_refused_holdout --freeze $R/freeze_v3.json 2>&1 | tail -1
python -c "import json; r = json.load(open('$S/dev5_refused_holdout/freeze_check.json')); print('matches', r['matches'], len(r['mismatches']), 'mismatches:', sorted({m['what'].split(' ')[0] for m in r['mismatches']}))"
ls $S/dev5_refused_holdout
echo "-- the full dev suite against the draft:"
python eval/block_fp8_guarantee/run_suite.py dev $S/dev5 --freeze $S/freeze_v4_draft.json 2>&1 \
  | grep --line-buffered -v "WARNING\|INFO" | cut -c1-200
python -c "import json; r = json.load(open('$S/dev5/freeze_check.json')); print('dev5 freeze check matches:', r['matches'], [m['what'] for m in r['mismatches']][:5])"
python eval/block_fp8_guarantee/aggregate.py $S/dev5 2>&1 | grep -v "WARNING\|INFO" | cut -c1-900
echo "-- the engine's rounds and median, after the freeze by label: graphs, structure, 2 rounds, 0.80"
BFG_FREEZE=$S/freeze_v4_draft.json BFG_POST_FREEZE=1 BFG_ENGINE_CONFIGS=graphs BFG_ENGINE_MODES=structure \
  BFG_ENGINE_ROUNDS=graphs=2 BFG_ENGINE_GPU_MEM=0.80 \
  python eval/block_fp8_guarantee/engine_smoke.py $S/dev5_engine 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-200 | tail -5
python -c "
import json
c = json.load(open('$S/dev5_engine/compare.json'))['graphs']
print('rounds', c['structure_steady_ratio_rounds'], 'median', c['structure_steady_ratio_median'], 'outputs equal', c['structure_outputs_equal_all_rounds'])
print(json.load(open('$S/dev5_engine/settings.json')))"
echo "dev5 done $(date)"
