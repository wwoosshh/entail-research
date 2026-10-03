source ~/venvs/vllm/bin/activate; export PATH=/usr/local/cuda/bin:$PATH
cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
R=$HOME/ai_compiler/lowlevel/l5/guarantee/results
python eval/block_fp8_guarantee/env_record.py $R/env_freeze_v2.json 2>&1 | tail -1
python eval/block_fp8_guarantee/freeze.py $R/calibration_v1.json $R/env_freeze_v2.json $R/freeze_v2.json --check static --integrity epoch --records changes --engine-budget 1.10 --what "M19 L5.4b block FP8 guarantee, check static (the consumer kernel's TTIR read once per launch configuration), integrity epoch: implementation, tolerances and harness frozen before the holdout" 2>&1 | tail -1
python -c "
import json; m=json.load(open('$R/freeze_v2.json')); print(m['entail_git'], m['plan']['check'], m['plan']['integrity'], m['plan']['records'], m['plan']['c_acc'], m['oracle']['c'], m['cost_budget'])"
