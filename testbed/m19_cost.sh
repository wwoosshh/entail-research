#!/bin/bash
# M19 L4 cost (testbed/M16_PROTOCOL.md 9): load time of testbed/m3_run_engine.py (the E2 harness: vLLM eager, SGLang
# without graphs or radix cache) with entail off, with every adapter, and with one adapter at a time (ENTAIL_ONLY), per
# model and engine, REPS repetitions in a rotated order of states. Same design as M18.6 (results/m18/cost).
# Usage: bash testbed/m19_cost.sh [REPS]; M19_COST_OUT (default testbed/results/m19/l4/cost) picks the folder.
set -u
ROOT=<workspace>
cd $ROOT
OUT=$ROOT/${M19_COST_OUT:-testbed/results/m19/l4/cost}
mkdir -p $OUT/runs
REPS=${1:-3}
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
echo "entail at $(cd $ROOT/entail && git rev-parse --short HEAD), $(date '+%F %T')" | tee -a $OUT/run.log

one() {  # venv engine model label state
  local venv=$1 eng=$2 model=$3 label=$4 state=$5 rep=$6
  local py=/home/<user>/venvs/$venv/bin/python name=${eng}_${label}_${state}_${rep}
  case $state in
    off) env -u PYTHONPATH -u ENTAIL timeout 1200 $py testbed/m3_run_engine.py $eng $model $OUT/runs/$name.json \
           > $OUT/runs/$name.log 2>&1 ;;
    all) env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$OUT/runs/$name.record.jsonl ENTAIL_LOG_DIR=$OUT/entail_logs \
           timeout 1200 $py testbed/m3_run_engine.py $eng $model $OUT/runs/$name.json > $OUT/runs/$name.log 2>&1 ;;
    *)   env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_ONLY=$state ENTAIL_RECORD=$OUT/runs/$name.record.jsonl \
           ENTAIL_LOG_DIR=$OUT/entail_logs timeout 1200 $py testbed/m3_run_engine.py $eng $model $OUT/runs/$name.json \
           > $OUT/runs/$name.log 2>&1 ;;
  esac
  echo "${eng}_${label} $state rep$rep $(grep -ho 'load=[0-9.]*' $OUT/runs/$name.log | tail -1)" | tee -a $OUT/run.log
}

VSTATES=(off all vllm_paths vllm_kernel_reference function_reference triton_launch)
SSTATES=(off all sglang_paths function_reference triton_launch)
for rep in $(seq 1 $REPS); do
  for m in "Qwen3-0.6B /home/<user>/models/Qwen3-0.6B" "Llama-3.2-3B /home/<user>/models/Llama-3.2-3B-Instruct" \
           "gemma-2-2b /home/<user>/models/gemma-2-2b-it"; do
    label=${m%% *}; model=${m##* }
    n=${#VSTATES[@]}
    for i in $(seq 0 $((n - 1))); do
      one vllm vllm $model $label ${VSTATES[$(( (i + rep) % n ))]} $rep
    done
  done
  n=${#SSTATES[@]}
  for i in $(seq 0 $((n - 1))); do
    one sglang sglang /home/<user>/models/Qwen3-0.6B Qwen3-0.6B ${SSTATES[$(( (i + rep) % n ))]} $rep
  done
done
echo "done $(date '+%F %T')" | tee -a $OUT/run.log
