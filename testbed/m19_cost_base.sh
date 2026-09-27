#!/bin/bash
# M19 L4 cost follow-up: where the fixed load cost of turning entail on comes from. vLLM (eager, the E2 harness),
# states: off; hook = the start-up hook with no adapter (ENTAIL_ONLY=__none__); all; each from the working copy on the
# Windows drive (the 9P mount the E2 runs use) and from a copy on the Linux disk (~/entail_native, git archive of the
# same commit: what a pip install gives). REPS repetitions in a rotated order.
# Usage: bash testbed/m19_cost_base.sh [REPS]; M19_COST_OUT (default testbed/results/m19/l4/cost_base).
set -u
ROOT=<workspace>
cd $ROOT
OUT=$ROOT/${M19_COST_OUT:-testbed/results/m19/l4/cost_base}
mkdir -p $OUT/runs
REPS=${1:-3}
NATIVE=/home/<user>/entail_native
echo "entail at $(cd $ROOT/entail && git rev-parse --short HEAD), native copy of $(cat $NATIVE/.commit 2>/dev/null || echo d53ff10), $(date '+%F %T')" | tee -a $OUT/run.log

one() {  # model label state rep
  local model=$1 label=$2 state=$3 rep=$4 py=/home/<user>/venvs/vllm/bin/python name=vllm_${2}_${3}_${4}
  local where=${state%%_*} kind=${state#*_} shim
  if [ "$where" = native ]; then shim="$NATIVE:$NATIVE/entail/adapters/autoinstall"; else shim="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"; fi
  case $kind in
    off)  env -u PYTHONPATH -u ENTAIL timeout 1200 $py testbed/m3_run_engine.py vllm $model $OUT/runs/$name.json \
            > $OUT/runs/$name.log 2>&1 ;;
    hook) env PYTHONPATH=$shim ENTAIL=load ENTAIL_ONLY=__none__ ENTAIL_LOG_DIR=$OUT/entail_logs \
            timeout 1200 $py testbed/m3_run_engine.py vllm $model $OUT/runs/$name.json > $OUT/runs/$name.log 2>&1 ;;
    all)  env PYTHONPATH=$shim ENTAIL=load ENTAIL_RECORD=$OUT/runs/$name.record.jsonl ENTAIL_LOG_DIR=$OUT/entail_logs \
            timeout 1200 $py testbed/m3_run_engine.py vllm $model $OUT/runs/$name.json > $OUT/runs/$name.log 2>&1 ;;
  esac
  echo "vllm_$label $state rep$rep $(grep -ho 'load=[0-9.]*' $OUT/runs/$name.log | tail -1)" | tee -a $OUT/run.log
}

STATES=(off mount_hook mount_all native_hook native_all)
for rep in $(seq 1 $REPS); do
  for m in "Qwen3-0.6B /home/<user>/models/Qwen3-0.6B" "Llama-3.2-3B /home/<user>/models/Llama-3.2-3B-Instruct"; do
    label=${m%% *}; model=${m##* }
    n=${#STATES[@]}
    for i in $(seq 0 $((n - 1))); do
      s=${STATES[$(( (i + rep) % n ))]}
      [ "$s" = off ] && s=x_off
      one $model $label $s $rep
    done
  done
done
echo "done $(date '+%F %T')" | tee -a $OUT/run.log
