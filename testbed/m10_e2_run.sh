#!/bin/bash
# M10 E2 (testbed/M10_PROTOCOL.md 2): healthy runs of the chosen models on transformers, vLLM and SGLang, each
# engine's defaults, entail off and on (the library hook only), with testbed/m3_run_engine.py.
# Usage: bash testbed/m10_e2_run.sh <list file: one "name<TAB>model dir" per line>
# Every run goes on after a failure; a run that hangs is stopped after 20 minutes. See run.log.
set -u
ROOT=<workspace>
cd $ROOT
R=${M10_E2_OUT:-$ROOT/testbed/results/m10/e2}   # M11.7: results/m11/e2_fresh for the 1.0.1 fresh sample
E=$R/engines
mkdir -p $E $R/entail_logs
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $R/run.log; }

run() {  # venv engine model name mode
  local venv=$1 eng=$2 model=$3 name=$4 mode=$5
  local py=/home/<user>/venvs/$venv/bin/python
  rm -f $E/$name.record.jsonl
  if [ "$mode" = load ]; then
    env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$E/$name.record.jsonl ENTAIL_LOG_DIR=$R/entail_logs \
      timeout 1200 $py $ROOT/testbed/m3_run_engine.py $eng $model $E/$name.json > $E/$name.log 2>&1
  else
    env -u ENTAIL -u PYTHONPATH timeout 1200 $py $ROOT/testbed/m3_run_engine.py $eng $model $E/$name.json \
      > $E/$name.log 2>&1
  fi
  log "$name: $(grep -h '^RESULT' $E/$name.log | cut -c1-200 || echo 'no RESULT line (exit or timeout)')"
}

while IFS=$'\t' read -r name model; do
  [ -z "$name" ] && continue
  for pair in "gpu transformers" "vllm vllm" "sglang sglang"; do
    set -- $pair
    run $1 $2 $model e2_${2}_${name}_off off
    run $1 $2 $model e2_${2}_${name}_load load
  done
done < "$1"
log "list $1 done"
