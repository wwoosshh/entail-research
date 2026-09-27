#!/bin/bash
# M15.6: the 38 models of M10/M11 E2 again with entail 1.1.0 (vocabulary v6) on (the library hook only), each engine's
# defaults, with testbed/m3_run_engine.py. Only the "load" runs: the "off" runs of M10 E2 are the control (same
# engines, venvs, models and greedy decoding), and are copied next to these results for the summarizer
# (m10_e2_summarize.py with M10_E2_DIR=<result folder>). The off runs of the 8 models added in M11 are in
# results/m11/e2_fresh/engines and are copied by hand.
# Usage: E2_DIR=testbed/results/m15/e2_final bash testbed/m15_e2_run.sh testbed/results/m11/e2/list_30.txt testbed/results/m11/e2/list_fresh.txt
# E2_DIR (default testbed/results/m15/e2, relative to the repository) picks the result folder.
# Every run goes on after a failure; a run that hangs is stopped after 20 minutes. See run.log.
set -u
ROOT=<workspace>
cd $ROOT
R=$ROOT/${E2_DIR:-testbed/results/m15/e2}
E=$R/engines
OLD=$ROOT/testbed/results/m10/e2/engines
FRESH=$ROOT/testbed/results/m11/e2_fresh/engines
mkdir -p $E $R/entail_logs
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $R/run.log; }

run() {  # venv engine model name
  local venv=$1 eng=$2 model=$3 name=$4
  local py=/home/<user>/venvs/$venv/bin/python
  rm -f $E/$name.record.jsonl
  env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$E/$name.record.jsonl ENTAIL_LOG_DIR=$R/entail_logs \
    timeout 1200 $py $ROOT/testbed/m3_run_engine.py $eng $model $E/$name.json > $E/$name.log 2>&1
  log "$name: $(grep -h '^RESULT' $E/$name.log | cut -c1-200 || echo 'no RESULT line (exit or timeout)')"
}

log "entail $(/home/<user>/venvs/gpu/bin/python -c 'import sys; sys.path.insert(0, "'$ROOT'/entail"); import entail; print(entail.__version__)') at $(cd $ROOT/entail && git rev-parse --short HEAD)"
for list in "$@"; do
  while IFS=$'\t' read -r name model; do
    [ -z "$name" ] && continue
    for pair in "gpu transformers" "vllm vllm" "sglang sglang"; do
      venv=${pair%% *}; eng=${pair##* }
      for f in $OLD/e2_${eng}_${name}_off.json $OLD/e2_${eng}_${name}_off.log \
               $FRESH/e2_${eng}_${name}_off.json $FRESH/e2_${eng}_${name}_off.log; do
        [ -f "$f" ] && cp -n "$f" $E/
      done
      run $venv $eng $model e2_${eng}_${name}_load
    done
  done < "$list"
  log "list $list done"
done
