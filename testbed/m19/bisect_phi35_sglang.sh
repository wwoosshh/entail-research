#!/bin/bash
# M19 L4: the final E2's SGLang + Phi-3.5-mini-instruct run died with a CUDA illegal memory access (entail on,
# d53ff10; the same run passed in M18 E2). Which part of entail sets it off? The E2 harness run under several states.
set -u
ROOT=<workspace>
cd $ROOT
OUT=$ROOT/testbed/results/m19/l4/bisect_phi35
mkdir -p $OUT
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
PY=/home/<user>/venvs/sglang/bin/python
M=/home/<user>/models/Phi-3.5-mini-instruct
run() {  # name, extra env...
  local name=$1; shift
  env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$OUT/$name.record.jsonl ENTAIL_LOG_DIR=$OUT/entail_logs "$@" \
    timeout 900 $PY testbed/m3_run_engine.py sglang $M $OUT/$name.json > $OUT/$name.log 2>&1
  echo "[$(date +%T)] $name: $(grep -h '^RESULT' $OUT/$name.log | cut -c1-160) $(grep -c 'illegal memory access' $OUT/$name.log) illegal"
}
for s in ${STATES:-all1 all2 nopaths notriton onlypaths off}; do
  case $s in
    all*)      run $s ;;
    nopaths)   run $s ENTAIL_NO_PATHS=1 ;;
    notriton)  run $s ENTAIL_SKIP=triton_launch ;;
    onlypaths) run $s ENTAIL_ONLY=sglang_paths ;;
    off)       env -u PYTHONPATH -u ENTAIL timeout 900 $PY testbed/m3_run_engine.py sglang $M $OUT/off.json > $OUT/off.log 2>&1
               echo "[$(date +%T)] off: $(grep -h '^RESULT' $OUT/off.log | cut -c1-160)" ;;
  esac
done
