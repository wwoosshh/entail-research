#!/bin/bash
# M19 L4, S4 with the whole library (testbed/m19_s4.py): ROUNDS rounds, each running the states on, off and control
# as separate processes in a rotated order. Usage: bash testbed/m19_s4.sh [ROUNDS]
# M19_S4_OUT (default testbed/results/m19/l4/s4) picks the result folder.
set -u
ROOT=<workspace>
cd $ROOT
OUT=$ROOT/${M19_S4_OUT:-testbed/results/m19/l4/s4}
mkdir -p $OUT
ROUNDS=${1:-4}
PY=/home/<user>/venvs/vllm/bin/python
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
echo "entail $(cd $ROOT/entail && git rev-parse --short HEAD), rounds $ROUNDS, $(date '+%F %T')" | tee -a $OUT/run.log
states=(on off control)
for r in $(seq 1 $ROUNDS); do
  k=$(( (r - 1) % 3 ))
  for i in 0 1 2; do
    s=${states[$(( (k + i) % 3 ))]}
    if [ "$s" = on ]; then
      env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_LOG_DIR=$OUT/entail_logs VLLM_LOGGING_LEVEL=WARNING \
        M19_S4_STATE=$s M19_S4_ROUND=$r M19_S4_OUT=$OUT $PY testbed/m19_s4.py > $OUT/${s}_$r.log 2>&1
    else
      env -u PYTHONPATH -u ENTAIL VLLM_LOGGING_LEVEL=WARNING \
        M19_S4_STATE=$s M19_S4_ROUND=$r M19_S4_OUT=$OUT $PY testbed/m19_s4.py > $OUT/${s}_$r.log 2>&1
    fi
    echo "[$(date +%T)] round $r $s exit $? $(grep -h 'B= 32' $OUT/${s}_$r.log | tail -1)" | tee -a $OUT/run.log
  done
done
