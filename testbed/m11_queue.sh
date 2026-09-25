#!/bin/bash
# M11.7 queue, one GPU: after the 30-model rerun (m11_e2_run.sh) finishes, the fresh sample off and on
# (m10_e2_run.sh into results/m11/e2_fresh), the test problems again (m91_run.sh B S T into results/m11/m91) and the
# E3 DSPARK case (sg33493 into results/m11/e3), one after another. See results/m11/queue.log.
# Usage: bash testbed/m11_queue.sh
set -u
ROOT=<workspace>
cd $ROOT
M=$ROOT/testbed/results/m11
mkdir -p $M
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $M/queue.log; }

until grep -q " done" $M/e2/run.log 2>/dev/null; do sleep 60; done
log "30-model rerun done"
until grep -q "^exit" /tmp/m11_download.log 2>/dev/null; do sleep 60; done
log "downloads done: $(tail -n 2 /tmp/m11_download.log | head -n 1)"
if [ -s $M/e2/list_fresh.txt ]; then
  M10_E2_OUT=$M/e2_fresh bash testbed/m10_e2_run.sh $M/e2/list_fresh.txt
  log "fresh sample done"
else
  log "no fresh models to run"
fi
M91_OUT=$M/m91 bash testbed/m91_run.sh B S T
log "test problems (m91 B S T) done"
E3_OUT=$M/e3 bash testbed/m10_e3/run_case.sh sglang sg33493 testbed/m10_e3/sg33493.py \
  /home/<user>/models/m10/openbmb__MiniCPM5-2B /home/<user>/models/m10/openbmb__MiniCPM5-2B-DSpark
log "e3 sg33493 done"
log "queue done"
