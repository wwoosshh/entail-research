#!/bin/bash
# M11.7, second pass: after the tie contract was changed to read what the loader left in the model (no full read of
# a shipped head), the 30 models again with entail on (m11_e2_run.sh), the load problems again in both policies
# (m3_problems.py into results/m11/m91/rerun and rerun_strict: rb-07 runs through transformers' tie_weights), then
# the problems file and the summary. One GPU, one after another. See results/m11/queue2.log.
# Usage: bash testbed/m11_rerun2.sh
set -u
ROOT=<workspace>
cd $ROOT
M=$ROOT/testbed/results/m11
G=$HOME/venvs/gpu/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $M/queue2.log; }

log "entail $($G -c 'import sys; sys.path.insert(0, "'$ROOT'/entail"); import entail; print(entail.__version__)') second pass"
bash testbed/m11_e2_run.sh $M/e2/list_30.txt
log "30-model rerun (second pass) done"
env TESTBED_RESULTS=$M/m91/rerun $G testbed/m3_problems.py > $M/m91/m3_problems_pass2.log 2>&1
log "m3_problems default done (exit $?)"
env TESTBED_RESULTS=$M/m91/rerun_strict ENTAIL_ON_BROKEN=stop $G testbed/m3_problems.py > $M/m91/m3_problems_strict_pass2.log 2>&1
log "m3_problems strict done (exit $?)"
env M91_DIR=$M/m91 $G testbed/m91_problems.py > $M/m91/problems_pass2.log 2>&1
log "m91_problems done (exit $?)"
$G testbed/m11_summarize.py > $M/summarize_pass2.log 2>&1
log "summary done (exit $?)"
log "second pass done"
