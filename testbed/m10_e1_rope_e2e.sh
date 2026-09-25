#!/bin/bash
# M10 E1 L1 end to end: GSM8K on a model E1 found exposed - untouched, override with entail off, override with entail on.
# Usage: bash testbed/m10_e1_rope_e2e.sh <model dir> <tag> [N]
set -u
ROOT=<workspace>
cd $ROOT
M=$1 T=$2 N=${3:-200}
R=$ROOT/testbed/results/m10/e1_llm/e2e
mkdir -p $R $R/entail_logs
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
PY=/home/<user>/venvs/vllm/bin/python
env -u ENTAIL -u PYTHONPATH $PY testbed/m10_e1_rope_e2e.py $M ${T}_untouched untouched $N > $R/${T}_untouched.log 2>&1
grep -h '^DONE' $R/${T}_untouched.log
env -u ENTAIL -u PYTHONPATH $PY testbed/m10_e1_rope_e2e.py $M ${T}_override_off override $N > $R/${T}_override_off.log 2>&1
grep -h '^DONE' $R/${T}_override_off.log
rm -f $R/${T}_override_on.record.jsonl
env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$R/${T}_override_on.record.jsonl ENTAIL_LOG_DIR=$R/entail_logs \
  $PY testbed/m10_e1_rope_e2e.py $M ${T}_override_on override $N > $R/${T}_override_on.log 2>&1
grep -h '^DONE' $R/${T}_override_on.log
