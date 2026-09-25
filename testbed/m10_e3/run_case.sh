#!/bin/bash
# M10 E3: run one reproduction script with entail off, then on (the library hook only), in the given venv.
# Usage: bash testbed/m10_e3/run_case.sh <venv> <case name> <script> [args...]
# Writes testbed/results/m10/e3/cases/<case>_{off,on}.{json,log,record.jsonl}.
set -u
ROOT=<workspace>
cd $ROOT
venv=$1 case=$2 script=$3; shift 3
O=${E3_OUT:-$ROOT/testbed/results/m10/e3/cases}   # M11.7: E3_OUT=.../results/m11/e3 reruns a case for 1.0.1
mkdir -p $O $O/entail_logs
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
PY=/home/<user>/venvs/$venv/bin/python
env -u ENTAIL -u PYTHONPATH HF_HUB_DISABLE_IMPLICIT_TOKEN=1 timeout 3600 $PY $script $O/${case}_off.json "$@" \
  > $O/${case}_off.log 2>&1
echo "off: $(grep -h '^RESULT' $O/${case}_off.log | cut -c1-300 || tail -n 2 $O/${case}_off.log)"
rm -f $O/${case}_on.record.jsonl
env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$O/${case}_on.record.jsonl ENTAIL_LOG_DIR=$O/entail_logs \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 timeout 3600 $PY $script $O/${case}_on.json "$@" > $O/${case}_on.log 2>&1
echo "on:  $(grep -h '^RESULT' $O/${case}_on.log | cut -c1-300 || tail -n 2 $O/${case}_on.log)"
echo "decisions not pass (entail on): $(grep -h '"verdict"' $O/${case}_on.record.jsonl 2>/dev/null | grep -v '"verdict": "pass"' | wc -l)"
