#!/bin/bash
# M19 L3: run definition cases with entail loaded as a user loads it (ENTAIL=load, the start-up shim), one process
# per case.  bash lowlevel/l2/run_l3.sh <out dir> <venv> <case> [<case> ...]
set -u
out=$1; venv=$2; shift 2
source ~/venvs/$venv/bin/activate
cd ~/ai_compiler
mkdir -p "$out"
export PYTHONPATH=~/ai_compiler/entail:~/ai_compiler/entail/entail/adapters/autoinstall
export ENTAIL=load ENTAIL_LOG_DIR=off
for c in "$@"; do
  rm -f "$out/$c.record.jsonl"      # the record is appended to: a case's file holds this run only
  ENTAIL_RECORD="$out/$c.record.jsonl" python lowlevel/l2/l3_definitions.py "$c" "$out/$c.json" > "$out/$c.log" 2>&1
  echo "rc=$? $(tail -1 "$out/$c.log")"
done
