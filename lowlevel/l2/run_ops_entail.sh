#!/bin/bash
# M19 L3.3b: the op-level generic checks (ops.py) with entail loaded (ENTAIL=load, the start-up shim), one process per
# spec; ENTAIL_SKIP from the caller's environment is kept.   bash lowlevel/l2/run_ops_entail.sh <out dir> <venv> <spec>...
set -u
out=$1; venv=$2; shift 2
source ~/venvs/$venv/bin/activate
cd ~/ai_compiler
mkdir -p "$out"
export PYTHONPATH=~/ai_compiler/entail:~/ai_compiler/entail/entail/adapters/autoinstall
export ENTAIL=load ENTAIL_LOG_DIR=off
for s in "$@"; do
  rm -f "$out/$s.record.jsonl"
  ENTAIL_RECORD="$out/$s.record.jsonl" python lowlevel/l2/ops.py "$out" "$s" > "$out/$s.log" 2>&1
  echo "rc=$? $(grep -v '^\[entail\]\|^INFO\|^WARNING' "$out/$s.log" | tail -1)"
done
