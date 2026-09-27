#!/bin/bash
# Repeat one engine spec off and on to tell the engine's own run-to-run variation from entail's effect.
# bash lowlevel/l2/run_repeat.sh <out dir> <venv> <spec.json> <name> <times>
set -u
out=$1; venv=$2; spec=$3; name=$4; times=$5
cd ~/ai_compiler
mkdir -p "$out"
for i in $(seq 1 "$times"); do
  for side in off on; do
    (
      source ~/venvs/$venv/bin/activate
      export VLLM_ENABLE_V1_MULTIPROCESSING=0
      if [ $side = on ]; then
        export PYTHONPATH=~/ai_compiler/entail:~/ai_compiler/entail/entail/adapters/autoinstall ENTAIL=load \
               ENTAIL_LOG_DIR=off ENTAIL_RECORD="$out/$name.$side$i.record.jsonl"
        rm -f "$out/$name.$side$i.record.jsonl"
      fi
      python lowlevel/l2/vllm_worker.py "$spec" "$out/$name.$side$i.json" > "$out/$name.$side$i.log" 2>&1
      echo "$name $side$i rc=$?"
    )
  done
done
