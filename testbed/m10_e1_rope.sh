#!/bin/bash
# M10 E1 L1: the RoPE override at config level, entail off and on (testbed/m10_e1_rope.py).
ROOT=<workspace>
cd $ROOT
R=$ROOT/testbed/results/m10/e1_llm
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
env -u ENTAIL -u PYTHONPATH ~/venvs/vllm/bin/python testbed/m10_e1_rope.py off > $R/rope_off.log 2>&1
tail -n 1 $R/rope_off.log
mkdir -p $R/entail_logs_rope
env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_LOG_DIR=$R/entail_logs_rope ~/venvs/vllm/bin/python testbed/m10_e1_rope.py on \
  > $R/rope_on.log 2>&1
tail -n 1 $R/rope_on.log
