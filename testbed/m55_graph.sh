#!/bin/bash
# M5.5, S4 on the CUDA graph path (testbed/m55_graph.py): vLLM default settings, the engine core in this process.
set -u
cd <workspace>
source ~/venvs/vllm/bin/activate
export PYTHONPATH=<workspace>/entail:<workspace>/entail/entail/adapters/autoinstall
export ENTAIL=load VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=WARNING
export M55_SYNC=${1:-0}
unset ENTAIL_SEED ENTAIL_ON_BROKEN ENTAIL_POLICY ENTAIL_RECORD
python testbed/m55_graph.py 2>&1 | grep -v "^INFO" | tail -${TAIL_LINES:-30}
