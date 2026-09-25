#!/bin/bash
# M5.5: market cases L07 and L13 simulated on a vLLM 0.30 server (testbed/m55_market_serve.py), one server at a time.
set -u
cd <workspace>
source ~/venvs/vllm/bin/activate
for tag in ${*:-clean_off bug_off bug_on bug_strict}; do
  python testbed/m55_market_serve.py $tag
done
