#!/bin/bash
# M5.4: the M5.3 servers under the policy that reports and goes on (testbed/m54_serve.py), then the M5.1 vLLM run with
# its KV block table planted one block short, under the same policy (the engine is not stopped by entail).
# Usage: bash testbed/m54_serve.sh [server:tag ... | kv]   default: every run below, in order
set -u
cd <workspace>
source ~/venvs/vllm/bin/activate
RUNS=${*:-"healthy:on healthy:note healthy:strict keep:on tool:on tool:observe tool:observe_strict kv"}
for r in $RUNS; do
  if [ "$r" = "kv" ]; then
    export PYTHONPATH=<workspace>/entail:<workspace>/entail/tools/autoinstall   # M9.3: the research tools' hook (the library's plus fault injection)
    M54=$(realpath -m ${TESTBED_RESULTS:-testbed/results})/m54   # M9.1: TESTBED_RESULTS moves the results
    mkdir -p $M54
    export VLLM_LOGGING_LEVEL=ERROR ENTAIL_SEED=short_blocks ENTAIL_SEED_AT=4 TESTBED_OUT=$M54
    rm -f $M54/vllm_seeded_report.jsonl
    ENTAIL=load ENTAIL_RECORD=$M54/vllm_seeded_report.jsonl \
      python testbed/m51_engine_run.py vllm seeded_report > $M54/vllm_seeded_report.log 2>&1
    echo "--- vllm seeded_report exit=$?"
    grep -E "^RESULT|entail-seed|broken at|refused at" $M54/vllm_seeded_report.log | cut -c1-260 | head -6
    unset ENTAIL_SEED ENTAIL_SEED_AT TESTBED_OUT PYTHONPATH
    continue
  fi
  python testbed/m54_serve.py ${r%%:*} ${r##*:}
done
