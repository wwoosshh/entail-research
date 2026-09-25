#!/bin/bash
# M5.1: the KV container contract on vLLM and SGLang after the rules moved to the core (testbed/m51_engine_run.py).
# Off, on (load), and on with a planted defect: vLLM's block table one block short (vllm_seed short_blocks),
# SGLang's books one slot short at decode batch 4 (sglang_seed kv_short).
# Output: testbed/results/m51/<engine>_<tag>.{log,json,jsonl}; the .jsonl is ENTAIL_RECORD (decisions, summaries).
# Usage: bash testbed/m51_engines.sh [run tag ...]
set -u
cd <workspace>
OUT=$(realpath -m ${TESTBED_RESULTS:-testbed/results})/m51   # absolute: ENTAIL_RECORD is given it as it is (M9.1)
mkdir -p $OUT
ONLY=" ${*:-} "
run () {  # run <venv> <engine> <tag> <ENTAIL> <seed or ->
  if [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $2_$3 "* ]]; then return; fi
  source ~/venvs/$1/bin/activate
  export PYTHONPATH=<workspace>/entail:<workspace>/entail/tools/autoinstall   # M9.3: the research tools' hook (the library's plus fault injection)
  export VLLM_LOGGING_LEVEL=ERROR
  if [ "$5" = "-" ]; then unset ENTAIL_SEED ENTAIL_SEED_AT; else export ENTAIL_SEED=$5 ENTAIL_SEED_AT=4; fi
  rm -f $OUT/$2_$3.jsonl
  ENTAIL=$4 ENTAIL_RECORD=$OUT/$2_$3.jsonl python testbed/m51_engine_run.py $2 $3 > $OUT/$2_$3.log 2>&1
  echo "--- $2 $3 (ENTAIL=$4, seed=$5) exit=$?"
  grep -E "^RESULT|entail-seed|refused at" $OUT/$2_$3.log | cut -c1-260 | head -4
  deactivate
}
run vllm   vllm   off     off  -
run vllm   vllm   on      load -
run vllm   vllm   seeded  load short_blocks
run sglang sglang off     off  -
run sglang sglang on      load -
run sglang sglang seeded  load kv_short
