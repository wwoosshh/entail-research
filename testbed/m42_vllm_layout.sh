#!/bin/bash
# M4.2: the signed repack boundary on a real vLLM (entail/adapters/vllm_layout.py v2, data/signatures.json).
# The same runs as entail/audits/D_LEDGER.md results 3, 4 and 4-1, which measured the check before it moved:
# healthy servers must pass; planted defects (vllm_seed.py) must be refused before the server answers.
# Output: testbed/results/m42/<run>.log and <run>.jsonl (ENTAIL_RECORD: every decision and timing line).
# Usage: bash testbed/m42_vllm_layout.sh [run tag ...]   (all seven runs when none is given)
set -u
source ~/venvs/vllm/bin/activate
export PYTHONPATH=<workspace>/entail:<workspace>/entail/tools/autoinstall   # M9.3: the research tools' hook (the library's plus fault injection)
export VLLM_LOGGING_LEVEL=ERROR
cd <workspace>
OUT=$(realpath -m ${TESTBED_RESULTS:-testbed/results})/m42   # absolute: ENTAIL_RECORD is given it as it is (M9.1)
mkdir -p $OUT
ONLY=" ${*:-} "   # optional run tags: bash m42_vllm_layout.sh transpose_on
run () {  # run <tag> <quant> <seed or -> <ENTAIL> [false-declaration policy]
  if [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $1 "* ]]; then return; fi
  if [ "$3" = "-" ]; then unset ENTAIL_SEED; else export ENTAIL_SEED=$3; fi
  if [ $# -ge 5 ]; then export ENTAIL_FALSE_DECLARATION=$5; else unset ENTAIL_FALSE_DECLARATION; fi
  rm -f $OUT/$1.jsonl
  start=$(date +%s.%N)
  ENTAIL=$4 ENTAIL_RECORD=$OUT/$1.jsonl python entail/audits/vllm_ledger_run.py $2 > $OUT/$1.log 2>&1
  code=$?
  end=$(date +%s.%N)
  echo "--- $1 (quant=$2, seed=$3, ENTAIL=$4, false_declaration=${5:-refuse}) exit=$code seconds=$(echo "$end - $start" | bc)"
  grep -E "^TEXT:|entail-seed|RoleError|quant_method" $OUT/$1.log | cut -c1-300 | head -6
}
run healthy_none    none -               load
run healthy_fp8     fp8  -               load
run roll_on         none roll_output     load
run roll_off        none roll_output     off
run transpose_on    none transpose_all   load
run strided_on      none strided_weights load
run strided_usedata none strided_weights load use_data
