#!/bin/bash
# M9.3: the three gaps M9.1 found, fixed before 1.0, measured again on real engines, into testbed/results/m93.
#   U  the unit tests
#   A  healthy runs: 3 models x 3 engines, entail off and on          -> S1 (the template now decided on transformers;
#                                                                         llama3's RoPE carried), S3 (no false alarm)
#   S  SGLang's OpenAI server: the model's template, a template file, a conversation template of its own, strict
#   W  vLLM's OpenAI server: the template decided once (vllm_serve), not again by the tokenizer
#   K  the transformers KV contract's cost (m51_transformers: dynamic and static caches, 8 pairs)  -> S4
#   R  the RoPE override with entail on and GSM8K, after vocabulary v4                             -> fd-rope
# Usage: bash testbed/m93_run.sh [U A S W K R ...]  (all when none). Every step goes on after a failure; see run.log.
set -u
ROOT=<workspace>
cd $ROOT
R=$ROOT/testbed/results/m93
mkdir -p $R/engines $R/rerun
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
PHASES="${*:-U A S W K R}"
G=~/venvs/gpu/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $R/run.log; }

engine() {  # venv engine model mode name [args for m3_run_engine.py]
  local venv=$1 eng=$2 model=$3 mode=$4 name=$5; shift 5
  local py=~/venvs/$venv/bin/python
  rm -f $R/engines/$name.record.jsonl
  if [ "$mode" = load ]; then
    env -u ENTAIL_ON_BROKEN PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$R/engines/$name.record.jsonl $py \
      $ROOT/testbed/m3_run_engine.py $eng $model $R/engines/$name.json "$@" > $R/engines/$name.log 2>&1
  else
    env -u ENTAIL -u PYTHONPATH $py $ROOT/testbed/m3_run_engine.py $eng $model $R/engines/$name.json "$@" \
      > $R/engines/$name.log 2>&1
  fi
  log "$name: $(grep -h '^RESULT' $R/engines/$name.log | cut -c1-220 || tail -n 1 $R/engines/$name.log)"
}

step() {  # name command...: run one step, log its end
  local name=$1; shift
  log "start $name"
  "$@" > $R/$name.log 2>&1
  log "end $name (exit $?): $(tail -n 2 $R/$name.log | tr '\n' ' ' | cut -c1-220)"
}

for phase in $PHASES; do
  case $phase in
    U)
      step unit_tests env PYTHON=$G bash $ROOT/entail/tests/run_all.sh ;;
    A)
      for m in Qwen3-4B Llama-3.2-3B-Instruct gemma-2-2b-it; do
        for pair in "gpu transformers" "vllm vllm" "sglang sglang"; do
          set -- $pair
          engine $1 $2 ~/models/$m off s3_${2}_${m}_off
          engine $1 $2 ~/models/$m load s3_${2}_${m}_load
        done
      done ;;
    S)
      for c in healthy file builtin strict; do
        step sglang_$c ~/venvs/sglang/bin/python testbed/m93_sglang_serve.py $c
      done ;;
    W)
      step vllm_healthy_on env TESTBED_RESULTS=$R/rerun ~/venvs/vllm/bin/python testbed/m54_serve.py healthy on ;;
    K)
      step m51_transformers env TESTBED_RESULTS=$R/rerun $G testbed/m51_transformers.py ;;
    R)
      cd $ROOT/issue_track/rope_override
      env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$R/engines/fd_rope_on.record.jsonl ~/venvs/vllm/bin/python \
        rope_override.py LC_m93 > $R/fd_rope_LC_m93.log 2>&1
      log "fd_rope LC_m93 (entail on): $(grep -h '^DONE' $R/fd_rope_LC_m93.log | cut -c1-220)"
      cd $ROOT ;;
  esac
done
log "all done: $PHASES"
