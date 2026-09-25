#!/bin/bash
# M9.1 (and the evaluation scores M9.2 reads): the milestones' measurements again, on the final code, into
# testbed/results/m91 - never over a milestone's own results (TESTBED_RESULTS moves every script's output).
#   A  healthy runs: 3 models x 3 engines, entail off and on          -> S1, S3, S4 (load share)
#   B  the test problems (PROBLEMS.md 1-3) that run without the researcher's ComfyUI -> S2
#   C  cost on the always-on path, per request, and with entail off    -> S4
#   R  the RoPE override with GSM8K: override off/on and the control   -> S2 fd-rope, and an evaluation score (S7)
#   S  the serve cases under the final policy as M5.4 measured them    -> S2 (mk-L07, L11, L13), S1 (requests)
#   T  the load and container problems again under the strict policy  -> S2's strict half
#   D  M7.3's scenarios: locating, and the diagnosis cost               -> S8, S4 (diagnosis)
#   E  M9.2: GSM8K as a post-hoc detector (testbed/M92_PROTOCOL.md)     -> S7
#   V  after M9.3 moved the research tools out of the package: the library's hook on healthy runs, the tools'
#      hook (entail/tools/autoinstall) on planted defects, and the off state of the release commit
# Usage: bash testbed/m91_run.sh [A B C R S T D E V ...]  (A B C R when none). Every step goes on after a failure;
# see run.log.
set -u
ROOT=<workspace>
cd $ROOT
R=${M91_OUT:-$ROOT/testbed/results/m91}   # M11.7: M91_OUT=.../results/m11/m91 reruns B S T for 1.0.1 without touching m91
export TESTBED_RESULTS=$R/rerun
mkdir -p $R/engines $TESTBED_RESULTS
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"   # the library's start-up hook
TOOLS="$ROOT/entail:$ROOT/entail/tools/autoinstall"           # the same plus the research tools (M9.3; phases
#                                                               A-E ran before the move, with the tools in the package)
PHASES="${*:-A B C R}"
G=~/venvs/gpu/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $R/run.log; }

engine() {  # venv engine model mode name [args for m3_run_engine.py]
  local venv=$1 eng=$2 model=$3 mode=$4 name=$5; shift 5
  local py=~/venvs/$venv/bin/python
  rm -f $R/engines/$name.record.jsonl
  if [ "$mode" = load ]; then
    env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$R/engines/$name.record.jsonl $py $ROOT/testbed/m3_run_engine.py \
      $eng $model $R/engines/$name.json "$@" > $R/engines/$name.log 2>&1
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
    A)
      for m in Qwen3-4B Llama-3.2-3B-Instruct gemma-2-2b-it; do
        for pair in "gpu transformers" "vllm vllm" "sglang sglang"; do
          set -- $pair
          engine $1 $2 ~/models/$m off s3_${2}_${m}_off
          engine $1 $2 ~/models/$m load s3_${2}_${m}_load
        done
      done ;;
    B)
      step m43_signed $G testbed/m43_signed.py
      step m3_problems $G testbed/m3_problems.py
      step m55_rolebench $G testbed/m55_rolebench.py
      step m52_rb10 $G testbed/m52_rb10.py
      mkdir -p $R/rb17
      step rb17_prepare $G testbed/m3_rb17.py prepare $R/rb17
      for b in torch_native triton; do
        env -u ENTAIL -u PYTHONPATH ~/venvs/sglang/bin/python rolebench/cases/17_sglang_torch_native_softcap/sglang_logprobs.py \
          $R/rb17/model $b $R/rb17/ids.json $R/rb17/lp_${b}_off.json > $R/rb17/$b.off.log 2>&1
      done
      rm -f $R/rb17/record.jsonl
      env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$R/rb17/record.jsonl ~/venvs/sglang/bin/python \
        rolebench/cases/17_sglang_torch_native_softcap/sglang_logprobs.py $R/rb17/model torch_native $R/rb17/ids.json \
        $R/rb17/lp_torch_native_load.json > $R/rb17/torch_native.load.log 2>&1
      step rb17_compare $G testbed/m3_rb17.py compare $R/rb17
      engine sglang sglang ~/models/gemma-2-2b-it load fd_softcap_sglang_torch_native torch_native
      engine sglang sglang ~/models/gemma-2-2b-it load fd_softcap_sglang_flex_attention flex_attention
      engine gpu transformers ~/models/gemma-2-2b-it load fd_softcap_transformers_paged sdpa --paged
      rm -f $R/engines/fd_shift.record.jsonl
      env PYTHONPATH=$TOOLS ENTAIL=load ENTAIL_SEED=corrupt_at_load ENTAIL_SOURCE=1 ENTAIL_RECORD=$R/engines/fd_shift.record.jsonl \
        ~/venvs/vllm/bin/python testbed/m3_run_engine.py vllm ~/models/Qwen3-4B $R/engines/fd_shift.json \
        > $R/engines/fd_shift.log 2>&1
      log "fd_shift: $(grep -h '^RESULT' $R/engines/fd_shift.log | cut -c1-220)"
      rm -f $R/engines/rope_user.record.jsonl
      env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$R/engines/rope_user.record.jsonl ~/venvs/vllm/bin/python \
        testbed/m3_rope_user.py $R/engines/rope_user.json > $R/engines/rope_user.log 2>&1
      log "rope_user: $(grep -h '^RESULT' $R/engines/rope_user.log | cut -c1-220)"
      step m51_engines bash testbed/m51_engines.sh
      step m51_transformers $G testbed/m51_transformers.py
      step m42_vllm_layout bash testbed/m42_vllm_layout.sh
      step m63_diffusers bash testbed/m63_diffusers.sh i04 m7 s3 lora vae i01
      step m55_l05 $G testbed/m55_l05.py
      step m53_serve bash testbed/m53_serve.sh
      step m55_market_serve bash testbed/m55_market_serve.sh ;;
    C)
      step m55_graph bash testbed/m55_graph.sh
      step m53_overhead env TESTBED_OUT=$TESTBED_RESULTS/m53 ~/venvs/vllm/bin/python testbed/m53_overhead.py
      rm -rf /tmp/entail_off_venv /tmp/entail_off_src && mkdir -p /tmp/entail_off_src         && git -C $ROOT/entail archive HEAD | tar -x -C /tmp/entail_off_src         && python3.12 -m venv /tmp/entail_off_venv         && /tmp/entail_off_venv/bin/python -m pip install -q /tmp/entail_off_src > $R/offstate_install.log 2>&1
      step offstate $G testbed/m91_offstate.py /tmp/entail_off_venv/bin/python ;;
    R)
      cd $ROOT/issue_track/rope_override
      env PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD=$R/engines/fd_rope_on.record.jsonl ~/venvs/vllm/bin/python rope_override.py \
        LC_m91 > $R/fd_rope_LC_m91.log 2>&1
      log "fd_rope LC_m91 (entail on): $(grep -h '^DONE' $R/fd_rope_LC_m91.log | cut -c1-220)"
      env -u ENTAIL -u PYTHONPATH ~/venvs/vllm/bin/python rope_override.py LC_off_m91 > $R/fd_rope_LC_off_m91.log 2>&1
      log "fd_rope LC_off_m91 (entail off): $(grep -h '^DONE' $R/fd_rope_LC_off_m91.log | cut -c1-220)"
      env -u ENTAIL -u PYTHONPATH ~/venvs/vllm/bin/python rope_override.py LA_m91 > $R/fd_rope_LA_m91.log 2>&1
      log "fd_rope LA_m91 (control): $(grep -h '^DONE' $R/fd_rope_LA_m91.log | cut -c1-220)"
      cd $ROOT ;;
    S)  # the serve cases under the final policy, as M5.4 measured them (m54), and the market case's healthy server
      step m54_serve bash testbed/m54_serve.sh
      step m55_market_clean_on bash testbed/m55_market_serve.sh clean_on
      step m54_summarize $G testbed/m54_summarize.py
      step m63_summarize $G testbed/m63_summarize.py ;;
    T)  # the strict half of S2: the load and container problems again with ENTAIL_ON_BROKEN=stop (into rerun_strict)
      export TESTBED_RESULTS=$R/rerun_strict ENTAIL_ON_BROKEN=stop
      mkdir -p $TESTBED_RESULTS
      step m3_problems_strict $G testbed/m3_problems.py
      rm -f $R/engines/fd_shift_strict.record.jsonl
      env PYTHONPATH=$TOOLS ENTAIL=load ENTAIL_SEED=corrupt_at_load ENTAIL_SOURCE=1 \
        ENTAIL_RECORD=$R/engines/fd_shift_strict.record.jsonl ~/venvs/vllm/bin/python testbed/m3_run_engine.py vllm \
        ~/models/Qwen3-4B $R/engines/fd_shift_strict.json > $R/engines/fd_shift_strict.log 2>&1
      log "fd_shift_strict: $(grep -h '^RESULT' $R/engines/fd_shift_strict.log | cut -c1-220)"
      step m42_strict bash testbed/m42_vllm_layout.sh roll_on transpose_on strided_on
      step m51_strict bash testbed/m51_engines.sh vllm_seeded sglang_seeded
      export TESTBED_RESULTS=$R/rerun
      unset ENTAIL_ON_BROKEN ;;
    D)  # M7.3's scenarios (S8, and the diagnosis cost of S4) on the final code
      step m73_locate $G testbed/m73_locate.py
      step m73_tolerance $G testbed/m73_tolerance.py
      step m73_summarize $G testbed/m73_summarize.py ;;
    E)  # M9.2 (S7): the evaluation score as a post-hoc detector (testbed/M92_PROTOCOL.md), into testbed/results/m92
      E=$ROOT/testbed/results/m92
      mkdir -p $E
      gsm() {  # venv engine model name [backend]: entail off
        env -u TESTBED_RESULTS -u ENTAIL -u PYTHONPATH ~/venvs/$1/bin/python testbed/m92_gsm8k.py $2 ~/models/$3 $4 ${5:-} \
          > $E/$4.log 2>&1
        log "m92 $4: $(grep -h '^DONE' $E/$4.log | cut -c1-220)"
      }
      gsm sglang sglang gemma-2-2b-it softcap_sglang_torch_native torch_native
      gsm sglang sglang gemma-2-2b-it softcap_sglang_triton triton
      gsm sglang sglang gemma-2-2b-it softcap_sglang_triton2 triton
      gsm vllm vllm Qwen3-4B shift_control
      gsm vllm vllm Qwen3-4B shift_control2
      rm -f $E/shift_seeded.record.jsonl
      env -u TESTBED_RESULTS PYTHONPATH=$TOOLS ENTAIL=load ENTAIL_SEED=corrupt_at_load ENTAIL_SOURCE=1 \
        ENTAIL_RECORD=$E/shift_seeded.record.jsonl ~/venvs/vllm/bin/python testbed/m92_gsm8k.py vllm ~/models/Qwen3-4B \
        shift_seeded > $E/shift_seeded.log 2>&1
      log "m92 shift_seeded: $(grep -h '^DONE' $E/shift_seeded.log | cut -c1-220)" ;;
    V)  # M9.3: after the research tools left the package - the library's hook alone on healthy runs, the tools' hook
        # (tools/autoinstall) on the planted defects, and the off state of the release commit
      step v_unit_tests env PYTHON=$G bash $ROOT/entail/tests/run_all.sh
      for pair in "gpu transformers" "vllm vllm" "sglang sglang"; do
        set -- $pair
        engine $1 $2 ~/models/Qwen3-4B load v_${2}_Qwen3-4B_load
      done
      rm -f $R/engines/v_fd_shift.record.jsonl
      env PYTHONPATH=$TOOLS ENTAIL=load ENTAIL_SEED=corrupt_at_load ENTAIL_SOURCE=1 \
        ENTAIL_RECORD=$R/engines/v_fd_shift.record.jsonl ~/venvs/vllm/bin/python testbed/m3_run_engine.py vllm \
        ~/models/Qwen3-4B $R/engines/v_fd_shift.json > $R/engines/v_fd_shift.log 2>&1
      log "v_fd_shift: $(grep -h '^RESULT\|broken at\|entail-seed' $R/engines/v_fd_shift.log | head -3 | cut -c1-200 | tr '\n' ' ')"
      export TESTBED_RESULTS=$R/rerun_v
      mkdir -p $TESTBED_RESULTS
      step v_m51_seeded bash testbed/m51_engines.sh vllm_seeded sglang_seeded
      step v_market_bug_on bash testbed/m55_market_serve.sh bug_on
      export TESTBED_RESULTS=$R/rerun_strict
      step v_m51_transformers_strict env ENTAIL_ON_BROKEN=stop $G testbed/m51_transformers.py
      export TESTBED_RESULTS=$R/rerun
      rm -rf /tmp/entail_off_venv /tmp/entail_off_src && mkdir -p /tmp/entail_off_src \
        && git -C $ROOT/entail archive HEAD | tar -x -C /tmp/entail_off_src \
        && python3.12 -m venv /tmp/entail_off_venv \
        && /tmp/entail_off_venv/bin/python -m pip install -q /tmp/entail_off_src > $R/offstate_v_install.log 2>&1
      step offstate_v env M91_OFFSTATE_OUT=$R/offstate_v.json $G testbed/m91_offstate.py /tmp/entail_off_venv/bin/python ;;
  esac
done
log "all done: $PHASES"
