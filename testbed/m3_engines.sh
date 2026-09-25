#!/bin/bash
# M3.5 engine runs on the GPU. Each run is its own process; entail is on only in "load" runs (the start-up shim on
# PYTHONPATH, ENTAIL=load, ENTAIL_RECORD collecting every decision and timing from every process of the engine).
# Results: testbed/results/m3/<name>.json, .log, .record.jsonl     Run in WSL:  bash ~/ai_compiler/testbed/m3_engines.sh
set -u
ROOT=~/ai_compiler
OUT=$ROOT/testbed/results/m3
SHIM="$ROOT/entail:$ROOT/entail/entail/adapters/autoinstall"
TOOLS="$ROOT/entail:$ROOT/entail/tools/autoinstall"   # M9.3: the research tools' hook, for the planted defect
mkdir -p "$OUT"

run() {  # venv engine model mode name [args for m3_run_engine.py]
  local venv=$1 engine=$2 model=$3 mode=$4 name=$5; shift 5
  local py=~/venvs/$venv/bin/python
  rm -f "$OUT/$name.record.jsonl"
  if [ "$mode" = load ]; then
    PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD="$OUT/$name.record.jsonl" \
      "$py" "$ROOT/testbed/m3_run_engine.py" "$engine" "$model" "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  else
    "$py" "$ROOT/testbed/m3_run_engine.py" "$engine" "$model" "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  fi
  grep -h "^RESULT" "$OUT/$name.log" || tail -n 3 "$OUT/$name.log"
}

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# S3, S4, S1: healthy models with each engine's default settings, entail off and on
for m in Qwen3-4B Llama-3.2-3B-Instruct gemma-2-2b-it; do
  for pair in "gpu transformers" "vllm vllm" "sglang sglang"; do
    set -- $pair
    run "$1" "$2" ~/models/$m off "s3_${2}_${m}_off"
    run "$1" "$2" ~/models/$m load "s3_${2}_${m}_load"
  done
done

# fd-softcap: Gemma 2 2B on the backends that drop softcap
run sglang sglang ~/models/gemma-2-2b-it load fd_softcap_sglang_torch_native torch_native
run sglang sglang ~/models/gemma-2-2b-it load fd_softcap_sglang_flex_attention flex_attention
run gpu transformers ~/models/gemma-2-2b-it load fd_softcap_transformers_paged sdpa --paged

# fd-shift: a row shifted while the checkpoint is read (vllm_seed), with the source check on
rm -f "$OUT/fd_shift.record.jsonl"
PYTHONPATH=$TOOLS ENTAIL=load ENTAIL_SEED=corrupt_at_load ENTAIL_SOURCE=1 ENTAIL_RECORD="$OUT/fd_shift.record.jsonl" \
  ~/venvs/vllm/bin/python "$ROOT/testbed/m3_run_engine.py" vllm ~/models/Qwen3-4B "$OUT/fd_shift.json" \
  > "$OUT/fd_shift.log" 2>&1
grep -h "^RESULT" "$OUT/fd_shift.log" || tail -n 3 "$OUT/fd_shift.log"

# fd-rope: Llama 3.2 given its own rope_scaling at launch (issue_track/rope_override LC), with the v2 adapters, and
# the same-day control without an override (LA)
cd "$ROOT/issue_track/rope_override"
rm -f "$OUT/fd_rope.record.jsonl"
PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD="$OUT/fd_rope.record.jsonl" ~/venvs/vllm/bin/python rope_override.py \
  LC_entail_v2 > results/LC_entail_v2.log 2>&1
grep -h "^DONE" results/LC_entail_v2.log | cut -c1-300
~/venvs/vllm/bin/python rope_override.py LA_m35 > results/LA_m35.log 2>&1
grep -h "^DONE" results/LA_m35.log | cut -c1-300
cd "$ROOT"

# rolebench 17: SGLang torch_native with a binding cap, off and on, against triton
~/venvs/gpu/bin/python "$ROOT/testbed/m3_rb17.py" prepare "$OUT/rb17"
for b in torch_native triton; do
  ~/venvs/sglang/bin/python "$ROOT/rolebench/cases/17_sglang_torch_native_softcap/sglang_logprobs.py" \
    "$OUT/rb17/model" $b "$OUT/rb17/ids.json" "$OUT/rb17/lp_${b}_off.json" > "$OUT/rb17/$b.off.log" 2>&1
done
rm -f "$OUT/rb17/record.jsonl"
PYTHONPATH=$SHIM ENTAIL=load ENTAIL_RECORD="$OUT/rb17/record.jsonl" ~/venvs/sglang/bin/python \
  "$ROOT/rolebench/cases/17_sglang_torch_native_softcap/sglang_logprobs.py" \
  "$OUT/rb17/model" torch_native "$OUT/rb17/ids.json" "$OUT/rb17/lp_torch_native_load.json" > "$OUT/rb17/torch_native.load.log" 2>&1
~/venvs/gpu/bin/python "$ROOT/testbed/m3_rb17.py" compare "$OUT/rb17"
echo "all done"
