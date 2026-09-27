#!/bin/bash
# M19 L5.2: the frozen research tools (lowlevel/l5/frozen_l2) on the replay 3/4 cases they can run, in the order
# and with the settings of lowlevel/l5/PROTOCOL.md section 3. entail is off. One GPU job at a time.
#   bash lowlevel/l5/run_l52.sh            (from WSL; writes lowlevel/l5/results/)
set -u
T=~/ai_compiler/lowlevel/l5/frozen_l2
R=~/ai_compiler/lowlevel/l5/results
mkdir -p "$R"
unset ENTAIL
NVFP4=/home/<user>/models/replay4/RedHatAI__Qwen3-8B-NVFP4
VL2B=/home/<user>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
Q4B=/home/<user>/models/m10/Qwen__Qwen3-4B-Instruct-2507
E_NVFP4='{"dtype": "float16", "max_model_len": 1024, "gpu_memory_utilization": 0.85}'
E_VL='{"max_model_len": 2048, "gpu_memory_utilization": 0.8, "limit_mm_per_prompt": {"image": 1, "video": 0}}'
E_Q4B='{"max_model_len": 1024, "gpu_memory_utilization": 0.85}'

log() { echo "$(date '+%F %T') $*" | tee -a "$R/run.log"; }
diskok() {
  local free_gb
  free_gb=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
  log "C: free ${free_gb} GB"
  if [ "$free_gb" -lt 5 ]; then log "STOP: C: free below 5 GB"; exit 3; fi
}

run() {  # name venv model engine_json images [modes...]
  local name=$1 venv=$2 model=$3 eng=$4 img=$5
  shift 5
  diskok
  log "start $name ($venv) modes: ${*:-all}"
  local t0 rc
  t0=$(date +%s)
  L2_ENGINE="$eng" L2_IMAGES="$img" VLLM_PY=~/venvs/$venv/bin/python python3 "$T/run_vllm.py" "$model" "$R/$name" "$@" \
    > "$R/$name.out" 2>&1
  rc=$?
  log "end $name rc=$rc wall=$(( $(date +%s) - t0 )) s"
}

layers() {  # name venv model engine_json
  local name=$1 venv=$2 model=$3 eng=$4
  local d="$R/layers_$name"
  diskok
  mkdir -p "$d"
  local t0 rc
  t0=$(date +%s)
  log "start layers $name ($venv)"
  (cd "$T" && timeout 3600 ~/venvs/$venv/bin/python layers.py vllm "$model" "$d/engine" "$eng") > "$d/engine.out" 2>&1
  rc=$?
  log "layers $name engine dump rc=$rc"
  if [ $rc -eq 0 ]; then
    (cd "$T" && timeout 3600 ~/venvs/gpu/bin/python layers.py hf "$model" "$d/ref_fp32" float32 "$d/engine") \
      > "$d/ref_fp32.out" 2>&1
    log "layers $name ref float32 rc=$?"
    (cd "$T" && timeout 3600 ~/venvs/gpu/bin/python layers.py hf "$model" "$d/ref_bf16" bfloat16 "$d/engine") \
      > "$d/ref_bf16.out" 2>&1
    log "layers $name ref bfloat16 rc=$?"
    (cd "$T" && ~/venvs/gpu/bin/python layers.py compare "$d/engine" "$d/ref_fp32" "$d/ref_bf16") > "$d/compare.out" 2>&1
    log "layers $name compare rc=$?"
  fi
  find "$d" -name '*.pt' -delete     # the dumps are large; meta.json and the comparison stay (PROTOCOL.md 3)
  log "end layers $name wall=$(( $(date +%s) - t0 )) s"
}

log "L5.2 start; $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"
log "tools sha256: $(cd "$T" && sha256sum *.py *.json | sha256sum | cut -c1-16) (sum over frozen_l2.sha256's files)"
run vl33560_0160 vllm0160 "$NVFP4" "$E_NVFP4" ""
run vl33560_0300 vllm     "$NVFP4" "$E_NVFP4" ""
run vl43602_0220 vllm0220 "$VL2B"  "$E_VL"    "1"
run vl43602_0300 vllm     "$VL2B"  "$E_VL"    "1"
run vl27390_0110 vllm0110 "$Q4B"   "$E_Q4B"   ""  base
run vl27390_0300 vllm     "$Q4B"   "$E_Q4B"   ""
layers vl43602_0220 vllm0220 "$VL2B" "$E_VL"
layers vl43602_0300 vllm     "$VL2B" "$E_VL"
layers vl27390_0110 vllm0110 "$Q4B"  "$E_Q4B"
layers vl27390_0300 vllm     "$Q4B"  "$E_Q4B"
log "L5.2 done"
