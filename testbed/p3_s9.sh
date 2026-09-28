#!/bin/bash
# S9 (ROADMAP product track P3; LIBRARY_DESIGN.md 13.6): the two safety modes on vLLM 0.30, starts in sequence.
# bash testbed/p3_s9.sh [a] [b] [c] [d] [e] [f] [g] [h]
#   a  the selective safe path (ENTAIL_SAFE unset: auto) on the two configurations whose paths disagreed in M19 L3.3c
#      (Qwen3.5-4B-NVFP4 and Nemotron-3-Nano-4B with n-gram speculative decoding): three starts each, one log folder
#   b  the explicit safe mode on Qwen3-0.6B (eager): clean, a planted fault inside the optimizations (custom kernel)
#      and one outside them (attention decode), each with ENTAIL_SAFE=off and =all, one log folder per start;
#      the kernel reference adapter is left out (ENTAIL_SKIP) so that it does not repair the planted kernel fault
#      first - two informational starts put it back and try auto
set -u
cd ~/ai_compiler
OUT=testbed/results/p3
mkdir -p $OUT
Q35Q=$(ls -d ~/.cache/huggingface/hub/models--AxionML--Qwen3.5-4B-NVFP4/snapshots/*/ | head -1)
NEMO=/home/<user>/models/m10/nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16
Q06=/home/<user>/models/Qwen3-0.6B
SPEC='"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}'

start() {   # start <log folder> <result name> <engine kwargs JSON> [env ...]
  local dir=$1 name=$2 kw=$3; shift 3
  (
    source ~/venvs/vllm/bin/activate
    export VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=~/ai_compiler/entail:~/ai_compiler/entail/entail/adapters/autoinstall \
           ENTAIL=load ENTAIL_LOG_DIR=$dir "$@"
    timeout 900 python testbed/p3_s9.py "$kw" "$OUT/$name.json" > "$OUT/$name.log" 2>&1
    echo "$name rc=$? $(grep -E '^\{|^pairs' "$OUT/$name.log" | tr '\n' ' ' | cut -c1-600)"
  )
}

parts=${*:-a b}
for part in $parts; do
  case $part in
    a)
      for cfg in q35q nemotron; do
        dir=$OUT/s9a_$cfg; rm -rf $dir; mkdir -p $dir
        if [ $cfg = q35q ]; then
          kw="{\"model\": \"$Q35Q\", \"enforce_eager\": true, \"max_model_len\": 1024, \"gpu_memory_utilization\": 0.85, \"max_num_seqs\": 4, $SPEC}"
        else
          kw="{\"model\": \"$NEMO\", \"enforce_eager\": true, \"max_model_len\": 2048, \"gpu_memory_utilization\": 0.85, \"max_num_seqs\": 4, $SPEC}"
        fi
        for n in 1 2 3; do
          start $dir s9a_$cfg.start$n "$kw"
          cp $dir/safe_paths.json $OUT/s9a_$cfg.start$n.safe_paths.json 2>/dev/null
        done
      done ;;
    b)
      kw="{\"model\": \"$Q06\", \"enforce_eager\": true, \"max_model_len\": 2048, \"gpu_memory_utilization\": 0.5}"
      for plant in clean inside outside; do
        for mode in off all; do
          dir=$OUT/s9b_${plant}_$mode; rm -rf $dir; mkdir -p $dir
          p=$plant; [ $plant = clean ] && p=""
          start $dir s9b_${plant}_$mode "$kw" ENTAIL_SAFE=$mode P3_PLANT=$p ENTAIL_SKIP=vllm_kernel_reference
        done
      done
      dir=$OUT/s9b_inside_off_full; rm -rf $dir; mkdir -p $dir
      start $dir s9b_inside_off_full "$kw" ENTAIL_SAFE=off P3_PLANT=inside
      dir=$OUT/s9b_inside_auto; rm -rf $dir; mkdir -p $dir
      start $dir s9b_inside_auto "$kw" P3_PLANT=inside ENTAIL_SKIP=vllm_kernel_reference ;;
    c|d)
      # c: the faults at the kernels themselves, with the table as it was (custom_kernels = custom_ops only);
      # d: the same after the table also turns the IR ops' kernels off (ir_op_priority), and the clean and outside
      # starts again. Each start its own log folder; "auto2" is the second start in the folder of "auto1".
      kw="{\"model\": \"$Q06\", \"enforce_eager\": true, \"max_model_len\": 2048, \"gpu_memory_utilization\": 0.5}"
      if [ $part = c ]; then runs="inside_kernel:off inside_kernel:all inside_op:off inside_op:all inside:auto2"
      else runs="clean:all inside_kernel:all inside_op:all outside:all inside_kernel:auto1 inside_kernel:auto2"; fi
      for r in $runs; do
        plant=${r%%:*}; mode=${r##*:}
        p=$plant; [ $plant = clean ] && p=""
        if [ $mode = auto2 ] && [ $part = c ]; then
          dir=$OUT/s9b_inside_auto; name=s9b_inside_auto2
        else
          name=s9${part}_${plant}_$mode; dir=$OUT/s9${part}_${plant}_${mode%[12]}
          [ $mode != auto2 ] && { rm -rf $dir; mkdir -p $dir; }
        fi
        env=(P3_PLANT=$p ENTAIL_SKIP=vllm_kernel_reference)
        case $mode in off|all) env+=(ENTAIL_SAFE=$mode) ;; esac
        start $dir $name "$kw" "${env[@]}"
      done ;;
    e)
      # e: what the found safe path costs in speed - each part-a configuration with speculative decoding on
      # (ENTAIL_SAFE=off, a fresh folder) and as its safe path starts it (auto, in the part-a folder: off)
      for cfg in q35q nemotron; do
        if [ $cfg = q35q ]; then
          kw="{\"model\": \"$Q35Q\", \"enforce_eager\": true, \"max_model_len\": 1024, \"gpu_memory_utilization\": 0.85, \"max_num_seqs\": 4, $SPEC}"
        else
          kw="{\"model\": \"$NEMO\", \"enforce_eager\": true, \"max_model_len\": 2048, \"gpu_memory_utilization\": 0.85, \"max_num_seqs\": 4, $SPEC}"
        fi
        dir=$OUT/s9e_${cfg}_spec; rm -rf $dir; mkdir -p $dir
        start $dir s9e_${cfg}_spec_on "$kw" ENTAIL_SAFE=off P3_BENCH=1
        start $OUT/s9a_$cfg s9e_${cfg}_safe_path "$kw" P3_BENCH=1
      done ;;
    f)
      # f: the explicit safe mode as a user meets it - the same configuration started with the optimizations on
      # (the fault shows), then with ENTAIL_SAFE=all, in one log folder: what entail says on each side
      kw="{\"model\": \"$Q06\", \"enforce_eager\": true, \"max_model_len\": 2048, \"gpu_memory_utilization\": 0.5}"
      for plant in inside_kernel outside; do
        dir=$OUT/s9f_$plant; rm -rf $dir; mkdir -p $dir
        start $dir s9f_${plant}_1off "$kw" ENTAIL_SAFE=off P3_PLANT=$plant ENTAIL_SKIP=vllm_kernel_reference
        start $dir s9f_${plant}_2all "$kw" ENTAIL_SAFE=all P3_PLANT=$plant ENTAIL_SKIP=vllm_kernel_reference
      done ;;
    g)
      # g: what a start costs while the safe path searches - the self-check's own time (paths_s) and the load, for
      # each part-a configuration with speculative decoding on (ENTAIL_SAFE=off, a fresh folder) and as its safe path
      # starts it (auto, the part-a folder)
      for cfg in q35q nemotron; do
        if [ $cfg = q35q ]; then
          kw="{\"model\": \"$Q35Q\", \"enforce_eager\": true, \"max_model_len\": 1024, \"gpu_memory_utilization\": 0.85, \"max_num_seqs\": 4, $SPEC}"
        else
          kw="{\"model\": \"$NEMO\", \"enforce_eager\": true, \"max_model_len\": 2048, \"gpu_memory_utilization\": 0.85, \"max_num_seqs\": 4, $SPEC}"
        fi
        dir=$OUT/s9g_${cfg}_spec; rm -rf $dir; mkdir -p $dir
        start $dir s9g_${cfg}_spec_on "$kw" ENTAIL_SAFE=off
        start $OUT/s9a_$cfg s9g_${cfg}_safe_path "$kw"
        dir=$OUT/s9g_${cfg}_nocheck; rm -rf $dir; mkdir -p $dir
        start $dir s9g_${cfg}_no_check "$kw" ENTAIL_SAFE=off ENTAIL_NO_PATHS=1
      done ;;
    h)
      # h: the pre-registered replication (PREREG_S9b2.md) - another model, two new kinds of fault and two kinds
      # again; each fault started with nothing off, then with ENTAIL_SAFE=all, in one log folder
      L32=/home/<user>/models/Llama-3.2-3B-Instruct
      kw="{\"model\": \"$L32\", \"enforce_eager\": true, \"max_model_len\": 2048, \"gpu_memory_utilization\": 0.8}"
      for plant in inside_rope outside_kvwrite inside_kernel outside; do
        dir=$OUT/s9h_$plant; rm -rf $dir; mkdir -p $dir
        start $dir s9h_${plant}_1off "$kw" ENTAIL_SAFE=off P3_PLANT=$plant ENTAIL_SKIP=vllm_kernel_reference
        start $dir s9h_${plant}_2all "$kw" ENTAIL_SAFE=all P3_PLANT=$plant ENTAIL_SKIP=vllm_kernel_reference
      done ;;
  esac
done
