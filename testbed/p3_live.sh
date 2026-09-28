#!/bin/bash
# P3 live checks the external evaluation asked for (its item 3): the safety modes where S9 did not go.
# bash testbed/p3_live.sh [graphs] [sglang] [serve]
#   graphs  vLLM 0.30 with its defaults (torch.compile and CUDA graphs on), ENTAIL_SAFE=off and =all: the explicit
#           safe mode turns CUDA graphs off live (S9's configurations were all eager)
#   sglang  SGLang 0.5.20 Engine, ENTAIL_SAFE=off and =all, its path check on (ENTAIL_PATHS=1)
#   serve   `vllm serve` (the OpenAI server) with ENTAIL_SAFE=all: the engine arguments are turned there too
# Qwen3-0.6B, one log folder per start, results in testbed/results/p3/live_*.
set -u
cd ~/ai_compiler
OUT=testbed/results/p3
mkdir -p $OUT
Q06=/home/<user>/models/Qwen3-0.6B
SHIM=~/ai_compiler/entail:~/ai_compiler/entail/entail/adapters/autoinstall

parts=${*:-graphs sglang serve}
for part in $parts; do
  case $part in
    graphs)
      kw="{\"model\": \"$Q06\", \"max_model_len\": 2048, \"gpu_memory_utilization\": 0.5}"
      for mode in off all; do
        dir=$OUT/live_graphs_$mode; rm -rf $dir; mkdir -p $dir
        (
          source ~/venvs/vllm/bin/activate
          export VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=$SHIM ENTAIL=load ENTAIL_LOG_DIR=$dir ENTAIL_SAFE=$mode
          timeout 900 python testbed/p3_s9.py "$kw" $OUT/live_graphs_$mode.json > $OUT/live_graphs_$mode.log 2>&1
          echo "graphs $mode rc=$? $(grep -E '^\{|^pairs' $OUT/live_graphs_$mode.log | tr '\n' ' ' | cut -c1-500)"
        )
      done ;;
    sglang)
      kw="{\"model_path\": \"$Q06\", \"mem_fraction_static\": 0.5, \"context_length\": 2048}"
      for mode in off all; do
        dir=$OUT/live_sglang_$mode; rm -rf $dir; mkdir -p $dir
        (
          source ~/venvs/sglang/bin/activate
          export PYTHONPATH=$SHIM ENTAIL=load ENTAIL_LOG_DIR=$dir ENTAIL_SAFE=$mode ENTAIL_PATHS=1
          timeout 900 python testbed/p3_sglang.py "$kw" $OUT/live_sglang_$mode.json > $OUT/live_sglang_$mode.log 2>&1
          echo "sglang $mode rc=$? $(grep -E '^\{' $OUT/live_sglang_$mode.log | cut -c1-600)"
        )
      done ;;
    serve)
      dir=$OUT/live_serve_all; rm -rf $dir; mkdir -p $dir
      (
        source ~/venvs/vllm/bin/activate
        export PYTHONPATH=$SHIM ENTAIL=load ENTAIL_LOG_DIR=$dir ENTAIL_SAFE=all
        vllm serve $Q06 --max-model-len 2048 --gpu-memory-utilization 0.5 --port 8011 > $OUT/live_serve_all.log 2>&1 &
        pid=$!
        up=no
        for i in $(seq 1 300); do
          if curl -sf localhost:8011/health > /dev/null; then up=yes; break; fi
          kill -0 $pid 2> /dev/null || break
          sleep 1
        done
        body="{\"model\": \"$Q06\", \"prompt\": \"The capital of France is\", \"max_tokens\": 8, \"temperature\": 0}"
        [ $up = yes ] && curl -s localhost:8011/v1/completions -H 'Content-Type: application/json' -d "$body" \
          > $OUT/live_serve_all.answer.json
        kill $pid 2> /dev/null; wait $pid 2> /dev/null
        echo "serve all up=$up answer=$(cut -c1-300 $OUT/live_serve_all.answer.json 2> /dev/null)"
        grep -o "enforce_eager=[A-Za-z]*\|enable_prefix_caching=[A-Za-z]*" $OUT/live_serve_all.log | sort | uniq -c
      ) ;;
  esac
done
