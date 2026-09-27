#!/bin/bash
# M19 L3.3: healthy models on real engines, entail off and on (ENTAIL=load, the start-up shim), the same probes.
# bash lowlevel/l2/run_l3_engines.sh <out dir> <config> [<config> ...]
# configs: q35_sglang_graph  fp8_vllm_default_triton  llama_vllm_eager  llama_vllm_default  gemma_vllm_eager
set -u
out=$1; shift
cd ~/ai_compiler
mkdir -p "$out"
L2=lowlevel/l2
PREV=$L2/results/l3_definitions/engines
Q35=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
Q35Q=$(ls -d ~/.cache/huggingface/hub/models--AxionML--Qwen3.5-4B-NVFP4/snapshots/*/ | head -1)
spec_from() {   # spec_from <name> <model> <engine json>: the probes of the earlier engine runs, another model/engine
  python3 - "$PREV/fp8_vllm_default.spec.json" "$out/$1.spec.json" "$2" "$3" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
s["model"], s["engine"] = sys.argv[3], json.loads(sys.argv[4])
json.dump(s, open(sys.argv[2], "w", encoding="utf-8"))
EOF
}
run() {   # run <name> <venv> <worker> <spec> [env ...]
  local name=$1 venv=$2 worker=$3 spec=$4; shift 4
  for side in ${SIDES:-off on}; do
    (
      source ~/venvs/$venv/bin/activate
      export VLLM_ENABLE_V1_MULTIPROCESSING=0 "$@"
      if [ $side = on ]; then
        export PYTHONPATH=~/ai_compiler/entail:~/ai_compiler/entail/entail/adapters/autoinstall ENTAIL=load \
               ENTAIL_LOG_DIR=off ENTAIL_RECORD="$out/$name.on.record.jsonl"
        rm -f "$out/$name.on.record.jsonl"
      fi
      python $L2/$worker "$spec" "$out/$name.$side.json" > "$out/$name.$side.log" 2>&1
      echo "$name $side rc=$? $(grep -o '"load_s": [0-9.]*' "$out/$name.$side.json" 2>/dev/null)"
    )
  done
}
for c in "$@"; do
  case $c in
    q35_sglang_graph) run $c sglang sglang_worker.py $PREV/q35_sglang_graph.spec.json ;;
    fp8_vllm_default_triton) run $c vllm vllm_worker.py $PREV/fp8_vllm_default_triton.spec.json \
        VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel ;;
    llama_vllm_eager) spec_from $c /home/<user>/models/Llama-3.2-3B-Instruct \
        '{"enforce_eager": true, "max_model_len": 2048, "gpu_memory_utilization": 0.8}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    llama_vllm_default) spec_from $c /home/<user>/models/Llama-3.2-3B-Instruct \
        '{"max_model_len": 2048, "gpu_memory_utilization": 0.8}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    gemma_vllm_eager) spec_from $c /home/<user>/models/gemma-2-2b-it \
        '{"enforce_eager": true, "max_model_len": 2048, "gpu_memory_utilization": 0.8}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    qwen3_vllm_default) spec_from $c /home/<user>/models/Qwen3-4B \
        '{"max_model_len": 2048, "gpu_memory_utilization": 0.8}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    q35_vllm_default) spec_from $c $Q35 '{"max_model_len": 1024, "gpu_memory_utilization": 0.89, "max_num_seqs": 4}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    q35_vllm_spec) spec_from $c $Q35 '{"enforce_eager": true, "max_model_len": 512, "gpu_memory_utilization": 0.89, "max_num_seqs": 4,
        "speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4,
        "prompt_lookup_min": 2}}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    q35q_vllm_spec) spec_from $c $Q35Q '{"enforce_eager": true, "max_model_len": 1024, "gpu_memory_utilization": 0.85,
        "max_num_seqs": 4, "speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4,
        "prompt_lookup_min": 2}}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    q35q_vllm_eager) spec_from $c $Q35Q '{"enforce_eager": true, "max_model_len": 1024, "gpu_memory_utilization": 0.85,
        "max_num_seqs": 4}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    qwen06_sglang_graph) spec_from $c /home/<user>/models/Qwen3-0.6B '{"mem_fraction_static": 0.7, "context_length": 2048}'
        run $c sglang sglang_worker.py $out/$c.spec.json ;;
    nemotron_vllm_spec) spec_from $c /home/<user>/models/m10/nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16 '{"enforce_eager": true,
        "max_model_len": 2048, "gpu_memory_utilization": 0.85, "max_num_seqs": 4, "speculative_config": {"method": "ngram",
        "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    nemotron_vllm_eager) spec_from $c /home/<user>/models/m10/nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16 '{"enforce_eager": true,
        "max_model_len": 2048, "gpu_memory_utilization": 0.85, "max_num_seqs": 4}'
        run $c vllm vllm_worker.py $out/$c.spec.json ;;
    *) echo "unknown config $c" ;;
  esac
done
