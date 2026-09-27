#!/bin/bash
# M19 L3.3d: which Triton kernels (and which engine functions launching them) normal runs reach, per model and
# engine, entail off (the census hook only).   bash lowlevel/l2/run_census.sh <out dir>
set -u
out=$1
cd ~/ai_compiler
mkdir -p "$out"
L2=lowlevel/l2
HUB=~/.cache/huggingface/hub
snap() { ls -d $HUB/models--$1/snapshots/*/ 2>/dev/null | head -1; }
mk() {   # mk <name> <model> <engine json>
  python3 - "$L2/results/l3_definitions/engines/fp8_vllm_default.spec.json" "$out/$1.spec.json" "$2" "$3" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
s["model"], s["engine"] = sys.argv[3], json.loads(sys.argv[4])
json.dump(s, open(sys.argv[2], "w", encoding="utf-8"))
EOF
}
go() {   # go <name> <venv> <worker>; a config already done is not run again
  [ -f "$out/$1.out.json" ] && { echo "$1 done"; return; }
  (
    source ~/venvs/$2/bin/activate
    # offline: a model whose weights are not on disk fails here instead of being downloaded
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=~/ai_compiler/$L2/census L3D_CENSUS="$out/$1"
    python $L2/$3 "$out/$1.spec.json" "$out/$1.out.json" > "$out/$1.log" 2>&1
    echo "$1 rc=$?"
  )
}
E='{"enforce_eager": true, "max_model_len": 2048, "gpu_memory_utilization": 0.8}'
for m in Llama-3.2-3B-Instruct Qwen3-4B Qwen3-0.6B gemma-2-2b-it gemma-3-1b-it Qwen2.5-3B-Instruct \
         Qwen2.5-3B-Instruct-AWQ Qwen3-4B-FP8; do
  mk "vllm_$m" /home/<user>/models/$m "$E"; go "vllm_$m" vllm vllm_worker.py
done
for m in trl-internal-testing--tiny-Qwen3MoeForCausalLM HuggingFaceTB--SmolLM2-135M-Instruct \
         AxionML--Qwen3.5-4B-NVFP4 Qwen--Qwen3-VL-2B-Instruct; do
  mk "vllm_${m##*--}" "$(snap $m)" "$E"; go "vllm_${m##*--}" vllm vllm_worker.py
done
for m in ibm-granite__granite-4.1-3b microsoft__Phi-4-mini-instruct nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16 \n         trl-internal-testing__tiny-Qwen3MoeForCausalLM; do
  mk "vllm_${m##*__}" /home/<user>/models/m10/$m '{"enforce_eager": true, "max_model_len": 2048, "gpu_memory_utilization": 0.85, "max_num_seqs": 4}'
  go "vllm_${m##*__}" vllm vllm_worker.py
done
S='{"mem_fraction_static": 0.75, "context_length": 2048, "disable_cuda_graph": true}'
for m in Qwen3-0.6B Llama-3.2-3B-Instruct gemma-2-2b-it; do
  mk "sglang_$m" /home/<user>/models/$m "$S"; go "sglang_$m" sglang sglang_worker.py
done
mk sglang_Qwen3.5-4B "$(snap Qwen--Qwen3.5-4B)" '{"mem_fraction_static": 0.92, "context_length": 1024, "max_running_requests": 4, "disable_cuda_graph": true}'
go sglang_Qwen3.5-4B sglang sglang_worker.py
