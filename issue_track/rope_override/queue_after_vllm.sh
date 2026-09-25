#!/bin/bash
# After the vLLM batch: the model-file-aware vLLM probe (CPU-side, needs the GPU free), then the SGLang Llama arm.
cd <workspace>/issue_track/rope_override
until ! pgrep -f "queue_next.sh|rope_override.py" > /dev/null; do sleep 5; done
(cd <workspace>/entail/audits && ~/venvs/vllm/bin/python override_probe_files.py vllm 2>&1 | grep -E "^ROW|Error" | tail -6)
bash run_sglang_llama.sh SA SA2 SC SE SC_rolecheck
