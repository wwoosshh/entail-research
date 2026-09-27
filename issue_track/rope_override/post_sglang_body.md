### Checklist

- [x] I searched related issues but found no solution.
- [x] The bug persists in the latest version.
- [x] Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.
- [x] If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.
- [x] Please use English. Otherwise, it will be closed.

### Describe the bug

Under Transformers v5, assigning `rope_scaling` on a built config replaces `rope_parameters` wholesale, so an
override given with `--json-model-override-args` leaves `rope_parameters` without `rope_theta`. `get_rope_config`
then falls back to 10000. #22739 added a fallback of 1e6 for Qwen3 dense models, which hides the problem there;
Llama (base 500000) and Qwen3-MoE (1e6) are not covered and run with base 10000, with no warning.

### Reproduction

Llama-3.2-3B-Instruct, restating the checkpoint's **own** `rope_scaling` (so the override should change nothing):

```bash
python -m sglang.launch_server --model-path Llama-3.2-3B-Instruct \
  --json-model-override-args '{"rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}}'
```

| run | RoPE base the model runs with | GSM8K (first 200, greedy) |
|---|---|---|
| untouched | 500000 | 161 |
| own `rope_scaling` restated at launch | **10000** (fallback) | **106** |

Outputs stay fluent. The same restatement through `config.json` keeps the base.

**Suggested fix**

When the override dict carries no `rope_theta`, keep the value from the config as loaded from the files before
replacing `rope_parameters` (the `config.json` path already does this), and warn when `rope_parameters` ends up
without a base instead of silently using 10000. A per-model fallback (as in #22739) does not cover checkpoints of
the same architecture with another base (Qwen3-2507: 5e6; Qwen2.5-1M: 1e7).

### Environment

SGLang 0.5.20 (transformers 5.12.1 in that environment at the time of the run), torch 2.13.0+cu130, Python 3.12, one RTX
4070 Ti, WSL2 Ubuntu 24.04. Script (`sglang_llama.py`), launch line and raw results:
https://github.com/wwoosshh/entail-research/tree/main/issue_track/rope_override. The same route on vLLM 0.30 is
reported at vllm-project/vllm#58675 (https://github.com/vllm-project/vllm/issues/58675); a config-level check of the 300 most-downloaded
models found 64 of 180 applicable ones change base under it on vLLM, and I have not repeated that count for SGLang.

```text
Python: 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]
CUDA available: True
GPU 0: NVIDIA GeForce RTX 4070 Ti
GPU 0 Compute Capability: 8.9
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 13.0, V13.0.88
CUDA Driver Version: 616.56
PyTorch: 2.13.0+cu130
sglang: 0.5.20
sglang-kernel: 0.4.7
flashinfer_python: 0.6.18
flashinfer_cubin: Module Not Found
flashinfer_jit_cache: Module Not Found
triton: 3.7.1
transformers: 5.12.1
torchao: Module Not Found
numpy: 2.3.5
aiohttp: 3.14.3
fastapi: 0.141.1
huggingface_hub: 1.32.0
interegular: 0.3.3
modelscope: 1.40.1
orjson: 3.12.0
outlines: 0.1.11
packaging: 26.3
psutil: 7.2.2
pydantic: 2.13.5
python-multipart: 0.0.32
pyzmq: 27.2.0
uvicorn: 0.53.0
uvloop: 0.22.1
vllm: Module Not Found
xgrammar: 0.2.1
openai: 2.6.1
tiktoken: 0.14.0
anthropic: 1.8.0
litellm: Module Not Found
torchcodec: 0.15.0
Hypervisor vendor:: Microsoft
ulimit soft: 10240
```
