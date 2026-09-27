### Checklist

- [x] I searched related issues but found no solution.
- [x] The bug persists in the latest version.
- [x] Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.
- [x] If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.
- [x] Please use English. Otherwise, it will be closed.

### Describe the bug

#### Summary

SGLang currently accepts requests that combine speculative decoding with `--grammar-backend outlines`, but the speculative grammar verifier is not compatible with the Outlines grammar object. 

#### Affected Configuration

- Server launched with `--grammar-backend outlines`
- Server launched with a speculative algorithm such as `EAGLE`, `EAGLE3`, `STANDALONE`, or `NGRAM`
- Request uses `json_schema`, `regex`, `ebnf`, or `structural_tag`

#### Serval Bugs Found

- `OutlinesGrammar` does not implement `rollback`, but speculative DFS temporarily accepts draft tokens and must roll those accepts back. https://github.com/sgl-project/sglang/blob/ef2b1b6d89977eca6872646ed569a5b1603c5adb/python/sglang/srt/constrained/outlines_backend.py#L42
- `spec_utils.traverse_tree()` assumes grammar masks are packed int32 bitmasks. However, Outlines uses a dense `torch.bool` mask. https://github.com/sgl-project/sglang/blob/ef2b1b6d89977eca6872646ed569a5b1603c5adb/python/sglang/srt/constrained/outlines_backend.py#L70
- The packed-bit membership check indexes `token_id // 32` and treats nonzero values as allowed. However, Outlines needs direct `token_id` indexing and uses `False` as allowed. https://github.com/sgl-project/sglang/blob/ef2b1b6d89977eca6872646ed569a5b1603c5adb/python/sglang/srt/speculative/spec_utils.py#L607
- There was no admission guard or documentation telling users that this combination is unsupported.

### Reproduction

```
2026-05-05 07:22:28,052 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
[2026-05-05 07:22:28] Unexpected error during package walk: cutlass.cute.experimental
Capturing batches (bs=1 avail_mem=12.26 GB):  96%|█████████████████████████████████████████████████████████████████████████████▍   | 22/23 [00:01<00:00, 18.33it/s]Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-05-05 07:22:29] Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Capturing batches (bs=1 avail_mem=12.26 GB): 100%|█████████████████████████████████████████████████████████████████████████████████| 23/23 [00:01<00:00, 11.75it/s]
Compiling num tokens (num_tokens=4): 100%|█████████████████████████████████████████████████████████████████████████████████████████| 58/58 [00:06<00:00,  8.72it/s]
Capturing num tokens (num_tokens=4 avail_mem=11.68 GB): 100%|██████████████████████████████████████████████████████████████████████| 58/58 [00:01<00:00, 32.49it/s]
/root/sglang/python/sglang/srt/constrained/outlines_backend.py:66: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
  tokens = torch.tensor(
[2026-05-05 07:22:47] Scheduler hit an exception: Traceback (most recent call last):
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 3818, in run_scheduler_process
    scheduler.run_event_loop()
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 1387, in run_event_loop
    dispatch_event_loop(self)
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 3690, in dispatch_event_loop
    scheduler.event_loop_normal()
  File "/root/sglang/.venv-spec-grammar-outline/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 1406, in event_loop_normal
    result = self.run_batch(batch)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 2868, in run_batch
    batch_result = self.model_worker.forward_batch_generation(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/speculative/ngram_worker.py", line 290, in forward_batch_generation
    vocab_mask = generate_token_bitmask(
                 ^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/speculative/spec_utils.py", line 669, in generate_token_bitmask
    traverse_tree(
  File "/root/sglang/python/sglang/srt/speculative/spec_utils.py", line 635, in traverse_tree
    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)
  File "/root/sglang/python/sglang/srt/speculative/spec_utils.py", line 615, in dfs
    dfs(
  File "/root/sglang/python/sglang/srt/speculative/spec_utils.py", line 624, in dfs
    grammar.rollback(1)
  File "/root/sglang/python/sglang/srt/constrained/base_grammar_backend.py", line 58, in rollback
    raise NotImplementedError()
NotImplementedError
```

### Environment

```
(.venv-spec-grammar-outline) root@d7cf8ae9d327:~/sglang# python3 -m sglang.check_env
Python: 3.11.10 (main, Sep  7 2024, 18:35:41) [GCC 11.4.0]
CUDA available: True
GPU 0: NVIDIA H100 PCIe
GPU 0 Compute Capability: 9.0
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 12.4, V12.4.131
CUDA Driver Version: 575.57.08
PyTorch: 2.9.1+cu128
sglang: 0.0.0.dev12032+g28ee08c17
sglang-kernel: 0.4.1.post1
flashinfer_python: 0.6.8.post1
flashinfer_cubin: 0.6.8.post1
flashinfer_jit_cache: Module Not Found
triton: 3.5.1
transformers: 5.6.0
torchao: 0.17.0+cu128
numpy: 2.3.5
aiohttp: 3.13.5
fastapi: 0.136.1
huggingface_hub: 1.13.0
interegular: 0.3.3
modelscope: 1.36.3
orjson: 3.11.8
outlines: 0.1.11
packaging: 26.2
psutil: 7.2.2
pydantic: 2.13.3
python-multipart: 0.0.27
pyzmq: 27.1.0
uvicorn: 0.46.0
uvloop: 0.22.1
vllm: Module Not Found
xgrammar: 0.1.32
openai: 2.6.1
tiktoken: 0.12.0
anthropic: 0.98.1
litellm: Module Not Found
torchcodec: 0.9.1+cu128
NVIDIA Topology: 
        GPU0    NIC0    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PHB     0-251   0               N/A
NIC0    PHB      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0


Hypervisor vendor:: KVM
ulimit soft: 1048576
```
