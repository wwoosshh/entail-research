### Checklist

- [x] I searched related issues but found no solution.
- [x] The bug persists in the latest version.
- [x] Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.
- [x] If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.
- [x] Please use English. Otherwise, it will be closed.

### Describe the bug

On a hybrid linear-attention (GDN) model with `--enable-hierarchical-cache`, a
prefix that has been evicted from the device pool and restored from the host
tier produces **different and wrong output** for a byte-identical request. There
is no crash, no warning and no accounting anomaly: the serve reports the hit at
the **full** `cached_tokens`, the request finishes with `finish_reason=stop`,
and the answer is fluent — it simply is not an answer to the prompt that was
sent.

Measured on **main HEAD `2929a39927a3943cee03e498f4e5f651185f1b1f`**
(2026-09-15): **20 of 20** host-tier hits wrong, against **10 of 10** correct
cold and **10 of 10** correct device-pool hits on the same prompts in the same
process.

This is the same defect class already reported on the **device** path —
#37836 (merged 2026-09-04), #39342, #31833 — but on the **host** path, where we
can find no report.

### Result, 10 prompts

| leg | n | answers correct | mean reported `cached_tokens` |
|---|---|---|---|
| cold | 10 | **10/10** | 0 |
| hit (device) | 10 | **10/10** | 27,628 |
| **host** | 10 | **0/10** | 27,628 |
| **host2** | 10 | **0/10** | 27,628 |

Every host leg reported a hit; none reported a miss; none errored.

### What the wrong answers look like

The prompt asks for three imperative one-line suggestions. Verbatim, same
prompt, cold then out of the host tier:

```
cold   Ask Orrin what your father was hiding in the glasshouse.
       Go to the glasshouse and look for what your father hid.
       Ask Orrin why your father stopped feeding the bees.
hit    (byte-identical to cold)
host   act:local_action
host2  act:speech local_action implied - -
```

```
cold   Let Jada go now.
       Keep Jada here until the auditor finishes.
       Ask the auditor to read the logbook aloud.
host   act horizon
host2  act horizon
```

```
host   act=speech act=normal act=act=act=act=act=act=act=act=act=act=act=act=
       act=act=act=act=act=act=act=act=... (hits the 96-token bound)
```

```
host   The Stranger releases you, stepping back with a sudden, jerky motion, and
       turns his shoulder deliberately toward the riders gathered near the front,
       his eyes fixed on the narrow door of the back office. "The writ is in the
       back office."
```

Read by eye, all 20 host legs:

| shape | n of 20 |
|---|---:|
| a line from a **different pass's wire format** (`act:local_action`, `act horizon`, `act=speech route=implied_npc target=implied_npc`, `cast_third intervene target:cast_second`) | 9 |
| a **paragraph of narration or two-speaker dialogue** — a different pass's output shape entirely | 6 |
| **degenerate repetition** to the token bound | 1 |
| fewer than the requested three lines, right voice | 2 |
| right shape, wrong voice (not an imperative the user could issue) | 2 |
| **an answer a user could be shown** | **0** |

Two properties that narrow it:

- **It is deterministic.** `host` and `host2` return the same wrong text
  repeatedly. Temperature is 0 and does not explain it.
- **It is not truncation. Measured on this build.** Appending a marker
  instruction to the END of the prompt ("Answer in the usual three lines, but
  begin every line with the word ZEBRA") is obeyed **3/3 lines on the host leg
  of both payloads tested**, while the content is wrong — the host leg answers
  `ZEBRA The player asks the Abbot whether the box has been moved.` where the
  cold leg answers `ZEBRA Ask the Abbot to show you the box.` So the tail of the
  restored prefix is present and attended to; the damage is in the body.
- **Waiting does not help.** 150 s of complete idle between the churn and the
  host leg (75 s after warming, 75 s after the churn) changes nothing, so an
  unfinished device->host copy does not explain it. (This one was measured on
  the older commit `4ccff141d`, not on main.)
- **Three other write/eviction settings do not help either**, all on this build:
  `--hicache-write-policy write_through_selective` (8 genuine restores, 0
  correct), `--mamba-max-states-per-path 4` (12 genuine restores, 0 correct),
  and `--max-mamba-cache-size 96` (6 genuine restores, 1 correct).

### Why this looks like the recurrent state, not the KV

The prefix match itself requires a Mamba checkpoint — in
`unified_cache/components/mamba_component.py` the MAMBA validator accepts a node
when `component_data[ct].value is not None or ... .host_value is not None` — so a
full-length `cached_input_len` means *a* checkpoint was present at the frontier.
The answers are fluent, in-world and about the right conversation; they are the
wrong PASS, not the wrong story. That is what a restored recurrent state taken
at a different position looks like, and it is exactly the symptom #37836 / #39342
describe on the device path.

### Two observations that may help whoever picks this up

1. **`--hicache-write-policy write_back` changes the outcome, and partly by
   caching less.** On the same build, 6 prompts: 4 of the 6 `host` legs reported
   `cached_tokens = 0` — a genuine miss, recomputed, correct by construction —
   and only **2** were genuine host restores. One of those two was usable and one
   was degraded (it named a person the campaign's cast does not contain). So
   `write_back` is not a fix; it is a smaller sample of the same event.
2. **`host2` is wrong too, and differently wrong.** By the time `host2` runs, the
   `host` leg has already pulled the prefix back into the device pool, so `host2`
   should be an ordinary device hit — and an ordinary device hit is correct
   10/10 in leg 2. It is not correct here. Whatever the host tier restores, it
   restores INTO the device pool in a state that stays wrong.

### Suggested starting points

`mem_cache/unified_cache/components/mamba_component.py`
(`build_hicache_transfers`, phases `BACKUP_HOST` / `LOAD_BACK`;
`commit_hicache_transfer` assigns `cd.host_value` and `cd.value` from cloned
index tensors with no visible synchronisation before use), and
`finalize_match_result_in_tree_core`, whose own comment is *"Full KV may extend
beyond the latest reusable Mamba state"* — the clip that has to hold on the host
path. The per-pool eviction metric and host-coverage boot line proposed in
**#39436** would say immediately whether a checkpoint was evicted under the
prefix whose KV rows survived.

### What would make this easy to confirm

A boot-time or per-request assertion that the Mamba checkpoint restored for a
host hit was taken at the same token position as the last KV page restored with
it. Today nothing checks it, and the failure is silent all the way to the user.

### A related omission in the host pool, with a proposed patch (not a fix for the above)

Reading the source for the cause, we found that the specialized HiCache Mamba host pool (`mem_cache/pool_host/mamba.py`) saves the GDN convolution and recurrent state of a checkpoint but never saves the model's registered PLE side state (the BF16 `[10240, 9]` PLE convolution window and its two int64 token-history entries, ~184 KB per checkpoint, registered as slot siblings in `memory_pool.py`). The ordinary device-side copy and the generic CPU offload path do preserve those siblings. The omission is present on the pin above, on main `2929a399` and on later main `63845a1`.

A proposal patch that carries the PLE siblings through the host pool (separate host arrays with the original per-slot shape and dtype, included in capacity accounting and buffer registration, restored at the first local Mamba layer with an event wait before the N-gram gather; whole-page serialization appends typed sections with size validated before any host tensor changes) is here: https://gist.github.com/JarJarBeatyourattitude/5bfe8db0b9f04378fc60962fb0622fb0

Two honest caveats about that patch:

- It was verified on CPU only (an AST-loaded round trip of the real classes with leaf DMA substituted); no GPU execution, no CUDA event timing, no performance measurement.
- We built and ran it on the same GPU and model. **The host restores were still wrong with the patch applied** (0 of 2 genuine host restores correct, same failure shapes as above), with the patched module demonstrably loaded. So the omission is real and worth fixing, but it is not the whole cause of the wrong output. We also checked for an adapter-id collision in the cache keys; main already namespaces them.

### Reproduction

Serve command (paths elided):

```
sglang serve --model-path <nvfp4-checkpoint> --load-format safetensors \
  --tp 1 --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.92 --context-length 65536 --page-size 64 \
  --max-running-requests 4 --chunked-prefill-size 4096 \
  --cuda-graph-max-bs-decode 4 --cuda-graph-max-bs-prefill 4 \
  --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 48 \
  --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --enable-cache-report --ple-offload-embedding \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-lora --max-lora-rank 32 --max-loras-per-batch 7 --lora-paths <7 adapters> \
  --enable-hierarchical-cache --hicache-size 96 \
  --hicache-write-policy write_through --hicache-mem-layout page_first \
  --hicache-io-backend direct
```

Boot lines, for the pool shapes:

```
Mamba Cache is allocated. max_mamba_cache_size: 48, conv_state size: 0.10GB,
  ssm_state size: 2.58GB intermediate_ssm_state_cache size: 1.05GB
  intermediate_conv_window_cache size: 0.02GB
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 240128, K 1.37 GB, V 1.37 GB
max_total_num_tokens=240128, ... available_gpu_mem=8.87 GB
Allocating kv hierarchical KV host pool: 3035904 tokens, 40.41 GB host memory,
  packed MTP KV layers: target_layers=12, draft_layers=1, total_layers=13.
Allocating 55.60 GB host memory for hierarchical Mamba cache (layout=page_first_direct).
Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=False
  hybrid_ssm=True hicache_attached=True streaming_wrapped=False
```

One request at a time, `temperature 0`, `max_tokens 96`, no grammar:

1. **cold** — send prompt P (10k-49k tokens). Prepend a unique random prefix so
   no radix node can match it.
2. **hit** — send P again immediately. Device-pool hit.
3. **churn** — send other large, mutually distinct prompts until the *uncached*
   token count pushed through exceeds 1.35x `max_total_num_tokens` (here
   ~324,000), so P is certainly evicted from the device pool.
4. **host** — send P again. The serve reports the full `cached_tokens`; the KV
   comes back from the host tier.
5. **host2** — send P again.

### Environment

| | |
|---|---|
| sglang | `2929a39927a3943cee03e498f4e5f651185f1b1f` (main, 2026-09-15), `0.5.20.dev728` |
| torch | `2.13.0+cu130`, CUDA 13.0 |
| GPU | 1x RTX PRO 6000 Blackwell WS, 97,887 MiB, driver 595.71.05, sm120 |
| model | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225242aacd3dbd3f9407468c2ee9a9d2594` (`Qwen4ExpForConditionalGeneration`, `qwen4_exp`), 48 layers, `full_attention_interval 4` -> 36 GDN + 12 full-attention |
| local delta | 134 lines in 4 files, LoRA enablement only (`lora/lora.py`, `lora/utils.py`, `models/qwen3_5.py`, `models/qwen4_exp.py`): `in_proj_ba` in `supported_lora_modules` / `_KNOWN_LORA_TARGET_MODULES`, and a nested-`text_config` `get_hidden_dim` on `Qwen4ExpForConditionalGeneration`. **Nothing in `mem_cache/`, `attention/` or `speculative/` is patched.** |

`python3 -m sglang.check_env` output. Note: this was captured from the same GPU model and image with the venv we serve production from, which is on the older pin `4ccff141d` with the LoRA delta noted above; the measurements in this report were taken on main `2929a39927a3943cee03e498f4e5f651185f1b1f` built the same way on a since-released machine (driver 595.71.05 there).

```
venv python: /root/models/sglang-official/.venv/bin/python
Python: 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]
CUDA available: True
GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
GPU 0 Compute Capability: 12.0
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 13.0, V13.0.88
CUDA Driver Version: 590.44.01
PyTorch: 2.13.0+cu130
sglang: 0.0.0.dev1+g4ccff141d
sglang-kernel: 0.4.6.post1+cu130
flashinfer_python: 0.6.17
flashinfer_cubin: Module Not Found
flashinfer_jit_cache: Module Not Found
triton: 3.7.1
transformers: 5.12.1
torchao: Module Not Found
numpy: 2.3.5
aiohttp: 3.14.3
fastapi: 0.141.1
huggingface_hub: 1.31.0
interegular: 0.3.3
modelscope: 1.40.0
orjson: 3.12.0
outlines: 0.1.11
packaging: 26.3
psutil: 7.2.2
pydantic: 2.14.0b2
python-multipart: 0.0.32
pyzmq: 27.2.0
uvicorn: 0.52.4
uvloop: 0.22.1
vllm: Module Not Found
xgrammar: 0.2.1
openai: 2.6.1
tiktoken: 0.14.0
anthropic: 1.5.0
litellm: Module Not Found
torchcodec: 0.15.0
```

