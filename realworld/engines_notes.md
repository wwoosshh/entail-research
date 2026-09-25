# Real-world engines vs. the locally measured "meaning erasure" losses: evidence notes

- Compiled: 2026-09-22. All pages and API responses were accessed 2026-09-22 unless another date is given.
- Method: GitHub source/docs read with `gh api` (GET only). Permalinks use the latest commit SHA touching each file on 2026-09-22. Blogs, docs and papers were fetched with WebFetch or curl. Download numbers come from the pypistats.org and Docker Hub JSON APIs.
- Conventions: one fact per bullet, each with its URL. Quotes are verbatim and at most 15 words. `[UNCERTAIN]` marks inference or incomplete evidence. `(date)` is the PR merge date, the issue/comment date, or the doc commit date.
- Local reference (from the lead, not a source): the transformers 5.17 SDPA path received the valid KV length as a dense bool mask, and GQA as a `repeat_kv` copy. Receiving these facts directly was 1.3–2.5x faster per decode step. torch.compile's guessed batch size caused 66–106 s recompiles.

---

## Summary of what the sources show (each point is backed by the bullets below)

- GPU serving engines (vLLM, SGLang, TensorRT-LLM, FlashInfer) pass explicit facts on their main paths: cumulative or used sequence lengths, page/block tables, window size, softcap, sinks. GQA is native, with no K/V replication (§1.1–1.4). The two losses measured locally were removed by hand in these engines.
- llama.cpp (and therefore Ollama and LM Studio's llama.cpp engine) still encodes validity, causality, SWA, chunking, ALiBi and even DSA sparsity in a dense KQ mask built on the CPU each ubatch. GQA is native through broadcasting. Its kernels have added auxiliary passes that rediscover the lost structure from the mask (§1.5, §5).
- In transformers itself, the local reference, the upstream code confirms both losses. The SDPA path uses `repeat_kv` whenever a mask exists, and PyTorch rejects masks for flash. HF docs admit the static-cache masking waste (§1.6).
- Hand removal has costs. New variants landed late, as slow fallbacks, as silent wrong output, or on restricted hardware. Engines keep large per-variant backend sets. Per-step metadata construction and plan() calls are a measured CPU/latency cost that engines keep engineering around (§2, §5).

---

## 1. How engines pass attention metadata to kernels

### 1.1 vLLM (V1)

- `FlashAttentionMetadata` holds `query_start_loc`, `max_query_len`, `max_seq_len`, `seq_lens`, `block_table` and `slot_mapping`, plus cascade fields. It has no dense mask field. https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/v1/attention/backends/flash_attn.py#L454-L470 (file commit 2026-09-16)
- The main FA call passes `cu_seqlens_q=cu_seqlens_q` (= `query_start_loc`), `seqused_k=seqused_k` (= `seq_lens`), `block_table`, `window_size`, `softcap`, `alibi_slopes` and `s_aux=self.sinks`. https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/v1/attention/backends/flash_attn.py#L1265-L1270 and #L1388-L1413
- The same call passes `mask_mod=rswa_mask_mod_fn or mm_mask_mod` only for special variants (reference-SWA, multimodal prefix). Masks are an exception path here. https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/v1/attention/backends/flash_attn.py#L1388-L1413
- The forward docstring gives key as `[num_tokens, num_kv_heads, head_size]` and kv_cache as `[num_blocks, num_kv_heads, block_size, 2 * head_size]`. K/V stay at KV-head count, with no replication. https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/v1/attention/backends/flash_attn.py#L1180-L1186
- FlashAttention's README: "Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads". https://github.com/Dao-AILab/flash-attention/blob/1f7ce2f7cb503473559f3d44d575ae05b1ed8557/README.md#L256-L257 (file commit 2026-07-06)
- `CommonAttentionMetadata` is documented as "Per-batch attention metadata, shared across layers and backends". Fields: `query_start_loc` (batch_size+1), `seq_lens`, `max_query_len`, `max_seq_len`, `block_table_tensor`, `slot_mapping`, `causal`. https://github.com/vllm-project/vllm/blob/651a88c09ac6cbb13c5dd36f2fdbe61a2c68eeef/vllm/v1/attention/backend.py#L384-L411 (file commit 2026-09-16)
- The vLLM "Transformers modeling backend" sets `self.config._attn_implementation = "vllm"` "so that vLLM's attention layer is used". HF model code running inside vLLM therefore gets vLLM's length/page-table attention, not the HF mask path. https://github.com/vllm-project/vllm/blob/6b858751f6e99c4f33913657dc0c2afd34ff13e6/docs/models/supported_models.md#L156 (file commit 2026-09-21)
- The same doc says the Transformers backend's performance "should be identical to a dedicated vLLM model implementation". https://github.com/vllm-project/vllm/blob/6b858751f6e99c4f33913657dc0c2afd34ff13e6/docs/models/supported_models.md#L18

### 1.2 FlashInfer (used by vLLM, SGLang, TensorRT-LLM)

- Signature: `BatchDecodeWithPagedKVCacheWrapper.plan(indptr, indices, last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size, ...)`. Keyword options include `pos_encoding_mode` (`ALIBI`), `window_left` and `logits_soft_cap`. https://github.com/flashinfer-ai/flashinfer/blob/d7a7447cb4bb29b4637f7fa02fe5eaeeb5e61a6e/flashinfer/decode.py#L1428-L1478 (file commit 2026-09-21)
- plan() docstring on GQA: "The num_qo_heads must be a multiple of num_kv_heads"; otherwise it uses grouped query attention. https://github.com/flashinfer-ai/flashinfer/blob/d7a7447cb4bb29b4637f7fa02fe5eaeeb5e61a6e/flashinfer/decode.py#L1528-L1530
- plan() docstring: "The plan method cannot be used in Cuda Graph or in torch.compile." https://github.com/flashinfer-ai/flashinfer/blob/d7a7447cb4bb29b4637f7fa02fe5eaeeb5e61a6e/flashinfer/decode.py#L1532
- Wrapper note: auxiliary data structures "can be reused across multiple batch decode attention calls (e.g. different Transformer layers)". https://github.com/flashinfer-ai/flashinfer/blob/d7a7447cb4bb29b4637f7fa02fe5eaeeb5e61a6e/flashinfer/decode.py#L937-L943
- With `use_cuda_graph=True`, "The batch_size cannot change during the lifecycle of this wrapper". https://github.com/flashinfer-ai/flashinfer/blob/d7a7447cb4bb29b4637f7fa02fe5eaeeb5e61a6e/flashinfer/decode.py#L971-L974
- KV-layout tutorial: the page table is `kv_indptr` + `kv_page_indices` + `kv_last_page_len`, and ragged Q uses `qo_indptr`. Custom masks are a separate option; "FlashInfer accepts both boolean mask and bit-packed mask". https://docs.flashinfer.ai/tutorials/kv_layout.html
- FlashInfer paper (arXiv 2501.01005) handles GQA by "head-group fusion of query heads with the query length dimension" (Figure 11). https://arxiv.org/html/2501.01005
- FlashInfer paper: "Supporting various attention variants in CUDA library is not sustainable". It uses a JIT-compiled customizable template instead. https://arxiv.org/html/2501.01005

### 1.3 SGLang

- SGLang's `FlashAttentionMetadata` is documented as "Metadata to be init once in the model forward pass" and reused by each layer. Fields: `cache_seqlens_int32`, `cu_seqlens_q`, `cu_seqlens_k`, `window_size`, `page_table`, `swa_page_table`. https://github.com/sgl-project/sglang/blob/970e946e4fdaa1271a4044fd36850650d315fecf/python/sglang/srt/layers/attention/flashattention_backend.py#L69-L95 (file commit 2026-09-21)
- The SGLang FlashInfer backend imports FlashInfer's `fast_decode_plan` and plans with `paged_kv_indptr`, `paged_kv_indices` and `paged_kv_last_page_len`. https://github.com/sgl-project/sglang/blob/61d0cf2074711db9c54c993a3732355b5377c906/python/sglang/srt/layers/attention/flashinfer_backend.py#L74-L80 and #L178-L200 (file commit 2026-09-22)
- The SGLang torch-native (SDPA) fallback loops per request over `seq_lens` and calls SDPA with an `enable_gqa` flag. It builds a sliding-window mask only when needed. https://github.com/sgl-project/sglang/blob/95d8a75bc93805534fb3ce0a75471bcbcf3ee4e6/python/sglang/srt/layers/attention/torch_native_backend.py#L100-L170 (file commit 2026-06-10)

### 1.4 TensorRT-LLM (PyTorch backend)

- Packed mode: "the user provides the operator with a 1D tensor containing the lengths". https://github.com/NVIDIA/TensorRT-LLM/blob/a56ec203967029af172349739d9b6bb5d646aa89/docs/source/features/attention.md#L111-L121 (doc commit 2026-09-05)
- Inflight batching "requires the input tensors to be packed (no padding)". https://github.com/NVIDIA/TensorRT-LLM/blob/a56ec203967029af172349739d9b6bb5d646aa89/docs/source/features/attention.md#L205-L212
- The backend forward takes k/v shaped `(num_tokens, num_kv_heads * head_dim)`. The XQA kernel is "another optimization for MQA/GQA in the generation phase". https://github.com/NVIDIA/TensorRT-LLM/blob/a56ec203967029af172349739d9b6bb5d646aa89/docs/source/features/attention.md#L98-L99 and #L186-L190
- Metadata lifecycle doc: "Before each forward step" the runtime calls `AttentionMetadata.prepare`. With CUDA graphs, metadata tensors must be pre-allocated and not re-allocated. https://github.com/NVIDIA/TensorRT-LLM/blob/a56ec203967029af172349739d9b6bb5d646aa89/docs/source/features/attention.md#L40-L48
- `AttentionMetadata` carries `seq_lens`, `seq_lens_kv` and `kv_cache_params`. The attention mask is an enum (`CAUSAL`, `FULL`) with a `CUSTOM` option. Forward args carry `attention_window_size` and `attention_sinks` as scalar/tensor facts. https://github.com/NVIDIA/TensorRT-LLM/blob/f9e3e06ee73930803cd3d947bc5f9dd8d6c2c8a3/tensorrt_llm/_torch/attention/backends/interface.py#L881-L920 (file commit 2026-09-18)

### 1.5 llama.cpp / ggml (also the engine under Ollama and LM Studio's GGUF path)

- API: `ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)`. The mask is `[n_kv, n_batch, ne32, ne33]`. The broadcast rule `n_head % n_head_kv == 0` gives native GQA with no K/V copy. https://github.com/ggml-org/llama.cpp/blob/37b53fd4545847188fdad29e38ba57875efc8228/ggml/include/ggml.h#L2476-L2495 (file commit 2026-09-16)
- The non-FA path uses `ggml_mul_mat`, which broadcasts over dims 2/3 (`[ne03 * x, ne02 * y, m, k]`), and `ggml_soft_max_ext(a, mask, scale, max_bias)`. https://github.com/ggml-org/llama.cpp/blob/37b53fd4545847188fdad29e38ba57875efc8228/ggml/include/ggml.h#L1479-L1485 and #L1805-L1818
- `build_attn_mha` passes the same `kq_mask` to `ggml_flash_attn_ext` (FA) or `ggml_soft_max_ext` (non-FA), and adds sinks with `ggml_flash_attn_ext_add_sinks`. https://github.com/ggml-org/llama.cpp/blob/96550613656e7f024df65f91cf8b2d80a83cf09e/src/llama-graph.cpp#L2598-L2720 (file commit 2026-09-21)
- The KQ mask is filled on the host (`GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))`), with 0 or -INFINITY for each (kv cell, token), every ubatch. https://github.com/ggml-org/llama.cpp/blob/96550613656e7f024df65f91cf8b2d80a83cf09e/src/llama-kv-cache.cpp#L1541-L1572 and #L1746-L1785
- The mask-fill function has a commented-out timer ("kq mask time"). No timing is reported in source. https://github.com/ggml-org/llama.cpp/blob/96550613656e7f024df65f91cf8b2d80a83cf09e/src/llama-kv-cache.cpp#L1765-L1785
- `n_kv` is padded to at least 256: "pad the n_kv value so that the graph remains constant across batches". https://github.com/ggml-org/llama.cpp/blob/96550613656e7f024df65f91cf8b2d80a83cf09e/src/llama-kv-cache.cpp#L1250-L1264
- ALiBi is folded into the KQ mask: `soft_max(KQ*scale + KQ_mask*m)` (PR #7192, merged 2024-05-11). https://github.com/ggml-org/llama.cpp/pull/7192
- A newer ggml API adds an explicit hint: "Use finite mask entries as a sparse K/V set." (`ggml_flash_attn_ext_set_n_kv_max`). https://github.com/ggml-org/llama.cpp/blob/37b53fd4545847188fdad29e38ba57875efc8228/ggml/include/ggml.h#L2505-L2509
- Ollama (current main): "llama_server.go wraps the llama-server binary as a subprocess". Flash attention is auto-detected by llama-server. https://github.com/ollama/ollama/blob/98acec40ae2b3ed361fc5117e5b2ae81a4bf5c18/llm/llama_server.go#L1-L13 (file commit 2026-09-15)
- Ollama pins its llama.cpp source through `LLAMA_CPP_VERSION` (value `b10969` on 2026-09-22). https://github.com/ollama/ollama/blob/2e036e7cdf7baccca93045a2e313b5d8d7730ae9/llama/README.md and https://github.com/ollama/ollama/blob/a43fad18b088095de20fbd7a8f0de50824cf5d27/LLAMA_CPP_VERSION
- LM Studio docs: "LM Studio supports running LLMs on Mac, Windows, and Linux using llama.cpp". MLX is also supported on Apple Silicon. https://lmstudio.ai/docs/app

### 1.6 Hugging Face transformers (the local reference path), for contrast

- `sdpa_mask` creates "a 4D boolean mask of shape (batch_size, 1, query_length, kv_length)". https://github.com/huggingface/transformers/blob/e2d83fdf9c25a0f6021e90a241263a4a59faa15f/src/transformers/masking_utils.py#L373-L392 (file commit 2026-09-22)
- `use_gqa_in_sdpa` on CUDA requires "attention_mask is None (otherwise it will fall back to the math kernel)". Otherwise `repeat_kv` copies K/V. https://github.com/huggingface/transformers/blob/2b296305d7bcf040831661b7b689e13a2e2e8d67/src/transformers/integrations/sdpa_attention.py#L27-L36 and #L97-L102 (file commit 2026-08-20)
- PyTorch's SDPA dispatcher rejects masks for flash: "Flash Attention does not support non-null attn_mask." https://github.com/pytorch/pytorch/blob/c18a1a0ece45a67aafc412cc94e431e0bd28f8fb/aten/src/ATen/native/transformers/sdp_utils_cpp.h#L276-L279 (file commit 2026-08-30)
- transformers' own comment on float bias masks: "will usually prevent sdpa from dispatching to the most efficient kernel implementations". https://github.com/huggingface/transformers/blob/2b296305d7bcf040831661b7b689e13a2e2e8d67/src/transformers/integrations/sdpa_attention.py#L39-L53
- In the FA2 path, transformers returns "the 2D mask which will then be used to extract the seq_lens". https://github.com/huggingface/transformers/blob/e2d83fdf9c25a0f6021e90a241263a4a59faa15f/src/transformers/masking_utils.py#L608-L650
- `_flash_attention_forward` calls `_upad_input` whenever `attention_mask is not None`. `_get_unpad_data` rebuilds `cu_seqlens` with `torch.nonzero` and `.max().item()`. [UNCERTAIN] Per source reading this runs inside each attention call, so per layer; not profiled. https://github.com/huggingface/transformers/blob/a56a23f31d2b62adbac5314020dc0c1233e7ceea/src/transformers/modeling_flash_attention_utils.py#L353-L378 and #L773-L790 (file commit 2026-09-15)
- HF KV-cache docs on StaticCache: "a lot of tokens will actually be masked" and "it incurs a waste of tokens in the attention computation". https://github.com/huggingface/transformers/blob/602f674cbe4024ae42f4ab1c9b942f905a8575a9/docs/source/en/kv_cache.md#L72 (doc commit 2026-08-24)
- HF continuous-batching docs: CB works better with flash "mostly because Flash does not require an attention mask". https://github.com/huggingface/transformers/blob/e972043bf80e23d9a4d3ddbaca267637e4ffc568/docs/source/en/continuous_batching.md#L417 (doc commit 2026-08-28)
- Same doc: the decode fast path uses `flash_attn_with_kvcache` "through a block table". https://github.com/huggingface/transformers/blob/e972043bf80e23d9a4d3ddbaca267637e4ffc568/docs/source/en/continuous_batching.md#L339

---

## 2. New attention variants: what broke, what fell back, how long

### 2.1 Mistral sliding window (Mistral 7B, Sept 2023)

- FlashAttention v2.3 added "Local (i.e., sliding window) attention", crediting Mistral AI. Tag v2.3.0 is dated 2023-09-27. https://github.com/Dao-AILab/flash-attention/blob/1f7ce2f7cb503473559f3d44d575ae05b1ed8557/README.md#L462-L466 and https://github.com/Dao-AILab/flash-attention/releases/tag/v2.3.0
- vLLM added Mistral-7B support in PR #1196 (merged 2023-09-28). https://github.com/vllm-project/vllm/pull/1196
- vLLM wrong output: "If the prompt contains more than 4k tokens, the model will begin generating nonsense." (issue #2064, 2023-12-12, Mistral and Mixtral). https://github.com/vllm-project/vllm/issues/2064
- Fixed by vLLM PR #2088 "Fix input positions for long context with sliding window" (merged 2023-12-13). https://github.com/vllm-project/vllm/pull/2088
- llama.cpp issue #3377 "support sliding window attention" was opened 2023-09-28 by ggerganov. https://github.com/ggml-org/llama.cpp/issues/3377
- ggerganov on #3377 (2024-04-04): "there hasn't been much interest in this technique". The issue closed as stale 2024-11-01. https://github.com/ggml-org/llama.cpp/issues/3377
- A user on #3377 (2024-06-28): "gemma 2 pretty much falls apart after 4k context or so using llama.cpp". https://github.com/ggml-org/llama.cpp/issues/3377
- llama.cpp got a real SWA KV cache (smaller cache, pruning) only in PR #13194 "kv-cache : add SWA support" (merged 2025-05-20). https://github.com/ggml-org/llama.cpp/pull/13194
- transformers issue #28980 (2024-02-12): "the latest version of transformers cannot use sliding window feature in mistral model" (SDPA path, tied to a PyTorch mask bug). https://github.com/huggingface/transformers/issues/28980
- The follow-up transformers PR #29407 closed unmerged on 2024-05-02. [UNCERTAIN] Exact date SDPA-path SWA was fixed not established. https://github.com/huggingface/transformers/pull/29407
- TensorRT-LLM issue #11538 (2026-02-16): Ministral 8B (SWA) was not faster than Llama 3.1 8B on long inputs in TRT-LLM 1.1.0. https://github.com/NVIDIA/TensorRT-LLM/issues/11538
- A TRT-LLM maintainer replied on #11538 (2026-04-03): "SWA was not working for Ministral model". https://github.com/NVIDIA/TensorRT-LLM/issues/11538
- The fix was TRT-LLM PR #12597 "Enable sliding window attention for Mistral/Mixtral" (closed 2026-04-03). https://github.com/NVIDIA/TensorRT-LLM/pull/12597

### 2.2 Gemma 2 (released 2024-06-27): attention logit softcapping + interleaved 4K SWA

- FlashAttention v2.6 "Softcapping ... as used in Gemma-2 and Grok": tag v2.6.0 dated 2024-07-11. The upstream PR #1025 merged 2024-07-08. https://github.com/Dao-AILab/flash-attention/blob/1f7ce2f7cb503473559f3d44d575ae05b1ed8557/README.md#L479-L481 and https://github.com/Dao-AILab/flash-attention/pull/1025
- vLLM PR #5908 (merged 2024-06-27): "this PR removes soft-capping as a temporary workaround". https://github.com/vllm-project/vllm/pull/5908
- Same PR: vLLM "ignores" Gemma 2's SWA, uses global attention, and "temporarily truncates the model's maximum length to 4K". https://github.com/vllm-project/vllm/pull/5908
- vLLM PR #6051 "logits_soft_cap for Gemma2 with flashinfer" merged 2024-07-04. https://github.com/vllm-project/vllm/pull/6051
- vLLM maintainer on issue #6173 (2024-07-08): "You need flashinfer to run gemma2" and "flashinfer only supports GPU with compute capability >= 8.0". V100 users were excluded. https://github.com/vllm-project/vllm/issues/6173
- vLLM warning text quoted in issue #6220 (2024-07-10): "Disabling sliding window and capping the max length to the sliding window size (4096)". https://github.com/vllm-project/vllm/issues/6220
- vLLM PR #9403 (merged 2024-10-20) moved SWA onto the FlashAttention backend: "We can use it instead of fall back to xformer." https://github.com/vllm-project/vllm/pull/9403
- vLLM PR #10584 "gemma2 full context length support" merged 2024-11-23, about 5 months after release. "sliding window is only used for computation". https://github.com/vllm-project/vllm/pull/10584
- SGLang added Gemma 2 in PR #592 (merged 2024-07-05). https://github.com/sgl-project/sglang/pull/592
- SGLang added window attention for Gemma 2 in PR #1056 (merged 2024-08-14). https://github.com/sgl-project/sglang/pull/1056
- SGLang v0.3 blog (2024-09-04): full 8K context via FlashInfer's window kernel "which skips computation instead of masking". https://lmsys.org/blog/2024-09-04-sglang-v0-3/
- llama.cpp added Gemma 2 in PR #8156 (merged 2024-06-28). https://github.com/ggml-org/llama.cpp/pull/8156
- llama.cpp PR #8197 (merged 2024-06-30) added softcapping: "attention soft-capping is not compatible with flash attention so flash attention is disabled". GGUFs had to be regenerated. https://github.com/ggml-org/llama.cpp/pull/8197
- llama.cpp PR #8227 (merged 2024-07-01): "a hack to support sliding window attention for gemma 2", done by masking past tokens. https://github.com/ggml-org/llama.cpp/pull/8227
- llama.cpp PR #8542 "CPU/CUDA: Gemma 2 FlashAttention support" (merged 2024-08-24) added a `logit_softcap` parameter to FA. Metal followed in PR #9159 (2024-08-26). https://github.com/ggml-org/llama.cpp/pull/8542 and https://github.com/ggml-org/llama.cpp/pull/9159
- transformers PR #31887 "[Gemma2] Support FA2 softcapping" merged 2024-07-11. https://github.com/huggingface/transformers/pull/31887
- transformers issue #32309 (2024-07-30): Gemma-2 gave nonsense output with `flash_attention_2`. Fix PR #32188 merged 2024-07-31. https://github.com/huggingface/transformers/issues/32309 and https://github.com/huggingface/transformers/pull/32188
- transformers maintainer on issue #32390 (2024-08-03): "the `sdpa` path does not support logit soft-capping (For Gemma2)". https://github.com/huggingface/transformers/issues/32390
- Current transformers main: Gemma2 still passes `softcap=` to the attention interface, and SDPA is the default implementation. https://github.com/huggingface/transformers/blob/7cd73d9df0c14b151c684b708a9f27d8d0349dfe/src/transformers/models/gemma2/modeling_gemma2.py#L269-L283 and https://github.com/huggingface/transformers/blob/d67c72935fc5eae6539a6c2fde8326dc9332b5fa/src/transformers/modeling_utils.py#L1841-L1876
- `sdpa_attention_forward` has no softcap handling: it calls SDPA with only mask/scale/is_causal. [UNCERTAIN] Source reading only, not executed: Gemma 2 on default SDPA would silently skip attention softcapping. https://github.com/huggingface/transformers/blob/2b296305d7bcf040831661b7b689e13a2e2e8d67/src/transformers/integrations/sdpa_attention.py#L157-L166
- TensorRT-LLM on issue #1984 (2024-07-24): "It's not supported now, and we are proactively looking." https://github.com/NVIDIA/TensorRT-LLM/issues/1984
- TensorRT-LLM v0.13.0 (2024-09-30) release notes: "Supported Gemma 2". https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.13.0
- TRT-LLM issue #2233 "gemma-2-27b bad outputs" (2024-09-17) was closed as stale 2024-11-03. [UNCERTAIN] Root cause not established. https://github.com/NVIDIA/TensorRT-LLM/issues/2233

### 2.3 Gemma 3 (5:1 local/global SWA) in llama.cpp

- llama.cpp issue #12637 (2025-03-29): a mask existed for SWA but no code reduced KV-cache use. A user estimated 62 GB → 10.4 GB at 128K for 27B if implemented [user estimate]. https://github.com/ggml-org/llama.cpp/issues/12637
- ggerganov closed #12637 on 2025-06-07: "The iSWA is now supported." The fix is PR #13194. https://github.com/ggml-org/llama.cpp/issues/12637

### 2.4 gpt-oss (released 2025-08-05): attention sinks + alternating 128-token SWA

- vLLM blog (2025-08-05): day-0 support needed a special build `vllm==0.10.1+gptoss`. It covered NVIDIA Blackwell/Hopper and AMD MI300x/MI355x, with FA3 on Hopper and Triton on AMD. https://vllm.ai/blog/2025-08-05-gpt-oss
- vLLM PR #22320 "Add attention sink in attention backends" merged 2025-08-06. https://github.com/vllm-project/vllm/pull/22320
- vLLM PR #22714 "[gpt-oss] Enable gpt-oss on ampere" merged 2025-08-12, 7 days after release. https://github.com/vllm-project/vllm/pull/22714
- vLLM PR #22478 "FA3 Attention Sinks Perf Boost" (merged 2025-08-15), gpt-oss-20b serving bench: mean TPOT 18.57 ms (main) → 14.87 ms. Mean TTFT 69.56 → 48.26 ms. https://github.com/vllm-project/vllm/pull/22478
- vLLM FA backend today: `supports_sink()` depends on `flash_attn_supports_sinks()`. https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/v1/attention/backends/flash_attn.py#L398-L401
- vLLM's rendered feature table lists sink support for FA3/FA4, Triton and FlashInfer trtllm-gen. [UNCERTAIN] The table was read through a summarizer. https://docs.vllm.ai/en/latest/design/attention_backends/
- llama.cpp PR #15091 "llama : add gpt-oss" merged 2025-08-05 (day 0). Its note: "currently only the vec FA kernels are implemented" with sinks. https://github.com/ggml-org/llama.cpp/pull/15091
- llama.cpp PR #15157 (merged 2025-08-08) added sinks to CUDA tensor-core FA. gpt-oss-20B pp16384 speedups: RTX 4090 1.02x (ubatch 1) to 3.99x (ubatch ≥2048). RTX 3090 up to 6.02x. The day-0 fallback was that much slower in prefill. https://github.com/ggml-org/llama.cpp/pull/15157
- llama.cpp PR #15178 (merged 2025-08-09) added sinks to the tile/wmma FA kernels for older GPUs. https://github.com/ggml-org/llama.cpp/pull/15178
- transformers gpt-oss today: `_supports_sdpa = False`. The eager path uses `repeat_kv` + additive mask + concatenated sink logits. https://github.com/huggingface/transformers/blob/7cd73d9df0c14b151c684b708a9f27d8d0349dfe/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L235-L262 and #L386-L388
- transformers gpt-oss flash implementations are limited to `kernels-community/vllm-flash-attn3`, `flash_attention_4` and `kernels-community/metal-flash-sdpa`. https://github.com/huggingface/transformers/blob/7cd73d9df0c14b151c684b708a9f27d8d0349dfe/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L399-L403

### 2.5 Llama 4 (released 2025-04-05): chunked local attention (iRoPE)

- vLLM PR #16104 "Support Llama4 in vLLM" merged 2025-04-06. https://github.com/vllm-project/vllm/pull/16104
- vLLM PR #21419 (merged 2025-07-23): conflicting changes made `use_irope` always False in V1, which silently disabled chunked local attention. https://github.com/vllm-project/vllm/pull/21419
- With that bug, Llama-4-Scout niah_multikey_2 at 32768 tokens was 0.000; after the fix it was 0.944. https://github.com/vllm-project/vllm/pull/21419
- vLLM PR #21761 (merged 2025-07-28): with the hybrid KV cache, Llama 4's "attn metadata and local virtual batches ... being constructed 3 times". Mean TPOT 28.46 ms (enabled) vs 26.48 ms (disabled). The PR disabled that path by default. https://github.com/vllm-project/vllm/pull/21761
- transformers PR #37307 "Add llama4" merged 2025-04-05. https://github.com/huggingface/transformers/pull/37307
- transformers issue #37351 (2025-04-08 comment): chunked attention errors; with eager or sdpa "it still throws an error if the input length exceeds 8K". https://github.com/huggingface/transformers/issues/37351
- transformers PR #37416 (merged 2025-04-10) marked Llama 4 unsupported with FA2. FA2 completed "Roses are red," as "of the1 in"; flex gave "violets are blue". https://github.com/huggingface/transformers/pull/37416
- transformers issue #37465 (open since 2025-04-12): "Llama4 do not support flash_attn, leading huge GPU MEM consumption." https://github.com/huggingface/transformers/issues/37465
- llama.cpp PR #12791 (merged 2025-04-07) supported Llama 4 text with the chunked attention implemented as a chunked KQ mask. https://github.com/ggml-org/llama.cpp/pull/12791

### 2.6 DeepSeek MLA (V2 May 2024; V3 Dec 2024)

- vLLM PR #4650 "Support Deepseek-V2" merged 2024-06-28. Its note: only the MHA approach is implemented, not "the efficient inference mode". https://github.com/vllm-project/vllm/pull/4650
- vLLM PR #12528 "[Attention] MLA decode optimizations" merged 2025-01-31, about 7 months after #4650. It credits SGLang's Triton decode kernel. https://github.com/vllm-project/vllm/pull/12528
- SGLang PR #905 "Support MLA for DeepSeek-V2 with Triton - step 1" merged 2024-08-04. https://github.com/sgl-project/sglang/pull/905
- SGLang PR #1447 "Enable MLA by default" merged 2024-09-17. https://github.com/sgl-project/sglang/pull/1447
- SGLang v0.3 blog (2024-09-04): MLA optimizations give "3x to 7x higher throughput than the baseline system". https://lmsys.org/blog/2024-09-04-sglang-v0-3/
- llama.cpp PR #7519 (merged 2024-05-28) added DeepSeek-V2 without MLA (decompressed K/V). https://github.com/ggml-org/llama.cpp/pull/7519
- llama.cpp PR #12801 "DeepSeek V2/V3 MLA implementation" merged 2025-04-15, about 10.5 months after #7519. It needs new GGUFs and makes "the new MLA GGUF files appear to be MQA". https://github.com/ggml-org/llama.cpp/pull/12801
- llama.cpp PR #13306 "CUDA: FA support for Deepseek (Ampere or newer)" merged 2025-05-09. https://github.com/ggml-org/llama.cpp/pull/13306

### 2.7 DeepSeek V3.2 sparse attention (DSA; released 2025-09-29)

- vLLM blog (2025-09-29): day-0 support uses FlashMLA sparse + DeepGEMM indexer kernels. "We plan to expand the architectures supported beyond Hopper and Blackwell." https://vllm.ai/blog/2025-09-29-deepseek-v3-2
- llama.cpp feature request #16331 (opened 2025-09-29) was closed as `not_planned` on 2026-03-04. https://github.com/ggml-org/llama.cpp/issues/16331
- llama.cpp PR #18849 (2026-01-14, unmerged) was dense-attention-only. It said an earlier attempt "fell into degenerate generation at about 45k context". https://github.com/ggml-org/llama.cpp/pull/18849
- llama.cpp PR #19460 (GLM DSA arch, merged 2026-02-13) kept the indexer tensors unused: "The quality will be suboptimal". https://github.com/ggml-org/llama.cpp/pull/19460
- llama.cpp PR #23346 (merged 2026-05-29, about 8 months after release) "implemented sparse attention by masking KQ mask elements". https://github.com/ggml-org/llama.cpp/pull/23346
- The same PR warns: "Generic lightning indexer implementation uses very large compute buffers". https://github.com/ggml-org/llama.cpp/pull/23346
- llama.cpp PR #24231 (merged 2026-07-11) added a dedicated indexer op. CPU compute buffer went from 168,368 MiB to 5,808 MiB. https://github.com/ggml-org/llama.cpp/pull/24231

### 2.8 ALiBi

- FlashAttention v2.4 added ALiBi (tag v2.4.0 dated 2023-12-22). https://github.com/Dao-AILab/flash-attention/blob/1f7ce2f7cb503473559f3d44d575ae05b1ed8557/README.md#L468-L470 and https://github.com/Dao-AILab/flash-attention/releases/tag/v2.4.0
- vLLM PR #945 "Fix Alibi implementation in PagedAttention kernel" merged 2023-09-07. [UNCERTAIN] The PR body does not describe the bug's impact. https://github.com/vllm-project/vllm/pull/945
- vLLM PR #15231 (merged 2025-03-21): "FA3 does not support ALiBi positional encodings". FA3 had been the Hopper default since v0.7.0. https://github.com/vllm-project/vllm/pull/15231
- In v0.8.1, BLOOM failed with a RuntimeError; PR #15231 made vLLM fall back to FA2. https://github.com/vllm-project/vllm/pull/15231
- llama.cpp PR #7192 (merged 2024-05-11) implements ALiBi by adding the bias into `KQ_mask`. https://github.com/ggml-org/llama.cpp/pull/7192

### 2.9 Scale of hand-maintained variant support (cost indicators)

- vLLM's `AttentionBackendEnum` has 44 members on 2026-09-16. Several are model- or variant-specific, e.g. `FLASHMLA_SPARSE_DSV4`, `FLASHINFER_MLA_SPARSE_DSV41`, `MINIMAX_M3_SPARSE`. https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/v1/attention/backends/registry.py#L34-L202
- SGLang has 27 `*_backend.py` modules in `srt/layers/attention` on 2026-09-22. They include `deepseek_v4_backend.py`, `minimax_sparse_backend.py` and `qwen_sparse_attn_backend.py`. https://github.com/sgl-project/sglang/tree/4c81cd1b0904f8f7778e87c0974b464aab21b74c/python/sglang/srt/layers/attention

---

## 3. Who runs inference through transformers `generate()`

- lm-evaluation-harness HF backend: `_model_generate` calls `self.model.generate(input_ids=context, ..., use_cache=True)`. https://github.com/EleutherAI/lm-evaluation-harness/blob/3e093bc90a87cb42c148820b57452ffbb570ac8c/lm_eval/models/huggingface.py#L1153-L1188 (file commit 2026-08-20)
- `generate_until` feeds that call left-padded batches plus `attention_mask`, via `tok_batch_encode` with `padding_side="left"` by default. https://github.com/EleutherAI/lm-evaluation-harness/blob/3e093bc90a87cb42c148820b57452ffbb570ac8c/lm_eval/models/huggingface.py#L1653-L1671 and #L1068
- The lm-eval README quickstart examples use `lm_eval --model hf`. vLLM is offered separately "for faster inference". https://github.com/EleutherAI/lm-evaluation-harness/blob/ad8737ae7fad24cf64e50fc7fc31397bff586b9e/README.md#L126 and #L456-L458 (commit 2026-09-10)
- TRL `GRPOConfig.use_vllm` defaults to `False`. When True, vLLM is used "instead of the default model.generate()". https://github.com/huggingface/trl/blob/aa89588dc63ad5f4354614e94c2851759ebb617e/trl/trainer/grpo_config.py#L118-L120 and #L582-L588 (commit 2026-09-09)
- TRL's `use_transformers_continuous_batching` defaults to `False` and requires transformers>=5.8.0. https://github.com/huggingface/trl/blob/aa89588dc63ad5f4354614e94c2851759ebb617e/trl/trainer/grpo_config.py#L380-L382
- TRL docs: "generating them with the model's own `generate` is the bottleneck". https://github.com/huggingface/trl/blob/f6a8beb66310e463621c95e8a95582cda9d57248/docs/source/vllm_integration.md#L113-L117 (commit 2026-09-18)
- HF optimization overview: pass a fixed-size cache to `generate` "to trigger `torch.compile` automatically". It warns to avoid `torch.compile(model)` outside `generate` because of per-step recompiles. https://github.com/huggingface/transformers/blob/05e078a1d276d0f9dcfc08d1f2964cb8abb33b04/docs/source/en/optimization_overview.md#L40-L57 (commit 2026-09-01)
- HF KV-cache docs: `cache_implementation="static"` "will also turn on automatic compilation of the decoding stage". https://github.com/huggingface/transformers/blob/602f674cbe4024ae42f4ab1c9b942f905a8575a9/docs/source/en/kv_cache.md#L76
- `transformers serve` docs: "Use it for evaluation, experimentation, and moderate load deployments." https://github.com/huggingface/transformers/blob/80a79d1bef2bacfa10b1b543669eebffa3f0ec36/docs/source/en/serve-cli/serving.md#L19-L22 (commit 2026-09-12)
- Same doc: "For large scale production deployments, use vLLM or SGLang". https://github.com/huggingface/transformers/blob/80a79d1bef2bacfa10b1b543669eebffa3f0ec36/docs/source/en/serve-cli/serving.md#L22
- HF docs position transformers as the model definition used inside engines ("compatible with inference engines like vLLM and SGLang") through the AttentionInterface. https://github.com/huggingface/transformers/blob/e9af3a323f5f822db84d8c84a0c8bf802e928854/docs/source/en/community_integrations/transformers_as_backend.md
- TGI README: "text-generation-inference is now in maintenance mode." It recommends vLLM, SGLang, llama.cpp and MLX. The commit "Maintenance mode (#3344)" is dated 2025-12-11. https://github.com/huggingface/text-generation-inference/blob/52c6dddf97531e5ef9d77ef55de2c806563676c6/README.md#L1-L4
- The GitHub API reports `huggingface/text-generation-inference` as `archived: true`, last pushed 2026-03-21. https://api.github.com/repos/huggingface/text-generation-inference
- vLLM lists `transformers >= 5.10.4` in its common requirements, so installing vLLM also pulls in transformers. https://github.com/vllm-project/vllm/blob/c58532c86b8ab821fc1d819636480b7588bce7b5/requirements/common.txt#L10
- SGLang pins `"transformers==5.12.1"` in its pyproject. https://github.com/sgl-project/sglang/blob/b18ca9ca443302501b009ecc7b3f065b691f3f13/python/pyproject.toml#L95

### 3.1 PyPI downloads (pypistats `recent`, fetched 2026-09-22 12:44–12:56 UTC; site says data update once daily)

- transformers: last_month 96,077,696; last_week 21,743,557; last_day 3,254,753. https://pypistats.org/api/packages/transformers/recent
- vllm: last_month 2,145,771; last_week 407,725; last_day 65,711. https://pypistats.org/api/packages/vllm/recent
- sglang: last_month 12,186,349; last_week 390,878; last_day 30,769. https://pypistats.org/api/packages/sglang/recent
- llama-cpp-python: last_month 554,579; last_week 116,293; last_day 16,607. https://pypistats.org/api/packages/llama-cpp-python/recent
- ollama (Python client library, not the server): last_month 12,736,801; last_week 2,887,374. https://pypistats.org/api/packages/ollama/recent
- Noise example: sglang daily downloads (without mirrors) ranged from 6,549 (2026-09-05) to 487,153 (2026-09-10). The count was 11,953,859 on 2026-08-19. The source gives no reason. https://pypistats.org/api/packages/sglang/overall?mirrors=false
- Caveat: downloads are a noisy usage proxy. Per the requirements cited above, vLLM and SGLang installs also pull transformers. CI and automated installs are not separated in these counts. https://pypistats.org/api/

---

## 4. Consumer / local GPU usage

- GitHub stars (API, 2026-09-22 12:46 UTC): ollama/ollama 181,454; ggml-org/llama.cpp 129,173; huggingface/transformers 166,520. https://api.github.com/repos/ollama/ollama , https://api.github.com/repos/ggml-org/llama.cpp , https://api.github.com/repos/huggingface/transformers
- More GitHub stars (same time): vllm-project/vllm 92,421; sgl-project/sglang 36,321; NVIDIA/TensorRT-LLM 14,692; abetlen/llama-cpp-python 10,629; ml-explore/mlx-lm 7,099. https://api.github.com/repos/vllm-project/vllm (same pattern for the others)
- LM Studio's app is not open-source. Its public repos have lmstudio-ai/lms 5,307 and lmstudio-ai/lmstudio-js 1,782 stars; these do not measure app usage. https://api.github.com/repos/lmstudio-ai/lms
- Docker Hub cumulative pulls (2026-09-22): ollama/ollama 177,873,563; vllm/vllm-openai 36,107,881; lmsysorg/sglang 13,509,275. https://hub.docker.com/v2/repositories/ollama/ollama/ , https://hub.docker.com/v2/repositories/vllm/vllm-openai/ , https://hub.docker.com/v2/repositories/lmsysorg/sglang/
- Stack Overflow Developer Survey 2025, AI agent orchestration tools: Ollama 51.1% of 3,758 respondents (7.7% of the survey). The base is agent builders only, not all developers. https://survey.stackoverflow.co/2025/ai
- The survey page's summary: "Among developers building agents, Ollama (51%) and LangChain (33%) are the most-used frameworks." https://survey.stackoverflow.co/2025/ai
- No other survey was found with usage shares for llama.cpp, Ollama or LM Studio (web search 2026-09-22). [UNCERTAIN] Absence of evidence only.
- Attention path of these tools: Ollama wraps llama-server and LM Studio runs llama.cpp (or MLX). The GPU attention path is therefore llama.cpp's dense KQ mask + native GQA broadcast (§1.5). https://github.com/ollama/ollama/blob/98acec40ae2b3ed361fc5117e5b2ae81a4bf5c18/llm/llama_server.go#L1-L13 and https://lmstudio.ai/docs/app

---

## 5. Measured or stated costs of metadata construction / mask handling

### 5.1 Per-step CPU work in GPU serving engines

- vLLM v0.6.0 blog (2024-09-05), Llama 3 8B on 1xH100: "29% of the total execution time is spent on scheduling". That 29% includes preparing inputs. https://vllm.ai/blog/2024-09-05-perf-update
- Same blog: the HTTP API server took 33%, and "only 38% of the time was spent on the actual GPU execution". https://vllm.ai/blog/2024-09-05-perf-update
- vLLM V1 blog (2025-01-27): in V0, "input tensors and metadata for the model are recreated at each step". https://vllm.ai/blog/2025-01-27-v1-alpha-release
- The same blog says V1's Persistent Batch "caches the input tensors and only applies the diffs". https://vllm.ai/blog/2025-01-27-v1-alpha-release
- A vLLM FA backend code comment: under piecewise CUDA graphs the attention forward runs eagerly. It says "`view` and `slice` (or `[:n]`) operations are surprisingly slow". https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/v1/attention/backends/flash_attn.py#L1215-L1222
- vLLM's `build(..., fast_build)` flag makes metadata "prioritize speed of building over then speed at execution" (for spec-decode). https://github.com/vllm-project/vllm/blob/651a88c09ac6cbb13c5dd36f2fdbe61a2c68eeef/vllm/v1/attention/backend.py#L662-L664
- vLLM PR #21761 (2025-07-28): building Llama 4 local-attention metadata 3 times cost about 2 ms per output token (TPOT 28.46 vs 26.48 ms). https://github.com/vllm-project/vllm/pull/21761
- vLLM PR #23185 (merged 2025-08-20): metadata micro-optimization for Llama 4. `cu_seqlens_q_local` went 35 µs → 6 µs and `batch_indices` 53 µs → 34 µs. https://github.com/vllm-project/vllm/pull/23185
- SGLang v0.4 blog (2024-12-04): an unoptimized engine "can spend as much as half of its time on CPU overhead". https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- The same blog's fix is a scheduler that "runs one batch ahead" and prepares next-batch metadata, giving 1.1x throughput. https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- TensorRT-LLM's overlap scheduler overlaps CPU tasks (stop checks, response updates, next-batch scheduling) with GPU work. https://github.com/NVIDIA/TensorRT-LLM/blob/f53fb4c8032c0420c669e45963bd23205a58068e/docs/source/features/overlap-scheduler.md#L3

### 5.2 FlashInfer plan() and CUDA graphs

- FlashInfer paper: plan runs on CPU "per generation step". The cost is amortized because "the same plan information can be reused for all layers". https://arxiv.org/html/2501.01005
- FlashInfer paper: "plan function is not captured by CUDAGraph because it's on CPU". Kernels use a fixed grid size to stay graph-compatible. https://arxiv.org/html/2501.01005
- FlashInfer's `fast_decode_plan` exists to "Remove unnecessary host-to-device copy for the metadata buffers." It also removes device-to-device copies. https://github.com/flashinfer-ai/flashinfer/blob/d7a7447cb4bb29b4637f7fa02fe5eaeeb5e61a6e/flashinfer/decode.py#L4236-L4265
- SGLang's FlashInfer backend has an indptr override fast path: "This is used to remove some host-to-device copy overhead." https://github.com/sgl-project/sglang/blob/61d0cf2074711db9c54c993a3732355b5377c906/python/sglang/srt/layers/attention/flashinfer_backend.py#L173-L175
- No paper or doc read here gave a direct plan() latency number. [UNCERTAIN]
- vLLM CUDA-graph design doc: piecewise capture was built by "excluding cudagraph-unsupported operations (mainly attention)". https://github.com/vllm-project/vllm/blob/33f50773cbec56cda66af786443bd13409df9bd5/docs/design/cuda_graphs.md#L25 (commit 2026-06-23)
- The same doc says full graphs were limited ("only FlashAttention 3 supports it currently"). FlashInfer, FlashMLA and Mamba were decode-only. https://github.com/vllm-project/vllm/blob/33f50773cbec56cda66af786443bd13409df9bd5/docs/design/cuda_graphs.md#L25
- The doc's backend table: FA3 `ALWAYS`, FA2 `UNIFORM_BATCH`, FlashInfer `UNIFORM_SINGLE_TOKEN_DECODE`, FlashMLA `UNIFORM_BATCH`. https://github.com/vllm-project/vllm/blob/33f50773cbec56cda66af786443bd13409df9bd5/docs/design/cuda_graphs.md#L177-L186

### 5.3 How engines keep attention away from the compiler (the batch-size guessing problem)

- vLLM torch.compile design doc: "the only changing size to the computation graph, is the batch size". vLLM drops Dynamo's shape guards and offers backed/unbacked dynamic-shape modes. https://github.com/vllm-project/vllm/blob/3461e7efd8d1af0dd069900383fd5e33c7956c1f/docs/design/torch_compile.md#L32-L36 and #L171 (commit 2025-11-28)
- Same doc: "we wrap the whole attention operation into a PyTorch custom op" so Dynamo does not inspect it. https://github.com/vllm-project/vllm/blob/3461e7efd8d1af0dd069900383fd5e33c7956c1f/docs/design/torch_compile.md#L173
- Same doc: `compile_sizes` such as `[1, 2, 4, 8]` add statically specialized, autotuned kernels. It notes "This can be slow when you run it for the first time". https://github.com/vllm-project/vllm/blob/3461e7efd8d1af0dd069900383fd5e33c7956c1f/docs/design/torch_compile.md#L214-L217
- TensorRT-LLM doc: prefill still shows bubbles "primarily due to the attention operator's substantial host-side overhead". https://github.com/NVIDIA/TensorRT-LLM/blob/21dc97fbc8d431ff9f6aa1693db181e6e761936c/docs/source/features/torch_compile_and_piecewise_cuda_graph.md#L114 (commit 2026-09-13)
- The same doc wraps attention in a big custom op. Reasons include "The argument number exceeds the torch custom op's limitation" and MLA dynamic shapes that "may introduce recompilation". https://github.com/NVIDIA/TensorRT-LLM/blob/21dc97fbc8d431ff9f6aa1693db181e6e761936c/docs/source/features/torch_compile_and_piecewise_cuda_graph.md#L162-L166
- transformers continuous batching: "the varlen path skips compilation because `max_seqlen_k` triggers frequent recompilation". https://github.com/huggingface/transformers/blob/e972043bf80e23d9a4d3ddbaca267637e4ffc568/docs/source/en/continuous_batching.md#L335

### 5.4 llama.cpp: rediscovering structure from the dense mask

- CUDA PR #14924 (merged 2025-07-30) skips fully masked KV slices for all CUDA FA kernels. An auxiliary kernel scans the mask, with "overhead <1% end-to-end". https://github.com/ggml-org/llama.cpp/pull/14924
- #14924's pp8192 speedups reach 1.18x (RX 6800), 1.10x (P40) and 1.02x (RTX 4090) at large microbatches. Its stated main benefit is batched server throughput (plot only, no number). https://github.com/ggml-org/llama.cpp/pull/14924
- The #14924 author: it would be "possible to pre-compute the max. extents of the KV cache in CPU code". He judged it not worth the extra complexity. https://github.com/ggml-org/llama.cpp/pull/14924
- Metal PR #16372 (merged 2025-10-08): "run a quick pass over the mask to find all -INF blocks". Speedups were 1.01–1.08x on Gemma 3 prefill. https://github.com/ggml-org/llama.cpp/pull/16372
- Vulkan PR #19281 (merged 2026-02-05) preprocesses the FA mask to detect all -inf and all-zero blocks. https://github.com/ggml-org/llama.cpp/pull/19281
- Metal PR #19337 (merged 2026-02-06) skips loading all-zero mask blocks, giving 1.00–1.03x. https://github.com/ggml-org/llama.cpp/pull/19337
- HIP PR #28943 (open, unmerged as of 2026-09-22): with a unified multi-slot KV cache, the kernel "can still compute interior KV tiles that are fully masked". https://github.com/ggml-org/llama.cpp/pull/28943
- The same PR claims "third-request time drops by approximately 33.3% versus upstream" in one RX 7900 XTX scenario (40,960-token prompt). [UNCERTAIN] Single author, one workload, not merged. https://github.com/ggml-org/llama.cpp/pull/28943
- PR #27970 "add sparse-fa for DSV4/GLM" (merged 2026-09-02) passes an explicit hint: the "max number of live kv-entries per token". https://github.com/ggml-org/llama.cpp/pull/27970
- With that hint, #27970 reports CPU speedups of 1.19x at 32K depth and 2.03x at 524,288 depth (DeepSeek V4 2-bit, DGX Spark). https://github.com/ggml-org/llama.cpp/pull/27970

### 5.5 Counterpoint: explicit paging metadata also has costs

- vAttention paper (arXiv 2405.04437): PagedAttention makes the KV cache's virtual layout non-contiguous. The authors report "up to 1.23x" serving throughput over PagedAttention-based FA/FlashInfer kernels. https://arxiv.org/abs/2405.04437

---

## 6. Gaps and cautions

- No source read here isolates the cost of a dense mask versus explicit lengths inside vLLM, SGLang or TRT-LLM. Their main paths do not use dense masks, so the local 1.3–2.5x has no direct production analogue there. The closest production analogues are the llama.cpp mask-scanning PRs (§5.4). [UNCERTAIN]
- The Gemma 2 + SDPA softcap drop in current transformers comes from static source reading only (§2.2). It should be checked by running it. [UNCERTAIN]
- vLLM's rendered backend feature table was read through a summarizing fetcher. Treat §2.4's table bullet as indicative. [UNCERTAIN]
- Download and star counts are noisy proxies (§3.1, §4). The only survey found samples agent builders, not local-inference users in general.
