# Items

Each item: an id, the software project, the issue title, and two short descriptions written earlier by two
independent readers of the issue and its fix. Rate every item. Do not look anything up.

## X001 (sglang)
Title: [Bug] Mooncake PD silently accepts mismatched Mamba SSM state dtypes/item lengths
Description 1: Prefill and decode used different Mamba SSM dtypes, and the transfer applied the source item length as the destination stride with no check, corrupting decode state.
Description 2: SGLang accepts mismatched prefill/decode Mamba SSM state dtypes without validation, and the transfer copies using only the source item length, so BF16-vs-FP32 state layouts silently disagree.

## X002 (transformers)
Title: Wrong dim for torch.cumsum(token_priority) & Identical router_probs / router_logits Output
Description 1: The fix says the router returned the top-1 probabilities from torch.max as router_logits and computed expert capacity over the flattened batch, both computation errors inside the router.
Description 2: Fix corrects the router formula: router_logits held selected top-1 probabilities instead of raw classifier logits, plus a wrong-dim cumsum.

## X003 (vllm)
Title: [Bug]: Triton fused MoE incorrectly indexes per-channel weight scales with per-tensor activations
Description 1: The Quark scheme declares per-channel weight scales, but the Triton path reused the per-tensor activation flag to index them, so routed experts' channel scales were ignored; the fix sets per_out_ch_quant from the actual weight scheme.
Description 2: The fix separates per_out_ch_quant from per_act_token_quant, showing the kernel had been indexing per-channel weight scales using a flag that actually describes a different (activation) property.

## X004 (vllm)
Title: [Bug]: CPU MoE kernel produces NaN logits under torch.compile for Qwen3_5MoeForConditionalGeneration — reproduces with AND without GPTQ-Int4 quantization, fixed by --enforce-eager
Description 1: The reporter isolates the NaN to the torch.compile/inductor build of the CPU MoE path while eager is correct, but no mechanism or fix is given.
Description 2: Reporter isolates the defect to a torch.compile/inductor miscompilation of the CPU MoE path, since eager mode is unaffected.

## X005 (sglang)
Title: [Bug] compressed-tensors W4AFP8 MoE: cutlass_w4a8_moe passes a literal chunk_size=128, so a group_size≠128 checkpoint loads clean and serves silently wrong output
Description 1: Scale tensors are sized from the checkpoint's group_size, but cutlass_w4a8_moe passes a literal chunk_size=128 to the kernel, so a group_size 64 checkpoint is dequantized with the wrong stride.
Description 2: cutlass_w4a8_moe always passes a literal chunk_size=128 regardless of the checkpoint's actual group_size, so the kernel never learns the real value.

## X006 (vllm)
Title: [Bug]: DSV4 sparse MLA emits garbage (gsm8k ~0%) with the packed KV layout from #44577
Description 1: The packed KV layout makes each component's per-block stride the whole block span; the cache-store op honors it, but the FlashInfer sparse-MLA decode addresses tokens by flat per-token stride and ignores it.
Description 2: The packed-KV change makes each component's per-block stride the whole packed span, and the cache-store op honors this new stride while the sparse-MLA decode path still assumes the old per-token stride convention.

## X007 (vllm)
Title: [Bug]: ExampleHiddenStatesConnector returns nan for hybrid attention model
Description 1: The hidden-states KV-cache group index was hardcoded to 0, so on hybrid models the reader received another group's block IDs; the fix looks up the real group.
Description 2: _hs_group_idx is hardcoded to 0, so for hybrid models where the hidden-states KV cache group is not group 0, the reader is handed the wrong group's block IDs and reads uninitialized cache.

## X008 (sglang)
Title: [Bug] Qwen3.5 with  inductor ,the precision is incorrect
Description 1: Accuracy is wrong only with the inductor piecewise compiler, and the closed fix PR makes the MoE forward op a graph split under inductor.
Description 2: The reporter's own fix adds the MoE forward function to the inductor compiler's split-op list, implying it was missing a case already handled for the deepep/mooncake backends.

## X009 (sglang)
Title: [DFlash] Infinite loop when using repetition_penalty — missing token accumulation and scaling penalties
Description 1: The DFlash verify path never accumulated committed tokens or applied repetition penalties, so the parameter was silently ignored; the fixes add the missing penalty handling.
Description 2: BatchedRepetitionPenalizer is never registered in the penalty orchestrator, and DFlash's speculative verification path separately never accumulates committed tokens into penalizer state, so the penalty is never applied.

## X010 (transformers)
Title: WhisperFeatureExtractor: one non-finite input sample makes the entire feature matrix NaN, and the pipeline transcribes it silently
Description 1: The dynamic-range floor log_spec.max() - 8.0 turns a single NaN sample into NaN across the whole feature matrix, as both the report and the fix PR state.
Description 2: log_spec.max() returns NaN for one bad sample and np.maximum(x, NaN) propagates NaN to the entire feature matrix.

## X011 (vllm)
Title: [Bug][XPU] compressed-tensors FP8 W8A8 (dynamic) generates garbage output on Intel Arc Pro B70 (Battlemage)
Description 1: The fix says the XPU FP8 kernel requires a C-contiguous [K, N] weight but the post-load path left a non-contiguous transposed view.
Description 2: The post-load transpose leaves weights in a non-contiguous view while the XPU kernel requires a C-contiguous layout.

## X012 (vllm)
Title: DeepSeek-V4-Flash-0731: deterministic wrong token on deep-context exact retrieval at 1-in-4 prompt lengths (reasoning off) — vLLM 0.28.0 + SGLang, reproduced on two hosted providers (DeepInfra, Baidu); one provider (OpenInference) is correct
Description 1: The only (open) fix rounds seq_lens to a multiple of 4 to avoid an alignment bounds issue inside fp8_fp4_paged_mqa_logits, matching the prompt-length = 3 mod 4 trigger.
Description 2: Fix rounds seq_lens to a multiple of 4 to avoid an internal memory-alignment bounds computation in the kernel; the mechanism is stated only briefly.

## X013 (sglang)
Title: [Bug][NPU] ModelSlim W8A8 checkpoint of Qwen3.8-27B (GDN hybrid) loads with garbage weights: fused in_proj_qkvz scheme unresolved, vision encoder forced unquantized; NPU quant matmul crashes on 3-D vision input
Description 1: The fused in_proj_qkvz name misses the checkpoint's per-projection quantization scheme, so the loader defaults to unquantized and copies int8 weights raw into bf16 parameters.
Description 2: The scheme-lookup code resolves several checkpoint naming variants but has no case for the GDN model's fused in_proj_qkvz/in_proj_ba parameter names, so the lookup misses and the quantized weights load raw into bf16.

## X014 (vllm)
Title: [Bug]: GLM5 on B300 generates garbage output
Description 1: FlashInfer requires a bf16 routing bias for trtllm MoE, which vLLM converted for other paths but not for per-block FP8; the fix passes the required dtype.
Description 2: Fix states flashinfer requires bf16 routing-bias dtype for trtllm MoE, which had not been applied for the fp8 per-block path.

## X015 (diffusers)
Title: Division by zero in rescale_noise_cfg can produce NaNs during inference
Description 1: rescale_noise_cfg divides by std_cfg with no guard, producing NaN/inf when the std is zero; the fixes add an epsilon or clamp.
Description 2: rescale_noise_cfg divides by std_cfg with no guard against zero, producing NaN/inf when the guided prediction has zero variance.

## X016 (vllm)
Title: [Bug][ROCm] GLM-5.3-Flash indexer block-table mismatch on gfx942
Description 1: allocate_kv_cache honours storage_block_size but prepare_kernel_block_sizes does not, so block-table entries hold manager block IDs where physical page IDs are expected.
Description 2: allocate_kv_cache honors storage_block_size but prepare_kernel_block_sizes does not, so manager block IDs are used where physical page IDs are required.

## X017 (vllm)
Title: [Bug]: GLM-5.1-FP8 produces gibberish with RunAI streamer after ac3dac545
Description 1: The RunAI streamer reuses internal tensor buffers, so retained yielded tensors were overwritten; the fix yields clones.
Description 2: The fix states the RunAI Model Streamer reuses its internal tensor buffers, so a tensor retained past the current iterator step could observe contents already overwritten by the next weight.

## X018 (vllm)
Title: [Bug]: FlashInfer fused allreduce + residual RMSNorm + quant produces corrupted output with FP32 norm weights
Description 1: The fusion patterns selected the fused op for a BF16 input with an FP32 norm weight without checking dtypes, and the fix adds the dtype-match guard.
Description 2: The residual quantized fusion patterns skip the _norm_input_weight_dtype_match check that sibling patterns apply, so a fused kernel is selected for a graph where the activation is BF16 but the RMSNorm weight is FP32.

## X019 (vllm)
Title: [Bug]: DeepSeek V3.2 & V4 incorrect structured output when thinking enabled
Description 1: The engine-side structured-output gate instantiated the reasoning parser without the request's chat-template kwargs; the fix forwards reasoning_parser_kwargs to it.
Description 2: Fix states the engine-side structured-output gate must instantiate the reasoning parser with the same chat-template kwargs the frontend used, which it previously did not receive.

## X020 (sglang)
Title: [Bug] MiniMax-M3 incorrectly routes Ascend FuseEP through the normal MoE path
Description 1: The MoE forward dispatch checks only is_deepep() and omits is_ascend_fuseep(), sending FuseEP through forward_normal and its extra all-reduce; the fix reroutes it.
Description 2: MiniMaxM3MoE checks only is_deepep() and omits is_ascend_fuseep(), so FuseEP falls into forward_normal(), which then performs a TP all-reduce on output that FuseEP already globally combined.

## X021 (sglang)
Title: [Bug] Kimi K2.6 DEP8 produces garbage output but DP8 works fine
Description 1: The related fix says that in SUM_LEN mode the communicator did not signal reduce-scatter to dense-MLP producers, so they also all-reduced and the hidden states were reduced twice.
Description 2: Fix describes a reduce-scatter "producer contract" that dense MLP producers must follow but didn't in SUM_LEN mode, double-reducing hidden states.

## X022 (sglang)
Title: [Bug] OpenAI /v1/chat/completions: logprobs.content includes reasoning tokens, mismatches message.content
Description 1: The reasoning parser's split between reasoning and content is not carried to the logprobs builder, which reports logprobs for the whole raw token stream.
Description 2: logprobs.content is assembled from the entire raw output token stream while message.content is the post-reasoning-parser final answer, so the two fields cover different spans of the same generation.

## X023 (vllm)
Title: [Bug]: DeepGemm accuracy auto-disable (_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES) does not apply to the FP8 MoE path
Description 1: The config sets use_deep_gemm=False for the excluded model type, but the FP8 MoE backend selector never receives that flag; the fix propagates it into the selector.
Description 2: quant_config.use_deep_gemm=False is set by config verification but the MoE backend selector never consults it, checking only an env var.

## X024 (vllm)
Title: [Bug]: test_flashinfer_cutlass_mxfp4_fused_moe accuracy mismatch on H20 (sm90) — 89% mismatch vs 20% threshold
Description 1: The fix says the test fed the SM90 mixed-input GEMM weights and scales without the interleaved layout it requires, while the kernel itself is correct.
Description 2: Fix states the test interleaved the scales but never interleaved the packed weights, so the kernel's required storage layout wasn't met.

## X025 (vllm)
Title: [Bug]: [PD + SpecDec] Prefix-cache trimming drops wrong block when P has extra lookahead block
Description 1: Decode's prefix trimming assumes the surplus remote block is a cached prefix at the front, when it is the prefill node's lookahead block at the back, so it drops the wrong block.
Description 2: The trimming code assumes any extra remote blocks are a cached prefix at the front and trims with remote[-num_local:], but P's extra block is a lookahead block at the back, so the trim drops the wrong end.

## X026 (vllm)
Title: [Bug]: Qwen3.5 (Qwen3_5ForConditionalGeneration) FLA linear attention tensor format mismatch causes gibberish output
Description 1: The reporter attributes the gibberish to FLA ops receiving head-first tensors when time-first is expected, based only on a heuristic shape warning, and the linked PR concerns a different model.
Description 2: The FLA kernel warns that tensors are received in head-first [B,H,T,...] layout when [B,T,H,...] was expected, though the only linked PR is an unrelated ColQwen3.5 fix.

## X027 (vllm)
Title: [Bug]: ROCM_ATTN produces incorrect output for LiquidAI LFM2
Description 1: The fix says the ROCm paged-attention kernel baked in the standard contiguous KV layout formula and so misreads models whose KV cache layout differs.
Description 2: The ROCm paged-attention kernel bakes in a standard contiguous KV layout formula that does not match this model's actual KV cache layout.

## X028 (vllm)
Title: [Bug]: Gemma-4 31B with DFlash speculator produces gibberish/repetitive token loop
Description 1: The fix PR says KV cache groups with different block sizes could share the same raw KVCacheTensor and corrupt each other's KV.
Description 2: Fix states KV cache groups with different block sizes could share the same raw KVCacheTensor.

## X029 (sglang)
Title: [Bug] GPT-OSS with DP attention enabled produces garbage output (GSM8K 0/128 vs 124/128)
Description 1: With DP attention, forward_normal all-reduced expert partials that were reduced again later, and the fix removes the duplicate MoE reduction.
Description 2: With MoE all-to-all disabled and DP attention on, forward_normal() still performs a TP all-reduce on expert output that was already summed across ranks, double-counting it.

## X030 (vllm)
Title: [Bug]: FlexAttention paged K/V offsets overflow int32 once `num_gpu_blocks ≥ 2**31 / (block_size · 2 · num_kv_heads · head_size)` — crash *or* silent wrong output
Description 1: The paged K/V element offset is computed in int32 and wraps negative past a block-count bound; the fix rejects caches beyond the int32-addressable size.
Description 2: The paged K/V element offset is computed in int32 and wraps negative (integer overflow) once num_gpu_blocks crosses a documented threshold.

## X031 (transformers)
Title: Accuracy regression in DeepSeek-R1-Distill-Llama-8B after upgrading Transformers from 4.55 to 5.9
Description 1: The output shows undecoded byte-level markers and the fix only updates the tokenizer mapping so these models use TokenizersBackend, indicating the wrong tokenizer class was selected.
Description 2: Fix routes affected models to TokenizersBackend, implying the wrong tokenizer backend/class was previously selected for decoding.

## X032 (sglang)
Title: [Bug] DCP on GLM-5.3-Flash (norope MLA): virtual locs used raw against per-rank KV buffers — index-K OOB crash + state-dependent quality corruption past max_total_num_tokens
Description 1: DCP virtual KV locations are used without translation against per-rank buffers because the norope write kernel and the sparse read paths lack DCP rank localization.
Description 2: The norope KV-write kernel has no DCP_RANK/DCP_WORLD_SIZE translation that the rope variant has, so virtual (global) locations are used raw against per-rank-sized physical buffers.

## X033 (vllm)
Title: [Bug] modelopt NVFP4 MoE: mismatched w1/w3 global scales are detected, warned about, and then used anyway
Description 1: When fusing w13, the loader keeps only the gate's global scale, so the up-projection half is dequantized with a scale the checkpoint did not declare for it.
Description 2: The code detects that the gate and up projections' global scales differ, warns once, and then always dequantizes both halves using only the gate's declared scale.

## X034 (sglang)
Title: [Bug] GLM-5.2-FP8 (DeepSeek-V3.2 / DSA block-fp8) produces wrong output on gfx950 (MI350X/MI355X): aiter gemm_a8w8_blockscale_bpreshuffle is numerically incorrect
Description 1: The aiter CK bpreshuffle GEMM is numerically wrong on gfx950 with ROCm 7.2 (a known hipcc miscompile) and one fix changes its codegen, though a paired SGLang fix also corrects the scale layout.
Description 2: Source comment and fixes explicitly attribute the wrong numerics to hipcc miscompiling the bpreshuffle CK kernel on gfx950.

## X035 (sglang)
Title: [Bug] Silent NaN corruption when serving VLM model without visual encoder weights
Description 1: The trigger is a checkpoint missing its visual-encoder weights, whose uninitialized parameters give NaN that warmup caches (a prefix-slot reuse path then spreads it), and the fix only adds fail-fast validation of unloaded parameters.
Description 2: load_weights() never checks whether every parameter was actually populated from the checkpoint, so an uninitialized visual encoder is silently used and its NaNs get cached.

## X036 (vllm)
Title: [Bug]: NaN vision embeddings with multi-budget encoder CUDA graphs sharing a pool
Description 1: NaNs appear only when nine encoder graph budgets share one pool, and disappear with separate pools or zero-initialized output padding; the report says the mechanism is not established.
Description 2: A regression test shows unwritten output padding in a shared CUDA-graph pool surfaces as NaN when a later, larger-budget capture reads memory a smaller capture never wrote, though the report calls this an open investigation.

## X037 (vllm)
Title: [Bug]: DeepSeek V4/V32 parser can unwrap arguments using the wrong tool's schema when parallel tool calls share identical raw argument text
Description 1: The argument converter identified the tool slot by matching raw argument text instead of receiving the active slot, so parallel calls with identical text used the wrong tool's schema; the fix passes the slot context to the hook.
Description 2: _convert_args looks up which tool a raw argument string belongs to by exact text equality, so two parallel calls with identical raw text resolve to the first matching slot instead of the actual one.

## X038 (transformers)
Title: Mamba2Mixer: use_cache with seq_len > 1 silently produces incorrect results (both CPU and GPU paths)
Description 1: Mamba2Mixer takes the single-step decode path whenever a cached state exists without checking seq_len; the fix adds a separate dispatch path for seq_len > 1.
Description 2: Mamba2Mixer only implements the seq_len==1 decode case; use_cache with seq_len>1 has no correct dispatch path.

## X039 (transformers)
Title: Right-padded prefill produces a corrupted cache in Mamba-family models (zero conv state, decayed SSM state)
Description 1: After right-padded prefill the cache holds the state after the pad tokens (zero conv state, decayed SSM state) instead of at the last valid position, while left padding works.
Description 2: Right-padded prefill leaves pad-token positions to update the recurrent conv/SSM state like real tokens, unlike attention paths that mask padding, so the cache is corrupted only when padding is on the right.

## X040 (vllm)
Title: [Bug]: [Parser] kimi_k2 streaming tool-call args skip schema type coercion applied in non-streaming (silent type mismatch)
Description 1: In the streaming parser engine the arg_converter-is-None branch returns before _fix_arg_types, so kimi_k2 streaming skips the schema type coercion that the non-streaming path applies.
Description 2: _compute_arg_delta/_flush_arg_converter return early and skip _fix_arg_types whenever arg_converter is None, the only branch kimi_k2 uses.

## X041 (sglang)
Title: [Bug] When enabling `--num-shots 20` on Nemotron-3 Ultra evaluation, it will cause accuracy drop.
Description 1: The fix says Mamba2Metadata gated the prefill initial-state restore on the tracking mask, a wrong condition that broke long prefills (the fix text is truncated).
Description 2: Fix ties the bug to a has_initial_states tracking mask that incorrectly gates whether a long prefill restores its cached recurrent state.

## X042 (sglang)
Title: [Bug] GSM8K accuracy regression on DeepSeek-V4-Flash-FP8 (dp8ep8, gfx950) traced to #29275 "Fix gfx95 bpreshuffle FP8 activation scale layout"
Description 1: AITER writes transposed scale storage but returns contiguous [M, G] metadata, which the CK materializer trusted, pairing scales with the wrong token/group cells; the fix restores truthful strides at the producer.
Description 2: AITER writes CK-ready transposed scale storage but returns metadata that looks contiguous, so the materializer trusts the wrong layout and associates scales with the wrong token/group cells.

## X043 (vllm)
Title: [Bug]: Gemma 4 (31B / 26B-A4B) generates infinite repetition loops, especially with structured output (JSON schema)
Description 1: The reporter attributes the loops to a model-level tendency seen on every platform and the linked PRs only add loop detection, so no software defect fitting K1-K6 is identified.
Description 2: Issue states this is model-level behavior seen across multiple platforms, not a vLLM defect, and the fix only adds loop-detection mitigation.

## X044 (vllm)
Title: [Bug]: MiniMax-M3 NVFP4 produces garbage output + CUDA illegal memory access on Hopper (sm90) via Marlin FP4-MoE
Description 1: The Marlin MoE adapter substitutes plain-SiLU defaults instead of forwarding the model's declared swigluoai alpha/beta/limit; the fix plumbs these parameters through.
Description 2: Root cause states the Marlin MoE adapter substitutes plain-SiLU defaults instead of forwarding the model's swigluoai activation parameters.

## X045 (vllm)
Title: [Bug]: speculative decoding under pipeline parallelism produces wrong output with --no-async-scheduling
Description 1: next_decode_eligible_step (the step when a request's sampled token is back from the PP stages) is set only by AsyncScheduler, so under sync scheduling the base scheduler reads its default 0 and reschedules on stale token state; the fix sets it in the base scheduler.
Description 2: next_decode_eligible_step is only updated by AsyncScheduler; the base Scheduler used with --no-async-scheduling never receives updates to it and reads the stale init value of 0, letting it reschedule before the sampled token is relayed.

## X046 (vllm)
Title: [Hybrid SSM] Investigate accuracy divergence between `mamba_chunk_scan` and `selective_state_update` kernels
Description 1: Both explanations are bf16 precision: intermediate bf16 casts in one kernel per the issue, or bf16 lm_head ULP ties per the closed PR, which only stabilizes the test.
Description 2: The linked resolution finds the "regression" is bf16 lm_head ULP-tie argmax flips near a logit near-tie, and fixes the flaky test's threshold rather than either SSM kernel.

## X047 (sglang)
Title: [Bug] Gemma-4-26B-A4B NVFP4 quality drops 
Description 1: The linked fixes restore the FP8 scale hand-off (q/k/v scales) to the TRTLLM MHA kernel, though the follow-up says the root cause was fixed in other, undescribed PRs.
Description 2: Fix states the dynamic q_scale for FP8 Q was never computed or passed into the attention BMM scale, so the kernel used q_scale=1.0.

## X048 (sglang)
Title: [Bug] LoRA adapters with use_rslora=True are served with the wrong scale
Description 1: SGLang ignores use_rslora from the PEFT adapter config and always scales by alpha/r instead of alpha/sqrt(r); the fix reads the flag from the adapter config.
Description 2: LoRAAdapter never read the adapter config's use_rslora flag, always applying alpha/r scaling instead of the declared alpha/sqrt(r).

## X049 (sglang)
Title: Quality regression (LLM-as-judge scores) after upgrading from 0.5.9 to 0.5.10rc0 with Qwen3.5-27B-FP8 on Blackwell SM120
Description 1: The linked fixes explain that Qwen3.5 projection weights are not in the interleaved layout the fused GDN Triton kernel assumed, and re-enable the fusion with stride-aware kernels.
Description 2: Linked fixes describe Qwen3.5's checkpoint using a different, non-interleaved weight layout than the GDN fusion kernel assumed.

## X050 (vllm)
Title: [Bug]: vLLM producing incorrect output for GLM-OCR
Description 1: The mrope Triton kernel hardcoded NeoX-style rotation and ignored the is_neox_style=False that GLM-OCR declares; the fix branches on that flag.
Description 2: The mrope kernel hardcodes NeoX-style rotation regardless of the model's declared is_neox_style=False (GPT-J rotation).

## X051 (vllm)
Title: [Bug] Gemma4ToolParser streams incorrect float values (e.g., 108.2 → 108.02)
Description 1: The streaming tool parser emitted float("108.") as "108.0" before the next digit arrived, and its diff logic then appended "2"; the fix withholds bare values ending in "." in partial mode.
Description 2: The streaming parser emits a partial numeric token ("108.") before its final digit arrives, so the later diff against the already-sent prefix appends the wrong suffix.

## X052 (sglang)
Title: [Bug] Antropic endpoint have wrong checks for the format of prompts - Claude Code not works.
Description 1: The reporter's workaround widens the Anthropic endpoint's accepted message roles and re-parses the request JSON, pointing to request validation and parsing, but it is not a confirmed fix.
Description 2: The reporter's own unofficial workaround adds "system" to the allowed role literal, suggesting a validation model that rejects a legitimate role, but no maintainer fix confirms this.

## X053 (transformers)
Title: [Zamba2] Causality violation in torch_forward: inter-chunk state recurrence reduces over the wrong axis
Description 1: The inter-chunk state recurrence sums over the wrong axis, letting later chunks affect earlier ones; the fix restores the Mamba2 formula.
Description 2: The state-passing recurrence sums over the wrong tensor axis (dim=2 instead of the input-chunk axis) after a permute, letting later chunks leak into earlier ones.

## X054 (sglang)
Title: [Bug] [ROCM] Qwen3.5-397B-A17B-FP8 Generates Garbage Response
Description 1: The ck-tile blockscale GEMM ran on the default stream instead of the caller's current HIP stream, which breaks producer/consumer ordering under overlap scheduling; the fix uses the current stream.
Description 2: The ck-tile blockscale GEMM was launched on the default HIP stream regardless of the caller's actual (non-default, overlap-scheduling) stream, breaking the producer/consumer execution ordering the caller relied on.

## X055 (vllm)
Title: [Bug] Inconsistent parameter names (`thinking` vs `enable_thinking`) between reasoning parsers and chat templates causes content:null
Description 1: The chat template reads enable_thinking while the Kimi K2 reasoning parser reads thinking and defaults it to True, so the user's setting reaches one component but not the other.
Description 2: Chat templates and reasoning parsers read the same thinking-mode toggle under two different key names, thinking vs enable_thinking.

## X056 (vllm)
Title: [ROCm] test_rocm_mxfp4_moe_oracle is stale, and the fused-TRITON case fails accuracy on gfx950
Description 1: The xfail PR attributes the fused TRITON mismatch to the test reference using raw bf16 weights while the kernel receives weights converted to a different layout, though the reporter is unsure whether the test or the kernel is at fault.
Description 2: The fix states reference_moe() compares against raw bf16 weights while the fused kernel receives weights already converted to a different physical layout that the reference does not account for.

## X057 (sglang)
Title: [Bug] DeepSeek-V4 PD HiSparse shows lower accuracy on GPQA and SWE-Bench
Description 1: The fix says the HiSparse transfer path used host KV page indices for the device destination buffers although host and device pages are allocated independently, and now carries separate device indices.
Description 2: The direct-to-host transfer path used host KV page indices for the device destination buffer even though host and device page IDs are allocated independently.

## X058 (vllm)
Title: [Bug]: Triton attention softcap returns NaN for large attention logits
Description 1: The softcap's exp-based tanh overflows to inf/inf = NaN for large logits, and the fix uses a numerically stable form.
Description 2: apply_softcap computes tanh via raw exponentials that overflow to inf for large ratios, so the result becomes inf/inf = NaN.

## X059 (vllm)
Title: [Bug]: FP8 KV cache causes systematic decode/prefill logprob mismatch on Hopper FA3
Description 1: The root cause is inside the FA3 Hopper FP8 kernel, where decode and prefill use different KV tile widths (kBlockN 96 vs 192) and so compute numerically different results for the same token; the fix removes that divergence.
Description 2: Root cause: decode and prefill FlashAttention-3 pick different KV tile widths, giving a different summation and hence different logprobs.

## X060 (vllm)
Title: [Bug]: ROCm MI300X FP8 KV cache MiniMax-M3-MXFP8 accuracy issues
Description 1: The MiniMax-M3 sparse backend hardcodes float8_e4m3fn to reinterpret KV bytes instead of the platform's declared e4m3fnuz, and the fix uses current_platform.fp8_dtype().
Description 2: The sparse-attention backend hardcodes torch.float8_e4m3fn while gfx942's real FP8 KV format is float8_e4m3fnuz, a different byte encoding misinterpreted by the reinterpret-cast.

## X061 (transformers)
Title: DeepSeek-Coder v1 tokenizer produces incorrect output on transformers v5+ (gap in PR #44801's fix)
Description 1: Following the tokenizer_class declared in tokenizer_config.json, transformers loads SentencePiece-based LlamaTokenizer, which cannot apply the ByteLevel-BPE pipeline in tokenizer.json; the fix routes these models to TokenizersBackend.
Description 2: DeepSeek-Coder v1 declares model_type llama and matches the registered class exactly, so the MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS override never fires and it falls through to the wrong (SentencePiece) tokenizer class.

## X062 (vllm)
Title: [Bug]: [Gemma4] gibberish output for long inputs with images on SM90 FlashAttn4
Description 1: The FA4 mm_prefix mask dropped the model's sliding-window constraint (causal OR mm_prefix) and used a relative instead of an absolute q_idx; the fix restores both.
Description 2: The FA4 mask builder implements "causal OR mm_prefix" instead of "(causal AND sliding_window) OR mm_prefix", omitting the sliding-window term that the Triton backend's path effectively enforces.

## X063 (sglang)
Title: [diffusion] MiniMax-H3 with --use-fsdp-inference produces silently corrupted video/audio output
Description 1: The rank-local FSDP fast path bypassed the loader that reorders MiniMax-H3's per-head grouped QKV rows, leaving correctly shaped but wrongly laid-out weights.
Description 2: The rank-local FSDP loading fast path reads shape-compatible safetensors slices directly, bypassing the ordinary loader's required QKV row-reorder transform, so values load with valid shapes but wrong content.

## X064 (vllm)
Title: [Bug]: Llama-4-Scout-FP8 tool calls left in content (not parsed into tool_calls) — pythonic parser vs JSON output mismatch, breaks Claude Code
Description 1: The llama4_pythonic parser does not recognize the JSON tool-call form this checkpoint emits, and the fixes add a JSON fallback to the parser.
Description 2: With the Llama-3.1 JSON chat template this checkpoint emits JSON-formatted tool calls, but the selected llama4_pythonic parser only recognizes the pythonic form, so the two configured components disagree on output format.

## X065 (vllm)
Title: [Bug]: Triton MoE and block-FP8 GEMMs mishandle the K tile — an out-of-bounds weight read, and wrong scales when a tile spans two quantization groups
Description 1: The kernels lack K-tile guards: the last-iteration weight load is unmasked, and nothing clamps BLOCK_SIZE_K <= group_k as the sibling launcher does, so a tile spanning two quantization groups uses one scale.
Description 2: Both Triton GEMMs are missing a masking/clamping case that an equivalent sibling kernel already applies: one leaves the weight load unmasked at the final K tile, the other doesn't clamp BLOCK_SIZE_K to the quantization group size.

## X066 (vllm)
Title: [Bug]: vllm 0.19.0, gemma4, The format of the tool call returned by vllm is incorrect.
Description 1: The fix corrects the Gemma4 tool-call parser (_parse_gemma4_args and _parse_gemma4_array), which leaked the STRING_DELIM token into the JSON when parsing array string values.
Description 2: Fix targets _parse_gemma4_args/_parse_gemma4_array, which read array string values incorrectly and leaked a delimiter into the JSON output.

## X067 (sglang)
Title: [Bug] NemotronH --mamba-scheduler-strategy extra_buffer accuracy drop on AIME26 (Nemotron-3-Super-120B)
Description 1: The tracking code indexes packed intermediate SSM states by per-request chunk counts, while the kernel writes them on one global chunk grid over the batch, so the wrong state is cached; the merged fix corrects the indexing.
Description 2: _init_track_ssm_indices computes offsets assuming a per-request-local contiguous count, but the kernel it reads from indexes states by a global physical chunk grid, so unaligned earlier requests cause a wrong state to be read.

## X068 (vllm)
Title: [Bug] Qwen3.8-Flash-Next (qwen4_exp) + MTP: episodic 0% draft acceptance + repetition collapse in thinking block until max_tokens (SM120, bf16 KV, nightly @73029d4)
Description 1: The fix PRs say the fused PLE short-conv kernels did not honour the stride of the non-contiguous state_indices column slice, so rows after the first read the wrong state.
Description 2: Fix states prefill state indices are sliced into a non-contiguous 1-D tensor and the PLE conv kernels don't honor its actual stride.

## X069 (vllm)
Title: [Bug]: Qwen3.5-27B Disagg accuracy gsm8k collapses with async scheduling when TP==1
Description 1: Under async scheduling the delayed zeroing kernel for newly allocated KV blocks can run after the NIXL RDMA transfer and erase the just-received KV; the fix removes this race.
Description 2: Fix describes a KV-zeroing kernel racing an NIXL RDMA transfer, erasing newly-received KV blocks under async scheduling.

## X070 (vllm)
Title: [Bug]: NVFP4 garbage output under torch.compile when per-rank K is not 32-aligned (TP8 on SM120): padded block-scale column reads stale memory; eager/Marlin/aligned-TP correct
Description 1: The title blames a padded block-scale column that reads stale memory under torch.compile (eager is correct), although the body and the closed fix PR instead blame the SM12x kernels and gate them off.
Description 2: The merged fix stops auto-selecting FlashInferCutlassNvFp4LinearKernel on SM120, where it is documented as broken in the FlashInfer backend itself while correct on other compute capabilities.

## X071 (transformers)
Title: `AutoTokenizer` produces wrong token IDs for all Granite models (silent v4→v5 regression)
Description 1: AutoTokenizer routed Granite to GPT2Tokenizer, which hardcodes a ByteLevel pre-tokenizer in place of the one declared in tokenizer.json; the fix loads tokenizer.json faithfully.
Description 2: AutoTokenizer routes Granite models to GPT2Tokenizer, which hardcodes a GPT-2 pre-tokenizer regex, instead of respecting the checkpoint's own tokenizer.json pre-tokenizer that the BPE was actually trained with.

## X072 (vllm)
Title: [Bug]: Ngram speculative decoding produces corrupted output on hybrid GDN (Qwen3.5) models
Description 1: The fix PRs say num_accepted_tokens was not passed to the SSM metadata builders on non-spec steps, so the next step read stale state from slot 0 instead of the accepted-token offset.
Description 2: Fix states num_accepted_tokens was not passed to SSM metadata builders on non-spec steps, so the wrong state slot was read.

## X073 (sglang)
Title: [Bug] Qwen3 GDN produces wrong results when num_v_heads == num_k_heads (ratio=1)
Description 1: After split and reshape the a/b tensors are non-contiguous but the fused_gdn_gating Triton kernel indexes them as flat contiguous arrays; the fix adds .contiguous().
Description 2: After reshape the a/b tensors are non-contiguous, but the Triton kernel indexes them as flat contiguous arrays.

## X074 (vllm)
Title: [Bug]: MergedColumnParallelLinearWithLoRA fails under TP when fully_sharded_loras=False due to incorrect all_gather in apply
Description 1: apply() always took _mcp_apply, which all-gathers unconditionally even when lora_a is replicated, and the fix gates the all_gather on fully_sharded_loras.
Description 2: _mcp_apply() unconditionally all-gathers the shrink buffer with no branch for fully_sharded_loras=False, where lora_a is already replicated and no gather should happen.

## X075 (sglang)
Title: [Bug] Prefill input logprobs are served from the wrong request when a batch member is retracted or finished
Description 1: The skip branch for retracted or finished requests in the prefill result loop does not advance the logprob cursor, so later requests read shifted slices; the fix advances the offset.
Description 2: The branch that skips finished/retracted requests fails to advance the shared logprob_pt cursor, so later requests in the batch read a shifted slice belonging to a different request.

## X076 (sglang)
Title: Dynamic batching causes severe image quality degradation, line art distorted and subjects missing with same seed & steps
Description 1: The fix names the text encoder's default position_ids of shape [1, seq] used with batched hidden states, which applied RoPE with the wrong token positions for batched prompts, alongside other unspecified accuracy fixes.
Description 2: Fix cites text-encoder position_ids shaped [1, seq] regardless of actual batch size, misapplying RoPE's token layout under batching.

## X077 (vllm)
Title: [Bug]: #47327 dense-MHA split breaks FlashMLA sparse: OOB write in top-k index conversion, corrupted fp8_ds_mla context gather
Description 1: After the dense-MHA split, the FlashMLA sparse backend still passed the full-batch req_id_per_token and gathered context without understanding the fp8_ds_mla layout, i.e. it was not updated for the new contract.
Description 2: After #47327 changed forward_mqa to receive only decode tokens, downstream code still sized/assumed the full batch, causing an out-of-bounds write.

## X078 (vllm)
Title: [Bug]: Step3p5ReasoningParser strips the wrong newline once the discarded prefix is preserved (follow-up to #50918)
Description 1: Step3p5ReasoningParser strips the newline at index 0 assuming the base parser's content starts right after the end token, an assumption that breaks once the base prepends content_before.
Description 2: The strip logic assumes the target newline is always at index 0, an assumption that breaks once the base parser's contract changes to preserve a prefix.

## X079 (transformers)
Title: Beam search cache reorder silently skipped for Mamba, XLNet, RWKV, and Reformer models (wrong generation output)
Description 1: Beam-search cache reordering is triggered only when past_key_values is present, missing models that keep their cache under other names; the fix scans ALL_CACHE_NAMES.
Description 2: Beam-search reordering checks only the past_key_values key, ignoring the other cache-name declarations in ALL_CACHE_NAMES.

## X080 (vllm)
Title: [Bug]: Accuracy drops ~20% when `--enable-prefix-caching` is used together with MTP speculative decoding (Qwen3.6 35B-A3B)
Description 1: The linked fix adds a not-isinstance(spec, MambaSpec) gate so the eagle cache-peek margin in find_longest_cache_hit is not applied to recurrent groups, although that fix is in the Mooncake connector, which the reporter's setup does not show.
Description 2: Fix adds a missing not isinstance(spec, MambaSpec) gate; the cache-hit search previously applied ordinary lookahead logic to recurrent/Mamba state groups too.

## X081 (sglang)
Title: [Bug] Quantized DeepSeek-V4: fused wq_a+wkv path silently drops every layer's packed weights (server runs, output is garbage)
Description 1: The fused wq_a+wkv loader maps only unquantized .weight names, so packed qweight/qzeros/scales tensors are dropped; the fixes classify layers by the packed format actually built.
Description 2: The fused wq_a+wkv weight-loading branch is selected by enumerating literal ".weight" suffixes, so it is structurally unreachable for any packed (GPTQ/AWQ) tensor name and those tensors fall through and get dropped.

## X082 (sglang)
Title: [Bug] Qwen3.6-27B-FP8 (dense): FP8 weight_scale_inv silently dropped → garbage output (gate_gate_up_proj loop bug in qwen3_5.py)
Description 1: The reporter traces the dropped weight_scale_inv to the stacked_params_mapping loop in load_weights rewriting parameter names twice (gate_gate_up_proj), so the scales are never matched.
Description 2: Root cause is a string-substitution loop bug producing the malformed key gate_gate_up_proj, so the scale-parameter lookup fails.

## X083 (vllm)
Title: [Bug][ROCm]: DeepSeek V4 accuracy drops with MRV2 on MI350/MI355 when FULL_DECODE_ONLY graph
Description 1: The accuracy drop occurs only with FULL_DECODE_ONLY graphs on ROCm MI350/MI355, and the linked workaround PR says it is needed only until ROCm 7.14 is released.
Description 2: The issue itself gives no diagnosis, and the merged fix only disables FULL_DECODE_ONLY for this config, calling the accuracy risk "unresolved."

## X084 (vllm)
Title: [Bug]: Mixed prompt embedding masks are omitted from prefix cache keys, causing incorrect outputs
Description 1: The prefix-cache key hashes the normalized token IDs and supplied tensor but omits the prompt_is_token_ids mask that says which embedding each position stands for, so a request reuses KV computed for a different input.
Description 2: The prefix-cache key hash omits the prompt_is_token_ids embedding-selection mask, so a request can incorrectly reuse another request's cached KV.

## X085 (vllm)
Title: [Bug]: Gemma 4 31B FP8_BLOCK checkpoint produces garbage repetitive output — logit saturation at softcap wall due to absorbed activation scales being double-applied
Description 1: The reporter traces the garbage to vLLM re-applying activation quantization to a checkpoint whose activation scales are already absorbed into the weights (double scaling), though no fix landed.
Description 2: The FP8_BLOCK checkpoint already has activation scales absorbed into its weights, but the runtime code has no way to detect this and re-applies dynamic per-token activation quantization on top, double-scaling.

## X086 (vllm)
Title: [Bug]: Dense DP weight transfer selects the wrong IPC payload
Description 1: Reconfiguration resets data_parallel_rank to 0 while payload selection still reads it as the global DP identity; the fix uses data_parallel_index instead.
Description 2: reconfigure_for_independent_dp_rank() resets data_parallel_rank to 0, so payload selection uses the wrong identifier instead of data_parallel_index.

## X087 (vllm)
Title: [Bug]: Gemma4 parser classifies plain output without channel markers differently in streaming and non-streaming mode
Description 1: Unlike the non-streaming path, the streaming parser was pre-initialized to REASONING for any new model turn, and the fix narrows that initial state.
Description 2: The streaming parser pre-initializes its state to REASONING for any prompt ending in a new model turn, a condition broader than the one case (an open reasoning block) it was meant to cover.

## X088 (vllm)
Title: [Bug]: compressed-tensors W4A16 MoE: weight_scale not sharded along K under tensor parallelism, kernel computes wrong group_size
Description 1: The config declares group_size=128, but under TP the scale is not sharded along K, so the kernel derives the group size from its shape (K_per_rank/16); the reporter says the bug is purely in how vLLM passes scales to the kernel.
Description 2: weight_scale is not re-sharded along K when the paired expert weight is sharded under tensor parallelism, so group_size is derived from the original (unsharded) scale shape and comes out wrong.

## X089 (transformers)
Title: Regression in Kimi-K2.5 tokenizer from 5.3.0 to 5.4.0: incorrect codec handling and misleading fix_mistral_regex warning
Description 1: Transformers overrode Kimi-K2.5's hub-declared TikToken tokenizer with a backend that cannot reproduce its non-sequential token IDs, so IDs after 163588 were misassigned; the fix removes the override.
Description 2: The fix removes kimi_k25 from a routing list so it uses its correct TikTokenTokenizer backend instead of being converted through a Tokenizers backend that cannot reproduce its non-sequential added-token IDs.

## X090 (diffusers)
Title: MiniMax-H3: `references` argument ignored and documented multi-GPU example fails with CUDA device mismatch
Description 1: The references input is dropped as unexpected, and the only fix moves position_ids and encoder_hidden_states onto the device the transformer assumed.
Description 2: The transformer forward assumed position_ids/encoder_hidden_states were already on its own device, but the documented multi-GPU setup places the text encoder elsewhere.

## X091 (sglang)
Title: [Bug] GLM-5.2-NVFP4 + flashinfer_trtllm: long-context outputs collapse to '!!!!' (NaN logits) — fp32 correction_bias regression from #29783
Description 1: e_score_correction_bias is created as fp32 and passed unconverted to flashinfer's routing kernel, whose documented contract requires bfloat16, which yields NaN expert weights; the fix restores the bf16 hand-over.
Description 2: e_score_correction_bias is created fp32 and passed unconverted where flashinfer's kernel contract requires bfloat16.

## X092 (vllm)
Title: [Bug]: DeepSeek-V2-Lite with online FP8 on an RTX 4090 (TRITON_MLA): NaN logits on every request once three or more sequences are decoded in one CUDA-graph batch
Description 1: The fix PR says CUDA-graph pad rows get seq_len=0 and Triton MLA decode stage2 always computed acc / e_sum, so those rows became 0/0 = NaN.
Description 2: Fix shows the decode kernel always computes acc/e_sum, and padded empty rows have e_sum=0, producing 0/0 NaN.

## X093 (vllm)
Title: [Bug]: Streaming vs non-streaming content whitespace mismatch at tool-call boundaries (ParserEngine: qwen3/nemotron_v3/seed_oss/glm47_moe/gemma4)
Description 1: The streaming path lacks the content whitespace-strip step that the non-streaming path applies at tool-call boundaries.
Description 2: The streaming path's _events_to_delta has no equivalent whitespace-stripping step that the non-streaming path applies.

## X094 (sglang)
Title: [Bug] GPTQ gptq_gemm: uninitialized output accumulated into by multi-block atomicAdd (data race, silent wrong results)
Description 1: gptq_gemm allocates its output with torch::empty and only the z=0 block zeroes it while other blocks atomicAdd into it, so partial sums land on garbage or are clobbered.
Description 2: gptq_gemm allocates its output with uninitialized torch::empty and multiple blocks race to zero/accumulate it with no synchronization.

## X095 (vllm)
Title: [Bug]: DeepSeek-V4 NIXL failure returns corrupted reasoning with empty content
Description 1: A failed NIXL/HMA receive is reported as completion without the failure reaching the scheduler, so decode treats missing KV as valid and generates corrupted tokens.
Description 2: A failed NIXL/HMA KV receive is reported as a completion rather than a request-level failure reaching the scheduler, so decode resumes on missing KV data.

## X096 (sglang)
Title: [Bug] Kimi-K2.6 (W4A16 INT4, compressed-tensors) produces deterministic single-token repetition loops on greedy decode on Blackwell GB200 (SM 100 / aarch64 / cu130); identical setup works correctly on H100
Description 1: The identical setup is correct on H100, and the reporter locates the fault in the Marlin W4A16 kernel as built for cu130/aarch64/SM100.
Description 2: The same model and sglang version produce clean output on H100 but repetition loops on GB200, with the reporter localizing it to the Marlin W4A16 kernel as compiled for cu130/aarch64/SM100 specifically.

## X097 (vllm)
Title: [Bug]: Gemma4 Unified image requests produce all-NaN logits after BF16-to-FP16 fallback
Description 1: After the BF16-to-FP16 fallback, the vision patch_dense projection overflows to Inf in FP16 and LayerNorm turns it into NaN; the fix evaluates those layers in FP32.
Description 2: Fix states the vision patch_dense projection overflows in FP16 and LayerNorm expands the resulting Inf into all-NaN embeddings.

## X098 (vllm)
Title: [Bug]: User-authored <|image_pad|> in text is misrecognized as image placeholder, causing image binding mismatch
Description 1: Media binding matches placeholder tokens by value rather than provenance, so a user-typed image_pad token takes the image meant for the template slot.
Description 2: Placeholder-to-image binding matches on raw token value, not provenance, so a literal user-text token that happens to share the image placeholder's special-token id can consume the queued image update.

## X099 (vllm)
Title: [Bug][XPU]: AutoRound int4 models silently emit garbage under concurrent requests; ARK WOQ backend is auto-selected with no opt-out
Description 1: The auto-round ark.woqgemm kernel is not recorded under XPU graph capture for m > 1 and replays as zeros while eager is correct, and vLLM only adds an opt-out.
Description 2: The captured ark.woqgemm graph replays all zeros because the kernel's work ran eagerly during capture and was never recorded into the graph.

## X100 (vllm)
Title: [Bug]: Prefill misdispatched into spec-decode FULL cudagraph when prompt length == 1 + num_speculative_tokens → silent GDN state loss, garbage output (hybrid/Qwen3-Next models)
Description 1: _is_uniform_decode is a pure shape check, so a prefill of 1+k tokens is misclassified and dispatched into the spec-decode FULL cudagraph; the fix rejects hybrid prefills from uniform-decode dispatch.
Description 2: Root cause is an explicit "pure shape check" that misclassifies a prefill as uniform-decode and dispatches it into the wrong CUDA graph.

## X101 (sglang)
Title: [Bug] Heterogeneous TP PD disaggregation produces incorrect results when decode TP > num_kv_heads
Description 1: Under heterogeneous P/D TP, integer division maps every decode rank to KV head 0, and the Mamba conv-state slice ignores the [K,K,V] group layout, so decode receives the wrong data.
Description 2: Integer division num_kv_heads // tp_size truncates to 0 when tp_size exceeds num_kv_heads, and the old modulo formula for mapping decode ranks to source KV heads is wrong for this replication case.

## X102 (diffusers)
Title: `LTXEulerAncestralRFScheduler.set_timesteps(sigmas=...)` does not validate monotonicity, causing silent incorrect denoising
Description 1: The corruption comes from a caller-supplied non-monotone sigma schedule that the scheduler computes as written, and the issue was closed as not planned.
Description 2: set_timesteps is missing validation to reject non-monotone sigma schedules before they reach the stepping formula.

## X103 (vllm)
Title: [Bug]: moe_wna16_marlin_gemm applies wrong per-row topk weights (mul_topk_weights=True) at gpt-oss NVFP4 MoE shapes — corrupt output
Description 1: The fix PR says the kernel reinterprets topk_weights as fp32 without checking its dtype, so a wrong-dtype tensor is silently type-punned, consistent with other rows' weights being applied.
Description 2: The kernel takes topk_weights as an untyped pointer reinterpreted as fp32 without validating the tensor's actual dtype, type-punning the data.

## X104 (vllm)
Title: [Bug][DSA] Complete sparse top-k output after DeepSelect detects a NaN
Description 1: On a NaN row DeepSelect writes only slot 0 and returns, but vLLM treats every top-k slot of its reused buffer as a valid position; the proposed fix meets the selector's contract by replacing NaN logits with -inf first.
Description 2: DeepSelect's scratch col_indices_buffer is reused across steps without being cleared, so unwritten top-k slots retain a prior row's stale indices after a NaN.

## X105 (sglang)
Title: [Bug] Intel XPU: TP token-id sync corrupted — xccl all_reduce(MIN/MAX) silently does SUM, breaking grammar/structured decoding.
Description 1: Intel XPU's oneCCL backend silently computes SUM for integer all_reduce MIN/MAX (NCCL is correct), which corrupts the TP token-id sync.
Description 2: The xccl/oneCCL backend on Intel XPU does not implement integer MIN/MAX all_reduce and silently substitutes SUM, while the identical code is correct on NVIDIA/nccl.

## X106 (vllm)
Title: [Bug]: MiniMax-M3 fp8 KV cache on SM80 (A100/A800): coherent with --enforce-eager, garbage under CUDA graphs — root cause + working fix
Description 1: The fp8 path's _insert_kv allocates a fresh tensor and does an advanced-index scatter into the index-K cache, which corrupts that cache under CUDA-graph capture; the fix makes it break to eager during capture.
Description 2: _insert_kv allocates a fresh tensor and scatter-writes it under CUDA graph capture, corrupting the index-K cache on replay.

## X107 (vllm)
Title: [Bug][ROCm][Attention][KV Connector] DeepSeek-V3.2 / GLM-5.x DSA sparse-MLA decode is garbage under CUDA graphs in PD-disaggregation (remote-prefilled KV)
Description 1: The open fix says the ROCm AITER fused_add_rms_norm corrupts CUDA-graph replay for remote DSA decode and switches that path to the native kernel, without explaining the mechanism.
Description 2: Fix states AITER's fused_add_rms_norm "corrupts CUDA-graph replay" for remote DSA decode, addressed by forcing the native kernel instead.

## X108 (vllm)
Title: [Bug]: `prompt_logprobs` silently corrupted for some requests when MTP speculative decoding is enabled (Qwen3.5-family, chunked prefill; two builds, two checkpoints)
Description 1: The fix PR says the padded drafter overwrites the reusable CUDA-graph output buffer before prompt logprobs are computed from the target's hidden states.
Description 2: The padded GPU drafter overwrites the CUDA-graph hidden-states output buffer before prompt logprobs are computed from it.

## X109 (diffusers)
Title: IPNDMScheduler and KDPM2DiscreteScheduler return all-NaN on MPS
Description 1: Both schedulers are correct on CPU and all-NaN only on MPS; the cause is MPS-specific PyTorch behavior (a 0-dim CPU tensor with nonzero storage offset times an MPS tensor, and differing -inf handling in lerp), filed upstream.
Description 2: Both schedulers' NaNs are attributed to CPU-vs-MPS backend arithmetic divergence (one filed as pytorch/pytorch#191929, the other tied to pytorch/pytorch#111374 on -inf handling), i.e. the same code differs only by platform.

## X110 (sglang)
Title: [Bug] RunAI streamer (#17948): corrupted weights, missing quant init, and broken object-storage URIs for multimodal models
Description 1: Of three independent bugs, the silent corruption (all-zero logits) comes from tensor views kept after the RunAI streamer reuses its staging buffers, though a missing quant_config hand-over is also fixed.
Description 2: The most severe symptom, all-zero logits, is traced to stale tensor views over GPU staging buffers reused by the RunAI streamer across batches.
