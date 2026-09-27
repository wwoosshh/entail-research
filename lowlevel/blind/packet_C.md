# Items

Each item: an id, the software project, the issue title, and two short descriptions written earlier by two
independent readers of the issue and its fix. Rate every item. Do not look anything up.

## X005 (sglang)
Title: [Bug] compressed-tensors W4AFP8 MoE: cutlass_w4a8_moe passes a literal chunk_size=128, so a group_size≠128 checkpoint loads clean and serves silently wrong output
Description 1: Scale tensors are sized from the checkpoint's group_size, but cutlass_w4a8_moe passes a literal chunk_size=128 to the kernel, so a group_size 64 checkpoint is dequantized with the wrong stride.
Description 2: cutlass_w4a8_moe always passes a literal chunk_size=128 regardless of the checkpoint's actual group_size, so the kernel never learns the real value.

## X006 (vllm)
Title: [Bug]: DSV4 sparse MLA emits garbage (gsm8k ~0%) with the packed KV layout from #44577
Description 1: The packed KV layout makes each component's per-block stride the whole block span; the cache-store op honors it, but the FlashInfer sparse-MLA decode addresses tokens by flat per-token stride and ignores it.
Description 2: The packed-KV change makes each component's per-block stride the whole packed span, and the cache-store op honors this new stride while the sparse-MLA decode path still assumes the old per-token stride convention.

## X009 (sglang)
Title: [DFlash] Infinite loop when using repetition_penalty — missing token accumulation and scaling penalties
Description 1: The DFlash verify path never accumulated committed tokens or applied repetition penalties, so the parameter was silently ignored; the fixes add the missing penalty handling.
Description 2: BatchedRepetitionPenalizer is never registered in the penalty orchestrator, and DFlash's speculative verification path separately never accumulates committed tokens into penalizer state, so the penalty is never applied.

## X012 (vllm)
Title: DeepSeek-V4-Flash-0731: deterministic wrong token on deep-context exact retrieval at 1-in-4 prompt lengths (reasoning off) — vLLM 0.28.0 + SGLang, reproduced on two hosted providers (DeepInfra, Baidu); one provider (OpenInference) is correct
Description 1: The only (open) fix rounds seq_lens to a multiple of 4 to avoid an alignment bounds issue inside fp8_fp4_paged_mqa_logits, matching the prompt-length = 3 mod 4 trigger.
Description 2: Fix rounds seq_lens to a multiple of 4 to avoid an internal memory-alignment bounds computation in the kernel; the mechanism is stated only briefly.

## X033 (vllm)
Title: [Bug] modelopt NVFP4 MoE: mismatched w1/w3 global scales are detected, warned about, and then used anyway
Description 1: When fusing w13, the loader keeps only the gate's global scale, so the up-projection half is dequantized with a scale the checkpoint did not declare for it.
Description 2: The code detects that the gate and up projections' global scales differ, warns once, and then always dequantizes both halves using only the gate's declared scale.

## X043 (vllm)
Title: [Bug]: Gemma 4 (31B / 26B-A4B) generates infinite repetition loops, especially with structured output (JSON schema)
Description 1: The reporter attributes the loops to a model-level tendency seen on every platform and the linked PRs only add loop detection, so no software defect fitting K1-K6 is identified.
Description 2: Issue states this is model-level behavior seen across multiple platforms, not a vLLM defect, and the fix only adds loop-detection mitigation.

## X047 (sglang)
Title: [Bug] Gemma-4-26B-A4B NVFP4 quality drops 
Description 1: The linked fixes restore the FP8 scale hand-off (q/k/v scales) to the TRTLLM MHA kernel, though the follow-up says the root cause was fixed in other, undescribed PRs.
Description 2: Fix states the dynamic q_scale for FP8 Q was never computed or passed into the attention BMM scale, so the kernel used q_scale=1.0.

## X063 (sglang)
Title: [diffusion] MiniMax-H3 with --use-fsdp-inference produces silently corrupted video/audio output
Description 1: The rank-local FSDP fast path bypassed the loader that reorders MiniMax-H3's per-head grouped QKV rows, leaving correctly shaped but wrongly laid-out weights.
Description 2: The rank-local FSDP loading fast path reads shape-compatible safetensors slices directly, bypassing the ordinary loader's required QKV row-reorder transform, so values load with valid shapes but wrong content.

## X077 (vllm)
Title: [Bug]: #47327 dense-MHA split breaks FlashMLA sparse: OOB write in top-k index conversion, corrupted fp8_ds_mla context gather
Description 1: After the dense-MHA split, the FlashMLA sparse backend still passed the full-batch req_id_per_token and gathered context without understanding the fp8_ds_mla layout, i.e. it was not updated for the new contract.
Description 2: After #47327 changed forward_mqa to receive only decode tokens, downstream code still sized/assumed the full batch, causing an out-of-bounds write.

## X081 (sglang)
Title: [Bug] Quantized DeepSeek-V4: fused wq_a+wkv path silently drops every layer's packed weights (server runs, output is garbage)
Description 1: The fused wq_a+wkv loader maps only unquantized .weight names, so packed qweight/qzeros/scales tensors are dropped; the fixes classify layers by the packed format actually built.
Description 2: The fused wq_a+wkv weight-loading branch is selected by enumerating literal ".weight" suffixes, so it is structurally unreachable for any packed (GPTQ/AWQ) tensor name and those tensors fall through and get dropped.

## X083 (vllm)
Title: [Bug][ROCm]: DeepSeek V4 accuracy drops with MRV2 on MI350/MI355 when FULL_DECODE_ONLY graph
Description 1: The accuracy drop occurs only with FULL_DECODE_ONLY graphs on ROCm MI350/MI355, and the linked workaround PR says it is needed only until ROCm 7.14 is released.
Description 2: The issue itself gives no diagnosis, and the merged fix only disables FULL_DECODE_ONLY for this config, calling the accuracy risk "unresolved."

## X090 (diffusers)
Title: MiniMax-H3: `references` argument ignored and documented multi-GPU example fails with CUDA device mismatch
Description 1: The references input is dropped as unexpected, and the only fix moves position_ids and encoder_hidden_states onto the device the transformer assumed.
Description 2: The transformer forward assumed position_ids/encoder_hidden_states were already on its own device, but the documented multi-GPU setup places the text encoder elsewhere.

## X104 (vllm)
Title: [Bug][DSA] Complete sparse top-k output after DeepSelect detects a NaN
Description 1: On a NaN row DeepSelect writes only slot 0 and returns, but vLLM treats every top-k slot of its reused buffer as a valid position; the proposed fix meets the selector's contract by replacing NaN logits with -inf first.
Description 2: DeepSelect's scratch col_indices_buffer is reused across steps without being cleared, so unwritten top-k slots retain a prior row's stale indices after a NaN.
