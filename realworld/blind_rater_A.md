# Blind rater A: root-cause labels for 24 LLM-inference wrong-output issues

Rated 2026-09-22 using read-only `gh api` GET requests. For each issue I read the body, all comments, the full timeline and the title and body of each linked fix PR. I did not read any other rater's notes.

**Category key:**
- A: kernel arithmetic, numerics or a race inside one kernel
- B: a setting or property was ignored or not passed along
- C: hardware, driver, compiler or dependency version
- D: wrong index, offset, length, position, window or cache slot
- E: tokenizer, template, sampling or detokenization
- F: two components used different conventions for the same data
- G: wrong-time read or stale state
- H: not a real bug, user error, or closed with no identified fix
- I: cannot determine

**Column meanings:**
- **fix identified:** "yes" only when a merged PR is named as the fix.
- **silent:** "yes" means the wrong output appeared with no error or warning.
- **source URL:** where the quoted evidence phrase comes from.

| repo | issue # | category | fix identified | silent | one-sentence mechanism | evidence phrase (verbatim) | source URL |
|---|---|---|---|---|---|---|---|
| vllm-project/vllm | 40018 | F | yes (#43781) | yes | The maintainers verified #43781 as the fix: the ROCm sparse-indexer K-cache layout was hard-coded to SHUFFLE whatever the block_size, so cache writers and readers disagreed on the layout (the same PR fixed an indexer RoPE fast path that wrongly assumed in-place mutation); the reporter's skip_kv_gather patch was never merged. | "hard-coded to `SHUFFLE` and can cause erroneous cache reads" | https://github.com/vllm-project/vllm/pull/43781 |
| vllm-project/vllm | 42182 | G | yes (#48481) | yes | Under async scheduling, the zeroing kernel for newly allocated hybrid-model attention blocks could run after the NIXL RDMA transfer had already written those blocks, which erased the received KV. | "the delayed zeroing kernel can erase the received attention KV" | https://github.com/vllm-project/vllm/pull/48481 |
| vllm-project/vllm | 43602 | G | yes (#43617) | yes | During compile warmup the decoder ran with deepstack inputs set to None, so the torch.compile graph specialized to the no-deepstack path; real image requests then reused that graph and lost the visual deepstack additions. | "this can make the decoder graph specialize to the no-deepstack path" | https://github.com/vllm-project/vllm/pull/43617 |
| vllm-project/vllm | 47239 | F | yes (#47381) | unclear | Model Runner V2 sorted the batch by scheduled-token count, which broke the decode-first order that split_decodes_and_prefills assumes, so MTP decodes were handled as prefills (full-KV indexer gathers and an eager fallback); the PR blames the low aa_lcr score on the resulting timeouts and truncation. | "Model Runner V2 sorted requests purely by scheduled-token count" | https://github.com/vllm-project/vllm/pull/47381 |
| vllm-project/vllm | 47300 | D | yes (#47332) | yes | The FA4 mm_prefix mask_mod left out the sliding-window bound and compared the chunk-local q_idx with the absolute kv_idx, so long multimodal prompts got wrong attention masks. | "The mask implemented `causal OR mm_prefix` instead of `(causal AND sliding_window) OR mm_prefix`" | https://github.com/vllm-project/vllm/pull/47332 |
| vllm-project/vllm | 48058 | C | no | yes | The garbage FP8 W8A8 output on the XPU backend went away with version 0.1.11 of the vllm-xpu-kernels dependency; nobody confirmed a vLLM-side PR or mechanism (the contiguity PR #48108 was not merged) before the maintainer closed the issue. | "I verified that vllm-xpu-kernels=0.1.11(with vllm=0.23.1rc1.dev1162+g2bd895762.xpu) can fix this issue." | https://github.com/vllm-project/vllm/issues/48058 |
| vllm-project/vllm | 48611 | F | yes (#48642) | yes | After the dense-MHA split, the context gather read the packed 656-byte fp8_ds_mla cache entries as plain fp8 with one global scale, which silently corrupted the context K/V (the same PR fixed an out-of-bounds write caused by an unsliced req_id_per_token). | "Dense MHA could not gather the packed 656-byte FP8 cache" | https://github.com/vllm-project/vllm/pull/48642 |
| vllm-project/vllm | 48831 | G | yes (#48901) | yes | With torch.compile and chunked prefill, pooling read a reused (aliased) hidden-state buffer before the kernel producing it had finished; the fix added a device synchronization. | "pooled hidden states reading a reused/aliased buffer whose producing kernel has not completed" | https://github.com/vllm-project/vllm/pull/48901 |
| vllm-project/vllm | 49692 | H | no | yes | On a two-image prompt, the EPD output differed from the single-instance output; the test author judged this normal run-to-run deviation, and the issue was closed with no code change. | "it's actually normal to have slight deviation like that" | https://github.com/vllm-project/vllm/issues/49692 |
| vllm-project/vllm | 51063 | B | yes (#51665) | yes | vLLM decided lm_head tying only from the composite config's tie_word_embeddings (default True) and never checked the checkpoint, so it silently discarded a real, separate lm_head.weight. | "decides whether to tie `lm_head` purely from `tie_word_embeddings` and never looks at the checkpoint" | https://github.com/vllm-project/vllm/pull/51665 |
| vllm-project/vllm | 52276 | B | no | no | A failed NIXL/HMA receive was reported as a completed receive with no request-level failure, so kv_load_failure_policy=fail never took effect and decode ran on missing KV; the issue was closed after a config workaround, and fix PR #52232 is still open. | "reported as receive completion without a request-level failure reaching the scheduler" | https://github.com/vllm-project/vllm/issues/52276 |
| vllm-project/vllm | 52644 | C | no | yes | The reporter's workaround PR blames a bug in the CUDA-graph memory pool of ROCm versions below 7.14 that corrupts values during FULL_DECODE_ONLY replay; the issue was closed as no longer seen in recent nightlies, and no fix was named. | "ROCm < 7.14 has a graph-pool bug that silently corrupts values" | https://github.com/vllm-project/vllm/pull/52646 |
| sgl-project/sglang | 25218 | H | no | unclear | From the second turn on, Kimi-K2.6 produced malformed tool-call IDs; there was no diagnosis or fix, and the issue was auto-closed for inactivity. | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/25218 |
| sgl-project/sglang | 27125 | F | yes (#25985; reporter verified main) | yes | The inputs to Wan's causal Conv3d did not match the channels_last_3d memory format of the Conv3d weights, which garbled the VAE-decoded video; the fix landed after v0.5.12, and main resolved the issue. | "when Conv3d weights are stored in `channels_last_3d` memory format" | https://github.com/sgl-project/sglang/pull/25985 |
| sgl-project/sglang | 29748 | A | no | yes | gptq_gemm allocates its output with torch::empty and only z-block 0 zeroes it, so the atomicAdds of the other z-blocks race with that zero-store inside the kernel (lost updates or garbage); fix PR #29749 was not merged, and the issue was auto-closed. | "can `atomicAdd` onto uninitialized memory before block `z==0` zeroes it" | https://github.com/sgl-project/sglang/issues/29748 |
| sgl-project/sglang | 30176 | B | no | no | LongcatFlashConfig did not recognize the checkpoint's oe_* n-gram config keys, so use_ngram_embedding resolved to False and 32 n-gram embedding weights were dropped with only a loader warning; the reporter confirmed that an alias workaround fixes the output. | "Because none of the expected keys are present, `use_ngram_embedding` resolves to `False`" | https://github.com/sgl-project/sglang/issues/30176 |
| sgl-project/sglang | 30233 | B | no | yes | In PD mode, an abort on the prefill side (input longer than max_req_input_len) was not passed on to decode: the request still entered the queue, only 1 token of KV was sent, and decode generated from uninitialized KV (fix PR #30551 was not merged). | "the aborted request still enters the prefill queue, and send_kv_chunk transfers only 1 token" | https://github.com/sgl-project/sglang/issues/30233 |
| sgl-project/sglang | 31482 | D | yes (#31901) | yes | The HiSparse PD direct-to-host transfer used host page indices for the device buffers (C4 indexer and C128), but host and device pages are allocated separately, so KV was written to the wrong device pages. | "used host KV page indices for both host and device destination buffers" | https://github.com/sgl-project/sglang/pull/31901 |
| sgl-project/sglang | 31833 | D | yes (#37836) | yes | _init_track_ssm_indices computed per-request offsets into Mamba2's packed chunk states, which sit on one global chunk grid, so requests that do not start on a chunk boundary cached the SSM state from the wrong token position (the benchmark accuracy-drop claim was retracted, but the outputs did change). | "`_init_track_ssm_indices` indexes the packed per-chunk states `h` as a per-request concatenation of chunk grids" | https://github.com/sgl-project/sglang/pull/37836 |
| sgl-project/sglang | 33107 | H | no | yes | The reporter narrowed the garbage output down to the fusion-off shared-experts path but closed the issue within the hour, with no diagnosis or fix. | "is the only component our instrumentation does not exonerate" | https://github.com/sgl-project/sglang/issues/33107 |
| sgl-project/sglang | 34227 | F | yes (#34294) | yes | The rank-local FSDP loading fast path skipped MiniMax-H3's reorder of per-head grouped QKV rows into concatenated Q/K/V, so it loaded weights with valid shapes but the wrong row order. | "the rank-local FSDP fast path read shape-compatible safetensors slices directly and bypassed that loader" | https://github.com/sgl-project/sglang/pull/34294 |
| sgl-project/sglang | 36371 | F | no | yes | Mooncake PD copied raw Mamba SSM state from a BF16 prefill pool into an FP32 decode pool using the source item length and stride, which corrupted the decode recurrent state; the reporter confirmed this by aligning the dtypes, and closed the issue with no fix. | "uses the source item length for both the copy length and destination slot stride" | https://github.com/sgl-project/sglang/issues/36371 |
| sgl-project/sglang | 37187 | F | yes (#37199) | yes | With DP attention, GPT-OSS all-reduced the MoE expert partial sums and then reduce-scattered those already-reduced copies, which multiplied the MoE output by the TP size. | "Since reduce-scatter also sums its inputs, it combined TP replicated copies of `s`" | https://github.com/sgl-project/sglang/pull/37199 |
| sgl-project/sglang | 38605 | H | no | unclear | The reporter blamed the corrupted MiniMax-H3 video under layerwise offloading on too little host RAM and closed the issue; there was no diagnosis or fix. | "The 64G RAM+3090(24G) seems to RAM is not enough" | https://github.com/sgl-project/sglang/issues/38605 |

## Counts per category

| Category | Count | Issues |
|---|---|---|
| A | 1 | sglang 29748 |
| B | 4 | vllm 51063, vllm 52276, sglang 30176, sglang 30233 |
| C | 2 | vllm 48058, vllm 52644 |
| D | 3 | vllm 47300, sglang 31482, sglang 31833 |
| E | 0 | none |
| F | 7 | vllm 40018, vllm 47239, vllm 48611, sglang 27125, sglang 34227, sglang 36371, sglang 37187 |
| G | 3 | vllm 42182, vllm 43602, vllm 48831 |
| H | 4 | vllm 49692, sglang 25218, sglang 33107, sglang 38605 |
| I | 0 | none |
| **Total** | **24** | |

Totals for the other columns:
- **Fix identified:** yes 13, no 11.
- **Silent:** yes 19, no 2, unclear 3.

## How borderline cases were decided

- **Closed without a merged fix, mechanism still used:** sglang 29748, 30176, 30233 and 36371, and vllm 52276. In each, the reporter's own analysis, workaround or proposed patch shows the mechanism, so I used it.
- **Dependency-version problem (C):** vllm 48058 and 52644. In both, the discussion ties the fix to a newer dependency or runtime (vllm-xpu-kernels 0.1.11, and ROCm below 7.14), and no fix inside vLLM was merged for either.
- **Two defects fixed together:**
  - vllm 40018: labeled from the merged fix the maintainers verified (#43781), not from the reporter's unmerged skip_kv_gather theory.
  - vllm 48611: labeled from the defect the issue calls silently wrong (the fp8_ds_mla gather), not the out-of-bounds write.
- **vllm 47239:** the fix PR says the accuracy loss is indirect (timeouts and truncation from misclassified decodes). I still labeled the ordering-contract defect, but H is a defensible alternative.
- **sglang 31833:** the reporter retracted the accuracy-drop claim, but the discussion confirmed the wrong SSM state, and the merged PR #37836 explicitly closes this issue.
