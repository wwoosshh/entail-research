# llama.cpp output-correctness bugs: pilot root-cause study

- Repository: `ggml-org/llama.cpp` (the engine used by Ollama and LM Studio)
- Date of run: 2026-09-22 (searches ran at about 12:34 UTC)
- Mode: read-only. Only GET calls through `gh api`. Nothing on GitHub was commented on, labeled or changed.
- Question: how often do output-correctness bugs in a real inference engine come from role-class defects (R1-R4), compared with ordinary numeric or platform bugs?

## 1. Candidate set

### Queries (one call each, `per_page=100`)

```
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title wrong created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title incorrect created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title garbage created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title gibberish created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title nonsense created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title accuracy created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title mismatch created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title "different output" created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title corrupted created:2025-01-01..2026-09-22' -f per_page=100
gh api -X GET search/issues -f q='repo:ggml-org/llama.cpp is:issue in:title degraded created:2025-01-01..2026-09-22' -f per_page=100
```

Every `total_count` was 100 or less, so no page 2 was needed. `incomplete_results` was false for every query.

| Keyword | total_count |
|---|---:|
| wrong | 46 |
| incorrect | 55 |
| garbage | 45 |
| gibberish | 54 |
| nonsense | 5 |
| accuracy | 7 |
| mismatch | 19 |
| "different output" | 3 |
| corrupted | 16 |
| degraded | 9 |
| **Sum** | **259** |
| **Union by issue number** | **251** (8 issues matched two keywords) |

### Output-correctness filter

**176 of the 251** issues passed the filter. I judged each one by its title. For about 30 titles that were ambiguous, I also read the first part of the issue body.

- **Included:** reports where the model's generated tokens, logits, embeddings, rerank scores, or vision answers (descriptions, bounding boxes) were garbled, wrong, inconsistent, or degraded, on any backend. I also included:
  - kernel- or op-level wrong-result reports, such as `test-backend-ops` failures
  - NaN outputs
  - tokenizer, chat-template and grammar/sampling issues that change the generated tokens
  - accuracy-drop reports
- **Excluded:**
  - UI, display and web-asset problems
  - wrong API metadata: model name, port, stats, metrics, VRAM or memory reports
  - post-generation parsing of reasoning or tool-call text, for example #23535 (whitespace stripped from tool-call parameters) and #22577
  - performance-only reports
  - load-time shape or tensor-count errors
  - crashes or error messages with no wrong output
  - build, packaging and docs issues
  - feature requests

State of the 176 that passed: 92 closed as `completed`, 50 closed as `not_planned`, 6 closed as `duplicate`, 28 still open.

### Selection

I sorted the 92 `completed` issues by `closed_at`, newest first, and took the first 25. I then read each body. All 25 still passed, so none was replaced. The selected issues closed between 2026-04-06 and 2026-09-10. The next candidate, #19336, closed 2026-03-25.

Two inclusions are borderline, and I kept them:
- #25027 is an op-level mismatch found by reading the code. No model-level symptom was reported.
- #24020 is the reporter's own unmerged port of a new model.

<details><summary>The 176 issue numbers that passed the filter</summary>

29251 29235 29081 28827 28676 28648 28637 28581 28537 28527 28429 28211 28167 28113 27771 27769 27763 27683 27579 27387 27386 27237 27068 27022 27015 26845 26759 26565 26423 26363 26314 26282 26207 26197 26027 25866 25761 25734 25620 25582 25568 25518 25477 25455 25382 25376 25102 25027 24812 24365 24303 24201 24168 24020 23986 23953 23850 23827 23800 23758 23717 23574 23509 23447 23400 23321 23252 23241 23155 23074 23044 23022 22842 22785 22565 22281 22235 22011 21915 21893 21888 21887 21855 21734 21726 21721 21715 21675 21648 21589 21441 21371 21194 20789 20610 20550 20423 20133 20104 20097 20081 20052 20029 19881 19847 19792 19659 19563 19401 19336 19276 19128 19119 19118 19112 19068 19040 18973 18767 18452 18171 17797 17351 17302 17290 17106 17067 17013 16961 16960 16881 16880 16680 16657 16538 16424 16407 16188 15931 15846 15516 15513 15216 15112 15110 14885 14877 14795 14761 14759 14469 14211 14075 13725 13545 13461 13327 13310 13297 13256 13044 12912 12538 12411 12357 12340 12253 12211 12096 12012 11970 11951 11575 11463 11256 11092

</details>

### Evidence rules

For every issue I read the issue body, all of its comments, and the timeline events (cross-references, connected, closed). I also read the fixing PR's title and body, and where needed its discussion. Every evidence phrase in the table was checked by script to be an exact substring of the linked source. The only difference allowed was collapsed whitespace.

## 2. Results

| Issue | Title (shortened) | Closed | Backend (hardware) | Cat. | Silent | Evidence phrase | Source |
|---|---|---|---|---|---|---|---|
| [#28537](https://github.com/ggml-org/llama.cpp/issues/28537) | HIP: sequence joining another's decode batch gets corrupted logits | 2026-09-10 | HIP (Radeon 8060S / gfx1151 APU) | R4 | yes | "The host rewrites prompt tokens in the ROCm_Host-backed inp_tokens tensor" | https://github.com/ggml-org/llama.cpp/pull/25863#issuecomment-5098309288 |
| [#26845](https://github.com/ggml-org/llama.cpp/issues/26845) | SYCL garbage on the second prompt | 2026-09-07 | SYCL (Arc Pro B60 / B70) | R3 | yes | "reordered bytes as if they were still the unreordered layout" | https://github.com/ggml-org/llama.cpp/issues/26845#issuecomment-5365655377 |
| [#28113](https://github.com/ggml-org/llama.cpp/issues/28113) | CUDA/HIP MoE garbage on RDNA3.5 since #27621 | 2026-09-04 | HIP (gfx1151 APU) | N4 | yes | "I guess it is related to my own tunes, which collide" | https://github.com/ggml-org/llama.cpp/issues/28113 |
| [#25734](https://github.com/ggml-org/llama.cpp/issues/25734) | Vulkan wrong matmul on Adreno (subgroup 128): warptile WM > BM | 2026-08-26 | Vulkan (Adreno 650) | N1 | yes | "warptiles currently assume warp sizes <= 64, clamp to work around larger warps" | https://github.com/ggml-org/llama.cpp/pull/27726 |
| [#23321](https://github.com/ggml-org/llama.cpp/issues/23321) | Vulkan `-nkvo` gibberish on Qwen3-Coder-Next / Qwen3.6 | 2026-08-20 | Vulkan + CPU split (AMD gfx90c iGPU) | R4 | yes | "Splits without input were running concurrently with other splits, while potentially reusing memory" | https://github.com/ggml-org/llama.cpp/pull/26040 |
| [#25382](https://github.com/ggml-org/llama.cpp/issues/25382) | DeepSeek-V4 quantized K-cache garbage (attention rotation) | 2026-07-07 | All (graph-level; CPU and 8x V100 CUDA) | R1 | yes | "compressed caches (CSA and HCA) did not respect the hadamard rotation" | https://github.com/ggml-org/llama.cpp/pull/25202 |
| [#25027](https://github.com/ggml-org/llama.cpp/issues/25027) | Vulkan `op_step` vs CPU `ggml_step` mismatch at x==0 | 2026-06-27 | Vulkan | N1 | yes | "step unary op returns a different value at x == 0" | https://github.com/ggml-org/llama.cpp/issues/25027 |
| [#20081](https://github.com/ggml-org/llama.cpp/issues/20081) | mmproj on Vulkan degraded vs CUDA for specific images | 2026-06-19 | Vulkan (Radeon 780M iGPU) | N4 | yes | "Since github actions marked as stale im closing anyways." | https://github.com/ggml-org/llama.cpp/issues/20081 |
| [#23850](https://github.com/ggml-org/llama.cpp/issues/23850) | Vulkan / RX 6900 XT garbage tokens between b9370 and b9389 | 2026-06-07 | Vulkan via MoltenVK (RX 6900 XT, macOS) | N2 | yes | "This is probably another case of subgroup instructions being broken in moltenvk." | https://github.com/ggml-org/llama.cpp/issues/23850 |
| [#24020](https://github.com/ggml-org/llama.cpp/issues/24020) | [WIP] LocateAnything-3B spatial localization failure | 2026-06-05 | CUDA / CPU (GTX 1050 Ti), mtmd | N4 | yes | "I have a working in-progress integration of nvidia/LocateAnything-3B" | https://github.com/ggml-org/llama.cpp/issues/24020 |
| [#23400](https://github.com/ggml-org/llama.cpp/issues/23400) | Last token of saved session replayed at wrong position | 2026-06-02 | Backend-independent (common / llama-completion) | R2 | yes | "effectively replaying the same token in the wrong position" | https://github.com/ggml-org/llama.cpp/pull/23468 |
| [#18452](https://github.com/ggml-org/llama.cpp/issues/18452) | jina-embeddings-v2-base-zh vectors incorrect | 2026-05-31 | CPU (tokenizer; backend-independent) | N3 | yes | "llama.cpp's BPE tokenizer always applies GPT-2 byte encoding" | https://github.com/ggml-org/llama.cpp/issues/18452 |
| [#23574](https://github.com/ggml-org/llama.cpp/issues/23574) | GLM 5.x crash / gibberish with high ubatch and long prompts | 2026-05-28 | CUDA (flash-attention MMA kernel) | N1 | yes | "fix KQ mask offset integer overflow in flash attention MMA kernel" | https://github.com/ggml-org/llama.cpp/pull/23610 |
| [#23717](https://github.com/ggml-org/llama.cpp/issues/23717) | Gibberish when K and V cache use the same quant type (RTX 5060 Ti) | 2026-05-27 | CUDA (Blackwell RTX 5060 Ti) | R4 | yes | "missing a call to `ggml_cuda_pdl_sync`. As a consequence on Blackwell there is a race" | https://github.com/ggml-org/llama.cpp/pull/23690 |
| [#23447](https://github.com/ggml-org/llama.cpp/issues/23447) | Garbage output with Qwen3.5-122B MTP | 2026-05-20 | CUDA + RPC (RTX 3060 + RTX 4050) | N4 | unclear | "this appears to be user error" | https://github.com/ggml-org/llama.cpp/issues/23447 |
| [#21893](https://github.com/ggml-org/llama.cpp/issues/21893) | SYCL B70 nonsense output unless `GGML_SYCL_DISABLE_OPT=1` | 2026-05-04 | SYCL (Arc Pro B70) | N5 | yes | "Your issue could be fixed by https://github.com/ggml-org/llama.cpp/pull/21638." | https://github.com/ggml-org/llama.cpp/issues/21893 |
| [#20097](https://github.com/ggml-org/llama.cpp/issues/20097) | Vulkan gibberish on 3x Intel GPU after b8183 | 2026-04-30 | Vulkan multi-GPU (3x Arc A770) | R4 | yes | "events were set, but the wait command was never submitted to the queue" | https://github.com/ggml-org/llama.cpp/pull/20518 |
| [#21734](https://github.com/ggml-org/llama.cpp/issues/21734) | server-intel gibberish at second turn | 2026-04-20 | SYCL (Arc B580) | R3 | yes | "subsequent prompt processing read them with the standard dequantizer, producing corrupt output" | https://github.com/ggml-org/llama.cpp/pull/21638 |
| [#21715](https://github.com/ggml-org/llama.cpp/issues/21715) | Second interaction hangs / gibberish after SYCL Q8_0 reorder | 2026-04-17 | SYCL (Arc A770 / A380) | R3 | yes | "After the first tg pass reordered the weights, subsequent prompt processing read them" | https://github.com/ggml-org/llama.cpp/pull/21638 |
| [#21887](https://github.com/ggml-org/llama.cpp/issues/21887) | Dual-GPU gibberish on asymmetric PCIe topology | 2026-04-16 | CUDA multi-GPU (2x RTX 4000 SFF Ada) | N2 | yes | "for some motherboards and BIOS settings this seems to cause crashes or corrupted outputs" | https://github.com/ggml-org/llama.cpp/pull/21910 |
| [#21855](https://github.com/ggml-org/llama.cpp/issues/21855) | Nonsense from b8738 with multiple GPUs | 2026-04-16 | CUDA multi-GPU (2x Tesla P40, Windows) | N2 | yes | "I had naively enabled CUDA peer-to-peer access guarded only by `cudaDeviceCanAccessPeer`" | https://github.com/ggml-org/llama.cpp/pull/21910 |
| [#21589](https://github.com/ggml-org/llama.cpp/issues/21589) | SYCL Qwen3.5 garbage on the second prompt | 2026-04-16 | SYCL (Arc B580; also B70) | R3 | yes | "After token generation reorders the weights, prompt processing reads them with the wrong layout." | https://github.com/ggml-org/llama.cpp/issues/21589 |
| [#21648](https://github.com/ggml-org/llama.cpp/issues/21648) | Garbage on multi-GPU without peer memory access | 2026-04-15 | HIP multi-GPU (2x RX 9070 XT, Windows) | N2 | yes | "This is likely a bug in HIP itself." | https://github.com/ggml-org/llama.cpp/issues/21648 |
| [#21726](https://github.com/ggml-org/llama.cpp/issues/21726) | Gemma 4 gibberish after about 230 tokens with `-nkvo` | 2026-04-11 | CUDA (RTX 3080 / 3070 / 3060 Ti, `-nkvo`) | R4 | yes | "the extra srcs ne/nb can also change while keeping the `data` pointer same" | https://github.com/ggml-org/llama.cpp/pull/21736 |
| [#20423](https://github.com/ggml-org/llama.cpp/issues/20423) | SYCL Qwen3.5 gibberish and crashes on Arc A770 | 2026-04-06 | SYCL (Arc A770) | N5 | no | "the fused Gated Delta Net kernel is not implemented for the SYCL backend" | https://github.com/ggml-org/llama.cpp/issues/20423 |

### Per-issue root-cause notes

- **#28537 (R4).** PR #24233 turned on HIP `prop.integrated`, which enabled direct compute on host (`ROCm_Host`) buffers on the APU.
  - A scheduler sanitizer run on the same regression reported a write-after-read race: the host rewrote the input tokens while HIP was still reading them (PR #25863).
  - The issue was closed by revert PR #28604, a stop-gap. The proposed real fix is an input ring buffer, PR #27311, still open.
- **#26845 (R3).** The batch-1 DMMV path rewrites Q2_K weights into a new layout in place. Later multi-token paths (MMVQ, `mul_mat_sycl`) have no reader for that layout and read it as the original one.
  - No PR is linked. The diagnosis comes from a commenter's experiments (for example, `-ub 1` keeps the output clean).
  - The reporter confirmed the problem was gone on a later build. PR #26336 (merged 2026-08-21) added the missing reordered-Q2_K MMVQ reader. The link to this fix is inferred.
- **#28113 (N4).** The reporter could not reproduce on a stock build and blamed their own local tuning.
- **#25734 (N1).** The Vulkan warptile configuration set WM to the subgroup size. At subgroup size 128, WM > BM, so the shader overran shared memory and left some output columns uncomputed. Fixed by clamping in PR #27726.
  - Borderline: this could also be read as a contract mismatch between the host tile configuration and the shader.
- **#23321 (R4).** The ggml-backend split scheduler ran an input-less CPU split while an asynchronous Vulkan split was still pending. The CPU split reused `model.input_embed`'s memory before Vulkan read it.
  - Fixed by PR #26040, which runs splits sequentially.
  - The issue was closed by the stale bot first, then reopened.
- **#25382 (R1).** DeepSeek-V4's compressed CSA/HCA caches ignored the Hadamard rotation that a quantized KV cache turns on.
  - A maintainer said PR #25202 fixes it. A later issue, #26423 (closed not_planned), reports that garbage remained.
- **#25027 (N1).** Vulkan's step op used `>=` instead of `>`. Fixed by PR #25036. This is an op-level report only.
- **#20081 (N4).** The maintainer could not reproduce it. The reporter closed it after the stale label.
- **#23850 (N2).** The FWHT (Hadamard) shader relies on subgroup shuffle, which is broken on MoltenVK with AMD GPUs. PR #23964 disabled `subgroup_shuffle` on MoltenVK AMD.
- **#24020 (N4).** The defect is in the reporter's own unmerged port, not in llama.cpp. For the record, a commenter found three errors in the port, all layout or mapping mistakes:
  - the conv output layout
  - sectioned instead of interleaved 2D RoPE
  - position embeddings sliced instead of interpolated
- **#23400 (R2).** Session save stored n-1 tokens. On restore, token n-1 was replayed at position n, leaving a duplicate KV entry with a different RoPE position. Fixed by PR #23468.
- **#18452 (N3).** Jina's Whitespace pre-tokenizer was not supported. The BPE path byte-encoded Chinese text, which produced the wrong tokens.
  - Fixed by PR #18756.
  - The issue was closed by the stale bot first, then reopened.
  - Borderline R1: an unsupported property was accepted silently.
- **#23574 (N1).** `j_vram*stride_mask` overflowed int32 once the KQ mask exceeded INT32_MAX elements. Fixed by PR #23610. The crash appears only with backend sampling. Without it, the output silently becomes question marks.
- **#23717 (R4).** The FWHT kernel used for the quantized-KV rotation was launched with PDL (programmatic dependent launch) but without `ggml_cuda_pdl_sync`, so on Blackwell it read its input too early. Fixed by PR #23690.
- **#23447 (N4).** The reporter was running a stale RPC-server binary.
- **#21893 (N5).** A maintainer suggested PR #21638. The reporter closed the issue without confirming it. If #21638 was the fix, this would be R3.
- **#20097 (R4).** Two bugs in Vulkan's asynchronous cross-device copies: event waits were never submitted, and event resets raced. Fixed by PR #20518, confirmed by two users.
- **#21734, #21715, #21589 (R3), one root cause.** The SYCL Q8_0 reorder optimization (PR #21527) had no reorder-aware GEMM dequantizer. Prompt processing therefore read weights that token generation had already reordered. Fixed by PR #21638.
- **#21887 and #21855 (N2), one root cause.** PR #19378 turned CUDA P2P on by default, which corrupts output on some motherboard, BIOS or IOMMU setups. PR #21910 made P2P opt-in. The #21887 reporter did not confirm the fix; the #21855 reporter did.
- **#21648 (N2).** Under HIP on Windows, `cudaMemcpyPeerAsync` corrupts data when peer access is unavailable.
  - The reporter attributes this to HIP itself.
  - No llama.cpp change was made. The workaround is `GGML_CUDA_NO_PEER_COPY=ON`.
- **#21726 (R4).** The CUDA-graph reuse check compared data pointers but not the source tensors' `ne`/`nb`. With `-nkvo` the shapes changed behind an unchanged pointer, so a stale graph was replayed. Fixed by PR #21736.
- **#20423 (N5).** Earlier gibberish was fixed by a SYCL op-support PR (#20283). The reporter later closed the remaining instability, which came with HTTP 500s, without comment. PR #20455, which adds a SYCL `GATED_DELTA_NET` op, says it fixes this issue but gives no mechanism.

### Counts per category

| Category | Count | Issues |
|---|---:|---|
| R1 dropped / ignored meaning | 1 | #25382 |
| R2 range / position / offset | 1 | #23400 |
| R3 layout / order / mapping | 4 | #26845, #21734, #21715, #21589 |
| R4 stale or mistimed state | 5 | #28537, #23321, #23717, #20097, #21726 |
| N1 numerical / kernel arithmetic | 3 | #25734, #25027, #23574 |
| N2 platform / toolchain | 4 | #23850, #21887, #21855, #21648 |
| N3 tokenizer / template / sampling | 1 | #18452 |
| N4 not a bug / user error / no fix | 4 | #28113, #20081, #24020, #23447 |
| N5 undetermined | 2 | #21893, #20423 |
| **Total** | **25** | |

**R1-R4 share: 11 of 21 = 52%.**
- The denominator is 25 minus the 4 N4 issues, so it still includes the 2 N5 issues.
- Excluding N5 as well: 11 of 19 = 58%.
- Counting each shared fix once (the three Q8_0-reorder issues share #21638; #21887 and #21855 share #21910): 9 of 18 = 50%, or 9 of 16 = 56% without N5.

**Silent wrong output:**

| Group | Silent = yes |
|---|---|
| All 25 issues | 23 (92%); 1 no (#20423, HTTP 500s), 1 unclear (#23447) |
| 21 non-N4 issues | 20 (95%) |
| 11 R1-R4 issues | 11 (100%) |

**Sensitivity to borderline calls:**
- Toward more R-class, each could be read as R: #25734 (host/shader tile contract), #23574 (offset overflow as R2), #18452 (unsupported tokenizer property as R1), and #21893 (R3 if #21638 was the fix). That gives 15 of 21 = 71%.
- Toward fewer R-class: #23717 and #20097 could be read as synchronization races (N1). That gives 9 of 21 = 43%.

## 3. Patterns observed

- Four of the 11 role-class cases come from one SYCL optimization: in-place weight reordering on Intel Arc. The batch-1 kernel path rewrites a weight's layout, and a different path later reads it as the original layout. Every time, the symptom was "first answer correct, second prompt garbage", and it happened twice, first with Q8_0 in April and then with Q2_K in August.
- The R4 cases cluster where memory is shared asynchronously between host and device or between devices: HIP host buffers on the Strix Halo APU, Vulkan split scheduling and CUDA graphs with `-nkvo`, Vulkan multi-GPU event waits, and a missing PDL sync on Blackwell. Multi-GPU peer-to-peer problems account for 3 of the 4 N2 cases.
- Qwen3.5/3.6 (hybrid Gated DeltaNet) and Gemma 4 are the most frequent model families. The quantized-KV Hadamard-rotation path appears in three issues under three different categories (#25382 R1, #23717 R4, #23850 N2).
- 19 of the 25 reports were on consumer or prosumer hardware, as the reporters described it: Arc A380/A770/B580/Pro B60/Pro B70, GeForce RTX 30/40/50-series and GTX 1050 Ti, Radeon RX and Ryzen APU iGPUs, and Adreno. So were 9 of the 11 R-class cases.

## 4. Limitations

- This is a pilot. The sample is small, and it comes from title-keyword search, so reports with other wording ("broken", "repetition", "NaN") are missing.
- The output-correctness filter was judged mostly from titles.
- Some root causes rest on a maintainer's closing comment rather than a confirmed repro (#21887, #25382), or on a commenter's diagnosis with an inferred fix (#26845).
- Closing as `completed` does not guarantee a fix. All four N4 issues were closed by their reporters without any llama.cpp change.
- Raw JSON (search results, issues, comments, timelines, PRs) was saved to the session scratchpad `...\scratchpad\llamacpp\`. That directory is temporary. Rerunning the queries above with the same date bound reproduces the candidate set, apart from later edits to issues.
