# 상류 보고 초안 (게시 전, 연구자 허락 필요): SGLang 배포 fused-MoE 설정의 K 타일이 양자화 블록을 넘음

- 상태: **초안, 미게시.** 발견 2026-09-25(M15.2 검토 에이전트의 배포 설정 훑기), 커널 수준 재현 같은 날(`testbed/m15/moe_tile.py`, sm_89).
- 게시 전 확인(2026-09-25, GitHub GET): 같은 이슈 없음(검색 결과 무관한 #128뿐). 설정 파일은 main에 있음(3,255바이트, sha 60ea104e). 현재 튜너 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` 438행은 `block_k % config["BLOCK_SIZE_K"] == 0`인 후보만 남기므로, 이 항목(256 over 128)은 현재 튜너가 만들 수 없는 값이다(옛 튜너나 손 편집).
- 문구 규칙: `PUBLIC_CLAIMS.md` 3절. 엔진을 비난하지 않고, 잰 것만 적고, entail은 한 줄로만 밝힌다.

---

**Title:** [Bug] Shipped fused-MoE tuned config with BLOCK_SIZE_K larger than the fp8 quantization block: E=512,N=256 on H100 gets one scale per 256-wide tile over 128-wide blocks

**Checklist:** searched issues (none found for this config), persists on main (file present), environment and repro below, English.

### Describe the bug

`python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` carries `BLOCK_SIZE_K: 256` for the entries `M=64, 128, 256, 512`, while the quantization block along K is 128.

The block-FP8 fused-MoE Triton kernel loads one scale per K tile: `offs_ks = k_start // group_k` (`sgl-kernel`/`sglang/kernels/ops/moe/fused_moe_triton_kernels.py`, the block-quant branch), then `accumulator += tl.dot(a, b) * a_scale * b_scale`. With `BLOCK_SIZE_K=256` and `group_k=128`, the second 128 columns of every tile are multiplied by the first block's scale. The dense path (`w8a8_block_fp8_matmul_triton`) has a sanitiser that clamps tiles that are too small; `try_get_optimal_moe_config` applies none, so the shipped entry reaches the kernel as is.

### Reproduction (kernel level, any CUDA GPU; the H100 file is not needed for the mechanism)

```python
# testbed/m15/moe_tile.py in github.com/wwoosshh/entail-research (SGLang 0.5.20)
# A = ones (fp8), w1 = ones (fp8), w1_scale[:, :, 0] = 1, w1_scale[:, :, 1] = 3, block_shape=[128, 128], K = 256
# expected per output: 128*1 + 128*3 = 512
config = try_get_optimal_moe_config(w1.shape, w2.shape, top_k, "fp8_w8a8", M, block_shape=[128, 128])
#   under override_config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, ...})
invoke_fused_moe_kernel(A, w1, None, C, A_scale, w1_scale, None, topk_weights, topk_ids, sorted_ids, expert_ids,
                        n_post, False, top_k, config, tl.bfloat16, True, False, False, False, False, block_shape=[128, 128])
```

| BLOCK_SIZE_K | output | 
|---|---|
| 128 (= block) | 512 (correct) |
| 256 (the shipped entry's value) | **256** |

Measured on an RTX 4070 Ti (sm_89), SGLang 0.5.20, Triton 3.8.0. On an H100 the shipped file is picked automatically for those shapes when the Triton fused-MoE path is used; I could not run that end to end (no H100).

### Expected behaviour

The current MoE tuner keeps only candidates with `block_k % config["BLOCK_SIZE_K"] == 0`
(`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, the filter in the search loop), so this entry cannot
have come from it; the other shipped fp8 block files I checked all satisfy that constraint. Either the entry is
regenerated with the current tuner, or `try_get_optimal_moe_config` clamps `BLOCK_SIZE_K` to `block_shape[1]` the way
the dense getter (`get_w8a8_block_fp8_configs`) already clamps tiles that are too small.

### Environment

SGLang 0.5.20, Triton 3.8.0, torch 2.14.0+cu130, CUDA 13.0, RTX 4070 Ti (sm_89), WSL2 Ubuntu 24.04.

(Found while sweeping the shipped configs for a small correctness checker I maintain; the checker clamps the tile at the lookup, but the fix belongs here.)
