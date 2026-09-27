"""M15.7 finding, from the M15.2 review: SGLang 0.5.20's shipped fused-MoE config for
E=512, N=256, dtype=fp8_w8a8, block_shape=[128, 128] on NVIDIA_H100_80GB_HBM3 carries BLOCK_SIZE_K=256 for
M=64, 128, 256, 512. The block-FP8 fused-MoE kernel loads one scale per K tile (offs_ks = k_start // group_k,
fused_moe_triton_kernels.py), so a tile of 256 over a block of 128 multiplies the second half of every tile by the
first half's scale. Reproduced here at the kernel level on this card: the up-projection matmul
(invoke_fused_moe_kernel) with the config taken from the engine's own lookup (try_get_optimal_moe_config, under
override_config, the path entail wraps); the H100 file itself is not needed for the mechanism. A = ones and
per-block weight scales 1 and 3 along K give an exact expectation: 128*1 + 128*3 = 512 per output; a single
scale over the 256-wide tile gives 256. entail off and on from outside.

Run in ~/venvs/sglang: [ENTAIL=load ...] python testbed/m15/moe_tile.py <out.json>
"""
import json
import os
import sys


def main():
    import torch
    import triton.language as tl
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config, try_get_optimal_moe_config
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel

    torch.manual_seed(0)
    dev = "cuda"
    M, K, N, E, top_k, blk = 64, 256, 128, 4, 2, 128           # K = 2 blocks of 128 along the reduced dim
    A = torch.ones(M, K, device=dev).to(torch.float8_e4m3fn)
    A_scale = torch.ones(M, K // blk, device=dev)
    w1 = torch.ones(E, 2 * N, K, device=dev).to(torch.float8_e4m3fn)     # gate | up, all ones
    w2 = torch.ones(E, K, N, device=dev).to(torch.float8_e4m3fn)
    w1_scale = torch.ones(E, 2 * N // blk, K // blk, device=dev)
    w1_scale[:, :, 1] = 3.0                                               # the second K block scaled by 3
    topk_ids = torch.randint(0, E, (M, top_k), device=dev, dtype=torch.int32)
    topk_weights = torch.full((M, top_k), 1.0, device=dev, dtype=torch.float32)
    expected_right, expected_wrong = 128 * 1.0 + 128 * 3.0, 256 * 1.0

    def run(tile_k):
        cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": tile_k, "GROUP_SIZE_M": 1,
               "num_warps": 4, "num_stages": 3}
        with override_config(cfg):
            config = try_get_optimal_moe_config(tuple(w1.shape), tuple(w2.shape), top_k, "fp8_w8a8", M,
                                                block_shape=[blk, blk])
        sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
        C = torch.zeros(M, top_k, 2 * N, device=dev, dtype=torch.bfloat16)
        invoke_fused_moe_kernel(A, w1, None, C, A_scale, w1_scale, None, topk_weights, topk_ids, sorted_ids,
                                expert_ids, n_post, False, top_k, config, tl.bfloat16, True, False, False, False,
                                False, block_shape=[blk, blk])
        torch.cuda.synchronize()
        vals = C.float().unique().tolist()
        return {"tile_k_asked": tile_k, "tile_k_used": config["BLOCK_SIZE_K"], "values": vals[:6],
                "correct": bool(len(vals) == 1 and abs(vals[0] - expected_right) < 1e-3)}

    r128, r256 = run(128), run(256)
    row = {"entail": os.environ.get("ENTAIL", "off"), "shape": f"M={M},K={K},N={N},E={E},top_k={top_k},block={blk}",
           "expected_right": expected_right, "expected_wrong": expected_wrong, "tile_128": r128, "tile_256": r256,
           "reproduced": bool(r128["correct"] and not r256["correct"])}
    os.makedirs(os.path.dirname(sys.argv[1]) or ".", exist_ok=True)
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
