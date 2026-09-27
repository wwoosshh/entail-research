"""M16 case, vllm-project/vllm#48895 (testbed/M16_PROTOCOL.md 5): moe_wna16_marlin_gemm with mul_topk_weights=True
multiplies output rows by another row's routing weight (or by zero) at gpt-oss-20b NVFP4 MoE shapes, while the
same call with mul_topk_weights=False plus an external multiply matches the reference. The report's self-contained
script (random NVFP4 weights, E=32, top_k=4, N=K=2880 padded to 2944, group 16), unchanged except that the two
runs are recorded instead of printed and that the two op calls follow 0.30.0's signatures (gptq_marlin_repack lost
its perm argument; moe_wna16_marlin_gemm lost g_idx, perm and is_k_full). Reported on 0.25.1 with the kernel source
identical on main; here vLLM 0.30.0 on sm89 (the report ran on an H100; the Marlin kernel also builds for Ada).
Run in ~/venvs/vllm: python testbed/m16/cases/vl48895.py <out.json>
"""
import json
import os
import sys

E, N_OUT, K_RAW, K_PAD = 32, 2880, 2880, 2944
M, TOPK, BLOCK_M = 32, 4, 16
GROUP = 16


def run():
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        _nvfp4_compute_scale_factor, nvfp4_marlin_process_global_scale, nvfp4_marlin_process_scales)
    from vllm.scalar_type import scalar_types

    E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    codes = torch.randint(0, 256, (E, N_OUT, K_PAD // 2), dtype=torch.uint8, device=dev)
    codes[..., K_RAW // 2:] = 0
    scale_f = torch.rand(E, N_OUT, K_PAD // GROUP, device=dev) * 3 + 0.5
    scale_f[..., K_RAW // GROUP:] = 0
    scales_fp8 = scale_f.to(torch.float8_e4m3fn)
    gscale = torch.rand(E, device=dev) * 0.002 + 1e-4
    lo = E2M1_LUT.to(dev)[(codes & 0xF).long()]
    hi = E2M1_LUT.to(dev)[(codes >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).reshape(E, N_OUT, K_PAD)
    w = w * scales_fp8.float().repeat_interleave(GROUP, dim=2) * gscale.view(E, 1, 1)
    qw, ms = [], []
    csf = _nvfp4_compute_scale_factor(scales_fp8.to(dt), dt)
    for e in range(E):
        q = codes[e].view(torch.int32).T.contiguous()
        qw.append(ops.gptq_marlin_repack(b_q_weight=q, size_k=K_PAD, size_n=N_OUT, num_bits=4, is_a_8bit=False))
        s = marlin_permute_scales(s=scales_fp8[e].to(dt).T, size_k=K_PAD, size_n=N_OUT, group_size=GROUP,
                                  is_a_8bit=False)
        s, _ = nvfp4_marlin_process_scales(s, scale_factor=csf, a_dtype=dt)
        ms.append(s)
    w_marlin, s_marlin = torch.stack(qw), torch.stack(ms)
    g_marlin = nvfp4_marlin_process_global_scale(gscale.float(), dt) / csf
    router = torch.randn(M, E, device=dev)
    topk_w, topk_ids = torch.topk(torch.softmax(router, -1), TOPK, dim=-1)
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(topk_ids.to(torch.int32), BLOCK_M, E)
    act = torch.randn(M * TOPK, K_PAD, device=dev, dtype=dt) * 0.5
    act[:, K_RAW:] = 0
    tw_flat = topk_w.reshape(-1).float()
    row_expert = torch.full((M * TOPK,), -1, dtype=torch.long, device=dev)
    for b in range(len(expert_ids)):
        if expert_ids[b] < 0:
            continue
        blk = sorted_ids[b * BLOCK_M:(b + 1) * BLOCK_M]
        for sid in blk[blk < M * TOPK]:
            row_expert[sid] = expert_ids[b]
    ref = torch.zeros(M * TOPK, N_OUT, device=dev)
    for r in range(M * TOPK):
        e = row_expert[r].item()
        if e >= 0:
            ref[r] = (act[r].float() @ w[e].T) * tw_flat[r]

    def gemm(mul_in_kernel):
        c = torch.zeros(M * TOPK, N_OUT, device=dev, dtype=dt)
        out = ops.moe_wna16_marlin_gemm(
            act, c, w_marlin, None, s_marlin, None, g_marlin, None,
            torch.zeros(1024, dtype=torch.int, device=dev), sorted_ids, expert_ids, num_post_pad,
            topk_w.to(dt), moe_block_size=BLOCK_M, top_k=1, mul_topk_weights=mul_in_kernel,
            b_q_type=scalar_types.float4_e2m1f, size_m=M * TOPK, size_n=N_OUT, size_k=K_PAD,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        if not mul_in_kernel:
            out = out * tw_flat.view(-1, 1).to(out.dtype)
        return out.float()

    res = {}
    for label, mul in (("mul_topk_weights=True", True), ("mul_topk_weights=False+external", False)):
        o = gemm(mul)
        bad = maxrel = 0
        for r in range(M * TOPK):
            if row_expert[r] < 0:
                continue
            rel = ((o[r] - ref[r]).norm() / (ref[r].norm() + 1e-9)).item()
            maxrel = max(maxrel, rel)
            bad += rel > 0.05
        res[label] = {"bad_rows": int(bad), "rows": M * TOPK, "maxrel": maxrel, "out_absmax": float(o.abs().max())}
    return res


def main():
    import torch
    import vllm

    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__,
           "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
    try:
        res = run()
        row["runs"] = res
        row["reproduced"] = bool(res["mul_topk_weights=True"]["bad_rows"] > 0
                                 and res["mul_topk_weights=False+external"]["bad_rows"] == 0)
    except Exception as e:  # noqa: BLE001
        row["error"], row["reproduced"] = f"{type(e).__name__}: {e}"[:500], False
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
