"""Week 2, experiment A: isolate the attention-path cost seen in the compiled StaticCache decode.

One decode-step attention call, Qwen3-4B shapes (32 q heads, 8 kv heads, head dim 128), bf16.
Variants (per call, one layer):
  flash_gqa_nomask_full      SDPA flash, enable_gqa, no mask -> attends to ALL Lmax slots (speed reference; numerically wrong)
  flash_repeat_nomask_full   materialize repeat_kv (x4) each call + flash, no mask (speed reference)
  efficient_mask_repeat      current compiled path: bool mask + materialized repeat_kv + mem-efficient backend
  efficient_mask_gqa         bool mask + enable_gqa under mem-efficient backend (may be unsupported)
  cudnn_mask_repeat          bool mask + repeat_kv under cuDNN backend (may be unsupported)
  flash_mask_repeat          flash with a mask (expected unsupported)
  flash_sliced_gqa           slice cache to the (uniform) valid length, flash + enable_gqa -> dynamic shape approach
  triton_valid_gqa           this repo's Triton kernel: static shapes, per-row valid length, GQA by index
  repeat_kv_only             cost of the repeat_kv materialization alone
Correctness is checked against an fp32 reference over the valid prefix where the variant is meant to be exact.
"""
import json
import os
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import time_fn  # noqa: E402
from triton_decode_attn import decode_attention, reference  # noqa: E402

HQ, HKV, D = 32, 8, 128
GROUP = HQ // HKV


def make(B, Lmax):
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, HQ, 1, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, HKV, Lmax, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, HKV, Lmax, D, device="cuda", dtype=torch.bfloat16, generator=g)
    # stale slots beyond the valid length hold garbage (as in a real static cache after longer prompts)
    valid = torch.tensor([Lmax - 64 - 8 * b for b in range(B)], device="cuda", dtype=torch.int32)
    pos = torch.arange(Lmax, device="cuda")
    mask = (pos[None, None, None, :] < valid[:, None, None, None])  # [B,1,1,Lmax] bool, True = attend
    return q, k, v, valid, mask


def repeat_kv(x):
    B, H, L, Dd = x.shape
    return x[:, :, None].expand(B, H, GROUP, L, Dd).reshape(B, H * GROUP, L, Dd)


def variants(q, k, v, valid, mask):
    uniform_valid = int(valid.min())
    return {
        "flash_gqa_nomask_full": (SDPBackend.FLASH_ATTENTION,
                                  lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True), False),
        "flash_repeat_nomask_full": (SDPBackend.FLASH_ATTENTION,
                                     lambda: F.scaled_dot_product_attention(q, repeat_kv(k), repeat_kv(v)), False),
        "efficient_mask_repeat": (SDPBackend.EFFICIENT_ATTENTION,
                                  lambda: F.scaled_dot_product_attention(q, repeat_kv(k), repeat_kv(v), attn_mask=mask), True),
        "efficient_mask_gqa": (SDPBackend.EFFICIENT_ATTENTION,
                               lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True), True),
        "cudnn_mask_repeat": (SDPBackend.CUDNN_ATTENTION,
                              lambda: F.scaled_dot_product_attention(q, repeat_kv(k), repeat_kv(v), attn_mask=mask), True),
        "flash_mask_repeat": (SDPBackend.FLASH_ATTENTION,
                              lambda: F.scaled_dot_product_attention(q, repeat_kv(k), repeat_kv(v), attn_mask=mask), True),
        "math_mask_gqa": (SDPBackend.MATH,
                          lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True), True),
        "flash_sliced_gqa": (SDPBackend.FLASH_ATTENTION,
                             lambda: F.scaled_dot_product_attention(q, k[:, :, :uniform_valid], v[:, :, :uniform_valid], enable_gqa=True), "uniform"),
        "triton_valid_gqa": (None, lambda: decode_attention(q, k, v, valid), True),
        "repeat_kv_only": (None, lambda: (repeat_kv(k), repeat_kv(v)), False),
    }


def main():
    results = []
    for B in (1, 4, 8):
        for Lmax in (640, 2048, 8192):
            if B * Lmax > 8 * 2048 and Lmax == 8192 and B > 1:
                continue  # keep memory modest: B8 x 8192 repeat_kv variants would allocate >1 GB per call
            q, k, v, valid, mask = make(B, Lmax)
            ref = reference(q, k, v, valid).float()
            kv_bytes = 2 * B * HKV * int(valid.float().mean()) * D * 2
            floor_us = kv_bytes / 504e9 * 1e6
            print(f"\n=== B={B} Lmax={Lmax} valid={valid.tolist()} | KV read floor {floor_us:.1f} us ===", flush=True)
            for name, (backend, fn, check) in variants(q, k, v, valid, mask).items():
                row = {"B": B, "Lmax": Lmax, "variant": name}
                try:
                    ctx = sdpa_kernel(backend) if backend is not None else torch.no_grad()
                    with torch.no_grad(), ctx:
                        out = fn()
                        med, mn, mx = time_fn(fn, warmup=5, iters=50)
                    row.update({"ms": med, "min_ms": mn})
                    if check is True:
                        row["max_abs_err"] = (out.float() - ref).abs().max().item()
                    elif check == "uniform":
                        vu = torch.full_like(valid, int(valid.min()))
                        row["max_abs_err"] = (out.float() - reference(q, k, v, vu).float()).abs().max().item()
                    print(f"  {name:28s} {med*1e3:9.1f} us   err={row.get('max_abs_err', float('nan')):.4f}", flush=True)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                    print(f"  {name:28s} ERROR {row['error']}", flush=True)
                results.append(row)
            del q, k, v, mask
            torch.cuda.empty_cache()
    out_path = os.path.join(HERE, "results_attn_path.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print("\nsaved", out_path)


if __name__ == "__main__":
    main()
