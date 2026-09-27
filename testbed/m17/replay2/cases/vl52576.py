"""M17.6 case, vllm-project/vllm#52576 (testbed/M16_PROTOCOL.md 7), part 2: w8a8_triton_block_scaled_mm reads one
scale pair per K tile, indexed by (k * BLOCK_SIZE_K) // group_k, which is right only when BLOCK_SIZE_K divides the
quantization group; a tuned config with BLOCK_SIZE_K 256 over a group of 128 (the shipped wna16 configs carry
such values) makes the tile span two groups and the wrong scale is used. The report's construction: the library's
own launcher against a dequantize-and-matmul reference, block_shape [128, 128], the config lookup standing in for
a tuned configs/*.json (patched here to return the requested BLOCK_SIZE_K). Reproduced when the relative error
with BLOCK_SIZE_K 256 is large while 64 and 128 are exact. Part 1 (the out-of-bounds read) needs compute-sanitizer
and is not run.
Run in ~/venvs/vllm: python testbed/m17/replay2/cases/vl52576.py <out.json>
"""
import json
import os
import sys


def main():
    import torch
    import vllm
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    torch.manual_seed(0)
    M, N, K = 64, 256, 1024
    gn, gk = 128, 128
    dev = "cuda"
    A = (torch.randn(M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // gk, device=dev, dtype=torch.float32) * 0.9 + 0.1        # per-token, per-group
    Bs = torch.rand(N // gn, K // gk, device=dev, dtype=torch.float32) * 0.9 + 0.1  # per-block
    A_f = A.float() * As.repeat_interleave(gk, dim=1)
    B_f = B.float() * Bs.repeat_interleave(gn, dim=0).repeat_interleave(gk, dim=1)
    ref = A_f @ B_f.T

    orig = fp8_utils.get_w8a8_block_fp8_configs
    results = {}
    for bk in (64, 128, 256, 512):
        cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8, "num_warps": 4,
               "num_stages": 3}
        fp8_utils.get_w8a8_block_fp8_configs = lambda *a, **k: {M: cfg}
        try:
            out = fp8_utils.w8a8_triton_block_scaled_mm(A, B, As, Bs, [gn, gk], torch.float32)
            torch.cuda.synchronize()
            rel = float((out - ref).abs().max() / ref.abs().max())
            results[str(bk)] = {"max_relative_error": rel, "tile_groups": bk // gk if bk >= gk else 0}
        except Exception as e:  # noqa: BLE001
            results[str(bk)] = {"error": f"{type(e).__name__}: {e}"[:200]}
        finally:
            fp8_utils.get_w8a8_block_fp8_configs = orig
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "shape": [M, N, K],
           "block_shape": [gn, gk], "by_block_size_k": results}
    ok = lambda k: results.get(k, {}).get("max_relative_error")
    row["reproduced"] = bool(ok("128") is not None and ok("128") < 1e-3 and ok("256") is not None and ok("256") > 1e-2)
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
