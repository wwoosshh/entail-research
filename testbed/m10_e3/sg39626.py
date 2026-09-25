"""M10 E3, sgl-project/sglang#39626 (testbed/M10_PROTOCOL.md 3.3): the block-FP8 Triton matmul accepts a K tile
(BLOCK_SIZE_K=64) larger than the quantization block (32) from a manually supplied config and returns wrong values
(64 instead of 288) without an error. The report's kernel-level reproduction (only the config lookup is patched),
on SGLang 0.5.20 (the module is looked up where 0.5.20 keeps it), entail off or on from outside.
Run in ~/venvs/sglang: python testbed/m10_e3/sg39626.py <out.json>
"""
import importlib
import json
import os
import sys
from unittest.mock import patch


def main():
    import torch

    fk, where = None, None
    for name in ("sglang.kernels.ops.quantization.fp8_kernel", "sglang.srt.layers.quantization.fp8_kernel"):
        try:
            fk, where = importlib.import_module(name), name
            break
        except ImportError:
            continue
    a = torch.ones((16, 64), device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones((32, 64), device="cuda").to(torch.float8_e4m3fn)
    a_s = torch.tensor([1., 4.], device="cuda").repeat(16, 1)
    b_s = torch.tensor([[1., 2.]], device="cuda")
    expected = torch.full((16, 32), 288., device="cuda", dtype=torch.bfloat16)
    res = {}
    for bk in (32, 64):
        cfg = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, num_warps=4, num_stages=4)
        try:
            with patch.object(fk, "get_w8a8_block_fp8_configs", return_value={16: cfg}):
                out = fk.w8a8_block_fp8_matmul_triton(a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            res[bk] = {"values": out.float().unique().tolist()[:5], "correct": bool(torch.equal(out, expected))}
        except Exception as e:  # noqa: BLE001
            res[bk] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    row = {"entail": os.environ.get("ENTAIL", "off"), "module": where, "bk32": res[32], "bk64": res[64],
           "reproduced": bool(res[32].get("correct") and res[64].get("correct") is False)}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
