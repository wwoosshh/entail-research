"""M17.6 case, vllm-project/vllm#56578 (testbed/M16_PROTOCOL.md 7): apply_softcap in vLLM's Triton attention helpers
computes x * tanh(S / x) through exp(S/x) and exp(-S/x); both overflow to inf for |S/x| > ~88 and the ratio is NaN,
poisoning the attention row (Gemma-2's softcap 50: scores above ~4400). The report's own Triton probe, unchanged:
six scores through the helper against x * tanh(S / x). Reproduced when the helper returns a non-finite value.
Run in ~/venvs/vllm: python testbed/m17/replay2/cases/vl56578.py <out.json>
"""
import json
import math
import os
import sys


def main():
    import torch
    import vllm
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.ops.triton_attention_helpers import apply_softcap

    @triton.jit
    def _probe(s_ptr, out_ptr, n, cap, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        s = tl.load(s_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, apply_softcap(s, cap), mask=mask)

    cap = 50.0
    s = torch.tensor([1000.0, 3000.0, 5000.0, 10000.0, -10000.0, 4400.0], device="cuda", dtype=torch.float32)
    out = torch.empty_like(s)
    _probe[(1,)](s, out, s.numel(), cap, BLOCK=16)
    torch.cuda.synchronize()
    kernel = out.tolist()
    ref = (cap * torch.tanh(s / cap)).tolist()
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "scores": s.tolist(),
           "kernel": kernel, "tanh": ref,
           "non_finite": [x for x in kernel if not math.isfinite(x)],
           "max_abs_diff_where_finite": max((abs(k - r) for k, r in zip(kernel, ref) if math.isfinite(k)), default=0.0)}
    row["reproduced"] = bool(row["non_finite"])
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
