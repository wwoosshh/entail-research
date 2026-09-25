"""Detection check for #2: the UE8M0 kernel requires data and scales in the same scale format."""
import os
import sys

import entail as rc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402

FP32 = rc.Layout("fp8_block", scale_format="fp32")
UE8M0 = rc.Layout("fp8_block", scale_format="ue8m0")


@rc.boundary(name="DeepGEMM UE8M0 kernel", q=(UE8M0,), scale=(UE8M0,))
def gemm(*, q, scale):
    return None


def _defect(ctx):
    q = rc.tag(ctx["q"].clone(), FP32)             # data quantized for the fp32 scales
    s = rc.tag(ctx["s_pow2"].clone(), UE8M0)       # scales converted, data left as is
    gemm(q=q, scale=s)


def _fixed(ctx):
    q2, _ = case.quantize(case.dequant(ctx["q"], ctx["s"]), ctx["s_pow2"])
    gemm(q=rc.tag(q2, UE8M0), scale=rc.tag(ctx["s_pow2"].clone(), UE8M0))


ARMS = {"W": {"mode": "debug", "checks": "scale-format tag on data and scales at the kernel boundary",
              "defect": _defect, "fixed": _fixed}}
