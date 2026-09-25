"""Detection check for #3: the QKV split declares its output layout; each kernel declares the layouts it accepts."""
import os
import sys

import entail as rc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402

PACKED, STRIDED = rc.Layout("dense"), rc.Layout("strided")


@rc.boundary(name="kernel assuming packed Q rows", q=(PACKED,))
def kernel_packed(*, q, w):
    return case.kernel(q, w, q.shape[1])


@rc.boundary(name="kernel taking the row stride", q=(PACKED, STRIDED))
def kernel_strided(*, q, w):
    return case.kernel(q, w, q.stride(0))


def _q(ctx):  # the fused-QKV split declares what it returns
    return rc.tag(ctx["q"], PACKED if ctx["q"].is_contiguous() else STRIDED)


ARMS = {"W": {"mode": "debug", "checks": "layout tag (packed vs strided) at the kernel boundary",
              "defect": lambda ctx: kernel_packed(q=_q(ctx), w=ctx["w"]),
              "fixed": lambda ctx: kernel_strided(q=_q(ctx), w=ctx["w"])}}
