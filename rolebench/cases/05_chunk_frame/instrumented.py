"""Detection check for #5: position tensors carry their frame; the mask builder compares absolute positions only."""
import os
import sys

import entail as rc
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402

ABS = rc.Positions("absolute")


@rc.boundary(name="mask builder (compares absolute positions)", q_pos=ABS, kv_pos=ABS)
def build_mask(*, q_pos, kv_pos):
    return kv_pos[None, :] <= q_pos[:, None]


def _kv():
    return rc.tag(torch.arange(2 * case.C, device="cuda"), ABS)


def _defect(ctx):  # the chunk scheduler hands over chunk-relative query positions
    q = rc.tag(torch.arange(case.C, device="cuda"), rc.Positions("chunk_relative", offset=ctx["offset"]))
    build_mask(q_pos=q, kv_pos=_kv())


def _fixed(ctx):
    q = rc.tag(torch.arange(case.C, device="cuda") + ctx["offset"], ABS)
    build_mask(q_pos=q, kv_pos=_kv())


ARMS = {"W": {"mode": "debug", "checks": "position-frame tag at the mask builder", "defect": _defect,
              "fixed": _fixed}}
