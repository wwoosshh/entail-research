"""Detection check for #1: the reorder producer re-tags the buffer; each reader declares the layout it accepts."""
import os
import sys

import entail as rc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402

INTERLEAVED = rc.Layout("q8_0", packing="interleaved")
SPLIT = rc.Layout("q8_0", packing="split")


@rc.boundary(name="second-path dequantize (interleaved reader)", buf=(INTERLEAVED,))
def read_interleaved(*, buf):
    return case.read_interleaved(buf)


@rc.boundary(name="split reader", buf=(SPLIT,))
def read_split(*, buf):
    return case.read_split(buf)


def _reordered(ctx):
    t = {"data": rc.tag(ctx["data"].clone(), INTERLEAVED), "extra": {}}
    case.reorder_in_place(t)
    rc.tag(t["data"], SPLIT)  # the producer declares what it now holds
    return t


def _defect(ctx):
    read_interleaved(buf=_reordered(ctx)["data"])


def _fixed(ctx):
    t = _reordered(ctx)
    (read_split if t["extra"].get("reordered") else read_interleaved)(buf=t["data"])


ARMS = {"W": {"mode": "debug", "checks": "layout tag at the reader boundary", "defect": _defect, "fixed": _fixed}}
