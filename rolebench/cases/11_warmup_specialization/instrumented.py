"""Detection check for #11: a compiled artifact records the specialization it was built under; reuse rechecks it.

Defect: one artifact is reused for every call (guards skipped), so the recheck sees a different input.
Fixed: the dispatcher builds one artifact per specialization, so every reuse matches its record.
"""
import os
import sys

import entail as rc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402


def _dispatch(ctx, key_fn):
    assumed = {}

    def call(x, extra, w):
        spec = {"extra_rows_zero": extra.shape[0] == 0}
        key = key_fn(spec)
        if key in assumed:
            rc.require(assumed[key] == spec, "compiled graph reuse",
                       f"artifact built for {assumed[key]} reused for {spec}")
        else:
            assumed[key] = spec
        return case.f(x, extra, w)

    call(ctx["x"], ctx["empty"], ctx["w"])  # warm-up
    return call(ctx["x"], ctx["extra"], ctx["w"])


ARMS = {"W": {"mode": "debug", "checks": "recorded specialization rechecked on reuse",
              "defect": lambda ctx: _dispatch(ctx, lambda spec: "single"),
              "fixed": lambda ctx: _dispatch(ctx, lambda spec: tuple(sorted(spec.items())))}}
