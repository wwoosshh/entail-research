"""Detection check for #9: a captured graph records the valid length it assumed; replay rechecks it."""
import os
import sys

import entail as rc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402


class CheckedRunner(case.GraphRunner):
    def __init__(self, key_fn):
        super().__init__(key_fn)
        self.assumed = {}

    def __call__(self, q, k, v, lv):
        key = self.key_fn(q, k, v, lv)
        if key in self.assumed:
            rc.require(self.assumed[key] == lv, "CUDA graph replay",
                       f"graph captured for valid length {self.assumed[key]} replayed with valid length {lv}")
        else:
            self.assumed[key] = lv
        return super().__call__(q, k, v, lv)


def _run(ctx, key_fn):
    r = CheckedRunner(key_fn)
    r(ctx["q"], ctx["k"], ctx["v"], case.L0)
    return r(ctx["q"], ctx["k"], ctx["v"], case.L1)


ARMS = {"W": {"mode": "debug", "checks": "recorded capture assumption rechecked at replay",
              "defect": lambda ctx: _run(ctx, lambda q, k, v, lv: (q.data_ptr(), k.data_ptr(), v.data_ptr())),
              "fixed": lambda ctx: _run(ctx, lambda q, k, v, lv: (q.data_ptr(), k.data_ptr(), v.data_ptr(), lv))}}
