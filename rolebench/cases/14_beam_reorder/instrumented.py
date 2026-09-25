"""Detection check for #14: every cache entry with a beam axis must be reordered at each beam selection.

The contract is recorded per entry as "reordered at step t"; the next step requires all beam-axis entries current.
"""
import os
import sys

import entail as rc
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402

BEAM_AXIS = ("key_cache", "ssm_state")  # declared: both hold one row per beam


def _beam_checked(ctx, reorder_names):
    h0 = torch.zeros(1, case.HID, dtype=torch.float64)
    cache = {"key_cache": h0.repeat(case.BEAMS, 1), "ssm_state": h0.repeat(case.BEAMS, 1)}
    order_step = {n: 0 for n in BEAM_AXIS}
    toks = torch.full((case.BEAMS, 1), ctx["start"])
    scores = torch.tensor([0.0] + [-1e9] * (case.BEAMS - 1), dtype=torch.float64)
    for step in range(1, case.STEPS + 1):
        stale = [n for n in BEAM_AXIS if order_step[n] != step - 1]
        rc.require(not stale, "beam step", f"beam-axis cache entries not reordered: {stale}")
        h, logp = case._step(ctx, cache["ssm_state"], toks[:, -1])
        cache["key_cache"], cache["ssm_state"] = h.clone(), h
        total = (scores[:, None] + logp).reshape(-1)
        best = total.topk(case.BEAMS).indices
        parent, nxt = best // case.V, best % case.V
        for n in reorder_names:
            cache[n] = cache[n][parent]
            order_step[n] = step
        toks = torch.cat([toks[parent], nxt[:, None]], 1)
        scores = total[best]


ARMS = {"W": {"mode": "debug", "checks": "per-beam axis declared on every cache entry",
              "defect": lambda ctx: _beam_checked(ctx, ["key_cache"]),
              "fixed": lambda ctx: _beam_checked(ctx, ["key_cache", "ssm_state"])}}
