"""Detection check for #10, hooked into the installed transformers code path.

Read-time contract: the query offset a mask was built from must still hold when attention reads it.
  - Cache.get_query_offset is wrapped to record (tensor, value) at mask-building time.
  - The registered flex_attention function is wrapped to recheck every recorded tensor before it runs.
The fixed path returns a snapshot (clone), so the recorded tensor cannot change.
"""
import os
import sys

import entail as rc
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402


def _step_checked(ctx, snapshot):
    case._restore(ctx)
    model, cache = ctx["model"], ctx["cache"]
    model.set_attn_implementation("flex_attention")
    cls = type(cache)
    orig_get, orig_flex = cls.get_query_offset, ALL_ATTENTION_FUNCTIONS["flex_attention"]
    recorded = []

    def get(self, layer_idx=0):
        t = orig_get(self, layer_idx)
        if snapshot:
            t = t.clone()
        recorded.append((t, int(t)))
        return t

    def flex(module, *a, **kw):
        for t, v in recorded:
            rc.require(int(t) == v, "flex attention (read-time contract)",
                       f"query offset was {v} when the mask was built but is {int(t)} when attention reads it")
        return orig_flex(module, *a, **kw)

    cls.get_query_offset = get
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex
    try:
        with torch.no_grad():
            model(ctx["tok"], past_key_values=cache, use_cache=True)
    finally:
        cls.get_query_offset = orig_get
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = orig_flex


ARMS = {"W": {"mode": "debug", "checks": "read-time contract on the mask's query offset (real transformers path)",
              "defect": lambda ctx: _step_checked(ctx, False), "fixed": lambda ctx: _step_checked(ctx, True)}}
