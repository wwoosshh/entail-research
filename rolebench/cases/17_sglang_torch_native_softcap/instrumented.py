"""Detection check for #17: model properties vs SGLang attention backend capabilities.

Capabilities come from reading the installed SGLang 0.5.20 (2026-09-23): torch_native_backend.py never reads
logit_cap; triton_backend.py applies it.
"""
import json
import os

import entail as rc

CAPS = {"torch_native": rc.KernelCaps(softcap=False, sliding_window=True),
        "triton": rc.KernelCaps(softcap=True, sliding_window=True)}


def _check(ctx, backend):
    with open(os.path.join(ctx["dir"], "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    props = rc.ModelProps(softcap=cfg.get("attn_logit_softcapping"), sliding_window=cfg.get("sliding_window"))
    rc.check_props(props, CAPS[backend], f"SGLang attention backend '{backend}' (Gemma 2)")


ARMS = {"W": {"mode": "load", "checks": "model property vs attention-backend capability",
              "defect": lambda ctx: _check(ctx, "torch_native"), "fixed": lambda ctx: _check(ctx, "triton")}}
