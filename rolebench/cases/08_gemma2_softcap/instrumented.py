"""Detection check for #8: model properties vs the capabilities of transformers' attention implementations.

Capabilities come from reading the installed code (2026-09-23): integrations/sdpa_attention.py never reads
softcap; eager_attention_forward in modeling_gemma2 and integrations/flex_attention.py apply it.
"""
import entail as rc

CAPS = {"sdpa": rc.KernelCaps(softcap=False, sliding_window=True),
        "eager": rc.KernelCaps(softcap=True, sliding_window=True),
        "flex_attention": rc.KernelCaps(softcap=True, sliding_window=True)}


def _check(ctx, impl):
    cfg = ctx["m32"].config
    props = rc.ModelProps(softcap=cfg.attn_logit_softcapping, sliding_window=cfg.sliding_window)
    rc.check_props(props, CAPS[impl], f"transformers attention implementation '{impl}' (Gemma 2)")


ARMS = {"W": {"mode": "load", "checks": "model property vs attention-implementation capability",
              "defect": lambda ctx: _check(ctx, ctx["default_impl"]), "fixed": lambda ctx: _check(ctx, "eager")}}
