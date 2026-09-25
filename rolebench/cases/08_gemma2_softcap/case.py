"""#8 PROPERTY (real engine: transformers as installed): the default SDPA path drops Gemma 2's attention softcap.

Tiny random Gemma 2 built with the installed transformers. The model declares attn_logit_softcapping; the default
attention implementation (sdpa) receives softcap= and never applies it (see issue_track/gemma2_softcap for real
2B/9B checkpoints). q/k weights are enlarged so the raw scores exceed the cap, which makes the loss visible in a
tiny model; in real 2B the scores stay below 50 and the effect is at kernel-noise level (issue_track results).
  defect:    the default attention implementation, whatever transformers picks (recorded in ctx)
  fixed:     attn_implementation="eager" (applies the cap)
  reference: eager path in float64
"""
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

META = {
    "id": "08", "title": "default SDPA path ignores attention logit softcapping", "fact": "PROPERTY",
    "issue": "local finding (no upstream issue found, 2026-09-23); issue_track/gemma2_softcap",
    "engine": "transformers (installed version)", "kind": "real-engine",
    "boundary": "model config (attn_logit_softcapping) -> attention kernel (sdpa)",
    "trigger": {"engine": "transformers", "setting": "default attn_implementation", "model": "Gemma 2"},
    "symptom": "plausible_but_wrong", "expected_detection": "load (kernel capability vs model property)",
    "compare": "fp32",
}
CAP = 5.0


def _build(dtype):
    cfg = Gemma2Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=512,
                       attn_logit_softcapping=CAP, final_logit_softcapping=None, sliding_window=256)
    torch.manual_seed(0)
    m = Gemma2ForCausalLM(cfg)
    with torch.no_grad():  # enlarge raw scores so the cap binds
        for layer in m.model.layers:
            layer.self_attn.q_proj.weight.mul_(20.0)
            layer.self_attn.k_proj.weight.mul_(20.0)
    return m.to("cuda", dtype).eval()


def setup():
    m32 = _build(torch.float32)
    ids = torch.randint(0, 1000, (1, 64), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    return {"m32": m32, "m64": _build(torch.float64), "ids": ids, "default_impl": m32.config._attn_implementation}


def _logits(m, ids, impl):
    m.set_attn_implementation(impl)
    with torch.no_grad():
        return m(ids).logits.double()


def defect(ctx):
    return _logits(ctx["m32"], ctx["ids"], ctx["default_impl"])


def fixed(ctx):
    return _logits(ctx["m32"], ctx["ids"], "eager")


def reference(ctx):
    return _logits(ctx["m64"], ctx["ids"], "eager")


def probe(ctx):
    return {"default_attn_implementation": ctx["default_impl"]}
