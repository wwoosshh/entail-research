"""Does transformers' default SDPA path silently drop Gemma 2's attention logit softcapping?

Tiny random Gemma2 (no download). Same weights, same input; compare attention implementations:
  eager  applies softcap inside the attention scores
  sdpa   default implementation; torch SDPA has no softcap argument
With a strong cap (attn_logit_softcapping=1.0) the two differ if sdpa ignores the cap. Control: cap disabled.
Also records the default implementation chosen by from_config and any warnings emitted.
"""
import json
import logging
import os
import warnings

import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))


class Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage()[:200])


def run(cap):
    cfg = Gemma2Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=512,
                       attn_logit_softcapping=cap, final_logit_softcapping=None, sliding_window=256)
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(cfg).to("cuda", torch.float32).eval()
    default_impl = model.config._attn_implementation
    with torch.no_grad():  # random init gives tiny attention scores; enlarge them so a cap of 1.0 actually binds
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.mul_(20.0)
            layer.self_attn.k_proj.weight.mul_(20.0)
    ids = torch.randint(0, 1000, (1, 64), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    out = {}
    for impl in ("eager", "sdpa"):
        model.set_attn_implementation(impl)
        with torch.no_grad():
            out[impl] = model(ids).logits.float()
    # how large are the raw attention scores? (layer 0, before any cap)
    with torch.no_grad():
        emb = model.model.embed_tokens(ids) * (cfg.hidden_size ** 0.5)
        h = model.model.layers[0].input_layernorm(emb)
        q = model.model.layers[0].self_attn.q_proj(h).view(1, 64, 4, 64).transpose(1, 2)
        k = model.model.layers[0].self_attn.k_proj(h).view(1, 64, 2, 64).transpose(1, 2).repeat_interleave(2, 1)
        raw = (q @ k.transpose(-1, -2)) * cfg.query_pre_attn_scalar ** -0.5
    d = (out["eager"] - out["sdpa"]).abs()
    return {"cap": cap, "default_attn_implementation": default_impl,
            "raw_score_abs_p50": raw.abs().median().item(), "raw_score_abs_max": raw.abs().max().item(),
            "max_abs_eager_vs_sdpa": d.max().item(), "mean_abs_eager_vs_sdpa": d.mean().item(),
            "top1_agree": (out["eager"].argmax(-1) == out["sdpa"].argmax(-1)).float().mean().item()}


def main():
    col = Collect()
    logging.getLogger("transformers").addHandler(col)
    res = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for cap in (None, 1.0):
            r = run(cap)
            res.append(r)
            print(json.dumps(r), flush=True)
    res.append({"python_warnings": [str(x.message)[:200] for x in w], "transformers_log_warnings": col.msgs})
    print("warnings:", res[-1], flush=True)
    with open(os.path.join(HERE, "check_gemma2_softcap.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
