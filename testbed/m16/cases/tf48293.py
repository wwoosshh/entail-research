"""M16 case, huggingface/transformers#48293 (testbed/M16_PROTOCOL.md 5): SwitchTransformersTop1Router returns the
max routing probability where the raw logits belong (the third return value, recorded as router_logits and fed to
the z-loss), and accumulates token_priority over a singleton axis, so expert capacity is never enforced. Loads
google/switch-base-8 through from_pretrained (so the transformers adapter is in the path when entail is on), then
drives its first router with random hidden states at expert_capacity 1 and compares the returns with the
classifier's raw logits and with a per-expert count. In 5.17.0 the returns are (max probability, one-hot expert
mask, "router_logits"). Fixed after 5.17.0 (PR #48421, 2026-09-21), so 5.17.0 carries it.
Run in ~/venvs/gpu: python testbed/m16/cases/tf48293.py <out.json>
"""
import json
import os
import sys

MODEL = "google/switch-base-8"


def main():
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM
    from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router

    torch.manual_seed(0)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    router = next(m for m in model.modules() if isinstance(m, SwitchTransformersTop1Router))
    router.expert_capacity = 1
    B, S, D = 2, 8, model.config.d_model
    hidden = torch.randn(B, S, D)
    with torch.no_grad():
        outs = router(hidden)
        raw_logits = router.classifier(hidden.to(router.dtype)).float()          # (B, S, E)
    probs = torch.softmax(raw_logits, dim=-1)
    E = raw_logits.shape[-1]
    shapes = [list(o.shape) for o in outs]
    # the one-hot expert mask is the return with a trailing expert axis; the "router_logits" is the last return
    mask = next(o for o in outs if o.shape[-1] == E and o.dim() == 4).float().reshape(B, S, E)
    third = outs[2].float()
    third_is_max_prob = bool(third.numel() == B * S and torch.allclose(third.reshape(B, S), probs.max(-1).values, atol=1e-4))
    third_is_raw_logits = bool(third.shape == raw_logits.shape and torch.allclose(third, raw_logits, atol=1e-4))
    per_expert = mask.sum(dim=1)                                                 # (B, E) tokens routed per expert
    over_capacity = int((per_expert > router.expert_capacity).sum())
    dropped = int((mask.sum(-1) == 0).sum())
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__, "returns": shapes,
           "third_return_equals_max_probability": third_is_max_prob, "third_return_equals_raw_logits": third_is_raw_logits,
           "tokens": B * S, "tokens_dropped_for_capacity": dropped, "expert_capacity": router.expert_capacity,
           "expert_slots_over_capacity": over_capacity, "max_tokens_on_one_expert": int(per_expert.max()),
           "reproduced": bool(third_is_max_prob and not third_is_raw_logits and over_capacity > 0 and dropped == 0)}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
