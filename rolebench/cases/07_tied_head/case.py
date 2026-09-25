"""#7 PROPERTY: the config says the output head is tied to the embeddings, but the checkpoint carries its own head.

Mechanism of vLLM #51063 (Mistral3 VLM: top-level config tie=true, real lm_head dropped; fixed by #51665, which
re-ties only if the tensors are equal). Tiny random Llama; the checkpoint holds distinct embed and lm_head tensors.
  defect:    loader trusts the declared tie: lm_head := embed, the checkpoint's lm_head is dropped silently
  fixed:     loader compares the two tensors and keeps the checkpoint's head when they differ
  reference: the model exactly as saved
  probe:     what transformers' own from_pretrained does with the same files (recorded, not judged)
"""
import json
import os
import tempfile

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

META = {
    "id": "07", "title": "declared tied head vs separate head in the checkpoint", "fact": "PROPERTY",
    "issue": "https://github.com/vllm-project/vllm/issues/51063", "engine": "loader logic (mechanism); transformers probe",
    "kind": "mechanism", "boundary": "checkpoint config (tie flag) -> weight loader",
    "trigger": {"model": "checkpoint whose config declares tie but stores a separate head"},
    "symptom": "plausible_but_wrong", "expected_detection": "load (declared tie vs actual tensors)", "compare": "fp32",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CFG = dict(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, max_position_embeddings=256)


def setup():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**CFG, tie_word_embeddings=False)).to(DEVICE).eval()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ids = torch.randint(0, CFG["vocab_size"], (1, 32), device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(1))
    return {"model": model, "state": state, "ids": ids}


def _load(ctx, trust_declared_tie):
    m = LlamaForCausalLM(LlamaConfig(**CFG, tie_word_embeddings=True)).to(DEVICE).eval()
    sd = dict(ctx["state"])
    head = sd.pop("lm_head.weight")
    m.load_state_dict(sd, strict=False)
    if not trust_declared_tie and not torch.equal(head, sd["model.embed_tokens.weight"]):
        m.lm_head.weight = torch.nn.Parameter(head.clone())  # untie: keep the checkpoint's own head
    return m


def _logits(m, ids):
    with torch.no_grad():
        return m(ids).logits.float()


def defect(ctx):
    return _logits(_load(ctx, True), ctx["ids"])


def fixed(ctx):
    return _logits(_load(ctx, False), ctx["ids"])


def reference(ctx):
    return _logits(ctx["model"], ctx["ids"])


def probe(ctx):
    """Save both tensors plus a config declaring the tie, then load with transformers."""
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as d:
        LlamaConfig(**CFG, tie_word_embeddings=True).save_pretrained(d)
        save_file({k: v.contiguous().cpu() for k, v in ctx["state"].items()}, os.path.join(d, "model.safetensors"))
        m = AutoModelForCausalLM.from_pretrained(d).to(DEVICE).eval()
        out = _logits(m, ctx["ids"])
        ref = reference(ctx)
        return {"max_abs_vs_reference": float((out - ref).abs().max()),
                "equals_reference": bool(torch.allclose(out, ref, atol=1e-5)),
                "lm_head_shares_embedding_storage": m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr(),
                "config_tie_after_load": bool(getattr(m.config, "tie_word_embeddings", None))}
