"""#15 MAPPING (real engine: transformers as installed): a config key the loader does not know is stored silently.

Mechanism of SGLang #30176 (LongCat-2.0: config keys under other names, feature silently off). Here the installed
transformers is the consumer: a checkpoint config states RoPE linear scaling under an unrecognized key name
("rope_scale", as another tool or a typo might write it). transformers keeps the unknown key as a plain
attribute, raises no warning (checked 2026-09-23), and builds the model with default RoPE.
  defect:    from_pretrained on the checkpoint with the alias key
  fixed:     the alias is mapped to the recognized field before building (or rejected loudly)
  reference: the same weights with rope_scaling = linear, factor 4
"""
import json
import os
import tempfile

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

META = {
    "id": "15", "title": "config key under an unrecognized name silently ignored", "fact": "MAPPING",
    "issue": "https://github.com/sgl-project/sglang/issues/30176 (mechanism); consumer here: transformers",
    "engine": "transformers (installed version)", "kind": "real-engine",
    "boundary": "checkpoint config (key names) -> model builder (recognized fields)",
    "trigger": {"model": "config written with a key name the loader does not recognize"},
    "symptom": "plausible_but_wrong", "expected_detection": "load (unknown or ignored config key is an error)",
    "compare": "fp32",
}
CFG = dict(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, max_position_embeddings=2048)
SCALING = {"rope_type": "linear", "factor": 4.0}


def setup():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**CFG)).eval()
    d = tempfile.mkdtemp(prefix="rolebench15_")
    model.save_pretrained(d)
    cfg = json.load(open(os.path.join(d, "config.json")))
    cfg.pop("rope_parameters", None)
    cfg.pop("rope_scaling", None)
    cfg["rope_scale"] = SCALING["factor"]  # the intended scaling, under a name the loader does not know
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    ids = torch.randint(0, CFG["vocab_size"], (1, 256), generator=torch.Generator().manual_seed(1))
    return {"dir": d, "ids": ids, "state": model.state_dict()}


def _logits(m, ids):
    with torch.no_grad():
        return m(ids).logits.float()


def defect(ctx):
    return _logits(AutoModelForCausalLM.from_pretrained(ctx["dir"]).eval(), ctx["ids"])


def fixed(ctx):
    cfg = LlamaConfig.from_pretrained(ctx["dir"])
    alias = getattr(cfg, "rope_scale", None)
    if alias is not None:  # map the alias to the recognized field
        cfg = LlamaConfig(**{**CFG, "rope_scaling": {"rope_type": "linear", "factor": float(alias)}})
    return _logits(AutoModelForCausalLM.from_pretrained(ctx["dir"], config=cfg).eval(), ctx["ids"])


def reference(ctx):
    m = LlamaForCausalLM(LlamaConfig(**CFG, rope_scaling=SCALING)).eval()
    m.load_state_dict(ctx["state"])
    return _logits(m, ctx["ids"])


def probe(ctx):
    cfg = LlamaConfig.from_pretrained(ctx["dir"])
    return {"unknown_key_kept_as_attribute": getattr(cfg, "rope_scale", None),
            "rope_parameters_used": getattr(cfg, "rope_parameters", None)}
