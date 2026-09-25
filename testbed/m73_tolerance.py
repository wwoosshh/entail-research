"""M7.3: what the layer comparison's tolerance should be, from data (entail/diagnose.py TOLERANCE).

diagnose.watch says a layer agrees with its reference when the largest difference, relative to the reference's largest
magnitude, is at most a tolerance per dtype. The first M7.3 run used 3e-2 for bfloat16 (a value set before measuring)
and found, on the first call of each layer of Qwen3-4B: healthy 0.0010 (attention), 0.0042 (MLP), 0.0055 (RMSNorm);
planted 0.040 (softmax scale x1.15) and 0.051 (GELU for SiLU). This measures both sides properly:
  healthy   every call of attention, MLP and RMSNorm during one prompt and 8 decode steps, on Qwen3-4B and
            Llama-3.2-3B (bfloat16, sdpa): the largest healthy difference is what the tolerance must stay above
  planted   Qwen3-4B with the softmax scale off by 1%, 2%, 5%, 10%, 15%, 30%, and the MLP with GELU for SiLU and with
            SiLU of a 5% larger input: the difference on the first call and over all calls, and whether the 32 greedy
            tokens changed - a fault too small to change the output matters less if it goes unseen
Writes testbed/results/m73/tolerance.json. Run in ~/venvs/gpu: python testbed/m73_tolerance.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
sys.path.insert(0, HERE)
os.environ["ENTAIL_LOG_DIR"] = "off"

import torch  # noqa: E402

import m73_locate as m  # noqa: E402
from entail import core, diagnose, load  # noqa: E402

OUT = os.path.join(os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results")), "m73",
                   "tolerance.json")   # M9.1: TESTBED_RESULTS moves the results
LLAMA = os.path.expanduser("~/models/Llama-3.2-3B-Instruct")
EVERY = 10 ** 6


def watch_all(family):
    """attention, MLP and RMSNorm of one model family, every call compared (tol=inf: record, never judge)."""
    import contextlib

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if family == "qwen3":
        from transformers.models.qwen3 import modeling_qwen3 as mod
        mlp, norm = mod.Qwen3MLP, mod.Qwen3RMSNorm
    else:
        from transformers.models.llama import modeling_llama as mod
        mlp, norm = mod.LlamaMLP, mod.LlamaRMSNorm
    stack = contextlib.ExitStack()
    stack.enter_context(diagnose.watch(ALL_ATTENTION_FUNCTIONS, "sdpa", m.manual_attention, label="attention",
                                       calls=EVERY, tol=float("inf")))
    stack.enter_context(diagnose.watch(mlp, "forward", m.mlp_reference, label="mlp", calls=EVERY, tol=float("inf")))
    stack.enter_context(diagnose.watch(norm, "forward", m.rmsnorm_reference, label="rmsnorm", calls=EVERY,
                                       tol=float("inf")))
    return stack


FORWARDS = 8   # generate(max_new_tokens=8): the prompt's forward pass and seven decode steps


def summary(entries):
    """Per layer: every call; the first call; the prompt's forward pass (the first 1/FORWARDS of the calls: every
    layer of the model once); and all calls."""
    out = {}
    for layer in sorted({e["layer"] for e in entries}):
        rel = [e["max_rel"] for e in entries if e["layer"] == layer and e["max_rel"] is not None]
        per_pass = len(rel) // FORWARDS
        out[layer] = {"calls": len(rel), "max": max(rel), "first": rel[0], "median": sorted(rel)[len(rel) // 2],
                      "prompt_pass_calls": per_pass, "prompt_pass_max": max(rel[:per_pass]),
                      "every_call": [round(x, 6) for x in rel]}
    return out


def run(model, ids, family, n_new=8):
    load.LEDGER.layers.clear()
    core.set_mode("debug")
    try:
        with watch_all(family), torch.no_grad():
            model.generate(ids, max_new_tokens=n_new, do_sample=False)
    finally:
        core.set_mode("off")
    return summary(load.LEDGER.layers)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3 import modeling_qwen3 as q3

    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__, "dtype": "bfloat16",
           "metric": "largest |layer - reference| / largest |reference|, per call", "healthy": {}, "planted": {}}
    for family, path in (("llama", LLAMA), ("qwen3", m.QWEN)):
        tok = AutoTokenizer.from_pretrained(path)
        ids = m.prompt_ids(tok, m.QUESTION)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        res["healthy"][family] = run(model, ids, family)
        print(family, {k: {x: v[x] for x in ("calls", "max", "prompt_pass_max", "first")}
                       for k, v in res["healthy"][family].items()}, flush=True)
        if family == "qwen3":
            base = m.generate(model, ids)
            real_sdpa, real_mlp = ALL_ATTENTION_FUNCTIONS["sdpa"], q3.Qwen3MLP.forward
            for err in (1.01, 1.02, 1.05, 1.10, 1.15, 1.30):
                def planted(module, query, key, value, attention_mask, scaling=None, _e=err, **kw):
                    s = (scaling if scaling is not None else query.shape[-1] ** -0.5) * _e
                    return real_sdpa(module, query, key, value, attention_mask, scaling=s, **kw)

                ALL_ATTENTION_FUNCTIONS["sdpa"] = planted
                try:
                    got = run(model, ids, family)
                    changed = m.generate(model, ids) != base
                finally:
                    ALL_ATTENTION_FUNCTIONS["sdpa"] = real_sdpa
                res["planted"][f"softmax scale x{err}"] = {"attention": got["attention"], "output_changed": changed}
                print(err, {x: got["attention"][x] for x in ("max", "prompt_pass_max", "first")}, changed, flush=True)
            for label, act in (("GELU for SiLU", torch.nn.functional.gelu),
                               ("SiLU of a 5% larger input", lambda x: torch.nn.functional.silu(1.05 * x))):
                def planted_mlp(self, x, _a=act):
                    return self.down_proj(_a(self.gate_proj(x)) * self.up_proj(x))

                q3.Qwen3MLP.forward = planted_mlp
                try:
                    got = run(model, ids, family)
                    changed = m.generate(model, ids) != base
                finally:
                    q3.Qwen3MLP.forward = real_mlp
                res["planted"][f"MLP: {label}"] = {"mlp": got["mlp"], "output_changed": changed}
                print(label, {x: got["mlp"][x] for x in ("max", "prompt_pass_max", "first")}, changed, flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
