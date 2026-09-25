"""M8.2: Qwen3-4B's decode step written with entail's front end, against transformers' own step.

The week-4 setting (phase0/week4/e4_fact_ablation.py): Qwen3-4B, int4 weight-only (torchao Int4WeightOnlyConfig,
group 128, tile_packed_to_4d), a static cache, batch 8, a prompt of 512 random tokens, one decode step at position
512. transformers fills the cache with the prompt; then one step is taken by transformers (sdpa, its default) and,
from the same cache contents, by the front-end program (entail/frontend/qwen3.py) with each attention lowering.
The front end is also asked to take the model with a wrong declaration - dense weights where int4 is stored, a
RoPE scaling no lowering computes - to see the load contract and the trace refuse before anything runs.

Writes testbed/results/m82/qwen3.json. Run in ~/venvs/gpu: python testbed/m82_qwen3.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))

import torch  # noqa: E402
import transformers  # noqa: E402

from entail.core import RoleError  # noqa: E402
from entail.facts import Rotary  # noqa: E402
from entail.frontend import qwen3  # noqa: E402

transformers.utils.logging.disable_progress_bar()
MODEL = os.path.expanduser("~/models/Qwen3-4B")
OUT = os.path.join(HERE, "results", "m82", "qwen3.json")
B, L = 8, 512
MAX_LEN = L + 128


def main():
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from transformers import AutoModelForCausalLM, StaticCache

    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__,
           "transformers": transformers.__version__, "batch": B, "prompt": L, "slots": MAX_LEN}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d"))
    res["attention_in_transformers"] = model.config._attn_implementation
    res["stored"] = {"q_proj": qwen3.layout_of(model.model.layers[0].self_attn.q_proj.weight),
                     "lm_head": qwen3.layout_of(model.lm_head.weight),
                     "embed": qwen3.layout_of(model.model.embed_tokens.weight)}
    cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)
    g = torch.Generator(device="cuda").manual_seed(1008)
    ids = torch.randint(0, model.config.vocab_size, (B, L), device="cuda", generator=g)
    with torch.no_grad():
        lg = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache, use_cache=True).logits
    tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
    saved = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    with torch.no_grad():
        ref = model(tok, cache_position=torch.tensor([L], device="cuda"), past_key_values=cache,
                    use_cache=True).logits[:, -1].float()
    res["logit_scale"] = ref.abs().max().item()
    ref_rows = [(layer.keys[:, :, L].clone(), layer.values[:, :, L].clone()) for layer in cache.layers]

    def restore():
        for layer, (k, v) in zip(cache.layers, saved):
            layer.keys.copy_(k)
            layer.values.copy_(v)

    positions = torch.tensor([L], device="cuda")
    until = torch.full((B,), L, device="cuda", dtype=torch.int64)
    res["frontend"] = {}
    for backend in ("torch", "triton", "flex"):
        restore()
        t0 = time.perf_counter()
        program = qwen3.trace_decode(model.config, B, MAX_LEN, attention=backend, quantized=True, model_path=MODEL)
        traced_s = time.perf_counter() - t0
        values = qwen3.tensors(model, cache, tok, positions, until)
        qwen3.check_layouts(program, values)
        step = program.bind(**values)
        with torch.no_grad():
            out = step()
        got = out["logits"].float()
        res["frontend"][backend] = {
            "nodes": len(program.graph.nodes), "trace_seconds": round(traced_s, 3), "notes": program.notes,
            "max_abs_vs_transformers": (got - ref).abs().max().item(),
            "mean_abs_vs_transformers": (got - ref).abs().mean().item(),
            "same_next_token": int((out["next"] == ref.argmax(-1)).sum().item()), "of": B,
            "cache_row_max_abs_vs_transformers": max(
                max((layer.keys[:, :, L].float() - k.float()).abs().max().item(),
                    (layer.values[:, :, L].float() - v.float()).abs().max().item())
                for layer, (k, v) in zip(cache.layers, ref_rows))}
        print(backend, json.dumps(res["frontend"][backend]), flush=True)
    # the load contract and the trace refuse a wrong declaration before anything runs
    refusals = {}
    restore()
    try:
        program = qwen3.trace_decode(model.config, B, MAX_LEN, attention="torch", quantized=False, model_path=MODEL)
        qwen3.check_layouts(program, qwen3.tensors(model, cache, tok, positions, until))
        refusals["declared dense, stored int4"] = "not refused"
    except RoleError as e:
        refusals["declared dense, stored int4"] = str(e)[:300]
    try:
        qwen3.trace_decode(model.config, B, MAX_LEN, attention="torch", rotary=Rotary("yarn", 1e6, 4.0, 32768))
        refusals["a RoPE scaling no lowering computes"] = "not refused"
    except RoleError as e:
        refusals["a RoPE scaling no lowering computes"] = str(e)[:300]
    try:
        program = qwen3.trace_decode(model.config, B, MAX_LEN + 1, attention="torch", model_path=MODEL)
        program.bind(**qwen3.tensors(model, cache, tok, positions, until))
        refusals["a cache declared one slot longer"] = "not refused"
    except RoleError as e:
        refusals["a cache declared one slot longer"] = str(e)[:300]
    res["refused"] = refusals
    print(json.dumps(refusals, indent=1), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
