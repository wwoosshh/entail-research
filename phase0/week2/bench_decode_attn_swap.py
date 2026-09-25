"""Week 2, experiment A3: swap the attention implementation inside the compiled StaticCache decode.

Baseline  : attn_implementation="sdpa" (transformers builds a mask -> mem-efficient backend + repeat_kv copy)
Candidate : attn_implementation="triton_decode" (this repo's static-shape, valid-length, GQA-aware Triton kernel,
            wrapped as a torch custom op so torch.compile + CUDA graphs treat it as an opaque node; no mask built)
Measured end-to-end decode step for bf16 and int4, B in {1,4,8}, mode compile_graphs (best week-1 mode).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import env_info, run_mode, save_results  # noqa: E402
from bench_llm_decode import build_model, make_static_cache, quantize_int4  # noqa: E402
from triton_decode_attn import decode_attention  # noqa: E402

VALID = torch.zeros(64, dtype=torch.int32, device="cuda")  # per-row valid cache length, updated each step


@torch.library.custom_op("phase0::decode_attention", mutates_args=())
def decode_attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor, scale: float) -> torch.Tensor:
    return decode_attention(q, k, v, valid, sm_scale=scale)


@decode_attention_op.register_fake
def _(q, k, v, valid, scale):
    return torch.empty_like(q)


def triton_decode_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers AttentionInterface signature. query [B,Hq,q_len,D]; key/value [B,Hkv,L,D] (static cache)."""
    B, Hq, q_len, D = query.shape
    scale = float(scaling) if scaling is not None else D ** -0.5
    if q_len == 1:
        out = decode_attention_op(query, key, value, VALID[:B], scale)
    else:  # prefill from an empty cache: causal over the prompt, GQA native; zeros beyond the prompt are never attended
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale, enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


def register_impl():
    from transformers import AttentionInterface
    AttentionInterface.register("triton_decode", triton_decode_attention_forward)
    try:
        from transformers.masking_utils import AttentionMaskInterface
        AttentionMaskInterface.register("triton_decode", lambda *a, **k: None)  # no mask tensor is ever built
        return "attention + mask interface registered"
    except Exception as e:
        return f"attention registered; mask interface unavailable ({type(e).__name__}) - a mask may still be built"


def set_impl(model, name):
    try:
        model.set_attn_implementation(name)
    except Exception:
        model.config._attn_implementation = name
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = name


def compiled_step_factory(model, B, prompt_len, vocab, mode, max_len):
    ids = torch.randint(0, vocab, (B, prompt_len), device="cuda")
    cache = make_static_cache(model, B, max_len)
    VALID[:B].fill_(prompt_len)
    with torch.no_grad():
        logits = model(ids, cache_position=torch.arange(prompt_len, device="cuda"),
                       past_key_values=cache, use_cache=True, return_dict=False)[0]
    state = {"tok": logits[:, -1].argmax(-1, keepdim=True), "pos": torch.tensor([prompt_len], device="cuda")}

    def decode_one(tok, pos):
        lg = model(tok, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)[0]
        return lg[:, -1].argmax(-1, keepdim=True)

    cstep = torch.compile(decode_one, mode=("reduce-overhead" if mode == "compile_graphs" else "default"))

    def step():
        VALID[:B].copy_((state["pos"] + 1).to(torch.int32).expand(B))  # token written at pos is attended
        state["tok"] = cstep(state["tok"], state["pos"]).clone()
        state["pos"] = state["pos"] + 1
    return step


def equivalence_check(model, vocab, B=2, prompt_len=64, max_len=128, steps=4):
    """Same random prompt AND the same forced decode tokens through both implementations (eager, static
    cache); compare per-step logits. Teacher forcing matters: feeding each implementation its own argmax
    lets one near-tie flip (random weights give nearly flat logits) send the sequences apart, which
    measures chaos, not the attention kernels."""
    torch.manual_seed(1)
    ids = torch.randint(0, vocab, (B, prompt_len), device="cuda")
    forced = torch.randint(0, vocab, (B, steps), device="cuda")
    outs = {}
    for impl in ("sdpa", "triton_decode"):
        set_impl(model, impl)
        cache = make_static_cache(model, B, max_len)
        VALID[:B].fill_(prompt_len)
        with torch.no_grad():
            lg = model(ids, cache_position=torch.arange(prompt_len, device="cuda"), past_key_values=cache,
                       use_cache=True, return_dict=False)[0]
            seq = [lg[:, -1].float()]
            pos = torch.tensor([prompt_len], device="cuda")
            for s in range(steps):
                VALID[:B].copy_((pos + 1).to(torch.int32).expand(B))
                lg = model(forced[:, s:s + 1], cache_position=pos, past_key_values=cache, use_cache=True,
                           return_dict=False)[0]
                seq.append(lg[:, -1].float())
                pos = pos + 1
        outs[impl] = torch.stack(seq)
    diff = (outs["sdpa"] - outs["triton_decode"]).abs()
    same_argmax = (outs["sdpa"].argmax(-1) == outs["triton_decode"].argmax(-1)).float().mean().item()
    return {"max_abs_logit_diff": diff.max().item(), "mean_abs_logit_diff": diff.mean().item(),
            "logit_scale": outs["sdpa"].abs().max().item(), "argmax_agreement": same_argmax,
            "note": "teacher-forced; prefill logits included as step 0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--mode", default="compile_graphs")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "int4"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "decode_attn_swap.json"))
    args = ap.parse_args()

    reg = register_impl()
    print(reg, flush=True)
    model, cfg, src, n_params = build_model()
    vocab = cfg.vocab_size
    max_len = args.prompt_len + 128
    results, meta = [], {"env": env_info(), "model_source": src, "params": n_params, "registration": reg}

    meta["equivalence_bf16"] = equivalence_check(model, vocab)
    print("equivalence bf16:", meta["equivalence_bf16"], flush=True)

    for dtype in args.dtypes:
        if dtype == "int4":
            torch._dynamo.reset()
            set_impl(model, "sdpa")
            t0 = time.perf_counter()
            quantize_int4(model)
            torch.cuda.synchronize()
            meta["int4_quantize_s"] = time.perf_counter() - t0
            meta["equivalence_int4"] = equivalence_check(model, vocab)
            print("equivalence int4:", meta["equivalence_int4"], flush=True)
        for impl in ("sdpa", "triton_decode"):
            set_impl(model, impl)
            torch._dynamo.reset()
            for B in args.batches:
                def make_fn(B=B):
                    return compiled_step_factory(model, B, args.prompt_len, vocab, args.mode, max_len)
                results.append(run_mode(
                    f"decode_{dtype}/{impl}/{args.mode}/B{B}", make_fn, iters=32, profile_steps=4,
                    extra={"workload": f"decode_{dtype}", "attn": impl, "mode": args.mode, "batch": B,
                           "prompt_len": args.prompt_len, "dtype": dtype}))
                torch.cuda.empty_cache()
    meta["results"] = results
    save_results(args.out, meta)


if __name__ == "__main__":
    main()
