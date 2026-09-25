"""Week 3, steps 2 and 3: interleaved A/B of decode-attention paths in the compiled (CUDA graphs) decode step.

All variants share one model, one static cache, the same input token and the same decode position:
  sdpa            transformers' StaticCache default: boolean mask -> mem-efficient SDPA + repeat_kv copy
  triton_decode   week-2 Triton kernel: per-row valid length + GQA by index mapping, no mask
  maskmod_decode  PyTorch FlexAttention. The valid-length meaning is given as code (mask_mod
                  `kv_idx < valid[b]`) instead of a boolean table, GQA via enable_gqa, and Inductor generates the
                  kernel. The block mask is built once for the measured position, so per-step block-mask
                  maintenance is not included (favourable to this variant).
Dynamo guards on the attention-implementation name, so one torch.compile wrapper holds one graph per variant.
Each variant is checked for (a) which attention kernels actually ran and (b) logits vs sdpa before timing.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import bench_decode_attn_swap as S  # noqa: E402  registers phase0::decode_attention and triton_decode
from ab_common import cuda_kernel_names, interleaved, print_timing, save_json, sync  # noqa: E402
from bench_llm_decode import build_model, make_static_cache, quantize_int4  # noqa: E402
from common import env_info  # noqa: E402

FLEX = {"bm": None}
IMPLS = ["sdpa", "triton_decode", "maskmod_decode"]
ATTN_KEYS = ("fmha", "flash", "_split_kernel", "_combine_kernel", "flex", "attention", "tem_")


def maskmod_decode_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    B, Hq, q_len, D = query.shape
    scale = float(scaling) if scaling is not None else D ** -0.5
    if q_len == 1:
        out = flex_attention(query, key, value, block_mask=FLEX["bm"], scale=scale, enable_gqa=True)
    else:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale, enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


def register_maskmod():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register("maskmod_decode", maskmod_decode_forward)
    AttentionMaskInterface.register("maskmod_decode", lambda *a, **k: None)


def build_block_mask(B, kv_len):
    def valid_len_mask(b, h, q_idx, kv_idx):
        return kv_idx < S.VALID[b]
    return create_block_mask(valid_len_mask, B=B, H=None, Q_LEN=1, KV_LEN=kv_len, device="cuda")


def expected_path(impl, names):
    low = [n.lower() for n in names]
    has_fmha = any("fmha" in n for n in low)
    has_split = any("_split_kernel" in n for n in low)
    if impl == "sdpa":
        return has_fmha and not has_split
    if impl == "triton_decode":
        return has_split and not has_fmha
    return not has_fmha and not has_split


def run_batch(model, vocab, B, dtype, args):
    L = args.prompt_len
    max_len = L + 128
    g = torch.Generator(device="cuda").manual_seed(1000 + B)
    ids = torch.randint(0, vocab, (B, L), device="cuda", generator=g)
    S.set_impl(model, "sdpa")
    cache = make_static_cache(model, B, max_len)
    with torch.no_grad():
        lg = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache,
                   use_cache=True, return_dict=False)[0]
    tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
    pos = torch.tensor([L], device="cuda")
    S.VALID.zero_()
    S.VALID[:B].fill_(L + 1)
    rec = {"batch": B, "dtype": dtype, "prompt_len": L, "variants": {}}
    try:
        FLEX["bm"] = build_block_mask(B, max_len)
    except Exception as e:
        FLEX["bm"] = None
        rec["block_mask_error"] = f"{type(e).__name__}: {str(e)[:300]}"

    def decode_one(t, p):
        return model(t, cache_position=p, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    compiled = torch.compile(decode_one, mode="reduce-overhead")
    variants, ref = {}, None
    for impl in IMPLS:
        if impl == "maskmod_decode" and FLEX["bm"] is None:
            rec["variants"][impl] = {"error": "block mask unavailable"}
            continue

        def setup(impl=impl):
            S.set_impl(model, impl)

        def step():
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                return compiled(tok, pos)

        v = {}
        try:
            setup()
            t0 = time.perf_counter()
            step()
            sync()
            v["first_call_s"] = time.perf_counter() - t0
            for _ in range(3):
                step()
            sync()
            out = step().float().clone()
            if ref is None:
                ref = out
            v["max_abs_logit_diff_vs_sdpa"] = (out - ref).abs().max().item()
            v["argmax_agree_vs_sdpa"] = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
            names = cuda_kernel_names(step)
            v["attention_kernels"] = [n[:120] for n in names if any(s in n.lower() for s in ATTN_KEYS)]
            v["n_kernel_types"] = len(names)
            v["uses_expected_path"] = expected_path(impl, names)
            if v["uses_expected_path"]:
                variants[impl] = (setup, step)
            else:
                v["error"] = "did not run the expected attention path; excluded from timing"
        except Exception as e:
            v["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        rec["variants"][impl] = v
        short = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "attention_kernels"}
        print(f"  B={B} {dtype} {impl}: {short}", flush=True)
        print(f"     attention kernels: {[n[:70] for n in v.get('attention_kernels', [])]}", flush=True)

    if "sdpa" in variants and len(variants) >= 2:
        rec["timing"] = interleaved(variants, rounds=args.rounds, k=args.steps, warmup_s=args.warmup_s,
                                    baseline="sdpa")
        print_timing(f"B={B} {dtype}", rec["timing"])
    del compiled, cache
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "int4"])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ab_attention.json"))
    args = ap.parse_args()

    for name, val in (("cache_size_limit", 32), ("recompile_limit", 32), ("automatic_dynamic_shapes", False)):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, val)
    print(S.register_impl(), flush=True)
    register_maskmod()
    model, cfg, src, n_params = build_model()
    res = {"env": env_info(), "model_source": src, "params": n_params, "args": vars(args), "runs": []}
    for dtype in args.dtypes:
        if dtype == "int4":
            S.set_impl(model, "sdpa")
            torch._dynamo.reset()
            quantize_int4(model)
        for B in args.batches:
            torch._dynamo.reset()
            res["runs"].append(run_batch(model, cfg.vocab_size, B, dtype, args))
            save_json(args.out, res)
            torch.cuda.empty_cache()
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
