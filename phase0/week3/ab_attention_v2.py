"""Week 3, steps 2 and 3 (v2 harness, see graph_ab.py): decode-attention paths inside the compiled decode step.

Variants (same model, same static cache, same token, fixed decode position):
  sdpa            transformers' StaticCache default: boolean mask -> mem-efficient SDPA + repeat_kv copy
  triton_decode   week-2 Triton kernel: per-row valid length + GQA by index mapping, no mask
  maskmod_decode  PyTorch FlexAttention: valid length given as code (mask_mod `kv_idx < valid[b]`) + enable_gqa;
                  Inductor generates the kernel. Block mask built once for the measured position.
Each compiled variant is checked against the eager model with the same variant, and eager variants are compared
with eager sdpa, before timing.
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import ab_attention as A  # noqa: E402  (maskmod registration, block mask, expected_path; main() not run)
import bench_decode_attn_swap as S  # noqa: E402
from ab_common import print_timing, save_json  # noqa: E402
from bench_llm_decode import build_model, make_static_cache, quantize_int4  # noqa: E402
from common import env_info  # noqa: E402
from graph_ab import build_and_check, time_graphs  # noqa: E402


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
    A.FLEX["bm"] = A.build_block_mask(B, max_len)
    rec = {"batch": B, "dtype": dtype, "prompt_len": L}
    # transformers 5.x ignores cache_position: StaticLayer writes at its own GPU counter `cumulative_length`
    # and advances it on every call. Rewind it to L before every step so each call decodes position L.
    lens = [layer.cumulative_length for layer in cache.layers]

    def rewind():
        torch._foreach_zero_(lens)
        torch._foreach_add_(lens, L)

    eager = {}
    for impl in A.IMPLS + ["sdpa_again"]:
        S.set_impl(model, "sdpa" if impl == "sdpa_again" else impl)
        rewind()
        with torch.no_grad():
            eager[impl] = model(tok, cache_position=pos, past_key_values=cache, use_cache=True,
                                return_dict=False)[0][:, -1].float().clone()
    rec["eager_sdpa_rerun_bitwise"] = bool(torch.equal(eager["sdpa"], eager.pop("sdpa_again")))
    rec["eager_vs_eager_sdpa"] = {impl: {"max": (eager[impl] - eager["sdpa"]).abs().max().item(),
                                         "mean": (eager[impl] - eager["sdpa"]).abs().mean().item()}
                                  for impl in A.IMPLS}
    rec["logit_scale"] = eager["sdpa"].abs().max().item()
    print(f"  B={B} {dtype} eager vs eager sdpa: {rec['eager_vs_eager_sdpa']} (logit scale {rec['logit_scale']:.2f}) "
          f"| eager sdpa rerun bitwise identical: {rec['eager_sdpa_rerun_bitwise']}", flush=True)

    def decode_one(t, p):
        return model(t, cache_position=p, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    compiled = torch.compile(decode_one)
    setups = {impl: (lambda impl=impl: S.set_impl(model, impl)) for impl in A.IMPLS}
    graphs, rec["variants"] = build_and_check(setups, compiled, (tok, pos), eager, "sdpa", A.expected_path,
                                              pre=rewind)
    if "sdpa" in graphs and len(graphs) >= 2:
        rec["timing"] = time_graphs(graphs, "sdpa", args.rounds, args.steps, args.warmup_s)
        print_timing(f"B={B} {dtype}", rec["timing"])
    rec["position_counter_after"] = int(lens[0].item())
    print(f"  position counter after all calls: {rec['position_counter_after']} (expected {L + 1})", flush=True)
    del compiled, cache, graphs
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "int4"])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--random-weights", action="store_true", help="hub-config random init instead of real weights")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ab_attention_v2.json"))
    args = ap.parse_args()

    for name, val in (("cache_size_limit", 32), ("recompile_limit", 32), ("automatic_dynamic_shapes", False)):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, val)
    print(S.register_impl(), flush=True)
    A.register_maskmod()
    if args.random_weights:
        model, cfg, src, n_params = build_model()
    else:
        from numerics_experiment import load
        _, model = load()
        cfg, src = model.config, "real Qwen3-4B weights (~/models/Qwen3-4B)"
        n_params = sum(p.numel() for p in model.parameters())
    res = {"env": env_info(), "model_source": src, "params": n_params, "args": vars(args), "harness": "v2+rewind",
           "runs": []}
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
