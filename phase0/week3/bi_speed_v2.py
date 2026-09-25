"""Week 3, step 4 part 3 (v2 harness, see graph_ab.py): end-to-end decode-step cost of batch invariance.

Variants on real Qwen3-4B bf16 weights:
  default   sdpa attention + cuBLAS linears + Inductor-fused RMSNorm
  tri_attn  week-2 Triton decode attention, everything else default
  bi        Triton decode attention + bi_linear + bi_rmsnorm (batch-invariant, verified bitwise in part 1)
Manual CUDA graph per variant, each checked against the eager model with the same variant, interleaved timing.
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import batch_invariance_cost as BI  # noqa: E402  (patch, use, VARIANTS; main() not run)
import bench_decode_attn_swap as S  # noqa: E402
from ab_common import print_timing, save_json  # noqa: E402
from bench_llm_decode import make_static_cache  # noqa: E402
from common import env_info  # noqa: E402
from graph_ab import build_and_check, time_graphs  # noqa: E402
from numerics_experiment import load  # noqa: E402


def path_check(name, names):
    low = [n.lower() for n in names]
    bi_lin = any("_bi_mm_partial" in n for n in low)
    tri = any("_split_kernel" in n for n in low)
    return {"default": not bi_lin and not tri, "tri_attn": tri and not bi_lin, "bi": tri and bi_lin}[name]


def run(model, B, args):
    L = args.prompt_len
    max_len = L + 128
    g = torch.Generator(device="cuda").manual_seed(2000 + B)
    ids = torch.randint(0, model.config.vocab_size, (B, L), device="cuda", generator=g)
    BI.use(model, "default")
    cache = make_static_cache(model, B, max_len)
    with torch.no_grad():
        lg = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache,
                   use_cache=True, return_dict=False)[0]
    tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
    pos = torch.tensor([L], device="cuda")
    S.VALID.zero_()
    S.VALID[:B].fill_(L + 1)
    rec = {"batch": B, "prompt_len": L}
    # transformers 5.x ignores cache_position; rewind the static cache's GPU position counter before every step.
    lens = [layer.cumulative_length for layer in cache.layers]

    def rewind():
        torch._foreach_zero_(lens)
        torch._foreach_add_(lens, L)

    eager = {}
    for name in BI.VARIANTS:
        BI.use(model, name)
        rewind()
        with torch.no_grad():
            eager[name] = model(tok, cache_position=pos, past_key_values=cache, use_cache=True,
                                return_dict=False)[0][:, -1].float().clone()
    rec["eager_vs_eager_default"] = {n: (eager[n] - eager["default"]).abs().max().item() for n in eager}
    print(f"  B={B} eager vs eager default (max abs logit diff): {rec['eager_vs_eager_default']}", flush=True)

    def decode_one(t, p):
        return model(t, cache_position=p, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    compiled = torch.compile(decode_one)
    setups = {n: (lambda n=n: BI.use(model, n)) for n in BI.VARIANTS}
    graphs, rec["variants"] = build_and_check(setups, compiled, (tok, pos), eager, "default", path_check,
                                              pre=rewind)
    if "default" in graphs and len(graphs) >= 2:
        rec["timing"] = time_graphs(graphs, "default", args.rounds, args.steps, args.warmup_s)
        print_timing(f"B={B} bf16", rec["timing"])
    rec["position_counter_after"] = int(lens[0].item())
    print(f"  position counter after all calls: {rec['position_counter_after']} (expected {L + 1})", flush=True)
    del compiled, cache, graphs
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "bi_speed_v2.json"))
    args = ap.parse_args()

    for name, val in (("cache_size_limit", 32), ("recompile_limit", 32), ("automatic_dynamic_shapes", False)):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, val)
    print(S.register_impl(), flush=True)
    tok, model = load()
    res = {"env": env_info(), "patched_norm_class": BI.patch(model), "args": vars(args), "harness": "v2", "runs": []}
    for B in args.batches:
        torch._dynamo.reset()
        res["runs"].append(run(model, B, args))
        save_json(args.out, res)
        torch.cuda.empty_cache()
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
