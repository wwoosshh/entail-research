"""Week 3, step 4: what does a batch-invariant decode cost on this card?

Batch invariance = a request's logits do not depend on which other requests share its batch. Week 2 showed that
cuBLAS (GEMV at M=1 vs GEMM at M>1) and ATen reductions break it. Variant "bi" makes every reduction in the decode
step use a kernel whose reduction order depends only on the reduced dimension:
  linear    -> bi_kernels.bi_linear   (split-K from (N, K) only, fixed-order reduction, fixed M tile)
  RMSNorm   -> bi_kernels.bi_rmsnorm  (one program per row, fixed block)
  attention -> week-2 Triton decode kernel (fixed split per (batch row, head))
Variants: default (sdpa + cuBLAS + ATen/Inductor norms), tri_attn (Triton attention only), bi (all three).
Real Qwen3-4B bf16 weights.

Part 1  invariance: 4 prompts, 64 positions fed one token at a time (every forward is a decode step, teacher-
        forced). Each prompt alone (B=1) vs its row inside B=4. Eager for all variants; compiled for default, bi.
Part 2  operator microbenchmark: bi_linear vs cuBLAS on the model's own weights (M = 1, 4, 8).
Part 3  end-to-end decode step, compiled + CUDA graphs, interleaved rounds: default / tri_attn / bi, B = 1, 4, 8.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import bench_decode_attn_swap as S  # noqa: E402
from ab_common import cuda_kernel_names, interleaved, op_interleaved, print_timing, save_json, sync  # noqa: E402
from bench_llm_decode import make_static_cache  # noqa: E402
from bi_kernels import bi_linear, bi_linear_op, bi_rmsnorm_op, choose_split  # noqa: E402
from common import env_info  # noqa: E402
from numerics_experiment import PROMPTS, load  # noqa: E402

MODE = {"bi": False}
VARIANTS = {"default": ("sdpa", False), "tri_attn": ("triton_decode", False), "bi": ("triton_decode", True)}


def patch(model):
    lin_orig = torch.nn.Linear.forward

    def lin_forward(self, x):
        if MODE["bi"]:
            y = bi_linear_op(x, self.weight)
            return y if self.bias is None else y + self.bias
        return lin_orig(self, x)

    torch.nn.Linear.forward = lin_forward
    rms_cls = type(model.model.norm)
    rms_orig = rms_cls.forward

    def rms_forward(self, x):
        if MODE["bi"]:
            return bi_rmsnorm_op(x, self.weight, float(self.variance_epsilon))
        return rms_orig(self, x)

    rms_cls.forward = rms_forward
    return rms_cls.__name__


def use(model, variant):
    impl, bi = VARIANTS[variant]
    S.set_impl(model, impl)
    MODE["bi"] = bi


def reset_cache(cache):
    try:
        cache.reset()
    except Exception:
        for layer in getattr(cache, "layers", []):
            layer.keys.zero_()
            layer.values.zero_()


@torch.no_grad()
def feed(model, tokens, cache, step_fn=None):
    """Feed tokens [B, T] one position at a time into `cache`; return fp32 logits [T, B, V]."""
    B, T = tokens.shape
    reset_cache(cache)
    outs = []
    for t in range(T):
        pos = torch.tensor([t], device="cuda")
        S.VALID[:B].fill_(t + 1)
        tok = tokens[:, t:t + 1].contiguous()
        if step_fn is None:
            lg = model(tok, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]
        else:
            lg = step_fn(tok, pos, cache)
        outs.append(lg.float().clone())
    return torch.stack(outs)


def compare_rows(full, singles):
    rows = []
    for i, single in enumerate(singles):
        a, b = full[:, i:i + 1], single
        d = (a - b).abs()
        rows.append({"bitwise_equal": bool(torch.equal(a, b)), "max_abs_diff": d.max().item(),
                     "mean_abs_diff": d.mean().item(),
                     "argmax_agree": (a.argmax(-1) == b.argmax(-1)).float().mean().item()})
    return rows


def part1_invariance(model, tok, T, res):
    ids = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    L = min(len(i) for i in ids)
    g = torch.Generator().manual_seed(7)
    tail = torch.randint(0, model.config.vocab_size, (len(PROMPTS), T - L), generator=g)
    tokens = torch.cat([torch.stack([i[:L] for i in ids]), tail], 1).cuda()
    Bf = tokens.shape[0]
    cache_b = make_static_cache(model, Bf, T + 8)
    cache_1 = make_static_cache(model, 1, T + 8)
    out = {"prompt_tokens": L, "positions": T}
    eager_logits = {}
    for name in VARIANTS:
        use(model, name)
        full = feed(model, tokens, cache_b)
        singles = [feed(model, tokens[i:i + 1], cache_1) for i in range(Bf)]
        eager_logits[name] = full
        out[f"eager_{name}"] = compare_rows(full, singles)
        r = out[f"eager_{name}"]
        print(f"  [eager {name:8s}] bitwise equal per prompt: {[x['bitwise_equal'] for x in r]} | "
              f"max diff {max(x['max_abs_diff'] for x in r):.4f}", flush=True)
    d = (eager_logits["bi"] - eager_logits["default"]).abs()
    out["eager_bi_vs_default_B4"] = {"max_abs_diff": d.max().item(), "mean_abs_diff": d.mean().item(),
                                     "argmax_agree": (eager_logits["bi"].argmax(-1) ==
                                                      eager_logits["default"].argmax(-1)).float().mean().item()}
    print(f"  [eager] bi vs default at B=4: {out['eager_bi_vs_default_B4']}", flush=True)

    def decode_one(t, p, cache):
        return model(t, cache_position=p, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    for name in ("default", "bi"):
        torch._dynamo.reset()
        use(model, name)
        step = torch.compile(decode_one)
        try:
            full = feed(model, tokens, cache_b, step)
            singles = [feed(model, tokens[i:i + 1], cache_1, step) for i in range(Bf)]
            out[f"compiled_{name}"] = compare_rows(full, singles)
            dd = (full - eager_logits[name]).abs()
            out[f"compiled_{name}_vs_eager_{name}_B4"] = {"max_abs_diff": dd.max().item(),
                                                         "mean_abs_diff": dd.mean().item()}
            r = out[f"compiled_{name}"]
            print(f"  [compiled {name:8s}] bitwise equal per prompt: {[x['bitwise_equal'] for x in r]} | "
                  f"max diff {max(x['max_abs_diff'] for x in r):.4f} | vs eager {name}: "
                  f"max {dd.max().item():.4f} mean {dd.mean().item():.5f}", flush=True)
        except Exception as e:
            out[f"compiled_{name}"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
            print(f"  [compiled {name}] ERROR {out[f'compiled_{name}']['error']}", flush=True)
    res["part1_invariance"] = out
    del cache_b, cache_1
    torch.cuda.empty_cache()


def part2_microbench(model, res):
    layer = model.model.layers[0]
    weights = {"q_proj": layer.self_attn.q_proj.weight, "k_proj": layer.self_attn.k_proj.weight,
               "o_proj": layer.self_attn.o_proj.weight, "gate_proj": layer.mlp.gate_proj.weight,
               "down_proj": layer.mlp.down_proj.weight, "lm_head": model.lm_head.weight}
    rows = []
    for name, w in weights.items():
        N, K = w.shape
        for M in (1, 4, 8):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            t = op_interleaved({"cublas": lambda: F.linear(x, w), "bi": lambda: bi_linear(x, w)})
            gbytes = N * K * 2 / 1e9
            y_c, y_b = F.linear(x, w), bi_linear(x, w)
            rows.append({"layer": name, "N": N, "K": K, "M": M, "split": choose_split(N, K)[0],
                         "cublas_us": t["cublas"] * 1e3, "bi_us": t["bi"] * 1e3,
                         "cublas_GBps": gbytes / (t["cublas"] / 1e3), "bi_GBps": gbytes / (t["bi"] / 1e3),
                         "bi_over_cublas": t["bi"] / t["cublas"],
                         "cublas_row0_invariant": bool(torch.equal(F.linear(x[:1], w), y_c[:1])),
                         "bi_row0_invariant": bool(torch.equal(bi_linear(x[:1], w), y_b[:1]))})
            r = rows[-1]
            print(f"  {name:9s} M={M} N={N:6d} K={K:5d}: cuBLAS {r['cublas_us']:7.1f} us ({r['cublas_GBps']:5.0f} GB/s) "
                  f"| bi {r['bi_us']:7.1f} us ({r['bi_GBps']:5.0f} GB/s) x{r['bi_over_cublas']:.2f} | "
                  f"row0 invariant cuBLAS {r['cublas_row0_invariant']} bi {r['bi_row0_invariant']}", flush=True)
    res["part2_microbench"] = rows


def part3_speed(model, B, args):
    L = args.prompt_len
    max_len = L + 128
    g = torch.Generator(device="cuda").manual_seed(2000 + B)
    ids = torch.randint(0, model.config.vocab_size, (B, L), device="cuda", generator=g)
    use(model, "default")
    cache = make_static_cache(model, B, max_len)
    with torch.no_grad():
        lg = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache,
                   use_cache=True, return_dict=False)[0]
    tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
    pos = torch.tensor([L], device="cuda")
    S.VALID.zero_()
    S.VALID[:B].fill_(L + 1)

    def decode_one(t, p):
        return model(t, cache_position=p, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    compiled = torch.compile(decode_one, mode="reduce-overhead")
    rec = {"batch": B, "variants": {}}
    variants, ref = {}, None
    for name in VARIANTS:
        def setup(name=name):
            use(model, name)

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
            v["max_abs_logit_diff_vs_default"] = (out - ref).abs().max().item()
            names = cuda_kernel_names(step)
            low = [n.lower() for n in names]
            v["uses_bi_linear"] = any("_bi_mm_partial" in n for n in low)
            v["uses_cublas"] = any(("gemv" in n or "gemm" in n) and "triton" not in n and "_bi_mm" not in n for n in low)
            v["uses_triton_attention"] = any("_split_kernel" in n for n in low)
            ok = {"default": not v["uses_bi_linear"] and not v["uses_triton_attention"],
                  "tri_attn": v["uses_triton_attention"] and not v["uses_bi_linear"],
                  "bi": v["uses_triton_attention"] and v["uses_bi_linear"] and not v["uses_cublas"]}[name]
            v["uses_expected_path"] = ok
            if ok:
                variants[name] = (setup, step)
            else:
                v["error"] = "did not run the expected kernels; excluded from timing"
        except Exception as e:
            v["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        rec["variants"][name] = v
        print(f"  B={B} {name}: {v}", flush=True)
    if "default" in variants and len(variants) >= 2:
        rec["timing"] = interleaved(variants, rounds=args.rounds, k=args.steps, warmup_s=args.warmup_s,
                                    baseline="default")
        print_timing(f"B={B} bf16", rec["timing"])
    del compiled, cache
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--parts", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "batch_invariance_cost.json"))
    args = ap.parse_args()

    for name, val in (("cache_size_limit", 32), ("recompile_limit", 32), ("automatic_dynamic_shapes", False)):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, val)
    print(S.register_impl(), flush=True)
    tok, model = load()
    res = {"env": env_info(), "patched_norm_class": patch(model), "args": vars(args)}
    if "1" in args.parts:
        print("== part 1: batch invariance (teacher-forced, token by token)", flush=True)
        part1_invariance(model, tok, args.positions, res)
        save_json(args.out, res)
    if "2" in args.parts:
        print("== part 2: linear microbenchmark (model weights)", flush=True)
        part2_microbench(model, res)
        save_json(args.out, res)
    if "3" in args.parts:
        print("== part 3: end-to-end decode step, compiled + CUDA graphs, interleaved", flush=True)
        res["part3_speed"] = []
        for B in args.batches:
            torch._dynamo.reset()
            res["part3_speed"].append(part3_speed(model, B, args))
            save_json(args.out, res)
            torch.cuda.empty_cache()
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
