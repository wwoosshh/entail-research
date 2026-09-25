"""E4: fact ablation. How much performance does each missing fact cost in the compiled decode step?

Same v2 harness as week 3 (real weights, fixed decode position, manual CUDA graph per variant, eager check,
interleaved rounds). Each variant gives the attention call a different set of facts:
  sdpa               neither fact: boolean mask table + materialised repeat_kv copy (transformers default)
  sdpa_gqa           sharing only: same boolean mask, heads shared by index via enable_gqa (no copy)
  maskmod_decode     both facts, general compiler: valid length as code (FlexAttention mask_mod) + sharing
  triton_decode      both facts, hand kernel: per-row valid length + sharing by index
  sdpa_bound_static  both facts, length known at compile time: cache sliced to the valid length, no mask,
                     so the flash backend is usable. Only possible because the position is fixed here; in real
                     decoding the length changes every step. Upper-bound reference, not a deployable path.
Also records the per-kernel GPU time of one graph replay, split into attention, GEMM and other kernels.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2"), os.path.join(PHASE0, "week3")]
import ab_attention as A  # noqa: E402
import ab_attention_v2 as V2  # noqa: E402
import bench_decode_attn_swap as S  # noqa: E402
from ab_common import print_timing, save_json  # noqa: E402
from bench_llm_decode import quantize_int4  # noqa: E402
from common import env_info  # noqa: E402
from numerics_experiment import load  # noqa: E402

BOUND = {"n": None}


def _causal_or_mask(query, key, value, attention_mask, scale):
    if attention_mask is None:
        return F.scaled_dot_product_attention(query, key, value, is_causal=query.shape[2] > 1, scale=scale,
                                              enable_gqa=True)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=scale, enable_gqa=True)


def sdpa_gqa_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    scale = float(scaling) if scaling is not None else query.shape[-1] ** -0.5
    return _causal_or_mask(query, key, value, attention_mask, scale).transpose(1, 2).contiguous(), None


def sdpa_bound_static_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    scale = float(scaling) if scaling is not None else query.shape[-1] ** -0.5
    if query.shape[2] == 1:
        n = BOUND["n"]
        out = F.scaled_dot_product_attention(query, key[:, :, :n], value[:, :, :n], scale=scale, enable_gqa=True)
    else:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale, enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


def register():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    try:
        from transformers.masking_utils import sdpa_mask
    except ImportError:
        sdpa_mask = AttentionMaskInterface._global_mapping["sdpa"]
    AttentionInterface.register("sdpa_gqa", sdpa_gqa_forward)
    AttentionMaskInterface.register("sdpa_gqa", sdpa_mask)
    AttentionInterface.register("sdpa_bound_static", sdpa_bound_static_forward)
    AttentionMaskInterface.register("sdpa_bound_static", lambda *a, **k: None)


def expected_path(impl, names):
    low = [n.lower() for n in names]
    fmha = any("fmha" in n for n in low)
    flash = any("flash" in n for n in low)
    split = any("_split_kernel" in n for n in low)
    # sdpa_gqa: whichever SDPA backend accepts mask + enable_gqa is part of what is measured
    return {"sdpa": fmha and not split and not flash,
            "sdpa_gqa": not split,
            "sdpa_bound_static": flash and not fmha and not split,
            "maskmod_decode": not fmha and not flash and not split,
            "triton_decode": split and not fmha and not flash}[impl]


def classify(name):
    n = name.lower()
    if any(s in n for s in ("fmha", "flash", "_split_kernel", "_combine_kernel", "flex", "tem_fused")):
        return "attention"
    if any(s in n for s in ("gemv", "gemm", "tinygemm", "cutlass", "_mm", "matmul")):
        return "gemm"
    if "efficient_attention" in n or "repeat" in n or "expand" in n:
        return "kv_copy_or_mask"
    return "other"


def kernel_breakdown(step):
    from torch.profiler import ProfilerActivity, profile
    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"e4_{os.getpid()}_{time.time_ns()}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    os.remove(path)
    by, per = {}, {}
    for e in tr["traceEvents"]:
        if e.get("cat") == "kernel" and "dur" in e:
            c = classify(e["name"])
            by[c] = by.get(c, 0.0) + e["dur"] / 1e3
            per[e["name"][:70]] = per.get(e["name"][:70], 0.0) + e["dur"] / 1e3
    out = {k: round(v, 3) for k, v in sorted(by.items(), key=lambda kv: -kv[1])}
    out["total"] = round(sum(by.values()), 3)
    out["top"] = [(n, round(v, 3)) for n, v in sorted(per.items(), key=lambda kv: -kv[1])[:8]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["int4:1", "int4:4", "int4:8", "bf16:8"])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e4_fact_ablation.json"))
    args = ap.parse_args()
    for name, val in (("cache_size_limit", 32), ("recompile_limit", 32), ("automatic_dynamic_shapes", False)):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, val)
    print(S.register_impl(), flush=True)
    A.register_maskmod()
    register()
    A.IMPLS = ["sdpa", "sdpa_gqa", "maskmod_decode", "triton_decode", "sdpa_bound_static"]
    A.expected_path = expected_path
    BOUND["n"] = args.prompt_len + 1
    _, model = load()
    res = {"env": env_info(), "args": vars(args), "runs": []}
    configs = [(c.split(":")[0], int(c.split(":")[1])) for c in args.configs]
    quantized = False
    for dtype in ("bf16", "int4"):
        for d, B in configs:
            if d != dtype:
                continue
            if dtype == "int4" and not quantized:
                S.set_impl(model, "sdpa")
                torch._dynamo.reset()
                quantize_int4(model)
                quantized = True
            torch._dynamo.reset()
            rec = V2.run_batch(model, model.config.vocab_size, B, dtype, args)
            res["runs"].append(rec)
            save_json(args.out, res)
            torch.cuda.empty_cache()
    print("saved", args.out, flush=True)


def patched_time_graphs(graphs, baseline, rounds, k, warmup_s):
    """Wrap V2's timing to also record a per-kernel breakdown of one replay per variant."""
    from ab_common import interleaved
    t = interleaved({n: ((lambda: None), g.replay) for n, g in graphs.items()},
                    rounds=rounds, k=k, warmup_s=warmup_s, baseline=baseline)
    t["kernel_breakdown_ms"] = {n: kernel_breakdown(g.replay) for n, g in graphs.items()}
    for n, b in t["kernel_breakdown_ms"].items():
        print(f"    kernels {n:18s} {({k: x for k, x in b.items() if k != 'top'})}", flush=True)
        print(f"      top {b['top'][:4]}", flush=True)
    return t


V2.time_graphs = patched_time_graphs

if __name__ == "__main__":
    main()
