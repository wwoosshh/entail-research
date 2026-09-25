"""Safe A/B of several compiled variants of one decode step (v2 harness).

v1 compiled one function with mode="reduce-overhead" (CUDA-graph trees) and switched variants between calls. The
non-first variants then disagreed with the first by the full logit scale and the first alternating round hit a
device-side assert. v2 instead:
  1. compiles with mode="default" (Inductor kernels, no CUDA-graph trees),
  2. captures one manual torch.cuda.CUDAGraph per variant, each with its own memory pool,
  3. checks every variant's graph output against an EAGER reference before it is allowed into the timing.
Timing then replays the graphs, interleaved, so no Python dispatch or guard evaluation is inside the measurement.
"""
import time

import torch

from ab_common import cuda_kernel_names, interleaved, sync


def capture(fn, warmup=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    sync()
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        out = fn()
    sync()
    return g, out


def build_and_check(variants, compiled, args, eager_ref, ref_name, path_check, tol=3.0, mean_tol=0.1, pre=None):
    """variants: {name: setup_fn}; compiled: torch.compile'd step; args: its inputs;
    eager_ref: {name: fp32 logits from the eager model with that variant}; path_check(name, kernel_names) -> bool.
    pre: GPU-side preparation run before every step and captured into the graph (e.g. rewinding the static
    cache's position counter, which transformers 5.x advances on every call).
    Returns ({name: graph}, {name: record})."""
    graphs, recs = {}, {}

    def run():
        if pre is not None:
            pre()
        return compiled(*args)

    for name, setup in variants.items():
        v = {}
        try:
            setup()
            t0 = time.perf_counter()
            with torch.no_grad():
                run()
            sync()
            v["compile_s"] = time.perf_counter() - t0
            g, out = capture(run)
            g.replay()
            sync()
            o = out.float().clone()
            v["max_abs_diff_vs_eager_same_variant"] = (o - eager_ref[name]).abs().max().item()
            v["mean_abs_diff_vs_eager_same_variant"] = (o - eager_ref[name]).abs().mean().item()
            v[f"max_abs_diff_vs_eager_{ref_name}"] = (o - eager_ref[ref_name]).abs().max().item()
            v["logit_scale"] = eager_ref[ref_name].abs().max().item()
            names = cuda_kernel_names(g.replay)
            v["n_kernel_types"] = len(names)
            v["uses_expected_path"] = path_check(name, names)
            v["kernels_of_interest"] = [n[:90] for n in names if any(
                s in n.lower() for s in ("fmha", "flash", "_split_kernel", "flex", "_bi_mm", "_bi_rmsnorm",
                                         "gemv", "tinygemm"))][:12]
            v["correct"] = (v["max_abs_diff_vs_eager_same_variant"] < tol and
                            v["mean_abs_diff_vs_eager_same_variant"] < mean_tol)
            if v["uses_expected_path"] and v["correct"]:
                graphs[name] = g
                v["timed"] = True
            else:
                v["timed"] = False
                v["excluded_because"] = ("wrong kernels" if not v["uses_expected_path"] else
                                         "graph output disagrees with its eager reference")
        except Exception as e:
            v["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            v["timed"] = False
        recs[name] = v
        shown = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "kernels_of_interest"}
        print(f"    {name}: {shown}", flush=True)
    return graphs, recs


def time_graphs(graphs, baseline, rounds, k, warmup_s):
    return interleaved({n: ((lambda: None), g.replay) for n, g in graphs.items()},
                       rounds=rounds, k=k, warmup_s=warmup_s, baseline=baseline)
