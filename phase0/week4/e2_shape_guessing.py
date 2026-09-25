"""E2: guessing versus declaring the batch dimension (one approach per process, cold caches).

A decode server sees batch sizes 1, 4, 8, 2 in that order (each with its own static cache). Approaches:
  auto          torch.compile defaults: specialise on what it sees, recompile and go dynamic when a guess fails
  dynamic_flag  torch.compile(dynamic=True): every dimension symbolic from the start (sizes 0/1 still specialised)
  mark_dynamic  declare only the batch dim with torch._dynamo.mark_dynamic (the documented per-dimension hint)
  unbacked      declare it with mark_unbacked (no 0/1 specialisation; data-dependent branches become errors);
                every tensor's batch dim gets its own symbol, so the compiler does not know they are equal
  unbacked_shared  mark_unbacked with one shape_id for all 73 tensors: also declares that they are the SAME
                batch (the relation), which is what a role system states once
  static        declare the set of batch sizes up front and compile each one before serving (no dynamic shapes);
                the practice of serving engines that capture one graph per batch size at start-up
Recorded: stall seen by the first step of each batch, graphs compiled, compile time, steady-state step time.
"""
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="e2cache_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(TMP, "inductor")
os.environ["TRITON_CACHE_DIR"] = os.path.join(TMP, "triton")
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import bench_decode_attn_swap as S  # noqa: E402
from bench_llm_decode import make_static_cache  # noqa: E402
from numerics_experiment import load  # noqa: E402

BATCHES = [1, 4, 8, 2]
L, STEPS = 512, 16
APPROACHES = ["auto", "dynamic_flag", "mark_dynamic", "unbacked", "unbacked_shared", "static"]


def compile_metrics():
    """{phase: (count, seconds)}; the Dynamo entry point is reported under both names used across versions."""
    try:
        from torch._dynamo.utils import compilation_time_metrics
        out = {k: (len(v), sum(v)) for k, v in compilation_time_metrics.items()}
        for k in list(out):
            if k.endswith("compile_inner"):
                out["compile_inner"] = out[k]
        return out
    except Exception:  # noqa: BLE001
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True, choices=APPROACHES)
    args = ap.parse_args()
    if args.approach == "static":
        torch._dynamo.config.automatic_dynamic_shapes = False
    for name in ("cache_size_limit", "recompile_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 32)
    S.register_impl()
    _, model = load()
    S.set_impl(model, "sdpa")

    def decode_one(t, c):
        return model(t, past_key_values=c, use_cache=True, return_dict=False)[0][:, -1]

    compiled = torch.compile(decode_one, dynamic=True if args.approach == "dynamic_flag" else None)

    def session(B):
        cache = make_static_cache(model, B, L + STEPS + 8)
        ids = torch.randint(0, model.config.vocab_size, (B, L), device="cuda")
        with torch.no_grad():
            lg = model(ids, past_key_values=cache, use_cache=True, return_dict=False)[0]
        tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
        if args.approach in ("mark_dynamic", "unbacked", "unbacked_shared"):
            for t in [tok] + [x for layer in cache.layers for x in (layer.keys, layer.values)]:
                if args.approach == "mark_dynamic":
                    torch._dynamo.mark_dynamic(t, 0)
                elif args.approach == "unbacked":
                    torch._dynamo.decorators.mark_unbacked(t, 0)
                else:
                    torch._dynamo.decorators.mark_unbacked(t, 0, shape_id="batch")
        return cache, tok

    res = {"approach": args.approach, "batches": BATCHES, "prompt_len": L, "steps": STEPS,
           "precompile_s": None, "per_batch": []}
    if args.approach == "static":
        t0 = time.perf_counter()
        for B in BATCHES:
            cache, tok = session(B)
            with torch.no_grad():
                compiled(tok, cache)
            torch.cuda.synchronize()
        res["precompile_s"] = time.perf_counter() - t0
        res["compiles_after_precompile"] = compile_metrics().get("compile_inner", (0, 0.0))[0]
        print(f"static: precompiled {len(BATCHES)} batch sizes in {res['precompile_s']:.1f} s", flush=True)

    for B in BATCHES:
        rec = {"batch": B}
        before = compile_metrics().get("compile_inner", (0, 0.0))
        try:
            cache, tok = session(B)
            ts = []
            for _ in range(STEPS):
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.no_grad():
                    compiled(tok, cache)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1e3)
            rec.update({"first_step_ms": ts[0], "steady_ms": statistics.median(ts[4:]),
                        "steady_min_ms": min(ts[4:]), "steady_max_ms": max(ts[4:])})
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:700]}"
        after = compile_metrics().get("compile_inner", (0, 0.0))
        rec["compiles_during_batch"] = after[0] - before[0]
        rec["compile_s_during_batch"] = after[1] - before[1]
        res["per_batch"].append(rec)
        print(f"{args.approach} B={B}: " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                                      for k, v in rec.items()}, ensure_ascii=False), flush=True)
    try:
        from torch._dynamo.utils import counters
        res["unique_graphs"] = counters["stats"].get("unique_graphs")
        res["dynamo_stats"] = dict(counters["stats"])
    except Exception:  # noqa: BLE001
        pass
    res["compile_metrics_total"] = {k: {"n": n, "s": s} for k, (n, s) in sorted(
        compile_metrics().items(), key=lambda kv: -kv[1][1])[:12]}
    print(f"{args.approach}: unique graphs {res.get('unique_graphs')} | compile_inner "
          f"{res['compile_metrics_total'].get('compile_inner')}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"e2_{args.approach}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
