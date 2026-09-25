"""E1: anatomy of torch.compile on the Qwen3-4B decode step, from cold caches.

How much of compilation is spent reconstructing what the program means (Dynamo tracing Python bytecode,
building guards = assumptions it must re-check on every call) versus producing code (Inductor lowering,
Triton compilation, autotuning)? What does a recompile after a batch-size change cost, and what fraction of a
steady-state call is guard evaluation?
Caches are pointed at fresh directories and the FX/AOT caches are disabled before torch is imported.
"""
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="e1cache_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(TMP, "inductor")
os.environ["TRITON_CACHE_DIR"] = os.path.join(TMP, "triton")
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"

import json  # noqa: E402
import logging  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2"), os.path.join(PHASE0, "week3")]
import bench_decode_attn_swap as S  # noqa: E402
from bench_llm_decode import make_static_cache  # noqa: E402
from numerics_experiment import load  # noqa: E402


class Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.msgs = []

    def emit(self, record):
        try:
            self.msgs.append(record.getMessage())
        except Exception:  # noqa: BLE001
            pass


def metrics_snapshot():
    try:
        from torch._dynamo.utils import compilation_time_metrics
        return {k: sum(v) for k, v in compilation_time_metrics.items()}
    except Exception:  # noqa: BLE001
        return {}


def delta(after, before):
    return {k: after[k] - before.get(k, 0.0) for k in after if after[k] - before.get(k, 0.0) > 1e-4}


def guard_stats(msgs):
    text = [m for m in msgs if "TREE_GUARD_MANAGER" in m or "GUARDS" in m]
    if not text:
        return {"guard_log_found": False}
    lines = text[-1].splitlines()
    nodes = [ln for ln in lines if "+-" in ln]
    leaves = [ln for ln in nodes if "GuardManager" not in ln]
    kinds = {}
    for ln in leaves:
        k = ln.split("+-", 1)[1].strip().split(":")[0].split("(")[0].strip()
        kinds[k] = kinds.get(k, 0) + 1
    return {"guard_log_found": True, "tree_nodes": len(nodes), "leaf_guards": len(leaves),
            "top_kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:12])}


def main():
    col = Collect()
    torch._logging.set_logs(guards=True, recompiles=True, graph_breaks=True)
    for name in ("torch", "torch._dynamo"):
        logging.getLogger(name).addHandler(col)
    print(S.register_impl(), flush=True)
    tok_, model = load()
    S.set_impl(model, "sdpa")
    L = 512
    res = {"cache_dir": TMP}

    def session(B):
        cache = make_static_cache(model, B, L + 1024)
        ids = torch.randint(0, model.config.vocab_size, (B, L), device="cuda")
        with torch.no_grad():
            lg = model(ids, past_key_values=cache, use_cache=True, return_dict=False)[0]
        lens = [layer.cumulative_length for layer in cache.layers]
        return cache, lg[:, -1].argmax(-1, keepdim=True).contiguous(), lens

    def decode_one(t, c):
        return model(t, past_key_values=c, use_cache=True, return_dict=False)[0][:, -1]

    compiled = torch.compile(decode_one)

    # 1. cold compile, B=1
    cache, tok, lens = session(1)
    m0 = metrics_snapshot()
    n0 = len(col.msgs)
    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(tok, cache)
    torch.cuda.synchronize()
    res["cold_compile_B1_s"] = time.perf_counter() - t0
    res["cold_compile_B1_phases_s"] = dict(sorted(delta(metrics_snapshot(), m0).items(), key=lambda kv: -kv[1]))
    res["cold_compile_B1_guards"] = guard_stats(col.msgs[n0:])
    try:
        from torch._dynamo.utils import counters
        res["dynamo_stats"] = {k: dict(v) for k, v in counters.items() if k in ("stats", "graph_break")}
    except Exception:  # noqa: BLE001
        pass
    print("cold compile B=1:", round(res["cold_compile_B1_s"], 1), "s", flush=True)
    print("guards:", res["cold_compile_B1_guards"], flush=True)
    print("phases (s):", {k: round(v, 2) for k, v in list(res["cold_compile_B1_phases_s"].items())[:14]}, flush=True)

    # 2. per-call cost of guard evaluation (steady state, B=1)
    def rewind():
        torch._foreach_zero_(lens)
        torch._foreach_add_(lens, L)

    def per_call(n=60):
        ts = []
        for _ in range(n):
            rewind()
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                compiled(tok, cache)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
        return statistics.median(ts[10:])

    res["steady_call_ms_with_guards"] = per_call()
    try:
        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res["steady_call_ms_skip_guards"] = per_call()
    except Exception as e:  # noqa: BLE001
        res["skip_guard_eval_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    print("steady call ms:", res.get("steady_call_ms_with_guards"), "skip guards:",
          res.get("steady_call_ms_skip_guards", res.get("skip_guard_eval_error")), flush=True)

    # 3. recompile when a new batch size arrives (B=4)
    cache4, tok4, _ = session(4)
    m1 = metrics_snapshot()
    n1 = len(col.msgs)
    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(tok4, cache4)
    torch.cuda.synchronize()
    res["recompile_B4_s"] = time.perf_counter() - t0
    res["recompile_B4_phases_s"] = dict(sorted(delta(metrics_snapshot(), m1).items(), key=lambda kv: -kv[1]))
    res["recompile_B4_guards"] = guard_stats(col.msgs[n1:])
    res["recompile_reasons"] = [m[:400] for m in col.msgs[n1:] if "ecompil" in m][:5]
    print("recompile B=4:", round(res["recompile_B4_s"], 1), "s", flush=True)
    print("recompile reasons:", res["recompile_reasons"][:2], flush=True)
    print("phases (s):", {k: round(v, 2) for k, v in list(res["recompile_B4_phases_s"].items())[:14]}, flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "e1_compile_anatomy.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
