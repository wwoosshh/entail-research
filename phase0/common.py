"""Shared helpers for Phase 0 measurements (RTX 4070 Ti, WSL2).

Collects, per workload and execution mode:
  - wall time per step (median/min/max, CUDA-synchronized)
  - first-call time (includes torch.compile compilation / CUDA graph capture)
  - peak GPU memory
  - kernel statistics from a torch.profiler chrome trace: kernels per step, GPU busy time,
    GPU idle fraction inside the step span, per-class breakdown, top kernels by time
"""
import json
import os
import re
import statistics
import tempfile
import time

import torch


def sync():
    torch.cuda.synchronize()


def time_fn(fn, warmup=3, iters=20):
    """Median/min/max wall time in ms per call, synchronized after each call."""
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


# Order matters: first match wins.
_CLASSES = [
    ("triton", re.compile(r"^triton_", re.I)),
    ("attention", re.compile(r"flash|fmha|sdpa|attention|attn", re.I)),
    ("conv_cudnn", re.compile(r"fprop|dgrad|wgrad|conv|cudnn|implicit_gemm|nchw|nhwc", re.I)),
    ("gemm_lib", re.compile(r"gemm|cublas|cutlass|xmma|nvjet|splitk|sm80_|sm86_|sm89_|sm90_|ampere_|ada_", re.I)),
    ("memcpy_memset", re.compile(r"memcpy|memset", re.I)),
    ("elementwise_reduce", re.compile(
        r"elementwise|vectorized|unrolled|fill|copy|CatArray|index|gather|scatter|where|clamp|softmax|"
        r"reduce|norm|cast|arange|embedding|topk|argmax|cumsum|sort|bitonic|distribution|silu|gelu|"
        r"upsample|interpolate|im2col|col2im", re.I)),
]


def classify(name):
    for cls, rx in _CLASSES:
        if rx.search(name):
            return cls
    return "other"


def profile_kernels(fn, steps=2):
    """Run fn `steps` times under torch.profiler and summarize GPU kernel activity per step.

    idle_frac = 1 - (union of GPU-busy intervals) / (span from first GPU event to last).
    torch.profiler adds CPU overhead, so idle_frac is an upper bound; cross-check with nsys.
    """
    if os.environ.get("PHASE0_NO_PROFILER"):
        # nsys and torch.profiler both use CUPTI; disable the in-process profiler when running under nsys.
        return {"skipped": "PHASE0_NO_PROFILER set"}
    from torch.profiler import ProfilerActivity, profile

    fn()
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            fn()
        sync()
    path = os.path.join(tempfile.gettempdir(), f"trace_{os.getpid()}_{int(time.time()*1e3)}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    os.remove(path)

    gpu = [e for e in tr["traceEvents"]
           if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e and "ts" in e]
    kern = [e for e in gpu if e["cat"] == "kernel"]
    if not gpu:
        return {"error": "no GPU events captured"}

    ivals = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    busy = 0.0
    cs, ce = ivals[0]
    for s, e in ivals[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = max(e["ts"] + e["dur"] for e in gpu) - min(e["ts"] for e in gpu)

    agg = {}
    for e in kern:
        a = agg.setdefault(e["name"], [0, 0.0])
        a[0] += 1
        a[1] += e["dur"]
    by_cls = {}
    for n, (c, d) in agg.items():
        b = by_cls.setdefault(classify(n), [0, 0.0])
        b[0] += c
        b[1] += d
    top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]

    return {
        "kernels_per_step": len(kern) / steps,
        "kernel_time_ms_per_step": sum(e["dur"] for e in kern) / steps / 1e3,
        "gpu_busy_ms_per_step": busy / steps / 1e3,
        "span_ms_per_step": span / steps / 1e3,
        "idle_frac_upper_bound": (1 - busy / span) if span else None,
        "by_class": {k: {"count": v[0] / steps, "ms": v[1] / steps / 1e3} for k, v in
                     sorted(by_cls.items(), key=lambda kv: -kv[1][1])},
        "top": [{"name": n[:100], "count": c / steps, "ms": d / steps / 1e3, "cls": classify(n)}
                for n, (c, d) in top],
    }


def env_info():
    p = torch.cuda.get_device_properties(0)
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": p.name,
        "cc": f"{p.major}.{p.minor}",
        "sms": p.multi_processor_count,
        "mem_gib": round(p.total_memory / 2**30, 1),
    }
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = None
    return info


def dynamo_stats():
    try:
        from torch._dynamo.utils import counters
        return dict(counters.get("stats", {}))
    except Exception:
        return {}


def run_mode(label, make_fn, iters=15, profile_steps=2, extra=None):
    """Execute one (workload, mode, batch) configuration and return a result dict.

    make_fn() -> callable step. Errors are recorded, not raised, so one failing mode
    does not stop the sweep (a failure is itself a compatibility/stability data point).
    """
    res = {"label": label}
    if extra:
        res.update(extra)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn = make_fn()
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
            sync()
        res["first_call_ms"] = (time.perf_counter() - t0) * 1e3
        with torch.no_grad():
            med, mn, mx = time_fn(fn, warmup=3, iters=iters)
        res.update({"step_ms_median": med, "step_ms_min": mn, "step_ms_max": mx})
        res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if profile_steps:
            try:
                with torch.no_grad():
                    res["kernels"] = profile_kernels(fn, steps=profile_steps)
            except Exception as e:  # profiling failure should not kill the measurement
                res["kernels"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        res["dynamo_stats"] = dynamo_stats()
        res["status"] = "ok"
    except Exception as e:
        res["status"] = f"error: {type(e).__name__}: {str(e)[:400]}"
    print(json.dumps({k: v for k, v in res.items() if k != "kernels"}, ensure_ascii=False), flush=True)
    return res


def save_results(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("saved", path, flush=True)
