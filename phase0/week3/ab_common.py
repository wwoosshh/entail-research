"""Shared helpers for week-3 A/B measurements on a noisy (display-attached) GPU.

Interleaved rounds: each round runs k synchronized steps of every variant, the variant order rotates each round
(and reverses every other cycle), and per-round medians give paired comparisons. Slow drifts such as screen
activity or temperature then hit all variants alike instead of biasing whichever ran later.
"""
import json
import os
import statistics
import tempfile
import time

import torch


def sync():
    torch.cuda.synchronize()


def cuda_kernel_names(step):
    """Names of all CUDA kernels launched by one call of `step` (also inside CUDA-graph replays)."""
    from torch.profiler import ProfilerActivity, profile
    step()
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        sync()
    path = os.path.join(tempfile.gettempdir(), f"k_{os.getpid()}_{time.time_ns()}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    os.remove(path)
    return sorted({e["name"] for e in tr["traceEvents"] if e.get("cat") == "kernel"})


def interleaved(variants, rounds=24, k=12, warmup_s=3.0, baseline=None):
    """variants: {name: (setup_fn, step_fn)}. Returns per-variant step-time stats and paired per-round ratios."""
    names = list(variants)
    base = baseline or names[0]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        for n in names:
            setup, step = variants[n]
            setup()
            step()
        sync()
    per_step = {n: [] for n in names}
    per_round = {n: [] for n in names}
    for r in range(rounds):
        shift = r % len(names)
        order = names[shift:] + names[:shift]
        if (r // len(names)) % 2 == 1:
            order = order[::-1]
        for n in order:
            setup, step = variants[n]
            setup()
            ts = []
            for _ in range(k):
                t = time.perf_counter()
                step()
                sync()
                ts.append((time.perf_counter() - t) * 1e3)
            per_step[n].extend(ts)
            per_round[n].append(statistics.median(ts))
    out = {}
    for n in names:
        s = sorted(per_step[n])
        rec = {"median_ms": statistics.median(s), "p10_ms": s[int(0.1 * (len(s) - 1))],
               "p90_ms": s[int(0.9 * (len(s) - 1))], "steps": len(s), "round_medians_ms": per_round[n]}
        if n != base:
            ratios = sorted(a / b for a, b in zip(per_round[n], per_round[base]))
            rec.update({"ratio_vs_base_median": statistics.median(ratios),
                        "ratio_vs_base_p10": ratios[int(0.1 * (len(ratios) - 1))],
                        "ratio_vs_base_p90": ratios[int(0.9 * (len(ratios) - 1))],
                        "rounds_faster_than_base": sum(x < 1 for x in ratios) / len(ratios)})
        out[n] = rec
    return {"baseline": base, "rounds": rounds, "steps_per_round": k, "warmup_s": warmup_s, "variants": out}


def op_interleaved(fns, rounds=20, k=20):
    """Microbenchmark for short GPU ops: CUDA-event time of k back-to-back calls, alternating order per round."""
    names = list(fns)
    for n in names:
        for _ in range(10):
            fns[n]()
    sync()
    samples = {n: [] for n in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(k):
                fns[n]()
            e.record()
            e.synchronize()
            samples[n].append(s.elapsed_time(e) / k)
    return {n: statistics.median(v) for n, v in samples.items()}


def print_timing(title, t):
    base = t["baseline"]
    print(f"  {title}: interleaved {t['rounds']} rounds x {t['steps_per_round']} steps, baseline {base}", flush=True)
    for n, r in t["variants"].items():
        extra = ""
        if n != base:
            extra = (f" | vs {base}: x{r['ratio_vs_base_median']:.3f} "
                     f"[p10 {r['ratio_vs_base_p10']:.3f}, p90 {r['ratio_vs_base_p90']:.3f}] "
                     f"faster in {r['rounds_faster_than_base']*100:.0f}% of rounds")
        print(f"    {n:16s} median {r['median_ms']:7.2f} ms (p10 {r['p10_ms']:.2f}, p90 {r['p90_ms']:.2f}){extra}",
              flush=True)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
