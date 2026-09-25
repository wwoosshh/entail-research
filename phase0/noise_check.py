"""Measure timing noise on this GPU (e.g. while it also drives the monitor).

For three micro-workloads it reports the distribution of per-iteration GPU time (CUDA events) and host
wall time: median, p5, p95, p99, max, coefficient of variation. It also reports VRAM held by other
processes (desktop, apps) and GPU clock/temperature, and a 10 s sustained-GEMM run in 1 s windows to
expose boost-clock drift as the card heats up.

Run once with the monitor on this GPU and once after moving the monitor to the iGPU, then compare:
  python noise_check.py --tag display_on_dgpu
  python noise_check.py --tag display_on_igpu
"""
import argparse
import json
import os
import statistics
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def smi():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu,"
             "memory.used,memory.total,pstate", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return f"n/a ({type(e).__name__})"


def dist(ts):
    ts = sorted(ts)
    n = len(ts)
    mean = statistics.mean(ts)
    return {"median": statistics.median(ts), "min": ts[0], "p5": ts[int(0.05 * (n - 1))],
            "p95": ts[int(0.95 * (n - 1))], "p99": ts[int(0.99 * (n - 1))], "max": ts[-1],
            "cv": statistics.pstdev(ts) / mean if mean else None}


def measure(fn, n, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu, host = [], []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        fn()
        e.record()
        e.synchronize()
        host.append((time.perf_counter() - t0) * 1e3)
        gpu.append(s.elapsed_time(e))
    return {"gpu_ms": dist(gpu), "host_ms": dist(host)}


def line(name, r):
    g, h = r["gpu_ms"], r["host_ms"]
    print(f"  {name:32s} GPU median {g['median']:.3f} ms | p95/median {g['p95']/g['median']:.3f} | "
          f"p99/median {g['p99']/g['median']:.3f} | max/median {g['max']/g['median']:.2f} | CV {g['cv']*100:.2f}%   "
          f"host p99/median {h['p99']/h['median']:.2f} max/median {h['max']/h['median']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="display_on_dgpu")
    ap.add_argument("--warmup-s", type=float, default=0.0,
                    help="seconds of sustained GEMM before measuring, so boost clocks settle first")
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    res = {"tag": args.tag, "vram_total_gib": total / 2**30, "vram_free_gib_at_start": free / 2**30,
           "vram_used_by_others_gib": (total - free) / 2**30, "smi_before": smi()}
    print(f"VRAM total {total/2**30:.2f} GiB, free {free/2**30:.2f} GiB -> held by other processes "
          f"{(total-free)/2**30:.2f} GiB", flush=True)
    print("nvidia-smi before:", res["smi_before"], flush=True)

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    x = torch.empty(1 << 28, device="cuda", dtype=torch.uint8)
    y = torch.empty_like(x)
    t = torch.randn(1024, device="cuda")

    def tiny():
        for _ in range(200):
            t.add_(1.0)

    res["warmup_s"] = args.warmup_s
    if args.warmup_s > 0:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.warmup_s:
            for _ in range(20):
                a @ b
            torch.cuda.synchronize()
        res["smi_after_warmup"] = smi()
        print(f"sustained warm-up {args.warmup_s:.1f} s done; nvidia-smi: {res['smi_after_warmup']}", flush=True)

    print("\nper-iteration distributions (300 iterations each)", flush=True)
    res["gemm_4096_fp16"] = measure(lambda: a @ b, 300)
    line("GEMM 4096 fp16 (compute-bound)", res["gemm_4096_fp16"])
    res["copy_256MiB"] = measure(lambda: y.copy_(x), 300)
    line("copy 256 MiB (bandwidth-bound)", res["copy_256MiB"])
    res["tiny_x200"] = measure(tiny, 300)
    line("200 tiny kernels (launch-bound)", res["tiny_x200"])

    print("\nsustained GEMM, 10 x 1 s windows (clock drift as the card heats up)", flush=True)
    windows = []
    for w in range(10):
        n = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            for _ in range(20):
                a @ b
            torch.cuda.synchronize()
            n += 20
        el = time.perf_counter() - t0
        tf = 2 * 4096**3 * n / el / 1e12
        windows.append(tf)
        print(f"  window {w}: {tf:.1f} TFLOPS", flush=True)
    res["sustained_tflops_windows"] = windows
    res["sustained_drop_first_to_last"] = 1 - windows[-1] / windows[0]
    res["smi_after"] = smi()
    print(f"  first -> last window change: {-res['sustained_drop_first_to_last']*100:+.1f}%", flush=True)
    print("nvidia-smi after:", res["smi_after"], flush=True)

    out = os.path.join(HERE, "results", f"noise_check_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
