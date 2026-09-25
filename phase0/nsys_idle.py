"""Run a command under Nsight Systems and report GPU busy/idle over the steady-state window plus top kernels.

Cross-check for the torch.profiler-based idle estimate in common.py (nsys has far lower overhead).
Set PHASE0_NO_PROFILER=1 in the environment so the benchmark does not also start torch.profiler
(both use CUPTI).

Usage (inside WSL, venv active):
  PHASE0_NO_PROFILER=1 python nsys_idle.py [--tail-frac 0.5] [--name eager_b1] -- python bench_llm_decode.py --modes eager --batches 1
  python nsys_idle.py --analyze results/nsys/eager_b1.nsys-rep      # re-analyze an existing report

Reports are written to <phase0>/results/nsys/ (not /tmp, which WSL clears when the distro restarts).
"""
import argparse
import csv
import io
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def load_gpu_events(rep_path):
    stats = run(["nsys", "stats", "--report", "cuda_gpu_trace", "--format", "csv", "--force-export", "true", rep_path])
    if stats.returncode != 0:
        print(stats.stderr[-2000:])
        sys.exit("nsys stats failed")
    rows = list(csv.reader(io.StringIO(stats.stdout)))
    hdr_idx, cols = None, None
    for i, row in enumerate(rows):
        low = [c.strip().strip('"').lower() for c in row]
        if any(c.startswith("start") for c in low) and any(c.startswith("duration") for c in low) and any(c == "name" for c in low):
            hdr_idx, cols = i, low
            break
    if hdr_idx is None:
        print("\n".join(stats.stdout.splitlines()[:15]))
        sys.exit("could not find CSV header in nsys stats output (first lines printed above)")
    i_start = next(i for i, c in enumerate(cols) if c.startswith("start"))
    i_dur = next(i for i, c in enumerate(cols) if c.startswith("duration"))
    i_name = cols.index("name")
    evs = []
    for row in rows[hdr_idx + 1:]:
        if len(row) <= max(i_start, i_dur, i_name):
            continue
        try:
            s = int(float(row[i_start].replace(",", "")))
            d = int(float(row[i_dur].replace(",", "")))
        except ValueError:
            continue
        evs.append((s, s + d, row[i_name]))
    return evs


def analyze(evs, tail_frac, top):
    if not evs:
        sys.exit("no GPU events in trace")
    evs.sort()
    t0, t1 = evs[0][0], max(e for _, e, _ in evs)
    cut = t1 - (t1 - t0) * tail_frac
    tail = [e for e in evs if e[0] >= cut]
    busy = 0
    cs, ce = tail[0][0], tail[0][1]
    for s, e, _ in tail[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = tail[-1][1] - tail[0][0]
    agg = {}
    for s, e, n in tail:
        a = agg.setdefault(n, [0, 0])
        a[0] += 1
        a[1] += e - s
    print(f"\n=== nsys steady-state window (last {tail_frac:.0%} of GPU timeline) ===")
    print(f"events: {len(tail)}  span: {span/1e6:.1f} ms  GPU busy: {busy/1e6:.1f} ms  idle frac: {1-busy/span:.3f}")
    print(f"\nTop {top} kernels by time:")
    for n, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"  {d/1e6:9.2f} ms  {c:6d}x  {n[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail-frac", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--name", default=None, help="report base name (default: timestamp)")
    ap.add_argument("--analyze", default=None, help="existing .nsys-rep to analyze instead of profiling")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.analyze:
        analyze(load_gpu_events(args.analyze), args.tail_frac, args.top)
        return

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        sys.exit("give the command after --")
    out_dir = os.path.join(HERE, "results", "nsys")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, args.name or time.strftime("nsys_%Y%m%d_%H%M%S"))
    env = dict(os.environ, PHASE0_NO_PROFILER="1")
    prof = subprocess.run(["nsys", "profile", "-t", "cuda", "-o", base, "--force-overwrite", "true"] + cmd,
                          check=False, text=True, capture_output=True, env=env)
    print(prof.stdout[-1500:])
    if prof.returncode != 0:
        print(prof.stderr[-2000:])
        sys.exit(f"nsys profile failed with code {prof.returncode}")
    rep = base + ".nsys-rep"
    print("report:", rep)
    analyze(load_gpu_events(rep), args.tail_frac, args.top)


if __name__ == "__main__":
    main()
