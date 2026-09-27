"""M17.5: break the E2 run on the v8 commit down against the 1.1.0 final run (M15, results/m15/e2_final3): unknown
lines by boundary and rule, the excluded runs, resolved decisions, and the load share's median/p90/max.
Run: python testbed/m17_e2_breakdown.py [new_dir] [old_dir]"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "m17", "e2")
OLD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "m15", "e2_final3")


def decisions(folder):
    out = collections.Counter()
    per_run = {}
    for path in sorted(glob.glob(os.path.join(folder, "engines", "*_load.record.jsonl"))):
        run = os.path.basename(path)[:-len("_load.record.jsonl")]
        c = collections.Counter()
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if "verdict" in o:
                key = (o["verdict"], o["boundary"], (o.get("rule") or "")[:60])
                c[key] += 1
        per_run[run] = c
        out.update(c)
    return out, per_run


def summary(folder):
    path = os.path.join(folder, "summary.json")
    return json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}


def main():
    new, new_runs = decisions(NEW)
    old, old_runs = decisions(OLD)
    print(f"new {NEW}: runs with records {len(new_runs)}; old {OLD}: {len(old_runs)}")
    for verdict in ("broken", "refused", "resolved", "unknown"):
        n_new = sum(v for k, v in new.items() if k[0] == verdict)
        n_old = sum(v for k, v in old.items() if k[0] == verdict)
        print(f"{verdict:9s} new {n_new:4d} old {n_old:4d}")
    print("\nunknown by boundary (new vs old):")
    keys = sorted({k for k in list(new) + list(old) if k[0] == "unknown"}, key=lambda k: (k[1], k[2]))
    by_b = collections.defaultdict(lambda: [0, 0])
    for k in keys:
        by_b[k[1]][0] += new.get(k, 0)
        by_b[k[1]][1] += old.get(k, 0)
    for b, (a, o) in sorted(by_b.items()):
        flag = "" if a == o else "   <-- changed"
        print(f"  {b:55s} new {a:3d} old {o:3d}{flag}")
    print("\nresolved (new):")
    for k, v in sorted(new.items()):
        if k[0] == "resolved":
            print(f"  {k[1]:55s} {v}")
    runs_changed = [r for r in sorted(set(new_runs) | set(old_runs))
                    if sum(v for k, v in new_runs.get(r, {}).items() if k[0] == "unknown")
                    != sum(v for k, v in old_runs.get(r, {}).items() if k[0] == "unknown")]
    print("\nruns whose unknown count changed:", len(runs_changed))
    for r in runs_changed[:20]:
        a = {k[1]: v for k, v in new_runs.get(r, {}).items() if k[0] == "unknown"}
        o = {k[1]: v for k, v in old_runs.get(r, {}).items() if k[0] == "unknown"}
        print(f"  {r}: new {a} old {o}")
    s = summary(NEW)
    rows = s.get("runs") or s.get("rows") or []
    shares = sorted(r.get("entail_share") or r.get("share") for r in rows if isinstance(r, dict)
                    and (r.get("entail_share") or r.get("share")) is not None) if rows else []
    if shares:
        print(f"\nload share n {len(shares)} median {shares[len(shares)//2]:.4f} p90 {shares[int(len(shares)*0.9)]:.4f} "
              f"max {shares[-1]:.4f}")
    else:
        print("\nsummary.json keys:", list(s)[:12])
    excluded = [r for r in rows if isinstance(r, dict) and r.get("valid") is False] if rows else []
    if excluded:
        print("excluded runs:", [(r.get("model"), r.get("engine"), (r.get("why") or r.get("reason") or "")[:60]) for r in excluded])


if __name__ == "__main__":
    main()
