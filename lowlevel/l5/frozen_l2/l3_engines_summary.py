"""M19 L3.3: what entail decided on healthy models and whether the engine's output moved (entail off against on).

  python lowlevel/l2/l3_engines_summary.py <out dir> [<config> ...]

Per config: the probes' generated tokens off against on (how many of the six are the same), the largest difference
of a chosen token's log-probability, the load time of each side, and the decisions entail recorded in the on run,
counted by boundary and verdict; every decision that is not a pass is printed with its note.
"""
import collections
import json
import os
import sys


def steps(res):
    plans = res.get("plans", {})
    return plans.get("alone", {})


def main():
    out = sys.argv[1]
    names = sys.argv[2:] or sorted({f.split(".")[0] for f in os.listdir(out) if f.endswith(".on.json")})
    summary = {}
    for name in names:
        try:
            off = json.load(open(os.path.join(out, f"{name}.off.json"), encoding="utf-8"))
            on = json.load(open(os.path.join(out, f"{name}.on.json"), encoding="utf-8"))
        except FileNotFoundError as e:
            print(name, "missing", e)
            continue
        a, b = steps(off), steps(on)
        same, worst = 0, 0.0
        for pid in a:
            ta, tb = a[pid]["tokens"], b.get(pid, {}).get("tokens")
            same += ta == tb
            if ta == tb:
                for sa, sb, t in zip(a[pid]["logprobs"], b[pid]["logprobs"], ta):
                    la, lb = sa.get(str(t)), sb.get(str(t))
                    if la is not None and lb is not None:
                        worst = max(worst, abs(la - lb))
        counts = collections.Counter()
        notes = []
        rec = os.path.join(out, f"{name}.on.record.jsonl")
        if os.path.exists(rec):
            for line in open(rec, encoding="utf-8"):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if "verdict" not in d:
                    continue
                counts[(d.get("boundary"), d.get("verdict"))] += 1
                if d.get("verdict") != "pass":
                    notes.append(f"{d.get('verdict')} {d.get('boundary')} {d.get('consumer')}: {(d.get('note') or '')[:300]}")
        summary[name] = {"same_tokens": f"{same}/{len(a)}", "max_chosen_logprob_diff": round(worst, 6),
                         "load_s": {"off": off.get("load_s"), "on": on.get("load_s")},
                         "decisions": {f"{k[0]} {k[1]}": v for k, v in sorted(counts.items(), key=str)}}
        print(f"== {name}: tokens {same}/{len(a)} same, max |logprob diff| {worst:.3g}, load off {off.get('load_s')} s"
              f" on {on.get('load_s')} s")
        for k, v in sorted(counts.items(), key=str):
            print(f"   {v:4d}  {k[0]}  {k[1]}")
        for n in notes[:40]:
            print("   -", n)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
