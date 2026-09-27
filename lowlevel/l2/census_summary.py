"""M19 L3.3d: rank the Triton kernels normal runs reach (results of run_census.sh) by how many engine configurations
reach them, and say which engine function launched each and whether entail already holds it to a definition.

  python lowlevel/l2/census_summary.py <census dir>

Covered means: the launching function is one entail/definitions.py defines, or the kernel is launched from a vLLM
custom op's dispatched path (the op carries its own native definition). The ranking says nothing about defects.
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "entail"))


def main():
    d = sys.argv[1]
    from entail import definitions

    defined = {t.replace(":", ".").rsplit(".", 1)[-1] for t in (x.target for x in definitions.DEFINITIONS)}
    per_kernel = collections.defaultdict(lambda: {"configs": set(), "launches": 0, "callers": set()})
    configs = set()
    for f in glob.glob(os.path.join(d, "*.*.json")):
        base = os.path.basename(f)
        if base.endswith((".out.json", ".spec.json")):
            continue
        cfg = base.rsplit(".", 2)[0]          # <config>.<pid>.json
        r = json.load(open(f, encoding="utf-8"))
        if not r.get("counts"):
            continue
        configs.add(cfg)
        for k, n in r["counts"].items():
            e = per_kernel[k]
            e["configs"].add(cfg)
            e["launches"] += n
            if k in r.get("callers", {}):
                e["callers"].add(r["callers"][k])
    rows = []
    for k, e in per_kernel.items():
        callers = sorted(e["callers"])
        covered = any(c.rsplit(":", 1)[-1] in defined for c in callers) or \
            any(("layernorm" in c or "rotary" in c or "activation" in c) and "forward_cuda" in c for c in callers)
        rows.append({"kernel": k, "configs": len(e["configs"]), "launches": e["launches"], "callers": callers,
                     "covered": covered, "engines": sorted({c.split("_", 1)[0] for c in e["configs"]})})
    rows.sort(key=lambda r: (-r["configs"], -r["launches"]))
    json.dump({"configs": sorted(configs), "kernels": rows}, open(os.path.join(d, "summary.json"), "w",
                                                                   encoding="utf-8"), indent=1)
    print(f"{len(configs)} configurations: {', '.join(sorted(configs))}")
    for r in rows[:60]:
        print(f"{r['configs']:3d} cfg {r['launches']:8d} launches {'COVERED ' if r['covered'] else '        '}"
              f"{r['kernel'][:70]:70s} <- {'; '.join(r['callers'])[:110]}")


if __name__ == "__main__":
    main()
