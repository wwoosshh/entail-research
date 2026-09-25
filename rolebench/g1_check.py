"""G1 check (EXECUTION_PLAN.md section 3.1): can the benchmark measure both directions? It picks no direction.

  breadth: at least 12 reproduced cases covering at least 6 fact kinds
  depth:   at least 4 reproduced real-engine cases
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rep, kinds, real, missing = [], set(), [], []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        if "meta" not in r:  # skip summary files such as this script's own output
            continue
        if r.get("reproduced"):
            rep.append(r["case"])
            kinds.add(r["meta"]["fact"])
            if r["meta"].get("kind") == "real-engine":
                real.append(r["case"])
        else:
            missing.append(r["case"])
    ok = len(rep) >= 12 and len(kinds) >= 6 and len(real) >= 4
    out = {"reproduced": len(rep), "fact_kinds": sorted(kinds), "real_engine": real, "not_reproduced": missing,
           "G1_measurable": ok}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "g1_check.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)


if __name__ == "__main__":
    main()
