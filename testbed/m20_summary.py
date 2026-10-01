"""M20 (testbed/M20_PROTOCOL.md 5): the numbers from results/m20/screening.json - rule 1 and the session's preliminary
K and V ratings (M20.2; the session knows the hypothesis, so these are preliminary). Wilson 95% intervals.
Writes results/m20/session_summary.json. Run: python testbed/m20_summary.py
"""
import json
import math
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "m20")
ENGINE = {"ollama/ollama": "Ollama", "ggml-org/llama.cpp": "llama.cpp", "lmstudio-ai/lmstudio-bug-tracker": "LM Studio"}
EARLIER = {"M16 (1.1.0 frozen)": (29, 87), "3rd replay": (36, 96), "4th replay": (25, 91)}   # consensus K1 / rule-1 passes


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def share(k, n):
    return {"k": k, "n": n, "share": round(k / n, 3) if n else None, "wilson95": wilson(k, n)}


def main():
    rows = list(json.load(open(os.path.join(OUT, "screening.json"), encoding="utf-8")).values())
    rows.sort(key=lambda r: r["position"])
    out = {"screened": len(rows), "positions": [rows[0]["position"], rows[-1]["position"]],
           "rule1": dict(Counter(r["decision"] for r in rows)), "by_engine": {}, "preliminary": True}
    groups = {"all": rows}
    for repo, name in ENGINE.items():
        groups[name] = [r for r in rows if r["repo"] == repo]
    for name, rs in groups.items():
        passed = [r for r in rs if r["decision"] == "pass"]
        n = len(passed)
        k1 = [r for r in passed if r.get("K") == "K1"]
        out["by_engine"][name] = {
            "screened": len(rs), "output_reports": n,
            "K": dict(sorted(Counter(r.get("K") for r in passed).items())),
            "V": dict(sorted(Counter(r.get("V") for r in passed).items())),
            "K1": share(len(k1), n),
            "K1_and_V1": share(sum(1 for r in k1 if r.get("V") == "V1"), n),
            "K1_and_V2": share(sum(1 for r in k1 if r.get("V") == "V2"), n),
            "K1_and_V3": share(sum(1 for r in k1 if r.get("V") == "V3"), n),
            "any_V1": share(sum(1 for r in passed if r.get("V") == "V1"), n),
        }
    out["earlier_class_shares_consensus"] = {k: share(*v) for k, v in EARLIER.items()}
    out["K1_items"] = [{"position": r["position"], "issue": f"{r['repo'].split('/')[1]}#{r['number']}", "V": r.get("V"),
                        "Kc": r.get("Kc"), "lost": r.get("lost")} for r in rows if r.get("K") == "K1"]
    out["V1_items"] = [{"position": r["position"], "issue": f"{r['repo'].split('/')[1]}#{r['number']}", "K": r.get("K"),
                        "evidence": r.get("evidence", "")[:160]} for r in rows if r.get("V") == "V1"]
    out["kept_aside_not_output"] = [{"position": r["position"], "issue": f"{r['repo'].split('/')[1]}#{r['number']}",
                                     "reason": r["reason"]} for r in rows if "Kept aside" in r.get("reason", "")]
    with open(os.path.join(OUT, "session_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("K1_items", "V1_items")}, ensure_ascii=False, indent=1))
    print("K1 items:")
    for x in out["K1_items"]:
        print(" ", x)
    print("V1 items:")
    for x in out["V1_items"]:
        print(" ", x["issue"], x["K"])


if __name__ == "__main__":
    main()
