"""Combine the two role-class replay tables (50 bugs) into approach coverage and fact distribution."""
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["replay_vllm_sglang.md", "replay_transformers_llamacpp.md"]
APPROACHES = ["S1", "S2", "S3", "S4", "S5"]


def parse_cell(c):
    c = c.replace("\\", "").strip()
    mark = next((ch for ch in c if ch in "PDT–-"), "–")
    return {"mark": "–" if mark in "–-" else mark, "star": "*" in c, "deg": "°" in c, "hw": "hw" in c}


def main():
    rows = []
    for f in FILES:
        for line in open(os.path.join(HERE, f), encoding="utf-8"):
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 11 or not re.search(r"\d{4,6}", cells[1]):
                continue
            fact = re.sub(r"\s*\(.*", "", cells[3]).strip()
            r = {"repo": cells[0], "issue": re.search(r"\d{4,6}", cells[1]).group(0), "fact": fact,
                 "trigger": cells[4], "new_logic": cells[10].startswith(("yes", "예"))}
            for i, a in enumerate(APPROACHES):
                r[a] = parse_cell(cells[5 + i])
            rows.append(r)
    n = len(rows)
    tot = {a: Counter(r[a]["mark"] for r in rows) for a in APPROACHES}
    strict = lambda r, a: r[a]["mark"] in "PD" and not r[a]["star"] and not r[a]["deg"]
    res = {
        "n": n,
        "per_approach": {a: dict(tot[a]) for a in APPROACHES},
        "per_approach_P_or_D_strict": {a: sum(strict(r, a) for r in rows) for a in APPROACHES},
        "S3_T_without_rare_hardware": sum(r["S3"]["mark"] == "T" and not r["S3"]["hw"] for r in rows),
        "union_P_or_D": sum(any(r[a]["mark"] in "PD" for a in APPROACHES) for r in rows),
        "union_P_or_D_strict": sum(any(strict(r, a) for a in APPROACHES) for r in rows),
        "union_P_only": sum(any(r[a]["mark"] == "P" for a in APPROACHES) for r in rows),
        "S1_S2_S4_union_strict": sum(any(strict(r, a) for a in ("S1", "S2", "S4")) for r in rows),
        "S2_S4_union_strict (no static types)": sum(any(strict(r, a) for a in ("S2", "S4")) for r in rows),
        "only_T_or_nothing": [f"{r['repo']}#{r['issue']}" for r in rows
                              if not any(r[a]["mark"] in "PD" for a in APPROACHES)],
        "new_logic_needed": sum(r["new_logic"] for r in rows),
        "facts": dict(Counter(r["fact"] for r in rows).most_common()),
    }
    print(json.dumps(res, indent=1, ensure_ascii=False))
    json.dump(res, open(os.path.join(HERE, "replay_totals.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
