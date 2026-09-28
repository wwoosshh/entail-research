"""Scores the pre-registered S9 (b) replication (testbed/results/p3/PREREG_S9b2.md) by the rules written there before
the run, and writes testbed/results/p3/prereg_s9b2.json.

  python testbed/p3_prereg_score.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "p3")
EXPECT = {"inside_rope": "inside", "outside_kvwrite": "outside", "inside_kernel": "inside", "outside": "outside"}


def load(name):
    return json.load(open(os.path.join(OUT, name + ".json"), encoding="utf-8"))


def main():
    rows = []
    for plant, expected in EXPECT.items():
        off, alls = load(f"s9h_{plant}_1off"), load(f"s9h_{plant}_2all")
        off_dis = [p["paths"] for p in off["pairs"] if p["differs"]]
        all_dis = [p["paths"] for p in alls["pairs"] if p["differs"]]
        seen = bool(off_dis) and off["planted_calls"] > 0
        said = " ".join(alls["said"])
        if not all_dis and "the cause is inside them" in said:
            told = "inside"
        elif all_dis and "the cause is outside them" in said:
            told = "outside"
        else:
            told = "unclear"
        rows.append({"fault": plant, "expected": expected, "seen": seen, "off_disagree": off_dis,
                     "off_pairs": {p["paths"]: [p["margin"], p["pdrift"]] for p in off["pairs"]},
                     "planted_calls": {"off": off["planted_calls"], "all": alls["planted_calls"]},
                     "all_disagree": all_dis, "said": alls["said"], "told": told if seen else None,
                     "result": ("right" if told == expected else "wrong" if told in ("inside", "outside")
                                else "unclear") if seen else "not applicable",
                     "answers": {"off": off["answer"], "all": alls["answer"]}})
    seen = [r for r in rows if r["seen"]]
    summary = {"rules": "PREREG_S9b2.md", "right": f"{sum(r['result'] == 'right' for r in seen)}/{len(seen)}",
               "not_applicable": [r["fault"] for r in rows if not r["seen"]], "rows": rows}
    with open(os.path.join(OUT, "prereg_s9b2.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    for r in rows:
        print(f"{r['fault']:16} expected {r['expected']:7} seen {str(r['seen']):5} told {str(r['told']):8} "
              f"-> {r['result']:14} calls {r['planted_calls']} off {r['off_pairs']}")
    print("right", summary["right"], "not applicable", summary["not_applicable"])


if __name__ == "__main__":
    main()
