"""M16.5 results (testbed/M16_PROTOCOL.md 5): joins the screening passes, the case runs (results/m16/cases), the
blind ratings (results/m16/ratings.json) and the hand-written verdicts (results/m16/verdicts.json: case ->
{"issue", "verdict", "note"}; verdict in resolved/reported/unknown_only/missed/false/not_reproduced) into
results/m16/results.json and prints the headline numbers: reproduced, verdict counts, the detection rate over the
reproduced consensus-K1 cases (and per rater), the class share among the rated reports, and the false count.
Run: python testbed/m16_results.py
The fourth replay (M16_PROTOCOL 9) adds, from the same rows: the low-level detection rate (reproduced consensus-K1
cases whose consensus level is D), "located" (a right-place verdict on a reproduced case outside the class, K2..K5:
not a false alarm and not a detection), and the share caught at the expected boundary over all reproduced cases.
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("M16_OUT") or os.path.join(HERE, "results", "m16")   # M17.6: results/m17/replay2
CASES = os.path.join(OUT, "cases")
DETECTED = {"resolved", "reported"}


def load(path, default=None):
    return json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else default


def record_counts(case):
    p = os.path.join(CASES, f"{case}_on.record.jsonl")
    c = Counter()
    if os.path.isfile(p):
        for line in open(p, encoding="utf-8"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "verdict" in d:
                c[d["verdict"]] += 1
    return dict(c)


def main():
    screening = load(os.path.join(OUT, "screening.json"))
    ratings = load(os.path.join(OUT, "ratings.json"), {"issues": {}})
    verdicts = load(os.path.join(OUT, "verdicts.json"), {})
    by_issue = {v["number"]: (u, v) for u, v in screening.items()}
    rows = []
    for case, vd in sorted(verdicts.items(), key=lambda kv: by_issue[kv[1]["issue"]][1]["position"]):
        url, s = by_issue[vd["issue"]]
        run = vd.get("run", case)          # the run that settled the case (a reported-version venv when one was needed)
        off, on = load(os.path.join(CASES, f"{run}_off.json"), {}), load(os.path.join(CASES, f"{run}_on.json"), {})
        r = ratings["issues"].get(url.rstrip("/"), {})
        rows.append({"position": s["position"], "issue": f"{s['repo'].split('/')[1]}#{s['number']}", "case": case,
                     "run": run, "engine_version": off.get("vllm") or off.get("sglang") or off.get("transformers")
                     or off.get("diffusers"), "title": s["title"][:100], "version": s.get("version"),
                     "expected_boundary": s.get("expected_boundary"),
                     # a verdict may settle reproduction itself when the case's own criterion was superseded by a
                     # control run after the fact (M18.6 replay 3, vl43602: disclosed in the verdict's note)
                     "reproduced_off": vd.get("reproduced", off.get("reproduced")), "reproduced_on": on.get("reproduced"),
                     "reproduced_by": "verdict" if "reproduced" in vd else "case",
                     "entail_decisions_on": record_counts(run), "rating_A": (r.get("A") or {}).get("k"),
                     "rating_B": (r.get("B") or {}).get("k"), "rating_C": (r.get("C") or {}).get("k"),
                     "consensus": r.get("consensus"), "level": r.get("level"), "verdict": vd["verdict"],
                     "note": vd.get("note", "")})
    reproduced = [x for x in rows if x["reproduced_off"]]
    k1 = [x for x in reproduced if x["consensus"] == "K1"]
    det = [x for x in k1 if x["verdict"] in DETECTED]
    per_rater = {}
    for rater in ("rating_A", "rating_B"):
        den = [x for x in reproduced if x[rater] == "K1"]
        per_rater[rater] = {"denominator": len(den), "detected": sum(1 for x in den if x["verdict"] in DETECTED)}
    low = [x for x in k1 if x["level"] == "D"]
    outside = [x for x in reproduced if x["consensus"] in ("K2", "K3", "K4", "K5")]
    rated = [v for v in ratings["issues"].values()]
    share = Counter(v.get("consensus") for v in rated)
    out = {"passes": len(rows), "reproduced": len(reproduced), "verdicts": dict(Counter(x["verdict"] for x in rows)),
           "detection": {"numerator": len(det), "denominator_reproduced_consensus_K1": len(k1),
                         "rate": (len(det) / len(k1)) if k1 else None, "per_rater": per_rater},
           "low_level_detection": {"numerator": sum(1 for x in low if x["verdict"] in DETECTED),
                                   "denominator_reproduced_consensus_K1_level_D": len(low)},
           "located_outside_class": {"cases": [x["case"] for x in outside if x["verdict"] in DETECTED],
                                     "reproduced_K2_to_K5": len(outside)},
           "caught_at_expected_boundary_all_reproduced": {
               "numerator": sum(1 for x in reproduced if x["verdict"] in DETECTED), "denominator": len(reproduced)},
           "false_alarms": {"cases_with_false": [x["case"] for x in rows if x["verdict"] == "false"],
                            "broken_or_refused_in_non_K1_reproduced": sum(
                                (x["entail_decisions_on"].get("broken", 0) + x["entail_decisions_on"].get("refused", 0))
                                for x in reproduced if x["consensus"] != "K1")},
           "class_share_among_rated": {"rated": len(rated), "consensus_counts": dict(share),
                                       "K1_share": (share.get("K1", 0) / len(rated)) if rated else None,
                                       "A_K1_share": ratings.get("counts_A", {}).get("K1", 0) / len(rated) if rated else None,
                                       "B_K1_share": ratings.get("counts_B", {}).get("K1", 0) / len(rated) if rated else None},
           "kappa_7": ratings.get("kappa_7"), "kappa_k1_vs_rest": ratings.get("kappa_k1_vs_rest"), "rows": rows}
    json.dump(out, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))
    for x in rows:
        print(f"{x['issue']:<22} {x['case']:<10} repro={x['reproduced_off']!s:<5} K={x['consensus']!s:<5} "
              f"L={x['level']!s:<5} {x['verdict']:<14} on={x['entail_decisions_on']}")


if __name__ == "__main__":
    main()
