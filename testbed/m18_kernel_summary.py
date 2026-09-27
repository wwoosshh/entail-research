"""M18.2 summary: every KernelReference decision in the given record files (entail on), one row per (run, op):
verdict, max |kernel - definition|, the definition's own noise (floor), scale, allowed tolerance, and the ratio
diff / floor - the distribution on healthy runs sets FACTOR and ATOL_ULPS (kernel_reference_contract.py).
Usage: python testbed/m18_kernel_summary.py <out.json> <record.jsonl>...
"""
import json
import os
import re
import sys

NUM = re.compile(r"max \|kernel - definition\| ([0-9.e+-]+) over (\d+) values(?:; the definition's own noise ([0-9.e+-]+)"
                 r"(?: \(typical ([0-9.e+-]+)\))?)?.*?scale ([0-9.e+-]+); (?:allowed ([0-9.e+-]+)|(\d+) values beyond"
                 r".*?worst ratio to the allowance ([0-9.e+-]+))")


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    rows = []
    for p in paths:
        run = os.path.basename(p).replace(".record.jsonl", "").replace("_on", "")
        for line in open(p, encoding="utf-8"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("name") != "KernelReference" or "verdict" not in d:
                continue
            note = d.get("note") or ""
            m = NUM.search(note)
            row = {"run": run, "consumer": d.get("consumer") or (d.get("contract") or {}).get("consumer"),
                   "verdict": d["verdict"], "note": note[:300]}
            if m:
                diff, n, floor, typical, scale, tol, beyond, margin = m.groups()
                row.update(diff=float(diff), values=int(n), floor=(float(floor) if floor else None), scale=float(scale),
                           typical=(float(typical) if typical else None), allowed=(float(tol) if tol else None),
                           beyond=(int(beyond) if beyond else None), margin=(float(margin) if margin else None))
                row["ratio"] = (row["diff"] / row["floor"]) if row["floor"] else None
                row["rel"] = row["diff"] / row["scale"] if row["scale"] else None
                row["bitwise"] = "reproduces the definition bitwise" in note
                row["definition_path"] = "calls the definition itself" in note
            rows.append(row)
    by_verdict = {}
    for r in rows:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
    ratios = sorted(r["ratio"] for r in rows if r.get("ratio") is not None)
    rels = sorted(r["rel"] for r in rows if r.get("rel") is not None)
    margins = sorted(r["margin"] for r in rows if r.get("margin") is not None and not r.get("definition_path"))
    summary = {"records": len(paths), "decisions": len(rows), "by_verdict": by_verdict,
               "bitwise_definition": sum(1 for r in rows if r.get("bitwise")),
               "definition_path": sum(1 for r in rows if r.get("definition_path")),
               "ratio_diff_over_floor": {"n": len(ratios), "median": ratios[len(ratios) // 2] if ratios else None,
                                         "p90": ratios[int(0.9 * len(ratios)) - 1] if len(ratios) >= 10 else None,
                                         "max": ratios[-1] if ratios else None},
               "rel_diff_over_scale": {"n": len(rels), "median": rels[len(rels) // 2] if rels else None,
                                       "max": rels[-1] if rels else None},
               # elementwise: the worst value's ratio to its allowance (<= 1 passes); sets FACTOR and ATOL_ULPS from data
               "margin_to_allowance": {"n": len(margins), "median": margins[len(margins) // 2] if margins else None,
                                       "p90": margins[int(0.9 * len(margins)) - 1] if len(margins) >= 10 else None,
                                       "max": margins[-1] if margins else None},
               "rows": rows}
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in rows:
        print(f"{r['run']:<34} {str(r['consumer']):<34} {r['verdict']:<8} diff={r.get('diff', '-')!s:<10} "
              f"floor={r.get('floor', '-')!s:<10} scale={r.get('scale', '-')!s:<8} allowed={r.get('allowed', '-')!s:<10} "
              f"ratio={('%.2f' % r['ratio']) if r.get('ratio') else '-'}")
    print("RESULT", json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
