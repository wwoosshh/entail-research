"""M17.5: summarise the M16 cases run again on the v8 commit (testbed/results/m17/m16_rerun/cases): per case the
reproduction line, entail's non-pass decisions with entail on (boundary, verdict, rule), and the two yardsticks -
false alarms outside the class (any broken/refused) and site coverage inside it (a decision at the expected
boundary). Writes testbed/results/m17/m16_rerun/summary.json and prints a table.
Run: python testbed/m17_m16_rerun_summary.py"""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "m17", "m16_rerun")
CASES = os.path.join(R, "cases")
OUTSIDE = ["tf47885", "tf48293", "df13411", "tf46032", "tf46710", "vl49316", "vl49412", "vl48231"]
INSIDE = {   # case -> the boundary (prefix) where M16 said the defect sits
    "sg40835": "load:sglang.adapter_config",
    "sg21843": "kernel:sglang.fused_gdn_gating",
    "tf46612": "request:transformers.generate.beam_reorder",
    "vl56655": "container:vllm.request.block_hashes",
    "vl48895": "kernel:",
    "vl43728_0220": "request:vllm.parser_settings",
    "vl42016_0220": "load:vllm.rotary_pairing",
}
# M16_PROTOCOL 7, pre-check (2): a case whose adapter is not installed in the reported version's environment is
# counted apart, not as a miss. vl43728 on vLLM 0.22.0: the serve adapter wraps ParserManager.get_parser (0.30);
# 0.22.0 has ReasoningParserManager and the case builds the parser in-process, so the site is never reached (the
# on-run log has no [entail] line at all; the rule itself resolves the case on the 0.22.0 row: tests/test_request_settings.py).
ADAPTER_NOT_INSTALLED = {"vl43728_0220": "vllm_serve targets vLLM 0.30's ParserManager; 0.22.0 has ReasoningParserManager"}


def record(case):
    path = os.path.join(CASES, f"{case}_on.record.jsonl")
    rows = []
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def result_line(case, which):
    path = os.path.join(CASES, f"{case}_{which}.log")
    if not os.path.isfile(path):
        return None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("RESULT"):
            return line.strip()[7:300]
    return "no RESULT line"


def main():
    out = {"cases": {}, "outside_false_alarms": 0, "inside_site_coverage": {}, "inside_verdicts": {}}
    for case in OUTSIDE + list(INSIDE):
        rows = record(case)
        decisions = [r for r in rows if "verdict" in r]
        by = collections.Counter((r["boundary"], r["verdict"]) for r in decisions)
        non_pass = {f"{b} {v}": n for (b, v), n in sorted(by.items()) if v != "pass"}
        broken = sum(n for (b, v), n in by.items() if v in ("broken", "refused"))
        row = {"off": result_line(case, "off"), "on": result_line(case, "on"), "decisions": len(decisions),
               "non_pass": non_pass, "broken_or_refused": broken,
               "boundaries": sorted({r["boundary"] for r in decisions})}
        if case in OUTSIDE:
            out["outside_false_alarms"] += broken
        else:
            at = [r for r in decisions if r["boundary"].startswith(INSIDE[case])]
            verdicts = collections.Counter(r["verdict"] for r in at)
            covered = bool(at)
            if any(v in ("resolved",) for v in verdicts):
                verdict = "resolved"
            elif any(v in ("broken", "refused") for v in verdicts):
                verdict = "reported"
            elif "unknown" in verdicts:
                verdict = "unknown_only"
            elif case in ADAPTER_NOT_INSTALLED:
                verdict = "adapter_not_installed"
            else:
                verdict = "missed"
            out["inside_site_coverage"][case] = covered
            out["inside_verdicts"][case] = verdict
            row["at_expected_boundary"] = dict(verdicts)
        out["cases"][case] = row
    n_in = len(INSIDE)
    out["site_coverage"] = f"{sum(out['inside_site_coverage'].values())}/{n_in}"
    out["retro"] = collections.Counter(out["inside_verdicts"].values())
    with open(os.path.join(R, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"outside the class: broken/refused {out['outside_false_alarms']} over {len(OUTSIDE)} cases")
    print(f"inside the class: site coverage {out['site_coverage']}, retro {dict(out['retro'])}")
    for case, row in out["cases"].items():
        print(f"{case:14s} decisions {row['decisions']:3d} non-pass {row['non_pass']} "
              f"{('at boundary ' + str(row.get('at_expected_boundary'))) if case in INSIDE else ''}")


if __name__ == "__main__":
    main()
