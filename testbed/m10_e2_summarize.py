"""M10 E2: testbed/results/m10/E2_SUMMARY.md and e2/summary.json from the healthy runs m10_e2_run.sh wrote
(results/m10/e2/engines: e2_<engine>_<model>_<off|load>.json, .record.jsonl, .log). Values are read from those files.
Definitions: testbed/M10_PROTOCOL.md 2.
Run: python testbed/m10_e2_summarize.py
"""
import glob
import json
import os
import re
import statistics
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("M10_E2_DIR") or os.path.join(HERE, "results", "m10", "e2")   # M11.7: results/m11/e2 for 1.0.1
E = os.path.join(R, "engines")
ENGINES = ("transformers", "vllm", "sglang")


def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def records(p):
    if not os.path.exists(p):
        return []
    out = []
    for x in open(p, encoding="utf-8"):
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


CAPS = {}


def caps_rows():
    if not CAPS:
        p = os.path.join(os.path.dirname(HERE), "entail", "entail", "data", "caps.json")
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            CAPS[(r["consumer"], r["fact"])] = r["evidence"]
    return CAPS


def props(value):
    m = re.search(r"ModelProps\(([^)]*)\)", str(value or ""))
    return dict(kv.split("=", 1) for kv in (m.group(1).split(", ") if m else []) if "=" in kv)


# A repair that is a rule of the core, not a route through the capability table (M14, M15): its evidence is the
# test problem it was measured on, not a caps row (protocol 2.3, changed 2026-09-26; M10_PROTOCOL.md 6)
RULE_REPAIRS = {("identity_recompute", "vllm"): "testbed/results/r4/e2e_on.json (vllm#49377: the stale block hash "
                                                "recomputed, the wrong 16-token cache hit gone)",
                ("clamp_tile_k", "sglang"): "testbed/results/m15/e3_39626_on.json (sglang#39626: the tile clamped, 288 "
                                            "where 64 was)",
                ("add_stops", "transformers"): "testbed/results/m15/stops_replay_on.json, stops_nemotron_on.json (the "
                                               "declared end added at load; generation stops at the end where it ran "
                                               "to 160 tokens without)"}


def backing(d):
    """A resolved decision against the capability table (protocol 2.3): the fields the declaration states and the
    consumer drops, each with the evidence of its row; backed when a measured row says the consumer drops one. A
    rule repair (RULE_REPAIRS) is backed by the measurement of its test problem."""
    engine = (d.get("consumer") or "").split(".")[0]
    if d.get("handle") is not None and (d.get("handle"), engine) in RULE_REPAIRS:
        return {"consumer": d.get("consumer"), "dropped": {d.get("name"): "rule repair"},
                "backed_by_measured_row": True, "rule_repair_evidence": RULE_REPAIRS[(d["handle"], engine)]}
    if d.get("handle") in {h for h, _ in RULE_REPAIRS}:
        return {"consumer": d.get("consumer"), "dropped": {d.get("name"): "rule repair"},
                "backed_by_measured_row": False,
                "rule_repair_evidence": f"no measurement of this repair on {engine} (M10_PROTOCOL.md 6, 2026-09-26)"}
    dec, cho = props((d.get("declared") or {}).get("value")), props((d.get("chosen") or {}).get("value"))
    dropped = [k for k, v in dec.items() if v not in ("None", None) and cho.get(k) in ("None", None)]
    ev = {k: caps_rows().get((d.get("consumer"), f"ModelProps.{k}")) for k in dropped}
    return {"consumer": d.get("consumer"), "dropped": ev,
            "backed_by_measured_row": any(v == "measured" for v in ev.values())}


def main():
    names = sorted({re.match(r"e2_(transformers|vllm|sglang)_(.+)_(off|load)\.json$", os.path.basename(p)).group(2)
                    for p in glob.glob(os.path.join(E, "e2_*_*.json"))
                    if re.match(r"e2_(transformers|vllm|sglang)_(.+)_(off|load)\.json$", os.path.basename(p))})
    runs = []
    for m in names:
        for e in ENGINES:
            off, on = jload(os.path.join(E, f"e2_{e}_{m}_off.json")), jload(os.path.join(E, f"e2_{e}_{m}_load.json"))
            if off is None or on is None:   # not run yet, or still running (a timeout still writes no json: see
                continue                    # run.log, "no RESULT line")
            rec = records(os.path.join(E, f"e2_{e}_{m}_load.record.jsonl"))
            dec = [r for r in rec if "verdict" in r]
            ms = sum(r.get("ms", 0) for r in rec if "timing" in r)
            row = {"model": m, "engine": e, "ok_off": bool(off and off.get("ok")), "ok_on": bool(on and on.get("ok")),
                   "error_off": (off or {}).get("error", "")[:300], "error_on": (on or {}).get("error", "")[:300],
                   "backend_off": (off or {}).get("backend_used"), "backend_on": (on or {}).get("backend_used"),
                   "same_output": bool(off and on and off.get("ok") and on.get("ok")
                                       and off.get("outputs") == on.get("outputs")),
                   "verdicts": dict(Counter(r["verdict"] for r in dec)),
                   "not_pass": [{"boundary": r.get("boundary"), "fact": r.get("name"), "verdict": r["verdict"],
                                 "rule": (r.get("rule") or "")[:160], "target": r.get("target"),
                                 "note": (r.get("note") or "")[:300],
                                 **({"backing": backing(r)} if r["verdict"] == "resolved" else {})}
                                for r in dec if r["verdict"] != "pass"],
                   "entail_ms": round(ms, 1), "load_s_on": (on or {}).get("load_seconds"),
                   "load_s_off": (off or {}).get("load_seconds")}
            if row["load_s_on"]:
                row["share"] = round(ms / 1e3 / row["load_s_on"], 4)
            runs.append(row)
    sel = jload(os.path.join(R, "selection.json")) or {}
    group = {}
    for g, key in (("top 20", "chosen"), ("per architecture", "by_architecture_extra")):
        for c in sel.get(key, []):
            group[c["id"].replace("/", "__")] = g
            group[c["id"].split("/")[1]] = group.get(c["id"].split("/")[1], g)   # local copies run under this name
    for r in runs:
        r["group"] = group.get(r["model"], "local")
    valid = [r for r in runs if r["ok_off"]]
    excluded = [r for r in runs if not r["ok_off"]]
    broke_run = [r for r in valid if not r["ok_on"]]
    alarms = [(r, d) for r in valid for d in r["not_pass"] if d["verdict"] in ("broken", "refused")]
    resolved = [(r, d) for r in valid for d in r["not_pass"] if d["verdict"] == "resolved"]
    unbacked = [(r, d) for r, d in resolved if not d["backing"]["backed_by_measured_row"]]
    unknown = [(r, d) for r in valid for d in r["not_pass"] if d["verdict"] == "unknown"]
    no_res = [r for r in valid if r["ok_on"] and not any(d["verdict"] == "resolved" for d in r["not_pass"])]
    shares = [r["share"] for r in valid if r.get("share") is not None]
    summary = {"models": len(names), "runs": len(runs), "valid_runs": len(valid), "excluded": len(excluded),
               "entail_broke_a_run": len(broke_run),
               "runs_with_broken_or_refused": len({(r["model"], r["engine"]) for r, _ in alarms}),
               "broken_or_refused_decisions": len(alarms),
               "alarms_by_boundary_fact": Counter(f"{d['boundary']} {d['fact']} {d['verdict']}" for _, d in alarms),
               "resolved_decisions": len(resolved),
               "resolved_not_backed_by_a_measured_row (false alarms by protocol 2.3)": len(unbacked),
               "false_alarm_runs (broken, refused or unbacked resolved)": len(
                   {(r["model"], r["engine"]) for r, _ in alarms + unbacked}),
               "unknown_decisions": len(unknown),
               "same_output_without_resolution": f"{sum(r['same_output'] for r in no_res)}/{len(no_res)}",
               "load_share_median": statistics.median(shares) if shares else None,
               "load_share_max": max(shares) if shares else None,
               "by_group": {g: {"models": len({r["model"] for r in valid if r["group"] == g}),
                                "valid_runs": sum(r["group"] == g for r in valid),
                                "runs_with_broken_or_refused": len({(r["model"], r["engine"]) for r, _ in alarms
                                                                    if r["group"] == g})}
                            for g in ("top 20", "per architecture", "local")}}
    with open(os.path.join(R, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": runs}, f, ensure_ascii=False, indent=1)

    L = ["# M10 E2: healthy runs of popular models (false alarms, outputs, cost)", "",
         "Generated by `testbed/m10_e2_summarize.py` from `results/m10/e2/engines`. Definitions: "
         "`testbed/M10_PROTOCOL.md` 2. Each run: 3 prompts, 16 greedy tokens, each engine's defaults "
         "(`testbed/m3_run_engine.py`), entail off and on (`ENTAIL=load`, the library hook only).", "",
         "## Totals", ""]
    L += [f"- {k}: {v}" for k, v in summary.items()]
    L += ["", "## Runs", "", "| model | group | engine | ok off/on | output same | backend off -> on | verdicts | entail "
          "share of load |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['model']} | {r['group']} | {r['engine']} | {r['ok_off']}/{r['ok_on']} | {r['same_output']} | "
                 f"{r['backend_off']} -> {r['backend_on']} | {r['verdicts']} | {r.get('share')} |")
    L += ["", "## Every decision other than pass (valid runs)", ""]
    for r in valid:
        for d in r["not_pass"]:
            L.append(f"- {r['model']} / {r['engine']}: {d['verdict']} at {d['boundary']} ({d['fact']})"
                     f"{' -> ' + str(d['target']) if d['target'] else ''}: {d['rule']} {d['note'][:160]}"
                     f"{' | backing: ' + json.dumps(d['backing']) if 'backing' in d else ''}")
    L += ["", "## Excluded (the engine fails with entail off too)", ""]
    L += [f"- {r['model']} / {r['engine']}: {r['error_off'][:200]}" for r in excluded] or ["- none"]
    L += ["", "## entail on failed where off succeeded", ""]
    L += [f"- {r['model']} / {r['engine']}: {r['error_on'][:300]}" for r in broke_run] or ["- none"]
    log = os.path.join(R, "run.log")
    lost = [x.strip() for x in open(log, encoding="utf-8")] if os.path.exists(log) else []
    lost = [x for x in lost if "no RESULT line" in x]
    L += ["", "## Runs that wrote no result (timeout or hard crash; from run.log)", ""]
    L += [f"- {x}" for x in lost] or ["- none"]
    out = "\n".join(L) + "\n"
    with open(os.path.join(os.path.dirname(R), "E2_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(out)
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
