"""M3.5: summarise the measurements in testbed/results/m3 (from m3_engines.sh and m3_problems.py) into SUMMARY.md and
SUMMARY.json. Only values read from those files are written. Run in ~/venvs/gpu: python testbed/m3_summarize.py

  S2  the M3 test problems (testbed/PROBLEMS.md): verdicts, and whether the output is right after the repair
  S3  healthy models with each engine's defaults: refusals, and every repair with where it went
  S4  load-time cost: the time entail spent (its own timing lines, every process) against the load time
  S1  for every fact the model declares at load, whether a decision reached its consumer or it was reported unknown
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "m3")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
from entail import caps, load  # noqa: E402

MODELS = ("Qwen3-4B", "Llama-3.2-3B-Instruct", "gemma-2-2b-it")
ENGINES = ("transformers", "vllm", "sglang")


def records(path):
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def jload(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def expected_facts(model_dir, engine):
    """(fact, boundary) pairs a load of this model should decide: what the files declare, where it is consumed."""
    facts = load.declared(model_dir)
    table = caps.default_table()
    out = [("Coverage (config keys)", "load:transformers.config")]
    props = [f for f in facts.get("ModelProps") if f.value is not None]
    if any(caps.project(f.value, caps.consumed(table, f"{engine}.attention")) for f in props):
        out.append(("ModelProps (attention)", f"load:{engine}.attention"))
    out.append(("ModelProps (tie)", f"load:{engine}.loader"))
    if facts.get("Rotary"):
        out.append(("Rotary", f"load:{engine}.config.rope_parameters"))
    if facts.get("Layout"):
        out.append(("Layout", f"load:{engine}.linear"))
    return out


def s3_s4_s1():
    rows, reach = [], []
    for m in MODELS:
        for e in ENGINES:
            off, on = jload(f"{OUT}/s3_{e}_{m}_off.json"), jload(f"{OUT}/s3_{e}_{m}_load.json")
            rec = records(f"{OUT}/s3_{e}_{m}_load.record.jsonl")
            dec = [r for r in rec if "verdict" in r]
            ms = sum(r["ms"] for r in rec if "timing" in r)
            row = {"model": m, "engine": e, "ok_off": bool(off and off.get("ok")), "ok_on": bool(on and on.get("ok")),
                   "load_s_off": off and off.get("load_seconds"), "load_s_on": on and on.get("load_seconds"),
                   "entail_ms": round(ms, 1), "processes": len({r["pid"] for r in rec}),
                   "same_output": bool(off and on and off.get("outputs") == on.get("outputs")),
                   # transformers reports the implementation it settled on; for vLLM and SGLang the runner only
                   # knows what it asked for, so a switch shows in the resolved decisions below
                   "backend_off": off and off.get("backend_used"), "backend_on": on and on.get("backend_used"),
                   "verdicts": {v: sum(r["verdict"] == v for r in dec) for v in ("pass", "resolved", "refused",
                                                                                 "unknown")},
                   "resolved": sorted({(r["consumer"], r.get("target")) for r in dec if r["verdict"] == "resolved"}),
                   "unknown": sorted({(r["boundary"], r["rule"], r.get("note", "")[:120]) for r in dec
                                      if r["verdict"] == "unknown"})}
            if row["load_s_on"]:
                row["entail_share_of_load"] = round(ms / 1e3 / row["load_s_on"], 4)
            rows.append(row)
            seen = {}
            for r in dec:
                seen.setdefault(r["boundary"], []).append(r["verdict"])
            for fact, boundary in expected_facts(os.path.expanduser(f"~/models/{m}"), e):
                verdicts = seen.get(boundary, [])
                state = ("reached" if any(v in ("pass", "resolved", "refused") for v in verdicts) else
                         "reported unknown" if verdicts else "not reached, not reported")
                reach.append({"model": m, "engine": e, "fact": fact, "boundary": boundary, "state": state,
                              "verdicts": verdicts})
    return rows, reach


def s2():
    p = jload(f"{OUT}/problems.json") or {}
    out = {}
    for k in ("rb-08", "rb-15", "rb-07", "rb-02", "rb-06"):
        out[k] = p.get(k)
    for name in ("fd_softcap_sglang_torch_native", "fd_softcap_sglang_flex_attention", "fd_softcap_transformers_paged",
                 "fd_shift"):
        run = jload(f"{OUT}/{name}.json")
        dec = [r for r in records(f"{OUT}/{name}.record.jsonl") if "verdict" in r]
        out[name] = {"ok": run and run.get("ok"), "backend_used": run and run.get("backend_used"),
                     "error": run and (run.get("error") or "")[:400],
                     "decisions": [(r["boundary"], r["consumer"], r["verdict"], r.get("target")) for r in dec
                                   if r["verdict"] != "pass"]}
    lc = jload(os.path.join(os.path.dirname(HERE), "issue_track", "rope_override", "results", "LC_entail_v2.json"))
    la = jload(os.path.join(os.path.dirname(HERE), "issue_track", "rope_override", "results", "LA_m35.json"))
    rope_dec = [r for r in records(f"{OUT}/fd_rope.record.jsonl") if "verdict" in r and "rope" in r["boundary"]]
    out["fd_rope"] = {
        "LC_entail_v2_gsm8k": lc and f"{sum(lc['gsm8k']['correct'])}/{lc['gsm8k']['n']}",
        "LC_entail_v2_rope_in_engine": lc and lc.get("rope_parameters_in_engine"),
        "LA_m35_gsm8k": la and f"{sum(la['gsm8k']['correct'])}/{la['gsm8k']['n']}",
        "earlier": "LA 379/500, LC 279/500, LC_rolecheck 378/500 (issue_track/rope_override/SUMMARY.json)",
        "decisions": [(r["boundary"], r["verdict"], r["rule"], (r.get("declared") or {}).get("source", {}).get("kind"))
                      for r in rope_dec]}
    out["rb-17"] = jload(f"{OUT}/rb17/compare.json")
    if out["rb-17"]:
        out["rb-17"].pop("record", None)
    return out


def main():
    rows, reach = s3_s4_s1()
    problems = s2()
    res = {"S3_S4": rows, "S1": reach, "S2": problems}
    with open(f"{OUT}/SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)

    L = ["# M3.5 measurements", "", "Values are read from the files in this folder (m3_engines.sh, m3_problems.py).", "",
         "## S3, S4: healthy models, each engine's default settings, entail off and on", "",
         "The backend column is what transformers settled on, and for vLLM and SGLang what the run asked for (a switch "
         "made by entail shows under Repairs).", "",
         "| model | engine | ok off/on | output same | backend off -> on | pass / resolved / refused / unknown | "
         "entail ms (processes) | load s on | entail share of load |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = r["verdicts"]
        L.append(f"| {r['model']} | {r['engine']} | {r['ok_off']}/{r['ok_on']} | {r['same_output']} | "
                 f"{r['backend_off']} -> {r['backend_on']} | {v['pass']} / {v['resolved']} / {v['refused']} / "
                 f"{v['unknown']} | {r['entail_ms']} ({r['processes']}) | {r['load_s_on']} | "
                 f"{r.get('entail_share_of_load')} |")
    L += ["", "Repairs and unknowns:", ""]
    for r in rows:
        for c, t in r["resolved"]:
            L.append(f"- {r['model']} / {r['engine']}: resolved {c} -> {t}")
        for b, rule, note in r["unknown"]:
            L.append(f"- {r['model']} / {r['engine']}: unknown at {b}: {rule}; {note}")
    L += ["", "## S1: declared facts at load", "", "| model | engine | fact | boundary | state | verdicts |",
          "|---|---|---|---|---|---|"]
    for x in reach:
        L.append(f"| {x['model']} | {x['engine']} | {x['fact']} | {x['boundary']} | {x['state']} | "
                 f"{', '.join(x['verdicts'])} |")
    counts = {}
    for x in reach:
        counts[x["state"]] = counts.get(x["state"], 0) + 1
    L += ["", f"Totals: {counts}", "", "## S2: test problems", ""]
    for k, v in problems.items():
        L.append(f"### {k}")
        L.append("```")
        L.append(json.dumps(v, ensure_ascii=False, indent=1, default=str)[:3000])
        L.append("```")
    text = "\n".join(L) + "\n"
    with open(f"{OUT}/SUMMARY.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text[:12000])


if __name__ == "__main__":
    main()
