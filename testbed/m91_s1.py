"""M9.1: S1 (preservation), S3 (false alarms) and the load share of S4, from the healthy runs of m91_run.sh phase A
(testbed/results/m91/engines), measured with M3's yardstick (m3_summarize.s3_s4_s1, unchanged) and wider.

S1 (LIBRARY_DESIGN.md 8): for each fact a model declares, whether it reached a consumer's decision point (a decision
passed, resolved, refused or broke there: the declaration was there to be compared) or was reported unknown - the two
should make 100%, with no silent default.
  M3's yardstick: the (fact, boundary) pairs a load should decide (m3_summarize.expected_facts).
  Wider: every fact the folder declares (load.declared), by name, against every decision in the run, and the
  problems the readers report (a declared key the vocabulary cannot carry). A fact is
    reached      some decision about it passed, resolved, refused or broke
    reported     decisions about it, all unknown
    silent       no decision about it in the run
  and whether the run consumed it at all: the runner (m3_run_engine.py) renders the chat template itself with
  transformers (tokenizer.apply_chat_template) and gives vLLM and SGLang the raw prompts; vLLM's renderer still
  renders once at start (its warm-up), and SGLang's offline engine not at all. A pass at a request boundary is
  counted, not recorded (corrected in M9.3: M9.1's first count read the records only). The request boundaries
  of the servers are measured in the serve runs of phases B and S (m53_serve, m54_serve, m55_market_serve).
Usage: python testbed/m91_s1.py [<engines folder> <out.json>]  (default: results/m91/engines, results/m91/s1_s3.json)
S3: broken, refused and resolved decisions in these runs, and whether the output with entail on equals off.
Run in ~/venvs/gpu.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m3_summarize as m3  # noqa: E402
from entail import load  # noqa: E402  (m3_summarize put the entail checkout on the path)

M91 = os.path.join(HERE, "results", "m91")
ENG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(M91, "engines")   # m93: python testbed/m91_s1.py <engines>
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(M91, "s1_s3.json")   # <out.json>
REACHED = ("pass", "resolved", "refused", "broken")
CONSUMES_TEMPLATE = {"transformers": "tokenizer.apply_chat_template in the runner",
                     "vllm": "the renderer's warm-up render at start (vllm_serve)",
                     "sglang": None}   # m3_run_engine.py: raw prompts to LLM.generate / Engine.generate


def counted(rec):
    """Passes at a request boundary are counted, not recorded one by one (request_contract._settle): the last count
    each process wrote, per boundary, as the fact it decides (the chat template and the reasoning history: Template).
    M9.1's first count of S1 missed these (corrected in M9.3)."""
    last = {}
    for r in rec:
        for b, s in (r.get("boundaries") or {}).items():
            if b.startswith("request:"):
                last[(r.get("pid"), b)] = s
    out = {}
    for (_, b), s in last.items():
        if "chat_template" in b or "reasoning_history" in b:
            out.setdefault("Template", {}).setdefault(b, 0)
            out["Template"][b] += s.get("checks", 0)
    return out


def wider(model, engine):
    folder = os.path.expanduser(f"~/models/{model}")
    declared = load.declared(folder)
    rec = m3.records(f"{ENG}/s3_{engine}_{model}_load.record.jsonl")
    dec = [r for r in rec if "verdict" in r]
    passes = counted(rec)
    rows = []
    for name in sorted(declared.facts):
        if not any(f.value is not None for f in declared.facts[name]):
            continue
        ds = [r for r in dec if r["name"] == name]
        verdicts = sorted({r["verdict"] for r in ds} | ({"pass"} if any(passes.get(name, {}).values()) else set()))
        state = ("reached" if any(v in REACHED for v in verdicts) else "reported" if ds else "silent")
        consumed = CONSUMES_TEMPLATE[engine] if name == "Template" else "at load"
        rows.append({"model": model, "engine": engine, "fact": name, "state": state, "verdicts": verdicts,
                     "boundaries": sorted({r["boundary"] for r in ds} | {b for b, n in passes.get(name, {}).items()
                                                                         if n}),
                     "consumed_in_run": consumed})
    return rows, declared.problems


def serve_template():
    """The request boundaries (the chat template, the reasoning history, the tool-call format, the request's
    fields) in the serve runs of phases B and S: the decisions recorded there (what did not pass), by boundary and
    verdict, and the counts each server process wrote last (a pass is counted, not recorded one by one)."""
    out, counted = {}, {}
    for sub in ("m53", "m54", "m55"):
        base = os.path.join(M91, "rerun", sub)
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if not f.endswith(".jsonl"):
                    continue
                last = {}
                for r in m3.records(os.path.join(root, f)):
                    if "verdict" in r and r["boundary"].startswith("request:"):
                        key = f"{r['boundary']} {r['name']} {r['verdict']}"
                        out[key] = out.get(key, 0) + 1
                    for b, s in (r.get("boundaries") or {}).items():
                        if b.startswith("request:"):
                            last[(r.get("pid"), b)] = s
                for (_, b), s in last.items():
                    c = counted.setdefault(b, {"checks": 0, "passed": 0, "broken": 0, "refused": 0, "skipped": 0})
                    c["checks"] += s.get("checks", 0)
                    c["passed"] += sum((s.get("passed") or {}).values())
                    for k in ("broken", "refused", "skipped"):
                        c[k] += s.get(k, 0)
    return {"recorded": out, "counted": counted}


def main():
    m3.OUT = ENG
    rows, reach = m3.s3_s4_s1()
    wide, problems = [], {}
    for m in m3.MODELS:
        for e in m3.ENGINES:
            w, p = wider(m, e)
            wide += w
            problems[m] = p
    alarms = []
    for m in m3.MODELS:
        for e in m3.ENGINES:
            for r in m3.records(f"{ENG}/s3_{e}_{m}_load.record.jsonl"):
                if r.get("verdict") in ("broken", "refused", "resolved"):
                    alarms.append({"model": m, "engine": e, "boundary": r["boundary"], "fact": r["name"],
                                   "verdict": r["verdict"], "target": r.get("target"), "note": r.get("note", "")[:300]})
    count = lambda xs, key: {s: sum(x[key] == s for x in xs) for s in sorted({x[key] for x in xs})}  # noqa: E731
    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "S1_m3_yardstick": {"pairs": reach, "totals": count(reach, "state")},
           "S1_wider": {"facts": wide, "totals": count(wide, "state"),
                        "consumed_but_silent": [x for x in wide if x["state"] == "silent" and x["consumed_in_run"]],
                        "not_consumed_in_run": [x for x in wide if not x["consumed_in_run"]],
                        "reader_problems": problems},
           "S1_serve_request_boundaries": serve_template(),
           "S3": {"runs": [{k: r[k] for k in ("model", "engine", "ok_off", "ok_on", "same_output", "backend_off",
                                              "backend_on", "verdicts")} for r in rows],
                  "broken_refused_resolved": alarms},
           "S4_load": [{k: r.get(k) for k in ("model", "engine", "entail_ms", "processes", "load_s_on",
                                              "entail_share_of_load")} for r in rows]}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({"S1_m3": res["S1_m3_yardstick"]["totals"], "S1_wider": res["S1_wider"]["totals"],
                      "consumed_but_silent": [(x["model"], x["engine"], x["fact"])
                                              for x in res["S1_wider"]["consumed_but_silent"]],
                      "reader_problems": problems, "serve": res["S1_serve_request_boundaries"],
                      "S3": [(r["model"], r["engine"], r["ok_off"], r["ok_on"], r["same_output"], r["verdicts"])
                             for r in rows], "alarms": alarms,
                      "S4_load_share": [(r["model"], r["engine"], r.get("entail_share_of_load")) for r in rows]},
                     ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
