"""M11.7: testbed/results/m11/SUMMARY.md - entail 1.0.1 against 1.0.0 on the M10 evidence, from the result files only.
  E1 L3   results/m10/e1_llm/static.json (1.0.0) and static_1.0.1.json (m10_e1_static.py, M10_E1_STATIC_OUT)
  E2      results/m10/e2/summary.json (1.0.0) and results/m11/e2/summary.json (1.0.1: m11_e2_run.sh, then
          m10_e2_summarize.py with M10_E2_DIR); the fresh sample in results/m11/e2_fresh (m10_e2_run.sh, M10_E2_OUT)
  S2      results/m91/problems.json (M9.1) and results/m11/m91/problems.json (m91_run.sh B S T with M91_OUT, then
          m91_problems.py with M91_DIR): the test problems' verdicts, compared value by value
  E3      results/m10/e3/cases/sg33493_on.record.jsonl (1.0.0) and results/m11/e3/sg33493_on.record.jsonl
Run in ~/venvs/gpu: python testbed/m11_summarize.py   (runs the E2 summarizer on the m11 folders first)
"""
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M10, M11 = os.path.join(RES, "m10"), os.path.join(RES, "m11")
sys.path.insert(0, HERE)


def j(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - a file a step did not write: reported as missing
        return None


def records(path):
    if not os.path.exists(path):
        return None
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


# --- E1 L3: the static check on the popular configs --------------------------------------------------------------

def e1_tally(rows):
    c, det = collections.Counter(), collections.defaultdict(list)
    for r in rows:
        for eng, e in r["engines"].items():
            if not isinstance(e, dict):
                continue
            for d in e.get("model_decisions", []):
                c[(eng, d["name"], d["verdict"])] += 1
                if d["verdict"] in ("broken", "refused"):
                    det[(eng, d["name"], d["verdict"])].append((r["id"], (d.get("note") or d.get("rule") or "")[:120]))
            for b, d in (e.get("backends") or {}).items():
                v = d.get("verdict") if isinstance(d, dict) else str(d)
                c[(eng, "attention backends", v)] += 1
    return c, det


def e1_section():
    old, new = j(os.path.join(M10, "e1_llm", "static.json")), j(os.path.join(M10, "e1_llm", "static_1.0.1.json"))
    if not old or not new:
        return ["## E1 L3: static check on the popular configs", "", "- missing: static.json or static_1.0.1.json", ""], {}
    c0, _ = e1_tally(old["rows"])
    c1, d1 = e1_tally(new["rows"])
    L = ["## E1 L3: entail's static check on the popular configs (no weights)", "",
         f"- 1.0.0: {len(old['rows'])} folders; 1.0.1: {len(new['rows'])} folders ({new.get('when', '')})", "",
         "| engine | fact | verdict | 1.0.0 | 1.0.1 |", "|---|---|---|---|---|"]
    for k in sorted(set(c0) | set(c1)):
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {c0.get(k, 0)} | {c1.get(k, 0)} |")
    left = [(k, m, n) for k, v in d1.items() for m, n in v]
    L += ["", f"- broken or refused model decisions left in 1.0.1: {len(left)}"]
    for k, m, n in left:
        L.append(f"  - {k[0]} {k[1]} {k[2]}: {m}: {n}")
    L.append("")
    cov = {v: (sum(n for (e, f, vv), n in c0.items() if f == "Coverage" and vv == v),
               sum(n for (e, f, vv), n in c1.items() if f == "Coverage" and vv == v)) for v in ("broken", "unknown", "pass")}
    return L, {"coverage": cov, "left": len(left)}


# --- E2: the healthy runs ----------------------------------------------------------------------------------------

def e2_summary(folder, name):
    """Run the E2 summarizer on `folder` (results/<..>/e2 or e2_fresh); returns its summary dict."""
    import m10_e2_summarize as m

    if not os.path.isdir(os.path.join(folder, "engines")):
        return None
    m.R, m.E = folder, os.path.join(folder, "engines")
    m.main()
    made = os.path.join(os.path.dirname(folder), "E2_SUMMARY.md")
    if os.path.exists(made) and name:
        os.replace(made, os.path.join(os.path.dirname(folder), f"E2_{name}_SUMMARY.md"))
    s = j(os.path.join(folder, "summary.json"))
    return s


def e2_rows(s):
    if not s:
        return {}
    x = s["summary"]
    return {"models": x["models"], "valid runs": x["valid_runs"], "entail broke a run": x["entail_broke_a_run"],
            "runs with broken or refused": x["runs_with_broken_or_refused"],
            "broken or refused decisions": x["broken_or_refused_decisions"],
            "resolved decisions": x.get("resolved_decisions"),
            "resolved, not backed by a measured row": x.get("resolved_not_backed_by_a_measured_row (false alarms by protocol 2.3)"),
            "false-alarm runs": x.get("false_alarm_runs (broken, refused or unbacked resolved)"),
            "unknown decisions": x.get("unknown_decisions"),
            "same output without resolution": x.get("same_output_without_resolution"),
            "load share median / max": f"{x.get('load_share_median')} / {x.get('load_share_max')}",
            "alarms by boundary and fact": json.dumps(x.get("alarms_by_boundary_fact"), ensure_ascii=False)}


def e2_section():
    old = j(os.path.join(M10, "e2", "summary.json"))
    new = e2_summary(os.path.join(M11, "e2"), "30")
    fresh = e2_summary(os.path.join(M11, "e2_fresh"), "FRESH")
    L = ["## E2: healthy runs of popular models, three engines each", "",
         "| | 1.0.0 (M10 E2, 30 models) | 1.0.1 (the same 30) | 1.0.1 (fresh sample) |", "|---|---|---|---|"]
    a, b, c = e2_rows(old), e2_rows(new), e2_rows(fresh)
    for k in a:
        L.append(f"| {k} | {a.get(k)} | {b.get(k, '-')} | {c.get(k, '-')} |")
    for label, s in (("1.0.1, the same 30", new), ("1.0.1, fresh sample", fresh)):
        if not s:
            continue
        L += ["", f"- {label}: decisions other than pass, by boundary, fact and verdict:"]
        cnt = collections.Counter()
        for r in s["runs"]:
            for d in r.get("not_pass", []):
                cnt[(d["boundary"], d["fact"], d["verdict"])] += 1
        for k, n in sorted(cnt.items()):
            L.append(f"  - {k[0]} {k[1]} {k[2]}: {n}")
        bad = [(r["model"], r["engine"], d) for r in s["runs"] for d in r.get("not_pass", [])
               if d["verdict"] in ("broken", "refused")]
        L.append(f"- {label}: broken or refused decisions: {len(bad)}")
        for m, e, d in bad:
            L.append(f"  - {m} / {e}: {d['verdict']} at {d['boundary']} ({d['fact']}): {d.get('note') or d.get('rule')}")
        odd = [r for r in s["runs"] if r.get("ok_on") and r.get("ok_off") and r.get("same_output") is False
               and not any(d["verdict"] == "resolved" for d in r.get("not_pass", []))]
        for r in odd:
            L.append(f"- {label}: {r['model']} / {r['engine']} gives another output than the M10 off run, with no "
                     f"resolution" + offcheck(r))
    L.append("")
    return L, {"old": a, "new": b, "fresh": c}


def offcheck(r):
    """A second off run made for a run whose output differed (results/m11/e2/offcheck/*.json): whether it equals
    the M10 off output and the 1.0.1 on output, so the difference is the engine's or entail's."""
    for p in glob.glob(os.path.join(M11, "e2", "offcheck", f"*{r['engine']}*.json")):
        again = j(p)
        if not again or r["model"].split("__")[-1] not in os.path.basename(p) and r["model"] not in os.path.basename(p):
            continue
        old = j(os.path.join(M10, "e2", "engines", f"e2_{r['engine']}_{r['model']}_off.json"))
        on = j(os.path.join(M11, "e2", "engines", f"e2_{r['engine']}_{r['model']}_load.json"))
        if old and on:
            return (f"; an off run made again after it ({os.path.basename(p)}) equals the M10 off output: "
                    f"{again.get('outputs') == old.get('outputs')}, equals the 1.0.1 on output: "
                    f"{again.get('outputs') == on.get('outputs')}")
    return ""


# --- S2: the test problems ---------------------------------------------------------------------------------------

def flat(x, prefix=""):
    """Every verdict-bearing value of a nested result, keyed by its path, lists folded into sorted tuples."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{prefix}.{k}"))
    elif isinstance(x, list):
        vals = [v for v in x if isinstance(v, (str, bool, int, float)) and v is not None]
        if vals:
            out[prefix + "[]"] = tuple(sorted(str(v) for v in vals))
        for v in x:
            if isinstance(v, (dict, list)):
                out.update(flat(v, prefix + "[]"))
    elif x is not None:
        out[prefix] = (str(x),)
    return out


VERDICT_WORDS = re.compile(r"\b(pass|resolved|broken|refused|unknown)\b")


def verdicts_only(values):
    """A value tuple with every entail line reduced to the verdict words it carries (its wording changes between
    versions; its verdicts are what S2 compares)."""
    return tuple(sorted("+".join(sorted(VERDICT_WORDS.findall(s))) if "[entail]" in s else s for s in values))


def s2_section():
    old, new = j(os.path.join(RES, "m91", "problems.json")), j(os.path.join(M11, "m91", "problems.json"))
    L = ["## S2: the test problems (m91_run.sh B S T, then m91_problems.py)", ""]
    if not old or not new:
        L += ["- missing: results/m91/problems.json or results/m11/m91/problems.json", ""]
        return L, {}
    same, differs, not_rerun, partly = [], [], [], {}
    for name, p0 in old["problems"].items():
        p1 = new["problems"].get(name)
        f0, f1 = flat(p0.get("final")), flat(p1.get("final")) if p1 else {}
        keys = [k for k in f0 if "verdict" in k.lower() or k.endswith("[]") or "stopped" in k or "right" in k]
        missing = [k for k in keys if k not in f1]
        if not f1 or all(v == ("see fd-softcap",) for v in f1.values()) or len(missing) == len(keys):
            not_rerun.append(name)
            continue
        if missing:
            partly[name] = missing
        diff = [(k, f0[k], f1[k]) for k in keys if k in f1 and verdicts_only(f0[k]) != verdicts_only(f1[k])]
        (differs if diff else same).append((name, diff))
    L += [f"- rerun {len(same) + len(differs)} of {len(old['problems'])} problems: same verdicts {len(same)}, differ "
          f"{len(differs)}; not rerun (need steps outside B S T, or ComfyUI): {', '.join(not_rerun) or '-'}"]
    for name, missing in partly.items():
        L.append(f"- {name}: the parts from other steps were not rerun ({', '.join(missing)}); the rest is compared")
    L.append("")
    for name, diff in differs:
        L.append(f"- {name} differs:")
        for k, a, b in diff[:6]:
            L.append(f"  - {k}: {a} -> {b}")
    L.append("")
    return L, {"same": len(same), "differs": len(differs), "not_rerun": not_rerun, "partly": partly}


# --- E3: the DSPARK case -----------------------------------------------------------------------------------------

def e3_section():
    old = records(os.path.join(M10, "e3", "cases", "sg33493_on.record.jsonl"))
    new = records(os.path.join(M11, "e3", "sg33493_on.record.jsonl"))
    L = ["## E3 #56 (sglang#33493): MiniCPM5-2B with its DSpark drafter, entail on", ""]
    for label, rec in (("1.0.0", old), ("1.0.1", new)):
        if rec is None:
            L.append(f"- {label}: no record")
            continue
        c = collections.Counter((d["verdict"], d["boundary"]) for d in rec if "verdict" in d)
        L.append(f"- {label}: " + ", ".join(f"{v} at {b}: {n}" for (v, b), n in sorted(c.items())))
        notes = {d.get("note", "")[:120] for d in rec if d.get("verdict") == "unknown" and "speculative" in (d.get("note") or "")}
        for n in sorted(notes):
            L.append(f"  - note: {n}")
    L.append("")
    return L, {}


def main():
    os.makedirs(M11, exist_ok=True)
    L = ["# M11.7: entail 1.0.1 against 1.0.0 on the M10 evidence", "",
         "Generated by `testbed/m11_summarize.py` from the result files named in its docstring. Definitions: "
         "`testbed/M10_PROTOCOL.md` (section 6 records the 1.0.1 rerun rules). Nothing here was typed in.", ""]
    e1, e1v = e1_section()
    e2, e2v = e2_section()
    s2, s2v = s2_section()
    e3, _ = e3_section()
    L += e1 + e2 + s2 + e3
    text = "\n".join(L) + "\n"
    with open(os.path.join(M11, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(text)
    json.dump({"e1": e1v, "e2": e2v, "s2": s2v}, open(os.path.join(M11, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
