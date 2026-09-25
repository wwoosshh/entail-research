"""M9.1 (and M9.2's S7): testbed/results/m91/SUMMARY.md from the files the M9 runs wrote. Only values read from those
files are written; where a file is missing the table says so.

  S1, S3, S4 load   results/m91/s1_s3.json          (m91_s1.py over m91_run.sh phase A)
  S2                results/m91/problems.json        (m91_problems.py), results/m91/s2.json (m91_s2.py)
  S4                results/m91/rerun: m55/graph.json, m53/overhead.json, m51/transformers.json, m73/cost.json;
                    results/m91/offstate.json
  S5, S6            results/m91/static.json          (m91_static.py)
  S7                results/m92/S7.json              (m92_s7.py)
Run: python testbed/m91_summarize.py
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M91 = os.path.join(RES, "m91")
RR = os.path.join(M91, "rerun")


def j(*parts):
    p = os.path.join(*parts)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def cell(x):
    if x is None:
        return "-"
    if isinstance(x, (list, tuple)):
        return ", ".join(map(str, x)) or "-"
    return str(x)


def s1_s3(L):
    d = j(M91, "s1_s3.json")
    if not d:
        return L.append("(s1_s3.json missing)")
    L += ["## S1: preservation", "",
          "M3's yardstick - the (fact, boundary) pairs a load should decide:", "",
          "| state | pairs |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in d["S1_m3_yardstick"]["totals"].items()]
    L += ["", "Wider - every fact a model folder declares, by name, against every decision in the run:", "",
          "| model | engine | fact | state | verdicts | consumed in the run |", "|---|---|---|---|---|---|"]
    for x in d["S1_wider"]["facts"]:
        L.append(f"| {x['model']} | {x['engine']} | {x['fact']} | {x['state']} | {cell(x['verdicts'])} | "
                 f"{cell(x['consumed_in_run'])} |")
    L += ["", f"Totals: {d['S1_wider']['totals']}", "", "Declared keys the vocabulary cannot carry (readers' problems):",
          ""]
    for m, ps in d["S1_wider"]["reader_problems"].items():
        L += [f"- {m}: {p}" for p in ps] or [f"- {m}: none"]
    sv = d["S1_serve_request_boundaries"]
    L += ["", "The request boundaries in the serve runs (phases B and S; vLLM 0.30 serving Qwen3-4B with pinned "
          "manifests), counted by each server process:", "",
          "| boundary | checks | passed | broken | refused | skipped |", "|---|---|---|---|---|---|"]
    L += [f"| {b} | {c['checks']} | {c['passed']} | {c['broken']} | {c['refused']} | {c['skipped']} |"
          for b, c in sorted(sv.get("counted", {}).items())] or ["| none found | | | | | |"]
    L += ["", "Recorded (what did not pass), by boundary, fact and verdict: " +
          ("; ".join(f"{k} x{v}" for k, v in sorted(sv.get("recorded", {}).items())) or "none")]
    L += ["", "## S3: false alarms (healthy runs, each engine's defaults)", "",
          "| model | engine | ok off/on | output same | backend off -> on | pass / resolved / refused / unknown |",
          "|---|---|---|---|---|---|"]
    for r in d["S3"]["runs"]:
        v = r["verdicts"]
        L.append(f"| {r['model']} | {r['engine']} | {r['ok_off']}/{r['ok_on']} | {r['same_output']} | "
                 f"{r['backend_off']} -> {r['backend_on']} | {v['pass']} / {v['resolved']} / {v['refused']} / "
                 f"{v['unknown']} |")
    L += ["", "Every broken, refused or resolved decision in these runs:", ""]
    L += [f"- {a['model']} / {a['engine']}: {a['verdict']} at {a['boundary']} ({a['fact']}) -> {a['target']}"
          for a in d["S3"]["broken_refused_resolved"]] or ["- none"]
    L += ["", "## S4: cost", "", "### At load (the time entail's own timing lines add up to, every process)", "",
          "| model | engine | entail ms (processes) | load s | share |", "|---|---|---|---|---|"]
    for r in d["S4_load"]:
        L.append(f"| {r['model']} | {r['engine']} | {r['entail_ms']} ({r['processes']}) | {r['load_s_on']} | "
                 f"{r['entail_share_of_load']} |")
    shares = [r["entail_share_of_load"] for r in d["S4_load"] if r.get("entail_share_of_load") is not None]
    if shares:
        L += ["", f"Largest share {max(shares):.4f}, median {statistics.median(shares):.4f}."]


def s4_rest(L):
    g = j(RR, "m55", "graph.json")
    L += ["", "### Always on, on vLLM's CUDA graph path (m55_graph: vLLM default settings, engine core in process)", ""]
    if g:
        L += ["| batch | on / off (median of 8) | control / off | checks while on | broken, refused | same tokens |",
              "|---|---|---|---|---|---|"]
        for b, v in g["batches"].items():
            L.append(f"| {b} | {v['median_on_over_off']:.3f} | {v['median_control_over_off']:.3f} | "
                     f"{v['checks_while_on']} | {v['broken_while_on']}, {v['refused_while_on']} | "
                     f"{v['same_tokens_on_off']} |")
    else:
        L.append("(rerun/m55/graph.json missing)")
    o = j(RR, "m53", "overhead.json")
    L += ["", "### Per request, vLLM server (m53_overhead)", ""]
    if o:
        L += ["| request | entail (us) | rendering (us) | entail / rendering |", "|---|---|---|---|"]
        L += [f"| {n} | {s['entail_us']} | {s['render_us']} | {s['ratio']} |" for n, s in o["shapes"].items()]
    else:
        L.append("(rerun/m53/overhead.json missing)")
    t = j(RR, "m51", "transformers.json")
    L += ["", "### transformers KV container contract (m51_transformers: 64 greedy tokens, 8 pairs)", ""]
    if t:
        L += ["| cache | on / off | control / off | same output |", "|---|---|---|---|"]
        for c in ("dynamic", "static"):
            L.append(f"| {c} | {t[c]['multiplier']} | {t[c]['control_multiplier']} | {t[c]['same_output']} |")
    else:
        L.append("(rerun/m51/transformers.json missing)")
    c = j(RR, "m73", "cost.json")
    L += ["", "### Diagnosis (m73_locate cost: Qwen3-4B, 64 tokens, 5 repetitions)", ""]
    if c:
        L += ["| attention | debug | debug + propagation | diagnosis (+ three layers watched) | same tokens |",
              "|---|---|---|---|---|"]
        for impl in ("eager", "sdpa"):
            m = c[impl]["median_x_off"]
            L.append(f"| {impl} | {m['debug']} | {m['debug_propagation']} | {m['diagnosis']} | "
                     f"{all(c[impl]['same_tokens_as_off'].values())} |")
    else:
        L.append("(rerun/m73/cost.json missing)")
    s = j(M91, "offstate.json")
    L += ["", "### Off (entail installed with its start-up hook, ENTAIL unset: m91_offstate)", ""]
    if s:
        L += [f"- Python start-up, median of {s['rounds'] * s['per_round']}: hook in place "
              f"{s['start_ms_median']['hook_in_place_entail_unset']} ms, hook removed "
              f"{s['start_ms_median']['hook_removed']} ms (difference {s['difference_ms']} ms)",
              f"- entail modules imported at start-up with ENTAIL unset: {cell(s['modules_imported_with_entail_unset'])}"]
    else:
        L.append("(offstate.json missing)")


def s2(L):
    p = j(M91, "problems.json")
    L += ["", "## S2: prevention, per test problem (final code)", ""]
    if not p:
        return L.append("(problems.json missing)")
    L += ["| problem | fact | expected | final (from the re-run files) |", "|---|---|---|---|"]
    for pid, r in p["problems"].items():
        L.append(f"| {pid} | {r['fact']} | {r['expected']} | "
                 f"{json.dumps(r['final'], ensure_ascii=False, default=str)[:900].replace('|', '/')} |")
    c = j(M91, "s2.json")
    if c:
        L += ["", "Against each milestone's own results (verdict-bearing values, list positions folded): "
              f"{c['counts']}", ""]
        for k, r in c["files"].items():
            if r["state"] == "differs":
                L.append(f"- {k}: " + "; ".join(f"{p}: {d['milestone']} -> {d['final']}"
                                                  for p, d in list(r["differs"].items())[:8]))
            elif r["state"] != "same":
                L.append(f"- {k}: {r['state']}")


def s5_s6(L):
    s = j(M91, "static.json")
    L += ["", "## S5: adapter thickness, S6: declaration burden", ""]
    if not s:
        return L.append("(static.json missing)")
    L += ["| engine | adapter files | code lines |", "|---|---|---|"]
    L += [f"| {e} | {', '.join(v['files'])} | {v['code_lines']} |" for e, v in s["S5"]["per_engine"].items()]
    L += ["", f"- v2 adapters with rule logic: {cell(s['S5']['v2 adapters with rule logic'])}",
          f"- research tools still in the package: {cell(s['S5']['research tools still in the package'])}",
          f"- code boundaries (rolebench, new code): {s['S6']['code_boundaries_total']}",
          f"- front end, Qwen3: {s['S6']['front_end_qwen3']}",
          f"- LLM model folders: {s['S6']['llm_model_folders']['user_lines']} lines",
          f"- image checkpoints that declare nothing: {s['S6']['image_checkpoints']['manifests']}"]


def s7(L):
    d = j(RES, "m92", "S7.json")
    L += ["", "## S7: next to post-hoc detection (M9.2, testbed/M92_PROTOCOL.md)", ""]
    if not d:
        return L.append("(results/m92/S7.json missing)")
    L += ["| run | defect | healthy | only healthy right / only defect right | McNemar p | evaluation detects | "
          "outputs differ (healthy vs healthy) | reference detects |", "|---|---|---|---|---|---|---|---|"]
    for k, e in d["evaluations"].items():
        if "missing" in e:
            L.append(f"| {k} | missing: {e['missing']} | | | | | | |")
            continue
        L.append(f"| {k} | {e['defect']} | {e['healthy']} | {e['only_healthy_right']} / {e['only_defect_right']} | "
                 f"{e['mcnemar_p']} | {e['evaluation_detects']} | {e.get('outputs_differ', '-')} "
                 f"({e.get('outputs_differ_between_healthy_runs', '-')}) | {e.get('reference_detects', '-')} |")
    L += ["", "| problem | entail | reference comparison (measured) | judged (rolebench/baselines.json): "
          "standard eval triggers / score moves / reference in practice |", "|---|---|---|---|"]
    for pid, r in d["problems"].items():
        ref = r.get("reference_measured") or {}
        jd = r.get("judged") or {}
        other = r.get("reference_other") or {}
        runs = [k for k in d["evaluations"] if k.split(" (")[0] == pid or (pid == "mk-L03" and
                                                                        k.startswith("fd-softcap"))]
        measured = (f"defect differs from the case's reference: {ref.get('defect_wrong')}" if ref else
                    f"{other.get('reference_detects')}: {other.get('measured')}" if other else
                    f"the GSM8K runs above: {', '.join(runs)}" if runs else "-")
        L.append(f"| {pid} | {cell(r['entail'])} | {measured.replace('|', '/')} | "
                 f"{cell(jd.get('standard_eval_triggers'))} / {cell(jd.get('score_moves_if_triggered'))} / "
                 f"{cell(jd.get('reference_in_practice'))} |")


def s8(L):
    d = j(RR, "m73", "SUMMARY.json")
    L += ["", "## S8: locating (M7.3's scenarios again, rerun/m73)", ""]
    if not d:
        return L.append("(rerun/m73/SUMMARY.json missing)")
    L += [f"Located {d['located']} of {d['of']}.", "", "| scenario | mode | output wrong | located in process | "
          "located from the record files |", "|---|---|---|---|---|"]
    for name, r in d["scenarios"].items():
        if name in ("reference", "cost"):
            continue
        L.append(f"| {name} | {r.get('mode')} | {r.get('output_wrong')} | {r.get('located_in_process')} | "
                 f"{r.get('located_from_records')} |")


def main():
    L = ["# M9.1: S1-S8 on the final code", "",
         "Generated by `testbed/m91_summarize.py` from the files named in each section (m91_run.sh phases A, B, C, R, "
         "S, T, D and E; RTX 4070 Ti 12 GB, WSL2). Values are read from those files only.", ""]
    s1_s3(L)
    s4_rest(L)
    s2(L)
    s5_s6(L)
    s7(L)
    s8(L)
    with open(os.path.join(M91, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L)[:6000])


if __name__ == "__main__":
    main()
