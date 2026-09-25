"""M9.2, S7: entail next to the post-hoc detectors on the same test problems, by testbed/M92_PROTOCOL.md.

Read, not judged here:
  entail              results/m91/problems.json (m91_problems.py: the final code's verdicts per problem)
  evaluation score    GSM8K 500, greedy; McNemar exact test on the paired answers, detected when p < 0.05
                        fd-rope             issue_track/rope_override/results LC_off_m91 (defect) vs LA_m91
                        fd-softcap SGLang   results/m92 softcap_sglang_torch_native vs softcap_sglang_triton
                        fd-shift            results/m92 shift_seeded vs shift_control
                        fd-softcap HF       issue_track/gemma2_softcap/results e3_2b, e3_9b (sdpa vs eager; earlier)
  reference compare   the same runs: questions whose output text differs, defect against healthy, detected when more
                      than differ between two healthy runs of the same configuration; and each problem's own
                      harness values (rolebench results: the defect's output against the case's reference)
  judgements          rolebench/baselines.json (a hypothesis-blind rater, 2026-09-23): b the standard evaluation
                      triggers the case, c its score would move, d a reference exists in practice - marked "judged"
Writes testbed/results/m92/S7.json (m91_summarize.py puts it in results/m91/SUMMARY.md).
Run: python testbed/m92_s7.py
"""
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
M92 = os.path.join(RES, "m92")
ROPE = os.path.join(ROOT, "issue_track", "rope_override", "results")
G2 = os.path.join(ROOT, "issue_track", "gemma2_softcap", "results")
RB = os.path.join(ROOT, "rolebench")


def j(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def wilson(k, n, z=1.96):   # as issue_track/gemma2_softcap/g2softcap.py
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def mcnemar_exact(b, c):    # as issue_track/gemma2_softcap/g2softcap.py
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def paired(defect, healthy, repeat=None, what=""):
    """The evaluation score and the output comparison of one defect run against its healthy run."""
    if defect is None or healthy is None:
        return {"what": what, "missing": [n for n, r in (("defect", defect), ("healthy", healthy)) if r is None]}
    a, h = defect["gsm8k"]["correct"], healthy["gsm8k"]["correct"]
    n = len(a)
    only_h = sum(1 for x, y in zip(a, h) if y and not x)
    only_d = sum(1 for x, y in zip(a, h) if x and not y)
    p = mcnemar_exact(only_h, only_d)
    out = {"what": what, "n": n, "defect": f"{sum(a)}/{n}", "healthy": f"{sum(h)}/{n}",
           "defect_wilson95": wilson(sum(a), n), "healthy_wilson95": wilson(sum(h), n),
           "only_healthy_right": only_h, "only_defect_right": only_d, "mcnemar_p": round(p, 4),
           "evaluation_detects": p < 0.05}
    to, ho = defect["gsm8k"].get("outputs"), healthy["gsm8k"].get("outputs")
    if to and ho:
        out["outputs_differ"] = sum(1 for x, y in zip(to, ho) if x != y)
        if repeat is not None and repeat["gsm8k"].get("outputs"):
            out["outputs_differ_between_healthy_runs"] = sum(1 for x, y in zip(repeat["gsm8k"]["outputs"], ho)
                                                             if x != y)
            out["reference_detects"] = out["outputs_differ"] > out["outputs_differ_between_healthy_runs"]
    return out


def softcap_hf(size):
    d = j(os.path.join(G2, f"e3_{size}.json"))
    if d is None:
        return {"what": f"Gemma 2 {size}, transformers sdpa (drops softcap) vs eager", "missing": ["e3"]}
    pr = d["paired"]
    s, e = d["paths"]["sdpa"]["correct"], d["paths"]["eager"]["correct"]
    return {"what": f"Gemma 2 {size} ({d['weights']}), transformers sdpa (drops softcap) vs eager; "
                    "issue_track/gemma2_softcap E3, measured earlier",
            "n": d["n"], "defect": f"{sum(s)}/{d['n']}", "healthy": f"{sum(e)}/{d['n']}",
            "only_healthy_right": pr["only_eager_correct"], "only_defect_right": pr["only_sdpa_correct"],
            "mcnemar_p": round(pr["mcnemar_exact_p"], 4), "evaluation_detects": pr["mcnemar_exact_p"] < 0.05,
            "reference_note": "E1 (logits against eager, teacher forcing): effect 1.4x (2B) / 2.5x (9B) the noise "
                              "between two correct kernels; confirmed only by the pattern test, which needs a "
                              "softcap-off control (issue_track/gemma2_softcap/RESULTS.md)"}


def rolebench():
    base = {c["id"]: c for c in (j(os.path.join(RB, "baselines.json")) or {}).get("cases", [])}
    out = {}
    for name in sorted(os.listdir(os.path.join(RB, "results"))):
        if not (name[:2].isdigit() and name.endswith(".json")):
            continue
        r = j(os.path.join(RB, "results", name))
        b = base.get(name[:2], {})
        out[f"rb-{name[:2]}"] = {
            "reference_measured": {"defect_wrong": r.get("defect_wrong"), "fixed_right": r.get("fixed_right"),
                                   "defect_silent": r.get("defect_silent"), "reproduced": r.get("reproduced")},
            "judged": {"standard_eval_triggers": b.get("b"), "score_moves_if_triggered": b.get("c"),
                       "reference_in_practice": b.get("d"), "trigger": b.get("trigger"),
                       "confidence": b.get("confidence")}}
    return out


def texts(path, prefix="TEXT:"):
    if not os.path.exists(path):
        return None
    return [line.strip() for line in open(path, encoding="utf-8", errors="replace") if line.startswith(prefix)]


def image_rows(summary, section, condition="off"):
    """The '<n> from reference' cells of one condition's row in an m63 SUMMARY.md section (m63_summarize.py)."""
    if not os.path.exists(summary):
        return None
    lines = open(summary, encoding="utf-8").read().split("\n")
    inside = False
    for line in lines:
        if line.startswith("### "):
            inside = line.startswith(f"### {section}")
        elif inside and line.startswith(f"| {condition} |"):
            return [c.strip() for c in line.split("|") if "reference" in c or "identical" in c]
    return None


def reference_other():
    """The reference comparison for the problems outside rolebench, from the harness files that already hold both
    outputs (M92_PROTOCOL 2: the defect's output with entail off against the reference output, same inputs)."""
    rr = os.path.join(RES, "m91", "rerun")
    out = {}
    off, seeded = j(os.path.join(rr, "m51", "vllm_off.json")), j(os.path.join(rr, "m51", "vllm_seeded.json"))
    if off and seeded:
        same = off.get("texts") == seeded.get("texts")
        out["fd-kv"] = {"reference_detects": not same, "measured": f"vLLM, block table one block short, 3 prompts x "
                        f"16 tokens: texts the same as entail off: {same} (rerun/m51); SGLang's own assertion stopped "
                        f"its seeded run after entail reported broken (rerun/m51/sglang_seeded.log)"}
    h = texts(os.path.join(rr, "m42", "healthy_none.log"))
    rep = {t: texts(os.path.join(rr, "m42", f"{t}.log")) for t in ("roll_off", "strided_on", "transpose_on")}
    if h is not None:
        out["fd-repack"] = {"reference_detects": {t: (None if v is None or not v else v != h) for t, v in rep.items()},
                            "measured": "vLLM Qwen3-4B, the TEXT line against the healthy run (rerun/m42): roll "
                                        f"{'differs' if rep['roll_off'] and rep['roll_off'] != h else 'same'}; strided "
                                        f"{'differs' if rep['strided_on'] and rep['strided_on'] != h else 'same'}; "
                                        "transpose: no text (the shapes stop matching, loud)"}
    s = os.path.join(rr, "m63", "SUMMARY.md")
    for pid, section in (("mk-I04", "i04"), ("fd-m7", "m7"), ("fd-vae", "vae")):
        cells = image_rows(s, section)
        if cells:
            out[pid] = {"reference_detects": not all(c.startswith("identical") for c in cells),
                        "measured": f"diffusers, entail off against the reference, 3 seeds: {'; '.join(cells)} "
                                    "(rerun/m63/SUMMARY.md)" + ("; ComfyUI (fd-m7, M6.3): 83-95/255" if pid == "fd-m7"
                                                                else "")}
    i01_off = j(os.path.join(rr, "m63", "diffusers_i01_off.json"))
    if i01_off:
        out["mk-I01"] = {"reference_detects": "only against a reference with the LoRA applied",
                         "measured": "diffusers: the image is pixel-identical to the model without a LoRA (rerun/m63); "
                                     "the same LoRA in the format diffusers reads changes it by 26-42/255"}
    out["fd-lora"] = {"reference_detects": "no reference: the LoRA cannot apply",
                      "measured": "ComfyUI 0.34.1 (M6.3, not re-run): the wrong LoRA's images are identical to no "
                                  "LoRA; diffusers refuses the file by itself (NoMatchingPeftModuleError)"}
    l05 = j(os.path.join(rr, "m55", "l05.json"))
    if l05:
        sc = l05["scenarios"]["default_context_model_has_room"]
        out["mk-L05"] = {"reference_detects": sc["off"].get("answer_right") is False,
                         "measured": f"the truncated prompt's answer right: {sc['off'].get('answer_right')} "
                                     f"(tokens seen {sc['off'].get('tokens_seen')} of {sc.get('prompt_tokens')})"}
    mk = j(os.path.join(rr, "m55", "market_clean_off.json"))
    if mk:
        r = mk["requests"]
        out["mk-L13"] = {"reference_detects": r["history_kept"].get("content") != r["history_dropped"].get("content"),
                         "measured": f"one request, thinking off: history kept -> {r['history_kept'].get('content')!r}, "
                                     f"dropped -> {r['history_dropped'].get('content')!r} (rerun/m55)"}
        bug = j(os.path.join(rr, "m55", "market_bug_off.json"))
        if bug:
            a, b = r["effort_high"], bug["requests"]["effort_high"]
            out["mk-L07"] = {"reference_detects": "not measurable here",
                             "measured": f"prompt tokens {a.get('prompt_tokens')} (effort rendered) vs "
                                         f"{b.get('prompt_tokens')} (dropped); the 16-token outputs were both "
                                         f"{a.get('content')!r} / {b.get('content')!r} (reasoning cut at 16 tokens)"}
    t_off, t_on = j(os.path.join(rr, "m53", "tool_off.json")), j(os.path.join(rr, "m53", "tool_on.json"))
    if t_off and t_on:
        a, b = t_off["requests"]["tool_call"], t_on["requests"]["tool_call"]
        out["mk-L11"] = {"reference_detects": bool(a.get("tool_calls")) != bool(b.get("tool_calls")),
                         "measured": f"entail off: {len(a.get('tool_calls') or [])} structured tool calls, the call "
                                     f"left in the text; repaired: {len(b.get('tool_calls') or [])} (rerun/m53)"}
    return out


def entail_detects(final):
    """Whether entail's verdicts on the defect include resolved, broken or refused (M92_PROTOCOL 2)."""
    words = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("fixed", "fixed_output_within", "healthy_none", "healthy_fp8", "diffusers_right_lora"):
                    continue
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x in ("resolved", "broken", "refused"):
            words.add(x)
    walk(final)
    return sorted(words)


def main():
    rope = {n: j(os.path.join(ROPE, f"{n}.json")) for n in ("LC_off_m91", "LA_m91", "LC_m91", "LA", "LA2")}
    m = {n: j(os.path.join(M92, f"{n}.json")) for n in ("softcap_sglang_torch_native", "softcap_sglang_triton",
                                                         "softcap_sglang_triton2", "shift_seeded", "shift_control",
                                                         "shift_control2")}
    evals = {
        "fd-rope": paired(rope["LC_off_m91"], rope["LA_m91"], rope["LA2"] if rope["LA"] else None,
                          "Llama-3.2-3B, vLLM: launch-time rope_scaling (theta lost) vs no override; noise: LA vs LA2"),
        "fd-rope (entail on)": paired(rope["LC_m91"], rope["LA_m91"], None,
                                      "the same override with entail on (repaired) vs no override"),
        "fd-softcap (SGLang)": paired(m["softcap_sglang_torch_native"], m["softcap_sglang_triton"],
                                      m["softcap_sglang_triton2"], "gemma-2-2b-it, SGLang torch_native vs triton"),
        "fd-softcap (HF 2B)": softcap_hf("2b"),
        "fd-softcap (HF 9B)": softcap_hf("9b"),
        "fd-shift": paired(m["shift_seeded"], m["shift_control"], m["shift_control2"],
                           "Qwen3-4B, vLLM: one weight's rows rolled at load vs not"),
    }
    if rope["LA"] and rope["LA2"]:   # the fd-rope noise is LA against LA2 (old runs), not a repeat of LA_m91
        la, la2 = rope["LA"]["gsm8k"]["outputs"], rope["LA2"]["gsm8k"]["outputs"]
        e = evals["fd-rope"]
        if "outputs_differ" in e:
            e["outputs_differ_between_healthy_runs"] = sum(1 for x, y in zip(la, la2) if x != y)
            e["reference_detects"] = e["outputs_differ"] > e["outputs_differ_between_healthy_runs"]
    problems = (j(os.path.join(RES, "m91", "problems.json")) or {}).get("problems", {})
    rb = rolebench()
    other = reference_other()
    table = {}
    for pid, p in problems.items():
        row = {"fact": p["fact"], "entail": entail_detects(p["final"])}
        if pid in rb:
            row.update(rb[pid])
        if pid in other:
            row["reference_other"] = other[pid]
        if pid == "mk-L03":   # Gemma 2's softcap on several engines: fd-softcap's mechanism and measurements
            row["entail"] = entail_detects(problems["fd-softcap"]["final"])
        for k, e in evals.items():
            if k.split(" (")[0] == pid or (pid == "mk-L03" and k.startswith("fd-softcap")):
                row.setdefault("evaluation_measured", {})[k] = e
        table[pid] = row
    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "evaluations": evals, "problems": table}
    os.makedirs(M92, exist_ok=True)
    with open(os.path.join(M92, "S7.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    for k, e in evals.items():
        print(k, json.dumps({x: y for x, y in e.items() if x != "what"}, ensure_ascii=False))
    for pid, row in table.items():
        print(pid, row["entail"], json.dumps(row.get("judged"), ensure_ascii=False)[:200])


if __name__ == "__main__":
    main()
