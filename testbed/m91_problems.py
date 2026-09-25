"""M9.1, S2 per test problem (testbed/PROBLEMS.md 1-3) on the final code: what entail said about the defect under the
default policy (report what is not repaired and go on) and under the strict one (ENTAIL_ON_BROKEN=stop; for the
code-boundary cases, debug mode), what it said about the fixed version, and whether the output was right, read from
the re-run files (m91_run.sh B, S, T; results/m91/rerun, rerun_strict, engines, rb17). The ComfyUI cases (fd-m7,
fd-lora) were not re-run - they need the researcher's install - and are read from M6.3's files, marked so.

Nothing is judged here: each cell is the verdicts found (a set, "-" when the file has none) or a value from the file.
The expected verdicts are PROBLEMS.md's as S2 reads them since M5.4 (LIBRARY_DESIGN.md 8): repaired -> resolved;
not repairable -> broken by default and refused under the strict policy.
Writes testbed/results/m91/problems.json. Run: python testbed/m91_problems.py
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M91 = os.environ.get("M91_DIR") or os.path.join(RES, "m91")   # M11.7: results/m11/m91 for the 1.0.1 rerun
RR, RS, ENG = os.path.join(M91, "rerun"), os.path.join(M91, "rerun_strict"), os.path.join(M91, "engines")
ROPE = os.path.join(os.path.dirname(HERE), "issue_track", "rope_override", "results")
OUT = os.path.join(M91, "problems.json")


def j(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def rec(path, prefix=""):
    """The verdicts a record file holds for boundaries that start with `prefix` (a sorted set), or None."""
    if not os.path.exists(path):
        return None
    out = set()
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "verdict" in r and r["boundary"].startswith(prefix):
            out.add(r["verdict"])
    return sorted(out)


def vs(decisions, prefix=""):
    """Verdicts in a list of decisions (dicts, or lists holding a verdict word)."""
    if decisions is None:
        return None
    out = set()
    for d in decisions:
        if isinstance(d, dict):
            if "verdict" in d and d.get("boundary", "").startswith(prefix):
                out.add(d["verdict"])
        else:
            if d and str(d[0]).startswith(prefix):
                out.update(x for x in d if x in ("pass", "resolved", "refused", "broken", "unknown"))
    return sorted(out)


def m43(case):
    d = j(os.path.join(RR, "m43", "SUMMARY.json"))
    if not d:
        return None
    c = d["cases"][case]
    return {"mode": "debug (code boundaries are checked in debug mode)",
            "strict": [c["defect"]["outcome"]], "fixed": [c["fixed"]["outcome"]],
            "defect_decisions": vs(c["defect"].get("decisions")),
            "repaired_output_within": (c["defect"].get("vs_reference") or {}).get("within"),
            "fixed_output_within": (c["fixed"].get("vs_reference") or {}).get("within"), "blocked": c.get("blocked")}


def m3(key):
    d, s = j(os.path.join(RR, "m3", "problems.json")), j(os.path.join(RS, "m3", "problems.json"))
    if not d:
        return None
    p, ps = d.get(key) or {}, (s or {}).get(key) or {}
    out = {}
    if "defect" in p:
        out["default"] = sorted(set(vs(p["defect"].get("decisions") or p["defect"].get("static")) or []))
        out["fixed"] = vs(p["fixed"].get("decisions") or p["fixed"].get("static"))
        out["output_right_defect"] = p["defect"].get("output_right")
        out["output_right_fixed"] = p["fixed"].get("output_right")
        if ps:
            out["strict"] = sorted(set(vs(ps["defect"].get("decisions") or ps["defect"].get("static")) or []))
            out["strict_stopped"] = ps["defect"].get("runtime_stopped")
    else:
        out["default"] = vs(p.get("decisions"))
        for k in ("defect_right", "routed_right", "defect_off_right", "defect_on_right", "used_with_entail",
                  "default_off", "default_on"):
            if k in p:
                out[k] = p[k]
        if ps:
            out["strict"] = vs(ps.get("decisions"))
    return out


def m55(case):
    d = j(os.path.join(RR, "m55", "rolebench.json"))
    if not d:
        return None
    c = d["cases"][case]
    return {"default": vs(c["on"]["defect"]["decisions"]), "default_stopped": c["on"]["defect"]["stopped"],
            "default_output_within": c["on"]["defect"]["vs_reference"].get("within"),
            "strict": vs(c["strict"]["defect"]["decisions"]), "strict_stopped": c["strict"]["defect"]["stopped"],
            "fixed": vs(c["on"]["fixed"]["decisions"]) or ["pass (no decision other than pass)"],
            "fixed_output_within": c["on"]["fixed"]["vs_reference"].get("within"),
            "off_defect_within": c["off"]["defect"]["vs_reference"].get("within")}


def rb10():
    d = j(os.path.join(RR, "m52", "rb10.json"))
    if not d:
        return None
    return {"default": vs(d["on_resolve"]["defect"]["decisions"]),
            "default_output_within": d["on_resolve"]["defect"]["vs_reference"].get("within"),
            "observe_refuse_policy": d["on_refuse"]["defect"].get("refused"),
            "fixed": vs(d["on_resolve"]["fixed"]["decisions"]) or ["pass (no decision other than pass)"],
            "off_defect_within": d["off"]["defect"]["vs_reference"].get("within"),
            "code_boundary_debug": m43("10")}


def rb17():
    d = j(os.path.join(M91, "rb17", "compare.json"))
    if not d:
        return None
    return {"default": vs(d.get("decisions"), "load:sglang.attention"),
            "defect_vs_triton_mean_abs": d["defect_vs_triton"]["mean_abs"],
            "entail_vs_triton_mean_abs": d["entail_torch_native_vs_triton"]["mean_abs"]}


def engine_run(name, prefix):
    r = j(os.path.join(ENG, f"{name}.json"))
    return None if r is None else {"ok": r.get("ok"), "backend_used": r.get("backend_used"),
                                   "verdicts": rec(os.path.join(ENG, f"{name}.record.jsonl"), prefix),
                                   "error": (r.get("error") or "")[:200] or None}


def rope():
    lc, lc_off, la = (j(os.path.join(ROPE, f"{n}.json")) for n in ("LC_m91", "LC_off_m91", "LA_m91"))
    score = lambda r: r and f"{sum(r['gsm8k']['correct'])}/{r['gsm8k']['n']}"  # noqa: E731
    rope_at = [v for b in ("load:transformers.config.rope", "load:vllm.config.rope")
               for v in (rec(os.path.join(ENG, "fd_rope_on.record.jsonl"), b) or [])]
    return {"default": sorted(set(rope_at)),
            "rope_in_engine_on": lc and lc.get("rope_parameters_in_engine"),
            "gsm8k_on": score(lc), "gsm8k_off": score(lc_off), "gsm8k_control": score(la)}


def kv():
    t = j(os.path.join(RR, "m51", "transformers.json")) or {}
    ts = j(os.path.join(RS, "m51", "transformers.json")) or {}
    k = j(os.path.join(RR, "m73", "kv_seeded.json")) or {}   # M7.3's kv_seeded: the default policy, with the record
    out = {"transformers_default (m73 kv_seeded)": {"broken": (k.get("located") or {}).get("broken"),
                                                    "output_wrong": k.get("output_wrong")},
           "transformers_default (m51, stops or not)": t.get("seeded_on"),
           "transformers_strict (m51)": ts.get("seeded_on")}
    for e, prefix in (("vllm", "container:vllm"), ("sglang", "container:sglang")):
        out[f"{e}_default"] = rec(os.path.join(RR, "m51", f"{e}_seeded.jsonl"), prefix)
        out[f"{e}_strict"] = rec(os.path.join(RS, "m51", f"{e}_seeded.jsonl"), prefix)
        r = j(os.path.join(RR, "m51", f"{e}_seeded.json"))
        out[f"{e}_default_run"] = r and (r.get("error") or "finished")
    return out


def repack():
    out = {}
    for tag in ("roll_on", "transpose_on", "strided_on", "strided_usedata", "healthy_none", "healthy_fp8"):
        out[tag] = rec(os.path.join(RR, "m42", f"{tag}.jsonl"), "load:vllm")
        if tag in ("roll_on", "transpose_on", "strided_on"):
            out[f"{tag}_strict"] = rec(os.path.join(RS, "m42", f"{tag}.jsonl"), "load:vllm")
    return out


def diffusers(tag, prefix="load:diffusers"):
    r = j(os.path.join(RR, "m63", f"diffusers_{tag}.json"))
    if r is None:
        return None
    out = {"verdicts": vs([d for d in r.get("decisions", []) if "verdict" in d], prefix)}
    for k in ("scheduler", "vae_scaling_factor", "outcome", "modules_with_the_adapter", "on_broken"):
        if k in r:
            out[k] = r[k]
    return out


def comfy(tag):
    r = j(os.path.join(RES, "m63", f"comfy_{tag}.json"))
    if r is None:
        return None
    return {"verdicts (M6.3, not re-run)": vs([d for d in r.get("decisions", []) if "verdict" in d], "load:comfyui")
            or vs([d for d in r.get("decisions", []) if "verdict" in d])}


def serve(folder, name, prefix="request:"):
    r = j(os.path.join(RR, folder, f"{name}.json"))
    if r is None:
        return None
    return {"verdicts": vs(r.get("decisions"), prefix),
            "statuses": {k: v.get("status") for k, v in (r.get("requests") or {}).items()}}


def l05():
    d = j(os.path.join(RR, "m55", "l05.json"))
    if not d:
        return None
    return {k: {"off_answer_right": s["off"].get("answer_right"), "on_answer_right": s["on"].get("answer_right"),
                "on": vs(s["on"].get("decisions"))} for k, s in d["scenarios"].items()}


PROBLEMS = [
    ("rb-01", "Layout", "refused", lambda: m43("01")),
    ("rb-02", "Layout(scale format)", "broken / refused", lambda: {"load": m3("rb-02"), "code": m43("02")}),
    ("rb-03", "Layout(stride)", "resolved", lambda: m43("03")),
    ("rb-04", "Reduction", "refused", lambda: m43("04")),
    ("rb-05", "Positions", "resolved or refused", lambda: m43("05")),
    ("rb-06", "ModelProps(sliding_window)", "resolved", lambda: m3("rb-06")),
    ("rb-07", "ModelProps(tie)", "broken / refused (the declaration is false)", lambda: m3("rb-07")),
    ("rb-08", "ModelProps(softcap)", "resolved", lambda: m3("rb-08")),
    ("rb-09", "Epoch, Assumed", "resolved or broken / refused", lambda: m55("09_stale_graph")),
    ("rb-10", "Epoch", "resolved (container) / refused (code, debug)", rb10),
    ("rb-11", "Assumed", "resolved or broken / refused", lambda: m55("11_warmup_specialization")),
    ("rb-12", "KvExtent", "broken / refused", lambda: m55("12_session_restore")),
    ("rb-14", "Epoch", "broken / refused", lambda: m55("14_beam_reorder")),
    ("rb-15", "Coverage(config keys)", "broken / refused", lambda: m3("rb-15")),
    ("rb-16", "Quantized", "resolved or refused", lambda: m43("16")),
    ("rb-17", "ModelProps(softcap)", "resolved", rb17),
    ("fd-rope", "Rotary", "resolved", rope),
    ("fd-softcap", "ModelProps(softcap)", "resolved", lambda: {
        n: engine_run(f"fd_softcap_{n}", "load:") for n in ("sglang_torch_native", "sglang_flex_attention",
                                                               "transformers_paged")}),
    ("fd-m7", "Prediction", "resolved", lambda: {"comfyui": comfy("m7_on"), "diffusers": diffusers("m7_on")}),
    ("fd-lora", "Coverage(LoRA)", "broken / refused", lambda: {
        "comfyui_on": comfy("lora_other_on"), "comfyui_strict": comfy("lora_other_strict"),
        "diffusers_on": diffusers("lora_other_on"), "diffusers_strict": diffusers("lora_other_strict"),
        "diffusers_right_lora": diffusers("lora_right_on")}),
    ("fd-kv", "KvExtent", "broken / refused", kv),
    ("fd-shift", "Layout(load source)", "broken / refused", lambda: {
        "default": engine_run("fd_shift", "load:vllm.weights"), "strict": engine_run("fd_shift_strict", "load:vllm")}),
    ("fd-repack", "Layout, Coverage", "broken / refused", repack),
    ("fd-vae", "LatentScale", "resolved (declared by a manifest)", lambda: {"on": diffusers("vae_on"),
                                                                           "observe": diffusers("vae_observe")}),
    ("mk-L03", "ModelProps(softcap)", "resolved (fd-softcap's mechanism)", lambda: "see fd-softcap"),
    ("mk-L05", "Valid, Origin", "resolved or unknown", l05),
    ("mk-L07", "Coverage(request)", "broken / refused", lambda: {
        "market_on": serve("m55", "market_bug_on"), "market_strict": serve("m55", "market_bug_strict"),
        "m54_healthy_on": serve("m54", "healthy_on"), "m54_healthy_strict": serve("m54", "healthy_strict")}),
    ("mk-L11", "Template(tool-call format)", "resolved or broken / refused", lambda: {
        "on": serve("m54", "tool_on", ""), "observe": serve("m54", "tool_observe", ""),
        "observe_strict": serve("m54", "tool_observe_strict", "")}),
    ("mk-L13", "Template(reasoning history)", "broken / refused", lambda: {
        "keep_on": serve("m54", "keep_on"), "market_on": serve("m55", "market_bug_on"),
        "market_strict": serve("m55", "market_bug_strict")}),
    ("mk-I01", "Coverage(LoRA keys)", "broken / refused (or resolved by a key map)", lambda: {
        "on": diffusers("i01_on"), "strict": diffusers("i01_strict")}),
    ("mk-I04", "Prediction", "resolved", lambda: {"on": diffusers("i04_on"), "observe": diffusers("i04_observe")}),
]


def main():
    rows = {}
    for pid, fact, expected, get in PROBLEMS:
        try:
            got = get()
        except Exception as e:  # noqa: BLE001 - a file of an unexpected shape is reported, not hidden
            got = {"error": f"{type(e).__name__}: {e}"}
        rows[pid] = {"fact": fact, "expected": expected, "final": got}
        print(pid, "|", expected, "|", json.dumps(got, ensure_ascii=False, default=str)[:600])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "problems": rows}, f, ensure_ascii=False, indent=1,
                  default=str)


if __name__ == "__main__":
    main()
