"""M19 L5.2 summary: the frozen research tools' results on the replay 3/4 cases, counted as lowlevel/l5/PROTOCOL.md
section 4 says (exposed, located, false alarm, other-site alarm, time).

  python lowlevel/l5/summarize_l52.py        (any python3; reads lowlevel/l5/results, writes results/summary.json)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")

# PROTOCOL.md section 2: (replay, consensus class, level, what the tools can do with it, bug run, fixed run)
CASES = {
    "vl58406": (3, "K1", "I", "no tool: video processor settings", None, None),
    "vl57353": (3, "K1", "I", "no tool: server parser", None, None),
    "tf47328": (3, "K1", "D", "no tool: audio generation", None, None),
    "tf49066": (3, "K1", "I", "no model: tokenizer files only", None, None),
    "vl49250": (3, "K1", "D", "no tool: KV connector", None, None),
    "df14213": (3, "K2", "D", "no tool: diffusion", None, None),
    "vl43602": (3, "split", "D", "run", "vl43602_0220", "vl43602_0300"),
    "tf48051": (3, "K4", "D", "no tool: video processor", None, None),
    "df12633": (4, "K1", "D", "no tool: diffusion", None, None),
    "tf43697": (4, "K1", "D", "no tool: object detection", None, None),
    "vl34186": (4, "K1", "D", "no tool: LoRA", None, None),
    "vl33560": (4, "K1", "D", "run", "vl33560_0160", "vl33560_0300"),
    "tf41494": (4, "K1", "I", "no tool: GGUF loading", None, None),
    "tf43450": (4, "K4", "D", "no tool: video processor", None, None),
    "vl33091": (4, "K1", "D", "no tool: audio", None, None),
    "vl35221": (4, "K4", "I", "no tool: server parser", None, None),
    "vl27390": (4, "K4", "I", "run", "vl27390_0110", "vl27390_0300"),
}
# PROTOCOL.md section 4: the signal prefixes that point at each case's expected boundary (none: nothing can locate it)
LOCATING = {"vl43602": ("graph_",), "vl33560": (), "vl27390": ()}
SITE = [("graph_", "compile/CUDA graph path"), ("native_", "custom-op kernel path"), ("triton_attn_", "attention backend"),
        ("chunk_", "chunked prefill"), ("spec_", "speculative decoding"), ("fp16_", "float16 path"),
        ("image_fp16", "float16 path"), ("batched_", "batching"), ("cache_", "prefix cache"),
        ("supplied_", "prefix cache"), ("decode_vs_prefill", "decode path"), ("layers", "layer"),
        ("nonfinite", "finiteness (no site)")]


def site(signal):
    return next((s for p, s in SITE if signal.startswith(p)), "?")


def read_run(name):
    """Signals of one run_vllm.py run and one layers comparison: {signal: detail}, failed modes, load times."""
    out = {"signals": {}, "failed_modes": None, "load_s": None, "ran": False, "layers": None}
    p = os.path.join(R, name, "compare.json")
    if os.path.exists(p):
        c = json.load(open(p, encoding="utf-8"))
        out["ran"] = bool(c.get("comparisons")) or bool(c.get("load_s"))
        out["failed_modes"], out["load_s"] = c.get("failed_modes"), c.get("load_s")
        for k, v in (c.get("comparisons") or {}).items():
            verdict = v.get("verdict")
            if isinstance(verdict, dict) and verdict.get("differs"):
                out["signals"][k] = f"{verdict.get('probe')}: {'; '.join(verdict.get('why') or [])}"
        if c.get("nonfinite_logprobs"):
            out["signals"]["nonfinite"] = c["nonfinite_logprobs"]
    lp = os.path.join(R, f"layers_{name}", "engine", "layers_compare.json")
    if os.path.exists(lp):
        lc = json.load(open(lp, encoding="utf-8"))
        lc.pop("_declared", None)
        hits = {pid: (r.get("first_diverging_layer"), r.get("first_diverging_layer_tok")) for pid, r in lc.items()
                if isinstance(r, dict) and (r.get("first_diverging_layer") is not None
                                            or r.get("first_diverging_layer_tok") is not None)}
        errs = {pid: r["error"] for pid, r in lc.items() if isinstance(r, dict) and "error" in r}
        out["layers"] = {"compared": len(lc) - len(errs), "errors": errs, "diverging": hits}
        if hits:
            out["signals"]["layers"] = hits
    elif os.path.isdir(os.path.join(R, f"layers_{name}")):
        out["layers"] = {"compared": 0, "errors": {"all": "no comparison written (see the layers .out files)"}}
    return out


def main():
    rows, tally = {}, {"in_class": {"n": 0, "exposed": 0, "located": 0, "D": [0, 0], "I": [0, 0], "ran": 0,
                                    "ran_exposed": 0},
                       "out_of_class": {"n": 0, "exposed": 0, "ran": 0},
                       "fixed_runs": {"n": 0, "false_alarms": 0}, "not_run": {}}
    for case, (rep, cls, lvl, what, bug, fixed) in CASES.items():
        row = {"replay": rep, "class": cls, "level": lvl, "tools": what}
        exposed = located = False
        if what == "run":
            b = read_run(bug)
            row["bug"] = b
            exposed = bool(b["signals"])
            located = any(s.startswith(LOCATING[case]) for s in b["signals"]) if LOCATING[case] else False
            row["other_site_alarms"] = {s: site(s) for s in b["signals"]
                                        if not (LOCATING[case] and s.startswith(LOCATING[case]))}
            f = read_run(fixed)
            row["fixed"] = f
            if f["ran"] or f["layers"]:
                tally["fixed_runs"]["n"] += 1
                if f["signals"]:
                    tally["fixed_runs"]["false_alarms"] += 1
        else:
            tally["not_run"][what] = tally["not_run"].get(what, 0) + 1
        row["exposed"], row["located"] = exposed, located
        rows[case] = row
        if cls == "K1":
            t = tally["in_class"]
            t["n"] += 1
            t["exposed"] += exposed
            t["located"] += located
            t[lvl][0] += exposed
            t[lvl][1] += 1
            if what == "run":
                t["ran"] += 1
                t["ran_exposed"] += exposed
        else:
            t = tally["out_of_class"]
            t["n"] += 1
            t["exposed"] += exposed
            t["ran"] += what == "run"
    json.dump({"rows": rows, "tally": tally}, open(os.path.join(R, "summary.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False, default=str)
    for case, r in rows.items():
        line = f"{case:<8} {r['class']:<5} {r['level']} {r['tools']:<36}"
        if r["tools"] == "run":
            b, f = r["bug"], r["fixed"]
            line += (f" bug: signals {sorted(b['signals']) or '-'} failed {b['failed_modes']} | fixed: signals "
                     f"{sorted(f['signals']) or '-'} failed {f['failed_modes']} | located {r['located']}")
        print(line)
    print(json.dumps(tally, indent=1))


if __name__ == "__main__":
    main()
