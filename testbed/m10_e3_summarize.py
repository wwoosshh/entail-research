"""M10 E3: testbed/results/m10/E3_SUMMARY.md from the census, the screening and the reproduction files
(results/m10/e3/{census,screening}.json, e3/cases/<case>_{off,on}.json and _on.record.jsonl).
Definitions: testbed/M10_PROTOCOL.md 3 and 6. The cause-class column is the author's judgment (hypothesis-aware)
and is not used for the headline.
Run: python testbed/m10_e3_summarize.py
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.join(HERE, "results", "m10", "e3")
C = os.path.join(E3, "cases")

# position, issue, short title, case file (None: no run), why not reproduced / scope, entail decisions related to the
# defect (judged from the records: every non-pass decision is listed in the table), cause class (judgment)
CASES = [
    (12, "vllm-project/vllm#53019", "NemotronParse: lm_head left untied, garbage", None,
     "not reproduced here: needs --trust-remote-code and packages this environment lacks (timm, albumentations, "
     "open_clip), not installed without the researcher's permission. Code reading: entail's vLLM tie rule takes the "
     "tie from the config vLLM holds (true), so a loader that fails to tie would pass", "declared tie not honoured "
     "by one loader (class; the check exists but reads the config, not the weights)"),
    (19, "sgl-project/sglang#25790", "FP8 KV cache: prefill and decode logprobs diverge", "sg25790",
     "not reproduced as reported: on Ada with the triton backend the bf16 control diverges too (the report: only "
     "with FP8, from index 96, on fa3)", "numerics (outside)"),
    (22, "sgl-project/sglang#37606", "breakable CUDA graph keeps weak refs: wrong greedy output", "sg37606",
     "reproduced at the mechanism (the report's model-free script); its model run needs TP8 on B300",
     "memory lifetime inside the engine (outside)"),
    (24, "vllm-project/vllm#49377", "token truncation leaves stale block hashes", "vl49449",
     "the same defect and code path as #49449 (Scheduler._update_request_as_session); reproduced once",
     "stale state (class by the broad definition; no fact for cache identity)"),
    (32, "sgl-project/sglang#35564", "tool-call parsers lose or change calls when streamed", "sg35564",
     "reproduced (the report's parser-level script on 0.5.20)", "parser logic (outside)"),
    (37, "vllm-project/vllm#49449", "streaming-session rebuild: stale prefix-cache key, wrong output", "vl49449",
     "reproduced on vLLM 0.30.0 exactly as reported", "stale state (class by the broad definition)"),
    (53, "huggingface/diffusers#14569", "DDPMScheduler fixed_large_log returns NaN", "df14569",
     "reproduced (the report's script and a DDPM pipeline)", "arithmetic inside a component (outside)"),
    (56, "sgl-project/sglang#33493", "DSPARK reads a misspelt penalty key: min_tokens ignored", "sg33493",
     "not reproduced: the misspelt key is no longer in SGLang 0.5.20's speculative code (grep), and with DSPARK "
     "(MiniCPM5-2B + its DSpark drafter) min_new_tokens=50 held (51 tokens; 55 without speculation); the issue is "
     "still open", "a setting under a name nobody reads (class; no hook on SGLang's sampler)"),
    (57, "vllm-project/vllm#49918", "prefill of K+1 tokens dispatched as spec decode: garbage", "vl49918",
     "not reproduced: the only recurrent-state model that fits (Nemotron-H 4B, Mamba-2; the report confirmed GDN "
     "hybrids) does not start on vLLM 0.30 with CUDA graphs, entail off too", "dispatch logic (outside)"),
    (62, "sgl-project/sglang#39626", "block-FP8 Triton matmul accepts a K tile over the quant block", "sg39626",
     "reproduced at kernel level with a manually supplied config (the report says the tuner filters it out)",
     "tile vs quantization block granularity (class; no fact for kernel configs)"),
    (68, "vllm-project/vllm#58138", "cross-encoder padding: wrong token_type_ids, scores change", "vl58138",
     "reproduced (vLLM 0.30's own functions and its /rerank server)",
     "the tokenizer's pad type not used (class; no fact for token types)"),
    (73, "huggingface/transformers#48967", "BERT tokenizer after 5.0: wrong ids", "tf48967_rev",
     "reproduced at the revision before the repository deleted its tokenizer.json (2026-09-22); the current files "
     "are fixed", "tokenizer vocabulary (32,000) against the model's (100,000) (class; no tokenizer fact)"),
]


def jl(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def records(p):
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def main():
    census = jl(os.path.join(E3, "census.json"))
    scr = jl(os.path.join(E3, "screening.json"))
    rows, seen_case = [], {}
    for pos, issue, title, case, note, cls in CASES:
        off = jl(os.path.join(C, f"{case}_off.json")) if case else None
        on = jl(os.path.join(C, f"{case}_on.json")) if case else None
        dec = [d for d in records(os.path.join(C, f"{case}_on.record.jsonl")) if "verdict" in d] if case else []
        not_pass = Counter(f"{d['verdict']} {d['boundary']} {d['name']}" for d in dec if d["verdict"] != "pass")
        reproduced = bool(off and off.get("reproduced"))
        still_wrong_on = bool(on and on.get("reproduced"))
        outcome = ("not reproduced" if not reproduced else
                   "detected (fixed)" if not still_wrong_on else "missed")
        rows.append({"position": pos, "issue": issue, "title": title, "case": case, "reproduced": reproduced,
                     "outcome": outcome, "entail_not_pass": dict(not_pass), "note": note, "cause_class_judged": cls,
                     "duplicate_of_case_run": seen_case.get(case)})
        if case:
            seen_case.setdefault(case, issue)
    repro = [r for r in rows if r["reproduced"]]
    out = {"census": {"issues": len(census["issues"]),
                      "by_population": dict(Counter("+".join(sorted(i["populations"])) for i in census["issues"]))},
           "screening": {k: scr[k] for k in ("screened", "passed", "failed_1_not_an_output_report",
                                             "failed_2_cannot_run_on_one_12gb_card")},
           "sample": len(rows), "reproduced": len(repro),
           "detected": sum(r["outcome"].startswith("detected") for r in rows),
           "missed": sum(r["outcome"] == "missed" for r in rows), "rows": rows}
    json.dump(out, open(os.path.join(E3, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    L = ["# M10 E3: real output bugs, replayed with entail off and on", "",
         "Generated by `testbed/m10_e3_summarize.py`. Definitions: `testbed/M10_PROTOCOL.md` 3 and 6. GitHub was read "
         "with GET only.", "",
         f"- Census (A: filed since the tested version and naming it; B: filed 2026-03-24..09-24 and still open, title "
         f"keywords of the study): {out['census']['issues']} issues {out['census']['by_population']}",
         f"- Screened in a seeded random order until 12 passed: {out['screening']['screened']} screened; "
         f"{out['screening']['failed_1_not_an_output_report']} not a wrong-output report; "
         f"{out['screening']['failed_2_cannot_run_on_one_12gb_card']} cannot run on one 12 GB card; "
         f"{out['screening']['passed']} passed",
         f"- **Reproduced {out['reproduced']} of {out['sample']}; entail detected {out['detected']}, missed "
         f"{out['missed']}.**", "",
         "| # | issue | report | reproduced | entail | entail's decisions other than pass (all unrelated to the defect) "
         "| note | cause class (judgment) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['position']} | {r['issue']} | {r['title']} | {r['reproduced']} | {r['outcome']} | "
                 f"{r['entail_not_pass'] or '-'} | {r['note']} | {r['cause_class_judged']} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(os.path.dirname(E3), "E3_SUMMARY.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
