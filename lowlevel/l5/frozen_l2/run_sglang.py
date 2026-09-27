"""L2 on SGLang: one model, several execution modes that must mean the same thing, compared with the base.

  python lowlevel/l2/run_sglang.py <model path> <out dir> [modes...]     (SGLANG_PY names the python; ~/venvs/sglang)
Modes (each its own engine):
  base        CUDA graphs off, radix cache on; plans alone, forced, batched, cache (the reference)
  graph       CUDA graphs on (the engine's default)
  nooverlap   the overlap scheduler off
  chunk       chunked_prefill_size 64
  triton      the Triton attention backend
  torch_native  the torch-native attention backend
  compile     torch.compile on
  spec        n-gram speculative decoding
Comparisons as in run_vllm.py: decode vs prefill, batched vs alone, cache hit vs cold, each mode vs base and that
mode's decode path vs the base's prefill path. Writes <out dir>/<mode>.json and compare.json.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare  # noqa: E402
from probes import CACHE_PAIRS, PROBES  # noqa: E402

SGLANG_PY = os.environ.get("SGLANG_PY", os.path.expanduser("~/venvs/sglang/bin/python"))
MARGIN = float(os.environ.get("L2_MARGIN", "1.0"))
PDRIFT = float(os.environ.get("L2_PDRIFT", "0.25"))
MODES = {
    "base": ({"disable_cuda_graph": True}, ["alone", "forced", "batched", "cache"]),
    "graph": ({"disable_cuda_graph": False}, ["alone"]),
    "nooverlap": ({"disable_cuda_graph": True, "disable_overlap_schedule": True}, ["alone"]),
    "chunk": ({"disable_cuda_graph": True, "chunked_prefill_size": 64}, ["alone"]),
    "triton": ({"disable_cuda_graph": True, "attention_backend": "triton"}, ["alone"]),
    "torch_native": ({"disable_cuda_graph": True, "attention_backend": "torch_native"}, ["alone"]),
    "compile": ({"disable_cuda_graph": False, "enable_torch_compile": True}, ["alone"]),
    "spec": ({"disable_cuda_graph": True, "speculative_algorithm": "NGRAM", "speculative_num_draft_tokens": 4},
             ["alone"]),
    # many requests at once (probes x4, every other one stopping after one token): each request's prompt log-probs
    # against its prompt's alone run in the same engine; "pressure" with a KV pool of 1024 tokens, admission at the
    # least conservative setting and mixed prefill/decode batches, so requests are retracted while batches mix
    # finishing requests with prefills (the log's "retracted_reqs" lines say whether it happened)
    "crowd": ({"disable_cuda_graph": True}, ["alone", "crowd"]),
    "pressure": ({"disable_cuda_graph": True, "max_total_tokens": 1024, "schedule_conservativeness": 0.01,
                  "enable_mixed_chunk": True, "chunked_prefill_size": 128, "log_level": "info"}, ["alone", "crowd"]),
}


def run_mode(model, out_dir, name):
    engine, plans = MODES[name]
    spec = {"model": model, "engine": engine, "plans": plans, "max_tokens": 24, "probes": PROBES,
            "cache_pairs": CACHE_PAIRS}
    sp = os.path.join(out_dir, f"{name}.spec.json")
    out = os.path.join(out_dir, f"{name}.json")
    if os.environ.get("L2_REUSE") and os.path.exists(out):
        return json.load(open(out, encoding="utf-8"))
    json.dump(spec, open(sp, "w", encoding="utf-8"))
    log = open(os.path.join(out_dir, f"{name}.log"), "w", encoding="utf-8")
    try:
        r = subprocess.run([SGLANG_PY, os.path.join(HERE, "sglang_worker.py"), sp, out], stdout=log,
                           stderr=subprocess.STDOUT, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.path.exists(out):
        return None
    return json.load(open(out, encoding="utf-8"))


def main():
    model, out_dir = sys.argv[1], sys.argv[2]
    modes = sys.argv[3:] or list(MODES)
    os.makedirs(out_dir, exist_ok=True)
    res = {m: run_mode(model, out_dir, m) for m in modes}
    base = res.get("base")
    rows = {}
    if base:
        alone = base["plans"]["alone"]
        if "forced" in base["plans"]:
            rows["decode_vs_prefill"] = compare.decode_vs_prefill(alone, base["plans"]["forced"])
        if "batched" in base["plans"]:
            rows["batched_vs_alone"] = compare.pair(alone, base["plans"]["batched"])
        for tgt, c in (base["plans"].get("cache") or {}).items():
            cold = {tgt: dict(alone[tgt], prompt_logprobs=None)}
            rows["cache_partial_vs_cold"] = compare.pair(cold, {tgt: c["partial"]})
            rows["cache_full_vs_cold"] = compare.pair(cold, {tgt: c["full"]})
        for m in MODES:
            if m == "base" or m not in modes:
                continue
            if res.get(m):
                rows[f"{m}_vs_base"] = compare.pair(alone, res[m]["plans"]["alone"])
                if "forced" in base["plans"]:
                    rows[f"{m}_decode_vs_prefill"] = compare.decode_vs_prefill(res[m]["plans"]["alone"],
                                                                               base["plans"]["forced"], alone)
            else:
                rows[f"{m}_vs_base"] = None
    for m in ("crowd", "pressure"):
        r = res.get(m)
        if r and r["plans"].get("crowd"):
            alone_m, a2 = r["plans"]["alone"], {}
            for key in r["plans"]["crowd"]:
                a2[key] = dict(alone_m[key.split("#")[0]], tokens=[], logprobs=[])
            rows[f"{m}_crowd_vs_alone"] = compare.pair(a2, r["plans"]["crowd"])
        elif m in modes:
            rows[f"{m}_crowd_vs_alone"] = None
    summary = {"model": model, "margin": MARGIN, "pdrift": PDRIFT,
               "load_s": {m: (r or {}).get("load_s") for m, r in res.items()},
               "failed_modes": [m for m, r in res.items() if r is None], "comparisons": {}}
    for name, cmp in rows.items():
        if cmp is None:
            summary["comparisons"][name] = {"verdict": "mode failed to run"}
            print(f"{os.path.basename(os.path.normpath(model)):<40} {name:<28} MODE FAILED")
            continue
        v = compare.verdict(cmp, MARGIN, PDRIFT)
        st = compare.summary(cmp)
        summary["comparisons"][name] = {"verdict": v, **st, "detail": cmp}
        print(f"{os.path.basename(os.path.normpath(model)):<40} {name:<28} {'DIFFERS' if v['differs'] else 'same   '} "
              f"dp gen {st['gen_pdrift']:.3f} prompt {st['prompt_pdrift']:.3f} | flip margin gen "
              f"{st['gen_flip_margin']:.2f} prompt {st['prompt_flip_margin']:.2f} | first_diff {st['first_diff']}"
              + (f" | {v['probe']}: {'; '.join(v['why'])}" if v["differs"] else ""))
    json.dump(summary, open(os.path.join(out_dir, "compare.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
