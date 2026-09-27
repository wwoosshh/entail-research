"""L2 on vLLM: one model, several execution modes that must mean the same thing, compared with the eager base.

  python lowlevel/l2/run_vllm.py <model path> <out dir> [modes...]      (run from WSL with any python3)
Modes (each its own engine process, `vllm_worker.py` in ~/venvs/vllm unless VLLM_PY names another python):
  base    enforce_eager, prefix caching on; plans alone, batched, cache (the reference for every comparison)
  graph   the engine's default: torch.compile and CUDA graphs
  chunk   eager with max_num_batched_tokens 64, so every long prompt is prefilled in chunks
  spec    eager with n-gram speculative decoding (4 tokens)
  native  eager with every custom op on its native definition (custom_ops none)
  triton_attn  eager with the Triton attention backend
  fp16    eager in float16 (the dtype an engine falls back to on GPUs without bfloat16)
  supplied  eager with prompt embeddings enabled: each probe as the same ids and embedding tensor under three
          prompt_is_token_ids masks, cold and right after each other one (the prefix cache must keep them apart)
Comparisons: decode vs prefill (the generated tokens teacher-forced), batched vs alone, cache hit (partial, full)
vs alone, each other mode vs base, every "supplied" request warm vs cold, and non-finite log-probabilities anywhere.
L2_IMAGES=1 adds the image probes (probes.IMAGE_PROBES) to base and fp16 for a multimodal model, compared the same
way; L2_ENGINE takes a JSON object of engine arguments for every mode (memory settings for a large model).
Writes <out dir>/<mode>.json, <out dir>/compare.json and prints one line per comparison.
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare  # noqa: E402
from probes import CACHE_PAIRS, IMAGE_PROBES, PROBES  # noqa: E402

VLLM_PY = os.environ.get("VLLM_PY", os.path.expanduser("~/venvs/vllm/bin/python"))
MARGIN = float(os.environ.get("L2_MARGIN", "1.0"))    # healthy maximum flip margin 0.62 (calibration 1)
PDRIFT = float(os.environ.get("L2_PDRIFT", "0.25"))  # healthy maximum 0.176 over 8 vLLM + 3 HF models (calibration 1)
MODES = {
    "base": ({"enforce_eager": True}, ["alone", "forced", "batched", "cache"], True),
    "graph": ({"enforce_eager": False}, ["alone"], True),
    "chunk": ({"enforce_eager": True, "max_num_batched_tokens": 64}, ["alone"], True),
    "spec": ({"enforce_eager": True, "speculative_config": {"method": "ngram", "num_speculative_tokens": 4,
                                                            "prompt_lookup_max": 4, "prompt_lookup_min": 2}},
             ["alone"], False),
    # every custom op on its native definition instead of its kernel (the kernels against their own definitions,
    # all at once, at the output)
    "native": ({"enforce_eager": True, "compilation_config": {"custom_ops": ["none"]}}, ["alone"], True),
    # another attention backend
    "triton_attn": ({"enforce_eager": True, "attention_backend": "TRITON_ATTN"}, ["alone"], True),
    "fp16": ({"enforce_eager": True, "dtype": "float16"}, ["alone"], True),
    "supplied": ({"enforce_eager": True, "enable_prompt_embeds": True}, ["supplied"], False),
}


def run_mode(model, out_dir, name):
    engine, plans, plp = MODES[name]
    engine = dict(engine, **json.loads(os.environ.get("L2_ENGINE", "{}")))
    if os.environ.get("L2_IMAGES") and name in ("base", "fp16"):
        plans = plans + ["image"]
    spec = {"model": model, "engine": engine, "plans": plans, "prompt_logprobs": plp, "max_tokens": 24,
            "probes": PROBES, "cache_pairs": CACHE_PAIRS, "image_probes": IMAGE_PROBES}
    sp = os.path.join(out_dir, f"{name}.spec.json")
    out = os.path.join(out_dir, f"{name}.json")
    if os.environ.get("L2_REUSE") and os.path.exists(out):     # recompute the comparisons from a finished run
        return json.load(open(out, encoding="utf-8"))
    json.dump(spec, open(sp, "w", encoding="utf-8"))
    log = open(os.path.join(out_dir, f"{name}.log"), "w", encoding="utf-8")
    r = subprocess.run([VLLM_PY, os.path.join(HERE, "vllm_worker.py"), sp, out], stdout=log, stderr=subprocess.STDOUT,
                       timeout=1800)
    if r.returncode != 0 or not os.path.exists(out):
        return None
    return json.load(open(out, encoding="utf-8"))


def nonfinite_logprobs(res):
    """{mode/plan: count} of log-probabilities that are NaN or +inf in any recorded step or prompt position."""
    bad = {}

    def walk(x, key):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, key)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, float) and (math.isnan(x) or x == math.inf):
            bad[key] = bad.get(key, 0) + 1

    for mode, r in res.items():
        for plan, recs in ((r or {}).get("plans") or {}).items():
            walk(recs, f"{mode}/{plan}")
    return bad


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
            rows["cache_partial_vs_cold"] = compare.pair({tgt: alone[tgt]}, {tgt: c["partial"]})
            rows["cache_full_vs_cold"] = compare.pair({tgt: alone[tgt]}, {tgt: c["full"]})
        for m in ("graph", "chunk", "spec", "native", "triton_attn", "fp16"):
            if res.get(m):
                rows[f"{m}_vs_base"] = compare.pair(alone, res[m]["plans"]["alone"])
                if "forced" in base["plans"]:
                    # the mode's decode path against the base's prefill path, up to the first token where it parts
                    rows[f"{m}_decode_vs_prefill"] = compare.decode_vs_prefill(res[m]["plans"]["alone"],
                                                                               base["plans"]["forced"], alone)
            elif m in modes:
                rows[f"{m}_vs_base"] = None
        if res.get("fp16") and base["plans"].get("image") and res["fp16"]["plans"].get("image"):
            rows["image_fp16_vs_base"] = compare.pair(base["plans"]["image"], res["fp16"]["plans"]["image"])
    sup = ((res.get("supplied") or {}).get("plans") or {}).get("supplied")
    if sup:
        cold, warm = {}, {}
        for pid, r in sup.items():
            for key, rec in r["warm"].items():
                cold[f"{pid}:{key}"] = r["cold"][key.split("_after_")[0]]
                warm[f"{pid}:{key}"] = rec
        rows["supplied_warm_vs_cold"] = compare.pair(cold, warm)
    elif "supplied" in modes:
        rows["supplied_warm_vs_cold"] = None
    bad = nonfinite_logprobs(res)
    if bad:
        print(f"{os.path.basename(model):<40} NON-FINITE log-probabilities: {bad}")
    summary = {"model": model, "margin": MARGIN, "pdrift": PDRIFT, "nonfinite_logprobs": bad,
               "load_s": {m: (r or {}).get("load_s") for m, r in res.items()},
               "failed_modes": [m for m, r in res.items() if r is None], "comparisons": {}}
    for name, cmp in rows.items():
        if cmp is None:
            summary["comparisons"][name] = {"verdict": "mode failed to run"}
            print(f"{os.path.basename(model):<40} {name:<24} MODE FAILED")
            continue
        v = compare.verdict(cmp, MARGIN, PDRIFT)
        st = compare.summary(cmp)
        summary["comparisons"][name] = {"verdict": v, **st, "detail": cmp}
        print(f"{os.path.basename(model):<40} {name:<24} {'DIFFERS' if v['differs'] else 'same   '} "
              f"dp gen {st['gen_pdrift']:.3f} prompt {st['prompt_pdrift']:.3f} | dlogp gen {st['gen_drift']:.3f} "
              f"prompt {st['prompt_drift']:.3f} | flip margin gen {st['gen_flip_margin']:.2f} prompt "
              f"{st['prompt_flip_margin']:.2f} | first_diff {st['first_diff']}"
              + (f" | {v['probe']}: {'; '.join(v['why'])}" if v["differs"] else ""))
    json.dump(summary, open(os.path.join(out_dir, "compare.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
