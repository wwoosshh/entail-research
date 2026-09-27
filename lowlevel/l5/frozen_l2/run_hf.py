"""L2 on transformers: one model, the execution modes of hf_worker.py, compared with the eager base.

  python lowlevel/l2/run_hf.py <model path> <out dir>        (HF_PY names the python; default ~/venvs/gpu)
Comparisons: sdpa, nocache, static, chunked, batch_left, pad_right against eager (alone); each beam search against
itself (its own score against its sequence recomputed without a cache). Writes <out dir>/hf.json and
<out dir>/compare.json, prints one line per comparison.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare  # noqa: E402

HF_PY = os.environ.get("HF_PY", os.path.expanduser("~/venvs/gpu/bin/python"))
MARGIN = float(os.environ.get("L2_MARGIN", "1.0"))
PDRIFT = float(os.environ.get("L2_PDRIFT", "0.25"))  # healthy maximum 0.176 over 8 vLLM + 3 HF models (calibration 1)
BEAM_GAP = float(os.environ.get("L2_BEAM_GAP", "0.05"))  # healthy gaps 0.001-0.007 (Llama-3.2-3B, Qwen3-0.6B, 5.17)


def main():
    model, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "hf.json")
    log = open(os.path.join(out_dir, "hf.log"), "w", encoding="utf-8")
    # a model whose kernels fall back to pure PyTorch here (no mamba_ssm / causal_conv1d) is too slow for every
    # mode: hf_modes_override.json names the modes to run for it, by folder name
    modes = []
    ov = os.path.join(HERE, "hf_modes_override.json")
    if os.path.exists(ov):
        for key, ms in json.load(open(ov, encoding="utf-8")).items():
            if key in os.path.basename(os.path.normpath(model)):
                modes = ms
    try:
        subprocess.run([HF_PY, os.path.join(HERE, "hf_worker.py"), model, out] + modes, stdout=log,
                       stderr=subprocess.STDOUT, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"{os.path.basename(model):<40} WORKER TIMED OUT (see hf.log)")
        return
    if not os.path.exists(out):
        print(f"{os.path.basename(model):<40} WORKER FAILED (see hf.log)")
        return
    res = json.load(open(out, encoding="utf-8"))
    plans = res["plans"]
    base = plans.get("eager")
    summary = {"model": model, "errors": res.get("errors"), "comparisons": {}}
    pairs = [(m, "eager") for m in ("sdpa", "nocache", "static", "chunked", "batch_left", "pad_right")]
    # beam search: two searches may part at a tie and end far apart (calibration 1: Llama-3.2-3B scored -1.39 and -0.80
    # per token with and without the cache, each consistent with a recompute), so they are not compared with each
    # other; each is held to itself - its own score for its sequence against that sequence recomputed without a cache
    for m in ("beam_cache", "beam_nocache"):
        if m in plans:
            gaps = {pid: abs(r["beam_internal"] - r["recomputed_mean"]) for pid, r in plans[m].items()
                    if "beam_internal" in r}
            worst = max(gaps.items(), key=lambda x: x[1]) if gaps else (None, 0.0)
            bad = worst[1] > BEAM_GAP
            summary["comparisons"][f"{m}_self"] = {"verdict": {"differs": bad, "probe": worst[0] if bad else None},
                                                   "internal_vs_recomputed_max": worst[1], "gaps": gaps}
            print(f"{os.path.basename(model):<40} {m + '_self':<24} {'DIFFERS' if bad else 'same   '} "
                  f"beam's own score vs the recomputed mean log-prob, worst gap {worst[1]:.4f} ({worst[0]})")
    for m, ref in pairs:
        if m not in plans or ref not in plans:
            summary["comparisons"][f"{m}_vs_{ref}"] = {"verdict": "mode failed", "error": (res.get("errors") or {}).get(m)}
            print(f"{os.path.basename(model):<40} {m + '_vs_' + ref:<24} MODE FAILED {(res.get('errors') or {}).get(m, '')[:120]}")
            continue
        cmp = compare.pair(plans[ref], plans[m])
        v = compare.verdict(cmp, MARGIN, PDRIFT)
        st = compare.summary(cmp)
        summary["comparisons"][f"{m}_vs_{ref}"] = {"verdict": v, **st, "detail": cmp}
        print(f"{os.path.basename(model):<40} {m + '_vs_' + ref:<24} {'DIFFERS' if v['differs'] else 'same   '} "
              f"dp gen {st['gen_pdrift']:.3f} prompt {st['prompt_pdrift']:.3f} | dlogp gen {st['gen_drift']:.3f} "
              f"prompt {st['prompt_drift']:.3f} | flip margin gen {st['gen_flip_margin']:.2f} prompt "
              f"{st['prompt_flip_margin']:.2f} | first_diff {st['first_diff']}"
              + (f" | {v['probe']}: {'; '.join(v['why'])}" if v["differs"] else ""))
    json.dump(summary, open(os.path.join(out_dir, "compare.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
