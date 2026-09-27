"""M19 L2: a tolerance for quantized checkpoints in the layer comparison, set from healthy engines.

The layer comparison (layers.py compare) flags the first layer where the engine leaves the float32 reference by more
than 4 x the reference's own bfloat16 noise + 0.02. The engine of a quantized checkpoint also carries the
quantization error, so a healthy quantized engine can cross that line (RESULTS.md, vllm#38643 fixed version). This
script puts every dumped case on the same ruler, two measures and four rules:
  frob  the current measure: relative Frobenius error over all tokens of a layer;
  tok   the median over tokens of each token's relative error (a few very large tokens cannot dominate it);
  R0    the current rule on frob;
  R1    R0 with an allowance for the checkpoint's declared quantization: the largest frob error a healthy engine with
        the same declared weight bits showed, taken from the calibration runs only (threshold 4 x max(noise, allowance)
        + 0.02);
  R2    R0 on tok;  R3  R1 on tok.
The declared weight bits come from the checkpoint's own quantization_config (none for an unquantized checkpoint).

  python lowlevel/l2/quant_tolerance.py        (python with torch; reads results/, writes results/quant_tolerance.json)
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layers import declared_bits  # noqa: E402
from probes import PROBES  # noqa: E402

RES = os.path.join(HERE, "results")
FACTOR, FLOOR = 4.0, 0.02

# name, engine dump, folder holding ref32/ and ref16/, role (calibration runs set the allowance; bugs must show)
CASES = [
    ("Qwen3-0.6B bf16, vLLM 0.30.0", "layers/qwen06_vllm0300/engine", "layers/qwen06_vllm0300", "healthy"),
    ("Llama-3.2-3B bf16, vLLM 0.30.0", "layers_quant/bf16_llama32_3b/engine", "layers_quant/bf16_llama32_3b", "healthy"),
    ("GLM-OCR bf16, vLLM 0.30.0 (fixed)", "layers/glmocr_vllm0300/engine", "layers/glmocr_vllm0300", "healthy"),
    ("Qwen2.5-3B AWQ, vLLM 0.30.0", "layers_quant/awq_qwen25_3b/engine", "layers_quant/awq_qwen25_3b", "calibration"),
    ("Qwen3-4B FP8, vLLM 0.30.0", "layers_quant/fp8_qwen3_4b/engine", "layers_quant/fp8_qwen3_4b", "calibration"),
    ("Qwen3.5-4B NVFP4, vLLM 0.30.0 (fixed)", "l4_qwen35/vllm0300", "l4_qwen35", "healthy"),
    ("GLM-OCR bf16, vLLM 0.22.0 (vllm#42016)", "layers/glmocr_vllm0220/engine", "layers/glmocr_vllm0220", "bug"),
    ("Qwen3.5-4B NVFP4, vLLM 0.19.0 (vllm#38643)", "l4_qwen35/vllm0190", "l4_qwen35", "bug"),
]


def measures(e, r, h):
    """Per layer: frob and tok error and noise, and how much of the frob error the single largest token carries."""
    rows = []
    for a, b, c in zip(e, r, h):
        a, b, c = a.double(), b.double(), c.double()
        de, dn = a - b, c - b
        tn = b.norm(dim=-1).clamp_min(1e-12)
        de2 = de.norm(dim=-1) ** 2
        rows.append({
            "frob_err": de.norm().item() / (b.norm().item() or 1.0),
            "frob_noise": dn.norm().item() / (b.norm().item() or 1.0),
            "tok_err": (de.norm(dim=-1) / tn).median().item(),
            "tok_noise": (dn.norm(dim=-1) / tn).median().item(),
            "top_token_share": (de2.max() / de2.sum().clamp_min(1e-30)).item(),
            "top_token_norm_x_median": (tn.max() / tn.median()).item(),
        })
    return rows


def judge(rows, m, allow):
    """First layer over the line and the largest err / threshold over all layers."""
    first, worst = None, 0.0
    for i, r in enumerate(rows):
        ratio = r[f"{m}_err"] / (FACTOR * max(r[f"{m}_noise"], allow) + FLOOR)
        worst = max(worst, ratio)
        if first is None and ratio > 1:
            first = i
    return first, worst


def main():
    data = {}
    for name, eng, ref, role in CASES:
        e_dir, r32, r16 = (os.path.join(RES, p) for p in (eng, f"{ref}/ref32", f"{ref}/ref16"))
        model = json.load(open(os.path.join(e_dir, "meta.json")))["model"]
        per = {}
        for p in PROBES:
            e, r, h = (torch.load(os.path.join(d, f"{p['id']}.pt")) for d in (e_dir, r32, r16))
            if e["ids"] != r["ids"] or len(e["layers"]) != len(r["layers"]):
                per[p["id"]] = None
                continue
            per[p["id"]] = measures(e["layers"], r["layers"], h["layers"])
        data[name] = {"role": role, "model": model, "bits": declared_bits(model), "probes": per}

    allow = {"frob": {}, "tok": {}}
    for d in data.values():
        if d["role"] != "calibration" or not isinstance(d["bits"], int):
            continue
        for rows in d["probes"].values():
            for m in allow:
                top = max(r[f"{m}_err"] for r in rows)
                allow[m][d["bits"]] = max(allow[m].get(d["bits"], 0.0), top)

    rules = {"R0": ("frob", False), "R1": ("frob", True), "R2": ("tok", False), "R3": ("tok", True)}
    out = {"factor": FACTOR, "floor": FLOOR, "allowance": {m: {str(k): v for k, v in a.items()} for m, a in allow.items()},
           "cases": {}}
    print("allowance by declared weight bits:", json.dumps(out["allowance"]))
    for name, d in data.items():
        probes = {k: v for k, v in d["probes"].items() if v}
        res = {"role": d["role"], "bits": d["bits"], "model": d["model"], "rules": {}}
        for rn, (m, use) in rules.items():
            a = 0.0
            if use and isinstance(d["bits"], int):
                a = allow[m].get(d["bits"], max(allow[m].values()))
            per = {pid: judge(rows, m, a) for pid, rows in probes.items()}
            firsts = [f for f, _ in per.values() if f is not None]
            res["rules"][rn] = {"allowance": a, "flagged": len(firsts), "probes": len(per),
                                "earliest_layer": min(firsts) if firsts else None,
                                "max_err_over_threshold": max(w for _, w in per.values()),
                                "per_probe": {pid: {"first": f, "max_ratio": w} for pid, (f, w) in per.items()}}
        res["max"] = {k: max(r[k] for rows in probes.values() for r in rows)
                      for k in ("frob_err", "tok_err", "frob_noise", "tok_noise", "top_token_share", "top_token_norm_x_median")}
        pid, li, r = max(((pid, i, r) for pid, rows in probes.items() for i, r in enumerate(rows)),
                         key=lambda x: x[2]["frob_err"])
        res["at_max_frob"] = {"probe": pid, "layer": li, **r}
        out["cases"][name] = res
        cells = "  ".join(f"{rn} {v['flagged']}/{v['probes']} L{v['earliest_layer']} x{v['max_err_over_threshold']:.3g}"
                          for rn, v in res["rules"].items())
        mx = res["max"]
        print(f"{name:<44} {d['role']:<11} bits={d['bits']}  frob<= {mx['frob_err']:.3g} tok<= {mx['tok_err']:.3g} "
              f"top-token share<= {mx['top_token_share']:.2f}\n    {cells}")
    json.dump(out, open(os.path.join(RES, "quant_tolerance.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
