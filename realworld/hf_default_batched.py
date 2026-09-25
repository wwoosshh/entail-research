"""How much does the head-sharing erasure cost in the DEFAULT transformers usage (no compile, no static cache)?

Typical batched use (evaluation harnesses, data generation, RL rollouts on the HF backend): several prompts of
different lengths, left padding, attention_mask, model.generate() with the default DynamicCache and sdpa, greedy.
transformers uses SDPA's native head sharing (enable_gqa) only when attention_mask is None; with a padding mask it
materialises repeat_kv copies of K and V in every layer and every step.
Variants (same prompts, same settings):
  stock         as shipped
  gqa_declared  use_gqa_in_sdpa patched to return True, i.e. the sharing fact is passed even with a mask
Also batch 1 without padding (no mask, stock path already shares heads) as a reference.
Reports generate() wall time, decode tokens/s, output equality, and a GPU kernel-time breakdown of one run.
"""
import json
import os
import statistics
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "phase0"), os.path.join(ROOT, "phase0", "week2")]
from numerics_experiment import load  # noqa: E402
import transformers.integrations.sdpa_attention as SA  # noqa: E402

STOCK = SA.use_gqa_in_sdpa
BASE = ("Compilers translate programs written by people into instructions for machines. A good compiler keeps "
        "track of what every value means: which data is only read, which is written, how long each sequence is, "
        "and which parts of the work can happen at the same time. ") * 40
NEW = 64


def classify(name):
    n = name.lower()
    if any(s in n for s in ("fmha", "flash", "attention")):
        return "attention"
    if any(s in n for s in ("gemm", "gemv", "cutlass", "cublas", "s16816", "xmma", "sm80_", "sm90_")):
        return "gemm"
    if any(s in n for s in ("copy", "cat", "index", "elementwise", "reduce", "softmax", "norm")):
        return "copy_elementwise"
    return "other"


def profile_breakdown(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"hfd_{os.getpid()}_{time.time_ns()}.json")
    prof.export_chrome_trace(path)
    tr = json.load(open(path))
    os.remove(path)
    by, per = {}, {}
    for e in tr["traceEvents"]:
        if e.get("cat") == "kernel" and "dur" in e:
            c = classify(e["name"])
            by[c] = by.get(c, 0.0) + e["dur"] / 1e3
            per[e["name"][:60]] = per.get(e["name"][:60], 0.0) + e["dur"] / 1e3
    return ({k: round(v, 1) for k, v in sorted(by.items(), key=lambda kv: -kv[1])},
            [(n, round(v, 1)) for n, v in sorted(per.items(), key=lambda kv: -kv[1])[:6]])


def main():
    tok, model = load()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_ids = tok(BASE).input_ids
    lengths = [64, 96, 128, 192, 256, 320, 384, 512]
    prompts = [tok.decode(base_ids[:n]) for n in lengths]
    res = {"lengths": lengths, "new_tokens": NEW, "runs": {}}

    def gen(batch):
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False)
        return out[:, enc.input_ids.shape[1]:]

    for label, batch, patch in (("B8_padded_stock", prompts, False), ("B8_padded_gqa_declared", prompts, True),
                                ("B1_nopad_stock", prompts[-1:], False)):
        SA.use_gqa_in_sdpa = (lambda attention_mask, key, value: True) if patch else STOCK
        try:
            gen(batch)  # warm-up
            torch.cuda.synchronize()
            ts, outs = [], None
            for _ in range(3):
                t = time.perf_counter()
                outs = gen(batch)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            bd, top = profile_breakdown(lambda: gen(batch))
            res["runs"][label] = {"generate_s_median": statistics.median(ts),
                                  "new_tokens_per_s": len(batch) * NEW / statistics.median(ts),
                                  "gpu_kernel_ms_by_class": bd, "top_kernels": top,
                                  "tokens": outs.tolist()}
        finally:
            SA.use_gqa_in_sdpa = STOCK
        r = res["runs"][label]
        print(label, round(r["generate_s_median"], 3), "s |", round(r["new_tokens_per_s"], 1), "tok/s |",
              r["gpu_kernel_ms_by_class"], flush=True)
        print("   top:", r["top_kernels"][:4], flush=True)
    a, b = res["runs"]["B8_padded_stock"]["tokens"], res["runs"]["B8_padded_gqa_declared"]["tokens"]
    same_rows = sum(x == y for x, y in zip(a, b))
    res["b8_identical_rows"] = same_rows
    print("B8 greedy outputs identical rows (stock vs gqa_declared):", same_rows, "of", len(a), flush=True)
    for run in res["runs"].values():
        run.pop("tokens")
    with open(os.path.join(HERE, "hf_default_batched.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
