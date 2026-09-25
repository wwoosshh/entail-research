"""Week 2, experiment B (v2): numerics and determinism with REAL Qwen3-4B weights, separating two effects.

  (1) Kernel-level numeric differences: TEACHER-FORCED comparisons. The compared configuration is fed the
      reference's tokens, so per-step logit differences measure only arithmetic differences
      (kernel selection, accumulation order, fusion, quantization).
  (2) Behavioural divergence: GREEDY comparisons. Each configuration follows its own argmax; we report the
      first step where the generated token differs. After that point logits are incomparable (chaos).

Configurations vs the eager bf16 B=4 reference:
  eager rerun (same batch)            -> run-to-run determinism
  eager B=1 (prompt 0 alone)          -> batch invariance (GEMV vs GEMM kernel choice)
  compile_default / compile_graphs    -> compiler-induced differences (static cache, Inductor fusion, CUDA graphs)
  int4 weight-only                    -> quantization error, for scale
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from bench_llm_decode import make_static_cache, quantize_int4  # noqa: E402
from numerics_experiment import MODEL_DIR, PROMPTS, encode, load  # noqa: E402


@torch.no_grad()
def run_eager(model, ids, steps, forced=None):
    """DynamicCache decode. forced: [B, steps] tokens to feed (teacher forcing) or None for greedy."""
    out = model(ids, use_cache=True)
    pkv = out.past_key_values
    logits = [out.logits[:, -1].float()]
    tok = out.logits[:, -1].argmax(-1, keepdim=True)
    toks = [tok]
    for s in range(steps - 1):
        inp = forced[:, s:s + 1] if forced is not None else tok
        o = model(inp, past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        logits.append(o.logits[:, -1].float())
        tok = o.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(tok)
    return torch.stack(logits), torch.cat(toks, 1)


@torch.no_grad()
def run_compiled(model, ids, steps, mode, forced=None):
    B, L = ids.shape
    cache = make_static_cache(model, B, L + steps + 8)
    lg = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache, use_cache=True, return_dict=False)[0]
    logits = [lg[:, -1].float()]
    tok = lg[:, -1].argmax(-1, keepdim=True)
    toks = [tok]
    pos = torch.tensor([L], device="cuda")

    def decode_one(t, p):
        return model(t, cache_position=p, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    step = torch.compile(decode_one, mode=("reduce-overhead" if mode == "compile_graphs" else "default"))
    for s in range(steps - 1):
        inp = forced[:, s:s + 1] if forced is not None else tok
        l = step(inp, pos).clone()
        pos = pos + 1
        logits.append(l.float())
        tok = l.argmax(-1, keepdim=True)
        toks.append(tok)
    return torch.stack(logits), torch.cat(toks, 1)


def forced_stats(name, ref_logits, logits):
    """Teacher-forced: all steps are comparable."""
    d = (ref_logits - logits).abs()
    top1_same = (ref_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()
    rec = {"max_abs_logit_diff": d.max().item(), "mean_abs_logit_diff": d.mean().item(),
           "p99_abs_logit_diff": d.flatten().kthvalue(int(0.99 * d.numel())).values.item(),
           "bitwise_identical": bool(d.max().item() == 0.0), "argmax_agreement_forced": top1_same}
    print(f"  [forced] {name:34s} max={rec['max_abs_logit_diff']:.4f} mean={rec['mean_abs_logit_diff']:.5f} "
          f"p99={rec['p99_abs_logit_diff']:.4f} argmax agree {top1_same*100:.1f}% bitwise={rec['bitwise_identical']}", flush=True)
    return rec


def greedy_stats(name, ref_toks, toks):
    agree = (ref_toks == toks)
    first_div = []
    for b in range(toks.shape[0]):
        idx = (~agree[b]).nonzero()
        first_div.append(int(idx[0]) if len(idx) else None)
    rec = {"first_divergent_step_per_row": first_div, "steps": int(toks.shape[1])}
    print(f"  [greedy] {name:34s} first divergent step per row: {first_div} (of {toks.shape[1]})", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--length", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "numerics_v2.json"))
    args = ap.parse_args()

    tok, model = load()
    ids = encode(tok, PROMPTS, args.length)
    res = {"steps": args.steps, "prompts": len(PROMPTS), "prompt_len": int(ids.shape[1])}

    print("== reference: eager bf16, B=4, greedy", flush=True)
    ref_l, ref_t = run_eager(model, ids, args.steps)
    res["reference_text_row0"] = tok.decode(ref_t[0])

    print("== run-to-run determinism (eager, same batch)", flush=True)
    l2, t2 = run_eager(model, ids, args.steps, forced=ref_t)
    res["eager_rerun_forced"] = forced_stats("eager rerun", ref_l, l2)

    print("== batch invariance (prompt 0 alone vs inside B=4)", flush=True)
    l1f, _ = run_eager(model, ids[:1], args.steps, forced=ref_t[:1])
    res["batch_invariance_forced"] = forced_stats("eager B=1 vs row0 of B=4", ref_l[:, :1], l1f)
    _, l1g_t = run_eager(model, ids[:1], args.steps)
    res["batch_invariance_greedy"] = greedy_stats("eager B=1 vs row0 of B=4", ref_t[:1], l1g_t)

    for mode in ("compile_default", "compile_graphs"):
        print(f"== {mode} (static cache) vs eager", flush=True)
        torch._dynamo.reset()
        t0 = time.perf_counter()
        cl, _ = run_compiled(model, ids, args.steps, mode, forced=ref_t)
        res[f"{mode}_forced"] = forced_stats(f"{mode} vs eager", ref_l, cl)
        res[f"{mode}_forced"]["wall_s_incl_compile"] = time.perf_counter() - t0
        cl2, _ = run_compiled(model, ids, args.steps, mode, forced=ref_t)
        res[f"{mode}_rerun_forced"] = forced_stats(f"{mode} rerun", cl, cl2)
        _, cg_t = run_compiled(model, ids, args.steps, mode)
        res[f"{mode}_greedy"] = greedy_stats(f"{mode} vs eager", ref_t, cg_t)
        cl1, _ = run_compiled(model, ids[:1], args.steps, mode, forced=ref_t[:1])
        res[f"{mode}_batch_invariance_forced"] = forced_stats(f"{mode} B=1 vs row0 of B=4", cl[:, :1], cl1)

    print("== int4 weight-only (torchao) vs bf16, eager", flush=True)
    torch._dynamo.reset()
    quantize_int4(model)
    ql, _ = run_eager(model, ids, args.steps, forced=ref_t)
    res["int4_forced"] = forced_stats("int4 vs bf16 eager", ref_l, ql)
    _, qg_t = run_eager(model, ids, args.steps)
    res["int4_greedy"] = greedy_stats("int4 vs bf16 eager", ref_t, qg_t)
    res["int4_text_row0"] = tok.decode(qg_t[0])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
