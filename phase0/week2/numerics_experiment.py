"""Week 2, experiment B: numerics and determinism with REAL Qwen3-4B weights (stability axis).

Questions:
  1. Does torch.compile (static cache, default / CUDA graphs) change the logits vs eager? By how much, and does
     greedy decoding diverge (first differing token, argmax agreement)?
  2. Is a given configuration bitwise reproducible run-to-run?
  3. Is inference batch-invariant? Same prompt alone (B=1) vs inside a batch (B=4): identical logits?
     (Different cuBLAS kernels for M=1 vs M=4 are the classic source of non-determinism in serving.)
  4. How large is the int4 quantization error relative to the compile-induced differences?
Prompts are truncated to a common token length so no padding/mask is needed.
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

MODEL_DIR = os.path.expanduser("~/models/Qwen3-4B")
PROMPTS = [
    "The GPU compiler research plan for consumer graphics cards begins with measuring where time is lost. First,",
    "서울대학교 컴퓨터공학부에서 진행하는 컴파일러 연구의 목표는 다음과 같다. 첫째,",
    "def flash_decoding_attention(q, k_cache, v_cache, valid_len):\n    \"\"\"Compute attention for one query token",
    "In 1972, the first message sent over the ARPANET was",
]


def load(dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    return tok, model


def encode(tok, prompts, length):
    full = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]
    length = min(length, min(len(i) for i in full))  # common length so no padding/mask is needed
    print(f"  using common prompt length {length} tokens (per-prompt lengths {[len(i) for i in full]})", flush=True)
    return torch.stack([i[:length] for i in full]).cuda()


@torch.no_grad()
def greedy_eager(model, ids, steps):
    """DynamicCache greedy decode; returns [steps, B, vocab] fp32 logits and tokens."""
    out = model(ids, use_cache=True)
    pkv, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    logits, toks = [out.logits[:, -1].float()], [tok]
    for _ in range(steps - 1):
        o = model(tok, past_key_values=pkv, use_cache=True)
        pkv, tok = o.past_key_values, o.logits[:, -1].argmax(-1, keepdim=True)
        logits.append(o.logits[:, -1].float()); toks.append(tok)
    return torch.stack(logits), torch.cat(toks, 1)


@torch.no_grad()
def greedy_compiled(model, ids, steps, mode):
    B, L = ids.shape
    cache = make_static_cache(model, B, L + steps + 8)
    lg = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache, use_cache=True, return_dict=False)[0]
    tok, pos = lg[:, -1].argmax(-1, keepdim=True), torch.tensor([L], device="cuda")
    logits, toks = [lg[:, -1].float()], [tok]

    def decode_one(t, p):
        return model(t, cache_position=p, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    step = torch.compile(decode_one, mode=("reduce-overhead" if mode == "compile_graphs" else "default"))
    for _ in range(steps - 1):
        l = step(tok, pos).clone()
        tok, pos = l.argmax(-1, keepdim=True), pos + 1
        logits.append(l.float()); toks.append(tok)
    return torch.stack(logits), torch.cat(toks, 1)


def compare(name, ref_logits, ref_toks, logits, toks):
    d = (ref_logits - logits).abs()
    agree = (ref_toks == toks)
    first_div = []
    for b in range(toks.shape[0]):
        idx = (~agree[b]).nonzero()
        first_div.append(int(idx[0]) if len(idx) else None)
    rec = {"max_abs_logit_diff": d.max().item(), "mean_abs_logit_diff": d.mean().item(),
           "bitwise_identical": bool(d.max().item() == 0.0),
           "token_agreement": agree.float().mean().item(), "first_divergent_step_per_row": first_div}
    print(f"  {name:40s} max|dlogit|={rec['max_abs_logit_diff']:.4f} mean={rec['mean_abs_logit_diff']:.5f} "
          f"tokens agree {rec['token_agreement']*100:.1f}% first div {first_div}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--length", type=int, default=24, help="common prompt length in tokens")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "numerics.json"))
    args = ap.parse_args()

    tok, model = load()
    ids = encode(tok, PROMPTS, args.length)
    res = {"steps": args.steps, "prompt_len": args.length, "prompts": len(PROMPTS)}

    print("== eager reference (B=4) and run-to-run reproducibility", flush=True)
    ref_l, ref_t = greedy_eager(model, ids, args.steps)
    l2, t2 = greedy_eager(model, ids, args.steps)
    res["eager_rerun"] = compare("eager vs eager (rerun)", ref_l, ref_t, l2, t2)
    res["sample_text"] = tok.decode(ref_t[0])
    print("  sample continuation:", repr(res["sample_text"][:120]), flush=True)

    print("== batch invariance: prompt 0 alone (B=1) vs inside B=4", flush=True)
    l1, t1 = greedy_eager(model, ids[:1], args.steps)
    res["batch_invariance_eager"] = compare("eager B=1 vs row0 of B=4", ref_l[:, :1], ref_t[:1], l1, t1)

    for mode in ("compile_default", "compile_graphs"):
        print(f"== {mode} (static cache) vs eager", flush=True)
        torch._dynamo.reset()
        t0 = time.perf_counter()
        cl, ct = greedy_compiled(model, ids, args.steps, mode)
        res[f"{mode}_vs_eager"] = compare(f"{mode} vs eager", ref_l, ref_t, cl, ct)
        res[f"{mode}_vs_eager"]["wall_s_incl_compile"] = time.perf_counter() - t0
        cl2, ct2 = greedy_compiled(model, ids, args.steps, mode)
        res[f"{mode}_rerun"] = compare(f"{mode} rerun", cl, ct, cl2, ct2)
        cl1, ct1 = greedy_compiled(model, ids[:1], args.steps, mode)
        res[f"{mode}_batch_invariance"] = compare(f"{mode} B=1 vs row0 of B=4", cl[:, :1], ct[:1], cl1, ct1)

    print("== int4 weight-only (torchao) vs bf16 eager", flush=True)
    torch._dynamo.reset()
    quantize_int4(model)
    ql, qt = greedy_eager(model, ids, args.steps)
    res["int4_vs_bf16_eager"] = compare("int4 eager vs bf16 eager", ref_l, ref_t, ql, qt)
    res["int4_sample_text"] = tok.decode(qt[0])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
