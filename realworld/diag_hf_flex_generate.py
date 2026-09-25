"""Does the transformers flex_attention + StaticCache off-by-one change outputs in NORMAL generate() use?

generate(cache_implementation="static") builds a fresh, zero-filled StaticCache on every call, so the extra slot the
flex mask admits is always empty. This script checks whether that empty slot still changes greedy outputs,
especially for short prompts where one extra zero key weighs more.
Three runs per prompt, eager (no torch.compile), bf16, greedy, 48 new tokens:
  sdpa        reference
  flex        transformers' built-in flex_attention path, as shipped
  flex_fixed  same, but the query offset is snapshotted (cloned) when the mask is built
If flex and flex_fixed agree with sdpa equally well, the defect is latent in normal use.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "phase0"), os.path.join(ROOT, "phase0", "week2")]
import bench_decode_attn_swap as S  # noqa: E402
from numerics_experiment import load  # noqa: E402
from transformers.cache_utils import Cache  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "Write a haiku about GPUs.",
    "List three prime numbers greater than 100 and explain why each is prime.",
    "In a distant future, compilers read programs the way people read sentences. Describe a day in the life of "
    "such a compiler, focusing on how it decides which parts of a program can run in parallel, how it keeps track "
    "of which data is only read and which is written, and what happens when a programmer makes a mistake about "
    "the role of an argument. Continue the story with dialogue between the compiler and the programmer.",
]
NEW = 48
ORIG = Cache.get_query_offset


def snapshot_offset(self, layer_idx=0):
    v = ORIG(self, layer_idx)
    return v.clone() if torch.is_tensor(v) else v


def run(model, tok, prompt, impl, fixed):
    S.set_impl(model, impl)
    Cache.get_query_offset = snapshot_offset if fixed else ORIG
    try:
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=NEW, do_sample=False, cache_implementation="static",
                                 output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0, ids.shape[1]:].tolist()
        scores = torch.stack([s[0].float() for s in out.scores])
        return ids.shape[1], seq, scores
    finally:
        Cache.get_query_offset = ORIG


def compare(a, b):
    sa, la = a
    sb, lb = b
    first = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), None)
    n = first if first is not None else min(len(sa), len(sb))
    d = (la[:n] - lb[:n]).abs() if n > 0 else torch.zeros(1)
    return {"identical_tokens": sa == sb, "first_divergence": first,
            "max_logit_diff_before_divergence": d.max().item() if n > 0 else None}


def main():
    S.register_impl()
    tok, model = load()
    res = []
    for p in PROMPTS:
        r = {}
        for name, impl, fixed in (("sdpa", "sdpa", False), ("flex", "flex_attention", False),
                                  ("flex_fixed", "flex_attention", True)):
            n_prompt, seq, scores = run(model, tok, p, impl, fixed)
            r[name] = (seq, scores)
        row = {"prompt_tokens": n_prompt,
               "flex_vs_sdpa": compare(r["flex"], r["sdpa"]),
               "flex_fixed_vs_sdpa": compare(r["flex_fixed"], r["sdpa"]),
               "flex_vs_flex_fixed": compare(r["flex"], r["flex_fixed"]),
               "sdpa_text": tok.decode(r["sdpa"][0])[:160],
               "flex_text": tok.decode(r["flex"][0])[:160]}
        res.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.endswith("_text")}), flush=True)
    with open(os.path.join(HERE, "diag_hf_flex_generate.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
