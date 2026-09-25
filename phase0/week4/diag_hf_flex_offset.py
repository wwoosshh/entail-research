"""Diagnosis: transformers' built-in flex_attention path with StaticCache attends to one slot too many?

Reading transformers 5.17 source:
  - Cache.get_query_offset() returns StaticLayer.cumulative_length, the live GPU counter tensor itself.
  - flex_attention_mask wraps the causal mask as mask_mod(q_idx + q_offset, kv_idx), holding that tensor by reference.
  - create_block_mask fixes the block structure when the mask is created (counter = p).
  - Each layer's cache update then does cumulative_length.add_(1) in place BEFORE that layer's attention runs,
    so the element-wise mask_mod evaluated inside the flex kernel reads p+1 and admits key slot p+1,
    which has not been written in this step. The sdpa path materialises its boolean mask at creation and is not
    affected.
Test in EAGER mode (no torch.compile, no CUDA graphs), bf16, batch 1, decode position p = 512:
  A. slot p+1 is empty (zeros, as in a fresh cache)       -> small deviation expected
  B. slot p+1 holds stale keys/values (a reused cache)     -> large deviation expected
  C. same as B, but the query offset is passed as a snapshot (clone) instead of the live counter -> no deviation
In every case compare with sdpa (keys 0..p) and with the hand kernel told to read keys 0..p+1.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import bench_decode_attn_swap as S  # noqa: E402
from bench_llm_decode import make_static_cache  # noqa: E402
from numerics_experiment import load  # noqa: E402

L, MAX_LEN = 512, 640


def main():
    S.register_impl()
    _, model = load()
    S.set_impl(model, "sdpa")
    g = torch.Generator(device="cuda").manual_seed(7)
    ids = torch.randint(0, model.config.vocab_size, (1, L), device="cuda", generator=g)
    cache = make_static_cache(model, 1, MAX_LEN)
    with torch.no_grad():
        lg = model(ids, past_key_values=cache, use_cache=True, return_dict=False)[0]
    tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
    lens = [layer.cumulative_length for layer in cache.layers]
    snap = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]

    def restore(stale):
        for layer, (k, v) in zip(cache.layers, snap):
            layer.keys.copy_(k)
            layer.values.copy_(v)
            if stale:  # plausible stale content in slot p+1: the keys/values of slot p-1 (a real token)
                layer.keys[:, :, L + 1].copy_(k[:, :, L - 1])
                layer.values[:, :, L + 1].copy_(v[:, :, L - 1])
        torch._foreach_zero_(lens)
        torch._foreach_add_(lens, L)

    def step(impl, valid=None):
        S.set_impl(model, impl)
        if valid is not None:
            S.VALID[:1].fill_(valid)
        with torch.no_grad():
            return model(tok, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1].float()

    def cmp(a, b):
        return {"max_abs": (a - b).abs().max().item(), "mean_abs": (a - b).abs().mean().item(),
                "same_top1": bool(a.argmax(-1).eq(b.argmax(-1)).all())}

    orig_offset = type(cache).get_query_offset
    res = {}
    for case, stale, snapshot in (("A_empty_slot", False, False), ("B_stale_slot", True, False),
                                  ("C_stale_slot_offset_snapshot", True, True)):
        out = {}
        for name, impl, valid in (("sdpa", "sdpa", None), ("hand_valid_p+1", "triton_decode", L + 1),
                                  ("hand_valid_p+2", "triton_decode", L + 2), ("hf_flex", "flex_attention", None)):
            restore(stale)
            if snapshot and impl == "flex_attention":
                type(cache).get_query_offset = lambda self, layer_idx=0: orig_offset(self, layer_idx).clone()
            try:
                out[name] = step(impl, valid)
            finally:
                type(cache).get_query_offset = orig_offset
        res[case] = {"hf_flex_vs_sdpa": cmp(out["hf_flex"], out["sdpa"]),
                     "hf_flex_vs_hand_reading_p+2": cmp(out["hf_flex"], out["hand_valid_p+2"]),
                     "hand_p+1_vs_sdpa (sanity)": cmp(out["hand_valid_p+1"], out["sdpa"]),
                     "logit_scale": out["sdpa"].abs().max().item()}
        print(case, json.dumps(res[case]), flush=True)
    res["note"] = __doc__
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "diag_hf_flex_offset.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
