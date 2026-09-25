"""Which decode-attention path is wrong at the model level?

Random-weight Qwen3-4B architecture (as in weeks 1-2), B=1, prompt prefilled with sdpa, one decode step at
position L. Before every decode call the KV cache is restored from a snapshot, so every path sees identical state.
Paths: sdpa (transformers default), sdpa again (determinism), triton_decode, maskmod_decode (FlexAttention), and
ref_decode: plain fp32 math over exactly the first `valid` cache slots, written for this test.
Forward hooks record each layer's attention output to find the first layer where paths diverge.
Run for two prompt lengths to see whether the divergence depends on context length.
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import ab_attention as A  # noqa: E402
import bench_decode_attn_swap as S  # noqa: E402
from bench_llm_decode import build_model, make_static_cache  # noqa: E402


def ref_decode_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    B, Hq, q_len, D = query.shape
    group = Hq // key.shape[1]
    scale = float(scaling) if scaling is not None else D ** -0.5
    if q_len == 1:
        outs = []
        for b in range(B):
            n = int(S.VALID[b])
            k = key[b:b + 1, :, :n].float().repeat_interleave(group, 1)
            v = value[b:b + 1, :, :n].float().repeat_interleave(group, 1)
            s = torch.einsum("bhqd,bhnd->bhqn", query[b:b + 1].float(), k) * scale
            outs.append(torch.einsum("bhqn,bhnd->bhqd", s.softmax(-1), v))
        out = torch.cat(outs).to(query.dtype)
    else:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale, enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


def register_ref():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register("ref_decode", ref_decode_forward)
    AttentionMaskInterface.register("ref_decode", lambda *a, **k: None)


def cache_tensors(cache):
    if hasattr(cache, "layers"):
        return [t for layer in cache.layers for t in (layer.keys, layer.values)]
    return list(cache.key_cache) + list(cache.value_cache)


def main():
    real = "--real" in sys.argv
    print(S.register_impl(), flush=True)
    A.register_maskmod()
    register_ref()
    if real:
        from numerics_experiment import load
        _, model = load()
        cfg = model.config
        print("using REAL Qwen3-4B weights", flush=True)
    else:
        model, cfg, _, _ = build_model()
        print("using RANDOM weights (hub config)", flush=True)
    attn_out = {}

    def hook(i):
        def f(mod, args, kwargs, out):
            attn_out[i] = out[0].detach().float().clone()
        return f

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(hook(i), with_kwargs=True)

    for L in (64, 512):
        B, max_len = 1, L + 128
        g = torch.Generator(device="cuda").manual_seed(1000 + L)
        ids = torch.randint(0, cfg.vocab_size, (B, L), device="cuda", generator=g)
        S.set_impl(model, "sdpa")
        cache = make_static_cache(model, B, max_len)
        with torch.no_grad():
            lg = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache,
                       use_cache=True, return_dict=False)[0]
        tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
        pos = torch.tensor([L], device="cuda")
        S.VALID.zero_()
        S.VALID[:B].fill_(L + 1)
        A.FLEX["bm"] = A.build_block_mask(B, max_len)
        snap = [t.clone() for t in cache_tensors(cache)]
        res, layers = {}, {}
        for name, impl in (("sdpa", "sdpa"), ("sdpa_again", "sdpa"), ("triton_decode", "triton_decode"),
                           ("triton_again", "triton_decode"), ("maskmod_decode", "maskmod_decode"),
                           ("ref_decode", "ref_decode"), ("ref_again", "ref_decode")):
            for t, s in zip(cache_tensors(cache), snap):
                t.copy_(s)
            S.set_impl(model, impl)
            attn_out.clear()
            with torch.no_grad():
                res[name] = model(tok, cache_position=pos, past_key_values=cache, use_cache=True,
                                  return_dict=False)[0][:, -1].float().clone()
            layers[name] = dict(attn_out)
        print(f"\n=== prompt {L}: logit scale {res['ref_decode'].abs().max().item():.2f}", flush=True)
        names = list(res)
        for a in names:
            row = "  ".join(f"{(res[a] - res[b]).abs().max().item():6.3f}" for b in names)
            print(f"  {a:15s} {row}", flush=True)
        print("  (columns in the same order; max |logit difference|)", flush=True)
        for a, b in (("sdpa", "sdpa_again"), ("triton_decode", "triton_again"), ("ref_decode", "ref_again")):
            print(f"  run-to-run bitwise identical {a}: {torch.equal(res[a], res[b])}", flush=True)
        scale0 = layers["ref_decode"][0].abs().max().item()
        print(f"  layer-0 attention output scale (ref): {scale0:.3f}", flush=True)
        for name in ("sdpa", "sdpa_again", "triton_decode", "maskmod_decode", "ref_again"):
            first = None
            for i in range(len(model.model.layers)):
                d = (layers[name][i] - layers["ref_decode"][i]).abs().max().item()
                if d > 0.05 and first is None:
                    first = (i, d)
            d0 = (layers[name][0] - layers["ref_decode"][0]).abs().max().item()
            print(f"  {name:15s} layer-0 attention output vs ref: {d0:.4f}; first layer with diff > 0.05: {first}",
                  flush=True)
        del cache, snap
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
