"""Locate where the sdpa and triton_decode paths diverge (prefill vs decode; per-layer attention output)."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import bench_decode_attn_swap as S
from bench_llm_decode import build_model, make_static_cache

print(S.register_impl())
model, cfg, src, _ = build_model()
B, L, MAXL = 2, 16, 64
torch.manual_seed(3)
ids = torch.randint(0, cfg.vocab_size, (B, L), device="cuda")

captured = {}
def hook_factory(name):
    def hook(mod, args, kwargs, out):
        captured.setdefault(name, []).append(out[0].detach().float().clone())
    return hook
attn0 = model.model.layers[0].self_attn
attnL = model.model.layers[-1].self_attn
h1 = attn0.register_forward_hook(hook_factory("attn0"), with_kwargs=True)
h2 = attnL.register_forward_hook(hook_factory("attnL"), with_kwargs=True)

res = {}
for impl in ("sdpa", "triton_decode"):
    S.set_impl(model, impl)
    print("impl now:", model.config._attn_implementation)
    captured.clear()
    cache = make_static_cache(model, B, MAXL)
    S.VALID[:B].fill_(L)
    with torch.no_grad():
        lg0 = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1].float()
        tok = lg0.argmax(-1, keepdim=True); pos = torch.tensor([L], device="cuda")
        S.VALID[:B].copy_((pos + 1).to(torch.int32).expand(B))
        lg1 = model(tok, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1].float()
    res[impl] = dict(prefill=lg0, decode=lg1, attn0_prefill=captured["attn0"][0], attn0_decode=captured["attn0"][1],
                     attnL_prefill=captured["attnL"][0], attnL_decode=captured["attnL"][1],
                     k0=cache.layers[0].keys.clone() if hasattr(cache, "layers") else None)
for key in ("prefill", "decode", "attn0_prefill", "attn0_decode", "attnL_prefill", "attnL_decode"):
    a, b = res["sdpa"][key], res["triton_decode"][key]
    print(f"{key:14s} shape {tuple(a.shape)}  max|diff| {(a-b).abs().max().item():.5f}  scale {a.abs().max().item():.3f}")
if res["sdpa"]["k0"] is not None:
    print("cache layer0 keys identical:", torch.equal(res["sdpa"]["k0"], res["triton_decode"]["k0"]), "shape", tuple(res["sdpa"]["k0"].shape))
