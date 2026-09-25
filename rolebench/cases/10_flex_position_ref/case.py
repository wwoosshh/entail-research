"""#10 TIME (real engine: transformers as installed): the flex mask holds the position counter by reference.

Found in week 4 (phase0/week4/diag_hf_flex_offset.py) with Qwen3-4B; here a tiny random Qwen3 so it runs anywhere.
With StaticCache, the query offset given to the flex mask is the cache's live cumulative_length tensor. Each layer
increments it in place before its attention runs, so the mask admits one key slot too many (slot p+1). If that
slot holds stale content (a reused cache), the output changes silently.
  defect:    built-in flex_attention decode step, live counter
  fixed:     same, but the query offset is passed as a snapshot (clone)
  reference: sdpa decode step (its boolean mask is materialised at creation)
The stale slot gets the key of slot p-1 and a value scaled by 50, so a tiny model shows the extra slot clearly.
"""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache

META = {
    "id": "10", "title": "flex mask reads the position counter after an in-place increment", "fact": "TIME",
    "issue": "phase0/week4/WEEK4_NOTES.md section 7.2 (latent in normal generate())",
    "engine": "transformers (installed version)", "kind": "real-engine",
    "boundary": "StaticCache position counter -> flex attention mask (read time)",
    "trigger": {"engine": "transformers", "setting": "flex_attention + StaticCache", "state": "stale slot p+1"},
    "symptom": "plausible_but_wrong", "expected_detection": "runtime (epoch/read-time contract)", "compare": "loose",
    # torch's create_block_mask deprecation notice about its _compile flag; raised identically on the fixed path,
    # so it says nothing about the defect (PROTOCOL.md section 2).
    "ignore_logs": [r"_compile flag on create_block_mask"],
}
L, MAX = 64, 128


def setup():
    cfg = Qwen3Config(vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=512)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).to("cuda", torch.float32).eval()
    ids = torch.randint(0, 1024, (1, L), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    cache = StaticCache(config=model.config, max_cache_len=MAX)
    model.set_attn_implementation("sdpa")
    with torch.no_grad():
        lg = model(ids, past_key_values=cache, use_cache=True).logits
    tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
    snap = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    return {"model": model, "cache": cache, "tok": tok, "snap": snap}


def _restore(ctx):
    for layer, (k, v) in zip(ctx["cache"].layers, ctx["snap"]):
        layer.keys.copy_(k)
        layer.values.copy_(v)
        layer.keys[:, :, L + 1].copy_(k[:, :, L - 1])  # stale content in slot p+1
        layer.values[:, :, L + 1].copy_(v[:, :, L - 1] * 50.0)
    lens = [layer.cumulative_length for layer in ctx["cache"].layers]
    torch._foreach_zero_(lens)
    torch._foreach_add_(lens, L)


def _step(ctx, impl, snapshot=False):
    _restore(ctx)
    model, cache = ctx["model"], ctx["cache"]
    model.set_attn_implementation(impl)
    orig = type(cache).get_query_offset
    if snapshot:
        type(cache).get_query_offset = lambda self, layer_idx=0: orig(self, layer_idx).clone()
    try:
        with torch.no_grad():
            return model(ctx["tok"], past_key_values=cache, use_cache=True).logits[:, -1].float()
    finally:
        type(cache).get_query_offset = orig


def defect(ctx):
    return _step(ctx, "flex_attention")


def fixed(ctx):
    return _step(ctx, "flex_attention", snapshot=True)


def reference(ctx):
    return _step(ctx, "sdpa")
