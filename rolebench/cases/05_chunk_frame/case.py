"""#5 FRAME: a mask function compares a chunk-relative query index with absolute key indices.

Mechanism of vLLM #47300 (Gemma 4 gibberish on long inputs): during chunked prefill the second chunk's queries
start at absolute position C, but flex_attention hands mask_mod the query index inside the chunk (0..C-1)
while key indices run over the whole cache (0..2C-1).
  defect:    mask_mod = kv <= q            (relative query vs absolute key; most of the prefix becomes invisible)
  fixed:     mask_mod = kv <= q + offset   (both absolute)
  reference: explicit float64 softmax attention with an absolute causal mask
"""
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

META = {
    "id": "05", "title": "chunk-relative query vs absolute key in a mask function", "fact": "FRAME",
    "issue": "https://github.com/vllm-project/vllm/issues/47300", "engine": "PyTorch FlexAttention (mechanism)",
    "kind": "mechanism", "boundary": "chunked-prefill scheduler (query offset) -> mask function (position frame)",
    "trigger": {"feature": "chunked prefill with a custom mask", "length": "> one chunk"},
    "symptom": "garbled", "expected_detection": "runtime (frame tag on query/key indices)", "compare": "loose",
}
H, D, C = 4, 64, 256
_flex = torch.compile(flex_attention)


def setup():
    g = torch.Generator("cuda").manual_seed(0)
    return {"q": torch.randn(1, H, C, D, device="cuda", generator=g),
            "k": torch.randn(1, H, 2 * C, D, device="cuda", generator=g),
            "v": torch.randn(1, H, 2 * C, D, device="cuda", generator=g), "offset": C}


def _run(ctx, mask_mod):
    bm = create_block_mask(mask_mod, 1, 1, C, 2 * C, device="cuda")
    with torch.no_grad():
        return _flex(ctx["q"], ctx["k"], ctx["v"], block_mask=bm)


def defect(ctx):
    return _run(ctx, lambda b, h, q, kv: kv <= q)


def fixed(ctx):
    off = ctx["offset"]
    return _run(ctx, lambda b, h, q, kv: kv <= q + off)


def reference(ctx):
    q, k, v = (ctx[x].double() for x in ("q", "k", "v"))
    s = q @ k.transpose(-1, -2) / D ** 0.5
    qa = torch.arange(C, device="cuda")[:, None] + ctx["offset"]
    ka = torch.arange(2 * C, device="cuda")[None, :]
    s = s.masked_fill(ka > qa, float("-inf"))
    return (torch.softmax(s, -1) @ v).float()
