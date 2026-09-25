"""#6 PROPERTY: passing a custom mask switches off the kernel's own sliding window; the mask lacks the window.

Mechanism of vLLM #47300 (FA4 path with images): the kernel convention is "a custom mask replaces the built-in
causal/window handling". The caller supplies a mask for another reason and still passes window=W, expecting both.
  defect:    kernel(window=W, mask=causal)            -> the window is silently ignored, keys beyond W are attended
  fixed:     kernel(window=W, mask=causal & window)   -> the caller folds the window into the mask
  reference: explicit float64 sliding-window causal attention
"""
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

META = {
    "id": "06", "title": "custom mask disables the kernel's sliding window", "fact": "PROPERTY",
    "issue": "https://github.com/vllm-project/vllm/issues/47300", "engine": "PyTorch FlexAttention (mechanism)",
    "kind": "mechanism", "boundary": "model property (sliding window) -> attention kernel call convention",
    "trigger": {"feature": "custom mask (e.g. multimodal) on a sliding-window layer", "length": "> window"},
    "symptom": "garbled", "expected_detection": "load (kernel capability vs model property)", "compare": "loose",
}
H, D, L, W = 4, 64, 1024, 256
_flex = torch.compile(flex_attention)


def kernel(q, k, v, window=None, mask_mod=None):
    """Toy kernel with the convention above: a custom mask_mod replaces the built-in causal/window mask."""
    if mask_mod is None:
        mask_mod = (lambda b, h, qi, ki: (ki <= qi) & (qi - ki < window)) if window else (lambda b, h, qi, ki: ki <= qi)
    bm = create_block_mask(mask_mod, 1, 1, L, L, device="cuda")
    with torch.no_grad():
        return _flex(q, k, v, block_mask=bm)


def setup():
    g = torch.Generator("cuda").manual_seed(0)
    return {x: torch.randn(1, H, L, D, device="cuda", generator=g) for x in ("q", "k", "v")}


def defect(ctx):
    return kernel(ctx["q"], ctx["k"], ctx["v"], window=W, mask_mod=lambda b, h, qi, ki: ki <= qi)


def fixed(ctx):
    return kernel(ctx["q"], ctx["k"], ctx["v"], window=W, mask_mod=lambda b, h, qi, ki: (ki <= qi) & (qi - ki < W))


def reference(ctx):
    q, k, v = (ctx[x].double() for x in ("q", "k", "v"))
    s = q @ k.transpose(-1, -2) / D ** 0.5
    i = torch.arange(L, device="cuda")
    allowed = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W)
    s = s.masked_fill(~allowed, float("-inf"))
    return (torch.softmax(s, -1) @ v).float()
