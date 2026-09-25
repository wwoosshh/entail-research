"""#2 LAYOUT (scale format): fp32 block scales are rounded up to powers of two, the FP8 data is not requantized.

Mechanism of transformers #47030 (FineGrainedFP8 + DeepGEMM silently wrong on SM100): the kernel wants UE8M0
scales (a power-of-two exponent per block). The conversion rounded each fp32 block scale up to the next power of
two but kept the FP8 values quantized for the old scale, so every block is multiplied by a factor in [1, 2).
  defect:    q (quantized with scale s) paired with scale 2^ceil(log2 s)
  fixed:     data requantized with the rounded scale (data and scale format agree)
  reference: x @ (q * s).T, the weight as originally quantized
Compared with the 'fp8' tolerance: the fixed path carries FP8 re-rounding error (PROTOCOL.md revision 1).
"""
import torch

META = {
    "id": "02", "title": "block scales rounded to powers of two without requantizing FP8 data", "fact": "LAYOUT",
    "issue": "https://github.com/huggingface/transformers/issues/47030",
    "engine": "transformers FineGrainedFP8 + DeepGEMM (mechanism)", "kind": "mechanism",
    "boundary": "checkpoint scale format (fp32) -> kernel scale format (UE8M0)",
    "trigger": {"hardware": "SM100 (DeepGEMM UE8M0 path)", "model": "block-FP8 checkpoint with fp32 scales"},
    "symptom": "plausible_but_wrong", "expected_detection": "load (declared scale format vs kernel requirement)",
    "compare": "fp8",
}
N, K, B = 128, 256, 64  # rows, columns, square block size
FP8_MAX = 448.0


def _blocks(w):
    return w.reshape(N // B, B, K // B, B).permute(0, 2, 1, 3)  # (nb_r, nb_c, B, B)


def _unblocks(b):
    return b.permute(0, 2, 1, 3).reshape(N, K)


def quantize(w, scale=None):
    b = _blocks(w)
    if scale is None:
        scale = (b.abs().amax(dim=(-1, -2)) / FP8_MAX).clamp_min(1e-12)
    q = (b / scale[..., None, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def dequant(q, scale):
    return _unblocks(q.float() * scale[..., None, None])


def setup():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(N, K, generator=g) * torch.logspace(-2, 1, K // B).repeat_interleave(B)[None, :]
    x = torch.randn(16, K, generator=g)
    q, s = quantize(w)
    return {"x": x, "w": w, "q": q, "s": s, "s_pow2": torch.exp2(torch.ceil(torch.log2(s)))}


def defect(ctx):
    return ctx["x"] @ dequant(ctx["q"], ctx["s_pow2"]).T


def fixed(ctx):
    q2, _ = quantize(dequant(ctx["q"], ctx["s"]), ctx["s_pow2"])
    return ctx["x"] @ dequant(q2, ctx["s_pow2"]).T


def reference(ctx):
    return ctx["x"] @ dequant(ctx["q"], ctx["s"]).T
