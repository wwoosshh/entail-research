"""#16 DTYPE: a consumer assumes BF16 activations but receives the FP8-quantized buffer (scale not applied).

Mechanism of vLLM #42007 (FP8 MoE models corrupted when serving LoRA adapters): the MoE path quantizes the
activations to FP8 with a per-tensor scale; the LoRA path reads the same buffer as if it were the original BF16
activations, so its delta is computed from x/s instead of x.
  defect:    lora_delta from xq.to(bf16)                (the FP8 values without their scale)
  fixed:     lora_delta from the dequantized buffer     (xq * s)
  reference: x @ W + (x @ A) @ B in float64, with x as the model's activations
Compared with the 'fp8' tolerance: both paths carry FP8 activation rounding (PROTOCOL.md revision 1).
"""
import torch

META = {
    "id": "16", "title": "FP8-quantized activations consumed as BF16", "fact": "DTYPE",
    "issue": "https://github.com/vllm-project/vllm/issues/42007", "engine": "vLLM FP8 MoE + LoRA (mechanism)",
    "kind": "mechanism", "boundary": "activation quantizer (FP8 + scale) -> LoRA consumer (expects BF16)",
    "trigger": {"model": "FP8 MoE checkpoint", "feature": "LoRA adapters"},
    "symptom": "garbled", "expected_detection": "runtime (dtype/scale tag on the shared buffer)", "compare": "fp8",
}
T, DIN, DOUT, R = 16, 256, 128, 8
FP8_MAX = 448.0


def setup():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(T, DIN, generator=g) * 3.0
    s = x.abs().max() / FP8_MAX
    return {"x": x, "s": s, "xq": (x / s).to(torch.float8_e4m3fn),
            "w": torch.randn(DIN, DOUT, generator=g) / DIN ** 0.5,
            "a": torch.randn(DIN, R, generator=g) / DIN ** 0.5, "b": torch.randn(R, DOUT, generator=g)}


def _main(ctx):
    return (ctx["xq"].float() * ctx["s"]) @ ctx["w"]  # the MoE path dequantizes correctly


def defect(ctx):
    x_seen = ctx["xq"].to(torch.bfloat16).float()  # read as if it were the BF16 activations
    return _main(ctx) + (x_seen @ ctx["a"]) @ ctx["b"]


def fixed(ctx):
    x_seen = ctx["xq"].float() * ctx["s"]
    return _main(ctx) + (x_seen @ ctx["a"]) @ ctx["b"]


def reference(ctx):
    x, w, a, b = (ctx[k].double() for k in ("x", "w", "a", "b"))
    return (x @ w + (x @ a) @ b).float()
