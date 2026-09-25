"""Detection check for #16: the LoRA consumer declares that it takes unquantized activations."""
import entail as rc

UNQUANTIZED = (rc.Quantized("float32", scale=None), rc.Quantized("bfloat16", scale=None))


@rc.boundary(name="LoRA path (expects unquantized activations)", x=UNQUANTIZED)
def lora(*, x):
    return x


def _defect(ctx):
    xq = rc.tag(ctx["xq"], rc.Quantized("float8_e4m3fn", scale=float(ctx["s"])))
    lora(x=xq)


def _fixed(ctx):
    x = rc.tag(ctx["xq"].float() * ctx["s"], rc.Quantized("float32", scale=None))
    lora(x=x)


ARMS = {"W": {"mode": "debug", "checks": "dtype/scale tag on the shared activation buffer",
              "defect": _defect, "fixed": _fixed}}
