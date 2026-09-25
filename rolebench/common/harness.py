"""Shared tools for rolebench cases: silence capture, output comparison, judging, environment record.

Definitions follow rolebench/PROTOCOL.md sections 2-3.
"""
import contextlib
import logging
import platform
import re
import warnings

import torch

TOL = {  # (atol, rtol): max|a-b| <= atol + rtol * max(1, max|ref|)
    "exact": (0.0, 0.0),
    "fp32": (1e-5, 1e-4),
    "loose": (2e-3, 2e-2),
    "bf16": (1e-2, 2e-2),
    "fp8": (1e-2, 5e-2),  # requantized FP8 weights (PROTOCOL.md revision 1)
}


@contextlib.contextmanager
def capture():
    """Collect Python warnings and WARNING+ log records emitted inside the block (library loggers included)."""
    logs = []

    class Grab(logging.Handler):
        def emit(self, r):
            logs.append(f"{r.levelname} {r.name}: {r.getMessage()[:300]}")

    h = Grab(level=logging.WARNING)
    loggers = [logging.getLogger(), logging.getLogger("transformers"), logging.getLogger("torch")]
    for lg in loggers:
        lg.addHandler(h)
    box = {"warnings": [], "logs": logs, "error": None}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            yield box
        except Exception as e:  # the case function raised: record it, do not crash the runner
            box["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        finally:
            box["warnings"] = [f"{x.category.__name__}: {str(x.message)[:300]}" for x in w]
            for lg in loggers:
                lg.removeHandler(h)


def silent(box, ignore=()):
    """True if no exception, no warning and no WARNING+ log (except messages matching an ignore regex)."""
    def kept(msgs):
        return [m for m in msgs if not any(re.search(p, m) for p in ignore)]
    return box["error"] is None and not kept(box["warnings"]) and not kept(box["logs"])


def diff(a, ref):
    a, ref = a.detach().double().cpu(), ref.detach().double().cpu()
    d = (a - ref).abs()
    out = {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "ref_max_abs": float(ref.abs().max())}
    if a.dim() >= 1 and a.shape[-1] > 1 and a.shape == ref.shape:
        out["top1_agree"] = float((a.argmax(-1) == ref.argmax(-1)).double().mean())
    return out


def within(d, kind):
    atol, rtol = TOL[kind]
    return d["max_abs"] <= atol + rtol * max(1.0, d["ref_max_abs"])


def env_info():
    info = {"torch": torch.__version__, "python": platform.python_version()}
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
