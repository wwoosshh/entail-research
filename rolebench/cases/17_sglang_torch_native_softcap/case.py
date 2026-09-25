"""#17 PROPERTY (real engine: SGLang 0.5.20 as installed): the torch_native attention backend ignores logit_cap.

Found by code reading on 2026-09-23. sglang/srt/layers/attention/torch_native_backend.py calls
scaled_dot_product_attention and never reads the layer's logit_cap, while the triton, flashinfer and
flashattention backends apply it.
Test with the real Gemma 2 2B weights. In the real checkpoint (cap 50) the raw scores stay below 40, so the cap
barely binds (issue_track/gemma2_softcap E1). To test whether the engine honours the declared property, a copy of
the config declares attn_logit_softcapping = 5.0, which binds. Every path uses that same config.
  defect:    SGLang, attention_backend = torch_native
  fixed:     SGLang, attention_backend = triton
  reference: transformers eager path (the model definition) with the same config
Output: log-prob of each next token over a fixed 256-token text. Compared in 'noise' mode (PROTOCOL.md section 3),
because SGLang triton and transformers eager are different bf16 kernels.
"""
import json
import os
import subprocess
import tempfile
import warnings

import torch

META = {
    "id": "17", "title": "SGLang torch_native attention ignores the declared logit cap", "fact": "PROPERTY",
    "issue": "local finding by code reading (2026-09-23); no upstream report searched yet",
    "engine": "SGLang 0.5.20 (installed)", "kind": "real-engine",
    "boundary": "model config (attn_logit_softcapping) -> SGLang attention backend (torch_native)",
    "trigger": {"engine": "SGLang", "setting": "--attention-backend torch_native (or a fallback to it)",
                "model": "Gemma 2 (logit cap)"},
    "symptom": "plausible_but_wrong", "expected_detection": "load (backend capability vs model property)",
    "compare": "noise",
}
SRC = os.path.expanduser("~/models/gemma-2-2b-it")
SGLANG_PY = os.path.expanduser("~/venvs/sglang/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 5.0


def setup():
    d = tempfile.mkdtemp(prefix="rolebench17_")
    for f in os.listdir(SRC):
        if f != "config.json" and not f.startswith("."):
            os.symlink(os.path.join(SRC, f), os.path.join(d, f))
    with open(os.path.join(SRC, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["attn_logit_softcapping"] = CAP
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    from transformers import AutoTokenizer

    with open("/usr/share/common-licenses/GPL-3", encoding="utf-8") as f:
        text = f.read()
    ids = AutoTokenizer.from_pretrained(SRC)(text).input_ids[:256]
    ids_path = os.path.join(d, "ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    return {"dir": d, "ids": ids, "ids_path": ids_path}


def _sglang(ctx, backend):
    out = os.path.join(ctx["dir"], f"lp_{backend}.json")
    r = subprocess.run([SGLANG_PY, os.path.join(HERE, "sglang_logprobs.py"), ctx["dir"], backend, ctx["ids_path"], out],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"sglang {backend} failed: {(r.stdout + r.stderr)[-1500:]}")
    log = r.stdout + r.stderr
    ctx[f"log_{backend}"] = log[-4000:]
    # The engine runs in another process: surface any line about the cap as a Python warning, so the
    # silence check (PROTOCOL.md section 2) sees it. Generic start-up warnings are not about the cap.
    for line in log.splitlines():
        if any(w in line.lower() for w in ("logit_cap", "softcap", "soft cap", "soft-cap")):
            warnings.warn(f"sglang[{backend}]: {line[:300]}")
    with open(out, encoding="utf-8") as f:
        return torch.tensor(json.load(f)["logprobs"], dtype=torch.float64)


def defect(ctx):
    return _sglang(ctx, "torch_native")


def fixed(ctx):
    return _sglang(ctx, "triton")


def reference(ctx):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(ctx["dir"], dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    ids = torch.tensor([ctx["ids"]], device="cuda")
    with torch.no_grad():
        lp = torch.log_softmax(m(ids).logits.float(), -1)[0]
    out = lp[:-1].gather(-1, ids[0, 1:, None])[:, 0].double().cpu()
    del m
    torch.cuda.empty_cache()
    return out


def probe(ctx):
    log = ctx.get("log_torch_native", "")
    hits = [line for line in log.splitlines() if any(w in line.lower() for w in ("cap", "softcap", "warn"))]
    return {"torch_native_engine_lines_mentioning_cap_or_warn": hits[:20]}
