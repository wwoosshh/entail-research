"""#12 RANGE: a restored session sets the position from the token count, which is one more than the KV count.

Mechanism of llama.cpp #23400 (last token of a saved session replayed at the wrong position). The session file
stores n tokens but only n-1 KV entries (the last token was not evaluated yet). On restore the loader resumes at
position n; the last token is written to slot n and slot n-1 stays empty but inside the attended range.
  defect:    resume position = number of saved tokens (n)
  fixed:     resume position = number of KV entries (n-1)
  reference: the whole sequence evaluated from scratch
Tiny one-layer attention decoder with additive sinusoidal positions; output = logits of the last token.
"""
import math

import torch

META = {
    "id": "12", "title": "session restore resumes one position too far", "fact": "RANGE",
    "issue": "https://github.com/ggml-org/llama.cpp/issues/23400", "engine": "session save/restore (mechanism)",
    "kind": "mechanism", "boundary": "session file (token count) -> KV cache (valid length)",
    "trigger": {"feature": "session save and restore", "turn": "resume after restore"},
    "symptom": "plausible_but_wrong", "expected_detection": "load (declared valid length vs actual KV entries)",
    "compare": "fp32",
}
V, D, N, MAXLEN = 64, 32, 12, 32


def setup():
    g = torch.Generator().manual_seed(0)
    return {"E": torch.randn(V, D, generator=g), "Wq": torch.randn(D, D, generator=g) / D ** 0.5,
            "Wk": torch.randn(D, D, generator=g) / D ** 0.5, "Wv": torch.randn(D, D, generator=g) / D ** 0.5,
            "O": torch.randn(D, V, generator=g), "toks": torch.randint(0, V, (N,), generator=g)}


def _pos(p):
    i = torch.arange(D // 2)
    ang = p / (10000 ** (2 * i / D))
    return torch.cat([torch.sin(ang), torch.cos(ang)])


def _embed(ctx, tok, p):
    return ctx["E"][tok] + _pos(p)


class KV:
    def __init__(self):
        self.k = torch.zeros(MAXLEN, D)
        self.v = torch.zeros(MAXLEN, D)


def _decode(ctx, kv, tok, p):
    x = _embed(ctx, tok, p)
    kv.k[p], kv.v[p] = x @ ctx["Wk"], x @ ctx["Wv"]
    s = (x @ ctx["Wq"]) @ kv.k[: p + 1].T / math.sqrt(D)
    return (torch.softmax(s, -1) @ kv.v[: p + 1]) @ ctx["O"]


def _saved_session(ctx):
    kv = KV()
    for p in range(N - 1):  # the last token is stored in the session but not evaluated yet
        _decode(ctx, kv, ctx["toks"][p], p)
    return {"tokens": ctx["toks"].tolist(), "kv": kv, "kv_count": N - 1}


def defect(ctx):
    s = _saved_session(ctx)
    return _decode(ctx, s["kv"], s["tokens"][-1], len(s["tokens"]))


def fixed(ctx):
    s = _saved_session(ctx)
    return _decode(ctx, s["kv"], s["tokens"][-1], s["kv_count"])


def reference(ctx):
    kv = KV()
    out = None
    for p in range(N):
        out = _decode(ctx, kv, ctx["toks"][p], p)
    return out
