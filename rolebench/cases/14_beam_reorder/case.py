"""#14 TIME: beam search reorders only caches with the standard names; a recurrent state is left in old order.

Mechanism of transformers #46612 (beam search cache reorder skipped for Mamba, XLNet, RWKV, Reformer). A tiny
recurrent language model keeps per-beam states under two names: "key_cache" (standard) and "ssm_state" (not).
After each step beam search picks parent beams; every per-beam state must be gathered by the parent index.
  defect:    reorder only the standard name; "ssm_state" keeps the pre-selection order
  fixed:     reorder every per-beam state
  reference: beam search that stores each beam's state inside the beam record (no reordering needed)
Output: token ids of the best beam and its score. Compared with the fp32 tolerance: the batched and per-beam
reference arithmetic may differ in the last bits of the score, while any token difference is >= 1.
"""
import torch

META = {
    "id": "14", "title": "beam search skips reordering a non-standard cache", "fact": "TIME",
    "issue": "https://github.com/huggingface/transformers/issues/46612", "engine": "beam search cache reorder (mechanism)",
    "kind": "mechanism", "boundary": "beam selection (parent indices) -> per-beam recurrent state",
    "trigger": {"feature": "beam search", "model": "recurrent or custom cache (Mamba, XLNet, RWKV, Reformer)"},
    "symptom": "plausible_but_wrong", "expected_detection": "runtime (per-beam axis role on every cache entry)",
    "compare": "fp32",
}
V, HID, BEAMS, STEPS = 32, 16, 3, 8


def setup():
    g = torch.Generator().manual_seed(0)
    f64 = dict(generator=g, dtype=torch.float64)
    return {"W": torch.randn(HID, HID, **f64) * 0.9, "U": torch.randn(V, HID, **f64),
            "O": torch.randn(HID, V, **f64), "start": 3}


def _step(ctx, h, tok):
    h = torch.tanh(h @ ctx["W"] + ctx["U"][tok])
    return h, torch.log_softmax(h @ ctx["O"], -1)


def _beam(ctx, reorder_names):
    h0 = torch.zeros(1, HID, dtype=torch.float64)
    cache = {"key_cache": h0.repeat(BEAMS, 1), "ssm_state": h0.repeat(BEAMS, 1)}
    toks = torch.full((BEAMS, 1), ctx["start"])
    scores = torch.tensor([0.0] + [-1e9] * (BEAMS - 1), dtype=torch.float64)
    for _ in range(STEPS):
        h, logp = _step(ctx, cache["ssm_state"], toks[:, -1])
        cache["key_cache"] = h.clone()
        cache["ssm_state"] = h
        total = (scores[:, None] + logp).reshape(-1)
        best = total.topk(BEAMS).indices
        parent, nxt = best // V, best % V
        for name in reorder_names:
            cache[name] = cache[name][parent]
        toks = torch.cat([toks[parent], nxt[:, None]], 1)
        scores = total[best]
    return torch.cat([toks[0].double(), scores[:1]])


def defect(ctx):
    return _beam(ctx, ["key_cache"])


def fixed(ctx):
    return _beam(ctx, ["key_cache", "ssm_state"])


def reference(ctx):
    beams = [(0.0, [ctx["start"]], torch.zeros(1, HID, dtype=torch.float64))]
    for _ in range(STEPS):
        cand = []
        for score, toks, h in beams:
            h2, logp = _step(ctx, h, torch.tensor([toks[-1]]))
            for t in range(V):
                cand.append((score + float(logp[0, t]), toks + [t], h2))
        cand.sort(key=lambda c: -c[0])
        beams = cand[:BEAMS]
    best = beams[0]
    return torch.tensor(best[1] + [best[0]], dtype=torch.float64)
