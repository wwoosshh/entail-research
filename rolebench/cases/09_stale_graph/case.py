"""#9 TIME: a CUDA Graph reuse check compares only buffer pointers; the shape changed, a stale graph replays.

Mechanism of llama.cpp #21726 (Gemma 4 gibberish after ~230 tokens with -nkvo; the graph reuse check missed
changed ne/nb). A graph is captured for attention over the first Lv keys of a KV buffer; the valid length is baked
into the captured kernels. Later Lv grows, the buffer pointer stays the same, and a pointer-only cache key
replays the old graph, which ignores the new keys.
  defect:    graph cache keyed by data pointers only
  fixed:     key also includes the valid length (the recorded assumption is rechecked)
  reference: eager attention over the first Lv keys
"""
import torch

META = {
    "id": "09", "title": "stale CUDA Graph replayed after the valid length changed", "fact": "TIME",
    "issue": "https://github.com/ggml-org/llama.cpp/issues/21726", "engine": "CUDA Graph reuse logic (mechanism)",
    "kind": "mechanism", "boundary": "graph capture (assumed shape) -> graph replay (actual shape)",
    "trigger": {"feature": "CUDA Graphs with a growing KV cache", "length": "crosses the captured length"},
    "symptom": "garbled", "expected_detection": "runtime (record specialization assumptions, recheck on replay)",
    "compare": "fp32",
}
H, D, MAXLEN, L0, L1 = 4, 64, 256, 100, 160


def attend(q, kv_k, kv_v, lv):
    s = (q @ kv_k[:, :lv].transpose(-1, -2)) / D ** 0.5
    return torch.softmax(s, -1) @ kv_v[:, :lv]


class GraphRunner:
    def __init__(self, key_fn):
        self.key_fn, self.graphs = key_fn, {}

    def __call__(self, q, k, v, lv):
        key = self.key_fn(q, k, v, lv)
        if key not in self.graphs:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                attend(q, k, v, lv)  # warm-up
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = attend(q, k, v, lv)
            self.graphs[key] = (g, out)
        g, out = self.graphs[key]
        g.replay()
        return out.clone()


def setup():
    g = torch.Generator("cuda").manual_seed(0)
    return {"q": torch.randn(H, 1, D, device="cuda", generator=g),
            "k": torch.randn(H, MAXLEN, D, device="cuda", generator=g),
            "v": torch.randn(H, MAXLEN, D, device="cuda", generator=g)}


def _run(ctx, key_fn):
    r = GraphRunner(key_fn)
    r(ctx["q"], ctx["k"], ctx["v"], L0)  # captured while the cache held L0 tokens
    return r(ctx["q"], ctx["k"], ctx["v"], L1)  # later step, same buffers, more tokens


def defect(ctx):
    return _run(ctx, lambda q, k, v, lv: (q.data_ptr(), k.data_ptr(), v.data_ptr()))


def fixed(ctx):
    return _run(ctx, lambda q, k, v, lv: (q.data_ptr(), k.data_ptr(), v.data_ptr(), lv))


def reference(ctx):
    with torch.no_grad():
        return attend(ctx["q"].double(), ctx["k"].double(), ctx["v"].double(), L1).float()
