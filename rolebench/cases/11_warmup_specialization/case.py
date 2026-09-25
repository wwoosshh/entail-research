"""#11 SPECIALIZATION: a graph compiled during warm-up with empty inputs is reused without guards; real input ignored.

Mechanism of vLLM #43602 (Qwen3-VL-2B accuracy below SGLang): the engine compiles once during warm-up and skips
Dynamo guards for speed. The warm-up call had no multimodal embeddings, so the compiled graph took the
"no extra input" branch; later calls with real embeddings reuse it and the embeddings are silently dropped.
  defect:    torch.compile with all guards skipped; warm-up with an empty extra input, then a real one
  fixed:     guards kept (the specialization assumption is rechecked, so it recompiles)
  reference: eager call with the real input
"""
import torch

META = {
    "id": "11", "title": "warm-up specialization reused without guards", "fact": "SPECIALIZATION",
    "issue": "https://github.com/vllm-project/vllm/issues/43602", "engine": "torch.compile guard skipping (mechanism)",
    "kind": "mechanism", "boundary": "compile-time assumption (empty extra input) -> runtime input (real embeddings)",
    "trigger": {"feature": "compile warm-up + guard skipping", "model": "multimodal input path"},
    "symptom": "plausible_but_wrong", "expected_detection": "runtime (record specialization, recheck on reuse)",
    "compare": "fp32",
}
D = 64


def f(x, extra, w):
    if extra.shape[0] == 0:  # no multimodal embeddings in this batch
        return x @ w
    return (x + extra.mean(0, keepdim=True)) @ w


def _skip_all_guards():
    fn = getattr(torch.compiler, "skip_all_guards_unsafe", None)
    return fn if fn is not None else (lambda entries: [False for _ in entries])


def setup():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(8, D, generator=g), "extra": torch.randn(5, D, generator=g),
            "w": torch.randn(D, D, generator=g), "empty": torch.zeros(0, D)}


def _run(ctx, options):
    torch._dynamo.reset()
    cf = torch.compile(f, dynamic=True, options=options)
    cf(ctx["x"], ctx["empty"], ctx["w"])  # warm-up: no extra input
    return cf(ctx["x"], ctx["extra"], ctx["w"])


def defect(ctx):
    return _run(ctx, {"guard_filter_fn": _skip_all_guards()})


def fixed(ctx):
    return _run(ctx, None)


def reference(ctx):
    return f(ctx["x"].double(), ctx["extra"].double(), ctx["w"].double()).float()
