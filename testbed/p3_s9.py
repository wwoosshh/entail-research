"""S9 (ROADMAP product track P3; LIBRARY_DESIGN.md 13.6): one engine start under entail's safety modes.

  python testbed/p3_s9.py <engine kwargs JSON> <out JSON>

Builds vLLM's offline LLM with the given engine arguments under ENTAIL=load (the start-up shim puts the adapters in:
vllm_safe turns optimizations off before the engine is built, vllm_paths compares the engine's paths once it is up and
moves the selective safe path on), generates one short answer, and writes what this start did: what the safety mode
turned off, the path check's numbers for every pair (passes too; the record keeps only their count), what entail said,
and the selective safe path's store after the start. P3_PLANT plants a defect (research only, not part of entail):
  inside   RMSNorm's CustomOp CUDA path (forward_cuda) scales its output by P3_SCALE on calls of at most 8 rows (the
           decode steps). In vLLM 0.30 that method only calls forward_native, which calls the IR op, so this is a
           fault in the op's dispatch wrapper, not in its kernel (kept as the first run measured it)
  inside_kernel  the rms_norm and fused_add_rms_norm kernels themselves (vllm_c, the IR ops' CUDA implementation)
           scale their output on calls of at most 8 rows: a custom-kernel fault that custom_ops does not reach
  inside_op  SiluAndMul's CUDA kernel path (forward_cuda, torch.ops._C.silu_and_mul) scales its output on calls of
           at most 8 rows: a custom-kernel fault that custom_ops=["none"] replaces with the op's definition
  outside  the attention backend scales its output by P3_SCALE on pure decode batches (max_query_len 1): a fault in
           no optimization the safety mode turns off
  inside_rope  the rotary embedding's CUDA kernel path (RotaryEmbedding.forward_cuda, ops.rotary_embedding) scales the
           query and key it returns on calls of at most 8 tokens (pre-registered, PREREG_S9b2.md)
  outside_kvwrite  the KV cache write of the FlashAttention backend (reshape_and_cache_flash) stores keys scaled by
           P3_SCALE on calls of at most 8 tokens (pre-registered, PREREG_S9b2.md)
P3_BENCH=1 also times a fixed workload after the start (four prompts, two of them asking to copy text back, the
kind n-gram speculative decoding is for; 128 greedy tokens each, one batch; the median of three runs after one warm-up):
what a safe path that turns an optimization off costs in speed.
Run it with VLLM_ENABLE_V1_MULTIPROCESSING=0 so that the engine core, and the planted defect, are in this process.
"""
import json
import os
import sys
import time

KW = json.loads(sys.argv[1])
OUT = sys.argv[2]
PLANT = os.environ.get("P3_PLANT", "")
SCALE = float(os.environ.get("P3_SCALE", "1.5"))
CALLS = {"planted": 0}


def plant():
    if PLANT == "inside":
        from vllm.model_executor.layers.layernorm import RMSNorm

        orig = RMSNorm.forward_cuda

        def forward_cuda(self, x, *args, **kwargs):
            out = orig(self, x, *args, **kwargs)
            if x.shape[0] <= 8:
                CALLS["planted"] += 1
                return (out[0] * SCALE,) + tuple(out[1:]) if isinstance(out, tuple) else out * SCALE
            return out

        RMSNorm.forward_cuda = forward_cuda
    elif PLANT == "inside_kernel":
        import vllm.kernels.vllm_c  # noqa: F401 - registers the vllm_c implementations
        from vllm import ir

        for op in (ir.ops.rms_norm, ir.ops.fused_add_rms_norm):
            impl = op.impls["vllm_c"]

            def wrap(orig):
                def impl_fn(x, *args, **kwargs):
                    out = orig(x, *args, **kwargs)
                    if x.shape[0] <= 8:
                        CALLS["planted"] += 1
                        if isinstance(out, tuple):
                            out[0].mul_(SCALE)
                        else:
                            out = out * SCALE
                    return out
                return impl_fn

            impl.impl_fn = wrap(impl.impl_fn)
    elif PLANT == "inside_op":
        from vllm.model_executor.layers.activation import SiluAndMul

        orig = SiluAndMul.forward_cuda

        def forward_cuda(self, x, *args, **kwargs):
            out = orig(self, x, *args, **kwargs)
            if x.shape[0] <= 8:
                CALLS["planted"] += 1
                return out * SCALE
            return out

        SiluAndMul.forward_cuda = forward_cuda
    elif PLANT == "inside_rope":
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        orig = RotaryEmbedding.forward_cuda

        def forward_cuda(self, positions, *args, **kwargs):
            out = orig(self, positions, *args, **kwargs)
            if positions.shape[0] <= 8 and isinstance(out, tuple):
                CALLS["planted"] += 1
                return tuple(t * SCALE if t is not None else None for t in out)
            return out

        RotaryEmbedding.forward_cuda = forward_cuda
    elif PLANT == "outside_kvwrite":
        import vllm.v1.attention.backends.flash_attn as fa

        orig = fa.reshape_and_cache_flash

        def reshape_and_cache_flash(key, value, *args, **kwargs):
            if key.shape[0] <= 8:
                CALLS["planted"] += 1
                key = key * SCALE
            return orig(key, value, *args, **kwargs)

        fa.reshape_and_cache_flash = reshape_and_cache_flash
    elif PLANT == "outside":
        import importlib

        for mod, cls in (("vllm.v1.attention.backends.flash_attn", "FlashAttentionImpl"),
                         ("vllm.v1.attention.backends.triton_attn", "TritonAttentionImpl"),
                         ("vllm.v1.attention.backends.flashinfer", "FlashInferImpl")):
            try:
                klass = getattr(importlib.import_module(mod), cls)
            except (ImportError, AttributeError):
                continue

            def wrap(orig):
                def forward(self, *args, **kwargs):
                    out = orig(self, *args, **kwargs)
                    md = kwargs.get("attn_metadata", args[5] if len(args) > 5 else None)
                    if md is not None and getattr(md, "max_query_len", 2) == 1 and out is not None:
                        CALLS["planted"] += 1
                        out.mul_(SCALE)
                    return out
                return forward

            klass.forward = wrap(klass.forward)


PARAGRAPH = ("The printing press, developed in the fifteenth century, changed how knowledge moved through Europe. "
             "Books that had taken months to copy by hand could be produced in days, and the price of a book fell so "
             "far that merchants, students and craftsmen could own them. Within fifty years presses were running in "
             "more than two hundred cities, and the number of books in circulation had grown into the millions.")
CODE = ("def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n"
        "            out.append(a[i])\n            i += 1\n        else:\n            out.append(b[j])\n            j += 1\n"
        "    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n")
BENCH = ["Write a short paragraph about the history of the printing press.",
         "Explain how a hash table works, step by step.",
         "Copy the following text exactly.\n\n" + PARAGRAPH + "\n\nCopy:\n",
         "Rewrite this Python function with type hints, keeping everything else the same.\n\n" + CODE + "\nRewritten:\n"]


def bench(llm):
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
    llm.generate(BENCH, sp, use_tqdm=False)
    times = []
    for _ in range(3):
        t = time.perf_counter()
        outs = llm.generate(BENCH, sp, use_tqdm=False)
        times.append(time.perf_counter() - t)
    tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    med = sorted(times)[1]
    return {"prompts": len(BENCH), "tokens": tokens, "runs_s": [round(x, 3) for x in times], "median_s": round(med, 3),
            "tok_s": round(tokens / med, 1)}


def main():
    from entail import path_contract

    pairs = []
    orig_check = path_contract.check

    def check(boundary, consumer, paths, v, where, *a, **k):
        pairs.append({"paths": paths, "differs": bool(v["differs"]), "margin": round(v["margin"], 4),
                      "pdrift": round(v["pdrift"], 4), "probe": v.get("probe"), "why": v.get("why")})
        return orig_check(boundary, consumer, paths, v, where, *a, **k)

    path_contract.check = check
    plant()
    from vllm import LLM, SamplingParams

    paths_s = []
    try:                        # the self-check's own time (the record keeps only its counts)
        from entail.adapters import vllm_paths

        orig_decide = vllm_paths.decide

        def decide(llm):
            t = time.perf_counter()
            try:
                return orig_decide(llm)
            finally:
                paths_s.append(round(time.perf_counter() - t, 3))

        vllm_paths.decide = decide
    except ImportError:
        pass

    t0 = time.perf_counter()
    llm = LLM(**KW)
    load_s = time.perf_counter() - t0
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0.0))
    from entail import record, safe_mode

    record.close_files()
    run = os.environ.get("ENTAIL_RUN_ID")
    folder = record.log_dir()
    lines = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("record-"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                lines += [x for x in (json.loads(s) for s in f if s.strip()) if x.get("run") == run]
    safe = [x for x in lines if x.get("boundary") == "start:vllm.safe_mode" and x.get("verdict")]
    said = [x["text"] for x in lines if x.get("said") == "start:vllm.safe_mode"]
    paths_lines = [x for x in lines if x.get("boundary") == "start:vllm.paths" and x.get("verdict")]
    result = {
        "kwargs": KW, "plant": PLANT, "scale": SCALE if PLANT else None, "planted_calls": CALLS["planted"],
        "mode": safe_mode.mode(), "run": run, "load_s": round(load_s, 2), "paths_s": paths_s,
        "turned_off": [{"rule": x.get("rule"), "resolution": x.get("resolution"), "verdict": x["verdict"]}
                       for x in safe],
        "pairs": pairs, "path_verdicts": [{"verdict": x["verdict"], "note": (x.get("note") or "")[:300]}
                                          for x in paths_lines],
        "said": said, "store": safe_mode.load_store(), "last": safe_mode.LAST.get("vllm"),
        "answer": out[0].outputs[0].text,
        "engine_spec": bool(getattr(llm.llm_engine.vllm_config, "speculative_config", None)),
        "engine_prefix_cache": bool(llm.llm_engine.vllm_config.cache_config.enable_prefix_caching),
        "engine_eager": bool(llm.llm_engine.model_config.enforce_eager),
    }
    if os.environ.get("P3_BENCH"):
        result["bench"] = bench(llm)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: result.get(k) for k in ("mode", "plant", "planted_calls", "load_s", "turned_off", "said",
                                                  "bench")}, ensure_ascii=False))
    print("pairs", json.dumps([(p["paths"], p["differs"], p["margin"], p["pdrift"]) for p in pairs]))


if __name__ == "__main__":
    main()
