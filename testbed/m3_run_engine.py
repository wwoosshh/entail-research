"""M3.5: load one model in one engine, time the load, decode a few prompts greedily, write what happened.

Run in the engine's own venv. entail is switched on from outside (PYTHONPATH with the autoinstall shim, ENTAIL=load,
ENTAIL_RECORD=<file>), so the engine's child processes get it too; this script only loads and decodes.
Usage: python m3_run_engine.py <transformers|vllm|sglang> <model dir> <out.json> [attention backend] [--paged]
"""
import json
import os
import sys
import time

PROMPTS = ["Explain in two sentences why the sky is blue.", "What is the capital of Australia?",
           "Write one sentence about a quiet library."]
N_NEW = 16


def run_transformers(model_dir, backend, paged):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
               add_special_tokens=False).input_ids for p in PROMPTS]
    t0 = time.perf_counter()
    kw = {"attn_implementation": backend} if backend else {}
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, **kw).cuda().eval()
    load_s = time.perf_counter() - t0
    chosen = model.config._attn_implementation
    if paged:
        gc = GenerationConfig(max_new_tokens=N_NEW, do_sample=False, eos_token_id=None)
        with torch.no_grad():
            res = model.generate_batch(inputs=ids, generation_config=gc)
        keys = list(res)
        outs = [list(res[keys[i]].generated_tokens)[:N_NEW] for i in range(len(ids))]
    else:
        outs = []
        for x in ids:
            with torch.no_grad():
                g = model.generate(torch.tensor([x], device="cuda"), max_new_tokens=N_NEW, do_sample=False)
            outs.append(g[0, len(x):].tolist())
    return load_s, chosen, [tok.decode(o) for o in outs]


def run_vllm(model_dir, backend, paged):
    if backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = backend
    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(model=model_dir, max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=True,
              disable_log_stats=True, seed=0)
    load_s = time.perf_counter() - t0
    outs = llm.generate(PROMPTS, SamplingParams(max_tokens=N_NEW, temperature=0), use_tqdm=False)
    return load_s, backend or "default", [o.outputs[0].text for o in outs]


def run_sglang(model_dir, backend, paged):
    import sglang as sgl

    t0 = time.perf_counter()
    kw = {"attention_backend": backend} if backend else {}
    engine = sgl.Engine(model_path=model_dir, mem_fraction_static=0.8, context_length=2048, log_level="error",
                        disable_cuda_graph=True, disable_radix_cache=True, random_seed=0, **kw)
    load_s = time.perf_counter() - t0
    try:
        outs = [engine.generate(p, {"max_new_tokens": N_NEW, "temperature": 0})["text"] for p in PROMPTS]
    finally:
        engine.shutdown()
    return load_s, backend or "default", outs


def main():
    engine, model_dir, out_path = sys.argv[1], os.path.expanduser(sys.argv[2]), sys.argv[3]
    rest = [a for a in sys.argv[4:] if not a.startswith("--")]
    backend, paged = (rest[0] if rest else None), "--paged" in sys.argv
    row = {"engine": engine, "model": os.path.basename(model_dir.rstrip("/")), "backend_asked": backend,
           "paged": paged, "entail": os.environ.get("ENTAIL", "off"), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        load_s, chosen, outs = {"transformers": run_transformers, "vllm": run_vllm, "sglang": run_sglang}[engine](
            model_dir, backend, paged)
        row.update(ok=True, load_seconds=round(load_s, 3), backend_used=chosen, outputs=outs)
    except Exception as e:  # noqa: BLE001 - a refusal stops the load; that is a result, recorded as such
        row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:2000]}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)
    print(f"RESULT {engine} {row['model']} entail={row['entail']} ok={row['ok']} "
          f"load={row.get('load_seconds')} used={row.get('backend_used')} {row.get('error', '')[:200]}", flush=True)


if __name__ == "__main__":
    main()
