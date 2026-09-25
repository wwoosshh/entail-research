"""Smoke test: can vLLM / SGLang load Qwen3-4B on this machine (RTX 4070 Ti 12 GB, WSL2) and generate?

Run inside the matching venv:
  ~/venvs/vllm/bin/python 08_engine_smoke.py vllm
  ~/venvs/sglang/bin/python 08_engine_smoke.py sglang
Prints the greedy continuation of a fixed prompt and the elapsed time. Writes logs/08_<engine>.json.
"""
import json
import os
import sys
import time

MODEL = os.path.expanduser("~/models/Qwen3-4B")
PROMPT = "The capital of France is"
HERE = os.path.dirname(os.path.abspath(__file__))


def run_vllm():
    """Settings for this machine: 12 GB with about 1.2 GB already in use, so vLLM cannot ask for more than about
    0.88 of the card (0.92 was refused: 10.78 GiB free vs 11.03 GiB requested), and 4096 tokens of context left
    too little KV cache for a 4B model (0.46 GiB free vs 0.56 GiB needed)."""
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.88, seed=0)
    out = llm.generate([PROMPT], SamplingParams(max_tokens=16, temperature=0.0))
    return {"version": vllm.__version__, "text": out[0].outputs[0].text}


def run_sglang():
    import sglang as sgl

    llm = sgl.Engine(model_path=MODEL, mem_fraction_static=0.8, context_length=4096, random_seed=0)
    try:
        out = llm.generate(PROMPT, {"max_new_tokens": 16, "temperature": 0.0})
    finally:
        llm.shutdown()
    return {"version": sgl.__version__, "text": out["text"]}


if __name__ == "__main__":
    engine = sys.argv[1]
    t0 = time.time()
    try:
        res = {"engine": engine, "ok": True, **(run_vllm() if engine == "vllm" else run_sglang())}
    except Exception as e:
        res = {"engine": engine, "ok": False, "error": f"{type(e).__name__}: {str(e)[:800]}"}
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "logs", f"08_{engine}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False), flush=True)
