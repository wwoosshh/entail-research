"""M5.1: one engine run with the KV container contract (installed by the sitecustomize shim in the engine's own
processes). The same runs as entail/audits/CACHE_CONTRACT.md measured before the rules moved to the core.

Usage (through testbed/m51_engines.sh, which sets PYTHONPATH, ENTAIL, ENTAIL_RECORD and the seed):
  python testbed/m51_engine_run.py vllm   <tag>     Qwen3-4B, eager, three prompts x 32 tokens
  python testbed/m51_engine_run.py sglang <tag>     Qwen3-4B, default backend, no CUDA graph, the same prompts
Prints one RESULT line and writes testbed/results/m51/<engine>_<tag>.json (or into TESTBED_OUT; M5.4 used it). What the contract checked is in the
ENTAIL_RECORD file: the engines check in child processes (vLLM's EngineCore, SGLang's scheduler).
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
MODEL = os.path.expanduser("~/models/Qwen3-4B")
PROMPTS = ["The capital of France is", "Explain gravity in one sentence.", "List three prime numbers."]


def vllm_run():
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, max_model_len=2048, gpu_memory_utilization=0.88, enforce_eager=True,
              disable_log_stats=True)
    outs = llm.generate(PROMPTS, SamplingParams(max_tokens=32, temperature=0))
    return [o.outputs[0].text.strip().replace("\n", " ")[:60] for o in outs]


def sglang_run():
    import sglang as sgl

    engine = sgl.Engine(model_path=MODEL, mem_fraction_static=0.92, max_total_tokens=2048, log_level="error",
                        disable_cuda_graph=True)
    try:
        outs = [engine.generate(p, {"max_new_tokens": 32, "temperature": 0}) for p in PROMPTS]
        return [o["text"].strip().replace("\n", " ")[:60] for o in outs]
    finally:
        engine.shutdown()


def main():
    engine, tag = sys.argv[1], sys.argv[2]
    t0 = time.time()
    res = {"engine": engine, "tag": tag, "entail": os.environ.get("ENTAIL", "off"),
           "seed": os.environ.get("ENTAIL_SEED"), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        res["texts"] = vllm_run() if engine == "vllm" else sglang_run()
        res["error"] = None
    except Exception as e:  # noqa: BLE001 - a refusal inside the engine surfaces here as the engine's own error
        blamed = [ln for ln in traceback.format_exc().splitlines() if "RoleError" in ln]
        res["texts"], res["error"] = None, (blamed[-1] if blamed else f"{type(e).__name__}: {e}")[:400]
    res["seconds"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
    out = os.path.join(os.environ.get("TESTBED_OUT") or os.path.join(RESULTS, "m51"), f"{engine}_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
