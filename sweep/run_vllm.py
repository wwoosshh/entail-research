"""Sweep one (fact, backend) pair in vLLM, by the rules in PROTOCOL.md.

Same shape as the SGLang runner: one pair per process, and the fact is perturbed by building two model
directories that share the weights and differ only in config.json. The backend is chosen with
VLLM_ATTENTION_BACKEND, which the shell script sets before starting this process.

Usage: VLLM_ATTENTION_BACKEND=FLASH_ATTN python sweep/run_vllm.py <model-dir> <fact>
"""
import copy
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facts import FACTS  # noqa: E402
from run_sglang import PROMPTS, N_NEW, RESULTS, model_copy  # noqa: E402


def decode(model_path, runs):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, max_model_len=2048, gpu_memory_utilization=0.88, enforce_eager=True,
              disable_log_stats=True)
    params = SamplingParams(max_tokens=N_NEW, temperature=0)
    return [[o.outputs[0].text for o in llm.generate(PROMPTS, params)] for _ in range(runs)]


def main():
    model_dir, fact_name = os.path.expanduser(sys.argv[1]), sys.argv[2]
    backend = os.environ.get("VLLM_ATTENTION_BACKEND", "default")
    name = os.path.basename(model_dir.rstrip("/"))
    fact = FACTS[fact_name]
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        base = json.load(f)
    if not fact["applies"](base):
        print(f"SKIP {fact_name}: {name} does not declare it", flush=True)
        return
    bound_cfg = copy.deepcopy(base)
    fact["bind"](bound_cfg)
    gone_cfg = copy.deepcopy(base)
    fact["bind"](gone_cfg)
    fact["remove"](gone_cfg)

    row = {"fact": fact_name, "kind": fact["kind"], "model": name, "engine": "vllm", "backend": backend,
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    bound_dir = gone_dir = None
    t0 = time.time()
    try:
        bound_dir = model_copy(model_dir, bound_cfg)
        gone_dir = model_copy(model_dir, gone_cfg)
        a, a2 = decode(bound_dir, runs=2)
        (b,) = decode(gone_dir, runs=1)
        if a != a2:
            row.update(verdict="void", why="the control run did not reproduce")
        else:
            row.update(verdict="dropped" if a == b else "honoured",
                       differing_prompts=sum(1 for x, y in zip(a, b) if x != y))
    except Exception as e:
        row.update(verdict="void", why=f"{type(e).__name__}: {str(e)[:160]}")
    finally:
        for d in (bound_dir, gone_dir):
            if d:
                shutil.rmtree(d, ignore_errors=True)
    row["seconds"] = round(time.time() - t0, 1)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"vllm_{name}_{fact_name}_{backend}.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    print(f"RESULT {fact_name} {backend} {row['verdict']} differs={row.get('differing_prompts')} "
          f"{row['seconds']}s {row.get('why', '')}", flush=True)


if __name__ == "__main__":
    main()
