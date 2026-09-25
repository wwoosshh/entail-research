"""M9.2 (S7): the evaluation score as a post-hoc detector - GSM8K, the first 500 problems, greedy (M92_PROTOCOL.md).
The problems, the prompt and the answer extraction are issue_track/rope_override/rope_override.py's (at most 512 new
tokens, thinking off). entail and a planted defect are switched on from outside, as m91_run.sh does it.

Usage (in the engine's venv): python testbed/m92_gsm8k.py <vllm|sglang> <model dir> <name> [attention backend]
Writes testbed/results/m92/<name>.json (TESTBED_RESULTS moves it).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
OUT = os.path.join(RESULTS, "m92")
sys.path.insert(0, os.path.join(ROOT, "issue_track", "rope_override"))
from rope_override import GSM_MAX, GSM_N, INSTR, extract_answer  # noqa: E402

DATA = os.path.join(ROOT, "issue_track", "gemma2_softcap", "data", "gsm8k_test.jsonl")


def problems():
    data = [json.loads(line) for line in open(DATA, encoding="utf-8")][:GSM_N]
    gold = [float(x["answer"].split("####")[-1].strip().replace(",", "")) for x in data]
    return data, gold


def prompts(tok, data):
    return [tok.apply_chat_template([{"role": "user", "content": INSTR + x["question"]}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False) for x in data]


def run_vllm(model_dir, backend, data):
    if backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = backend
    import vllm
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=model_dir, max_model_len=2048, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False,
              max_num_seqs=64, disable_log_stats=True)
    load_s = time.time() - t0
    ps = prompts(llm.get_tokenizer(), data)
    t1 = time.time()
    outs = llm.generate(ps, SamplingParams(max_tokens=GSM_MAX, temperature=0.0), use_tqdm=False)
    return {"engine_version": vllm.__version__, "load_seconds": round(load_s, 1),
            "generate_seconds": round(time.time() - t1, 1)}, [o.outputs[0].text for o in outs]


def run_sglang(model_dir, backend, data):
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    ps = prompts(tok, data)
    t0 = time.time()
    kw = {"attention_backend": backend} if backend else {}
    engine = sgl.Engine(model_path=model_dir, mem_fraction_static=0.8, context_length=2048, log_level="error",
                        disable_cuda_graph=True, disable_radix_cache=True, random_seed=0, **kw)
    load_s = time.time() - t0
    try:
        t1 = time.time()
        outs = engine.generate(ps, {"max_new_tokens": GSM_MAX, "temperature": 0})
        gen_s = time.time() - t1
    finally:
        engine.shutdown()
    return {"engine_version": sgl.__version__, "load_seconds": round(load_s, 1),
            "generate_seconds": round(gen_s, 1)}, [o["text"] for o in outs]


def main():
    engine, model_dir, name = sys.argv[1], os.path.expanduser(sys.argv[2]), sys.argv[3]
    backend = sys.argv[4] if len(sys.argv) > 4 else None
    data, gold = problems()
    row = {"name": name, "engine": engine, "model": os.path.basename(model_dir.rstrip("/")), "backend_asked": backend,
           "entail": os.environ.get("ENTAIL", "off"), "seed": os.environ.get("ENTAIL_SEED"),
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    info, texts = {"vllm": run_vllm, "sglang": run_sglang}[engine](model_dir, backend, data)
    answers = [extract_answer(t) for t in texts]
    correct = [a is not None and abs(a - g) < 1e-6 for a, g in zip(answers, gold)]
    row.update(info, gsm8k={"n": len(data), "correct": correct, "accuracy": sum(correct) / len(data),
                            "outputs": texts})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)
    print(f"DONE {name} {engine} {row['model']} backend={backend} entail={row['entail']} seed={row['seed']} "
          f"gsm8k={sum(correct)}/{len(data)}", flush=True)


if __name__ == "__main__":
    main()
