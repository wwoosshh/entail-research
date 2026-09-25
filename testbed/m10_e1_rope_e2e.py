"""M10 E1 L1, end to end (testbed/M10_PROTOCOL.md 1.2): the RoPE override on a model E1 found exposed, run on vLLM 0.30
with GSM8K (the first N problems, greedy; the prompt and the answer extraction of issue_track/rope_override).
  untouched                      no override
  override                       hf_overrides={"rope_scaling": <the YaRN value model cards give>}, entail off or on
The effect of the lost base is override (entail off) against override (entail on): both apply the same YaRN scaling;
only the base differs. The RoPE parameters the engine holds are written too.
Run in ~/venvs/vllm: python testbed/m10_e1_rope_e2e.py <model dir> <name> <untouched|override> [N]
Writes testbed/results/m10/e1_llm/e2e/<name>.json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "m10", "e1_llm", "e2e")
sys.path.insert(0, os.path.join(ROOT, "issue_track", "rope_override"))
from rope_override import GSM_MAX, INSTR, extract_answer  # noqa: E402

DATA = os.path.join(ROOT, "issue_track", "gemma2_softcap", "data", "gsm8k_test.jsonl")


def main():
    model, name, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    from vllm import LLM, SamplingParams

    cfg = json.load(open(os.path.join(model, "config.json"), encoding="utf-8"))
    kw = {}
    if mode == "override":
        mpe = cfg.get("max_position_embeddings") or 32768
        kw["hf_overrides"] = {"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                                               "original_max_position_embeddings": mpe}}
    data = [json.loads(x) for x in open(DATA, encoding="utf-8")][:n]
    gold = [float(x["answer"].split("####")[-1].strip().replace(",", "")) for x in data]
    llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False,
              max_num_seqs=64, disable_log_stats=True, **kw)
    held = llm.llm_engine.model_config.hf_text_config.rope_parameters
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": INSTR + x["question"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for x in data]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(max_tokens=GSM_MAX, temperature=0.0), use_tqdm=False)
    answers = [extract_answer(o.outputs[0].text) for o in outs]
    right = [a is not None and abs(a - g) < 1e-6 for a, g in zip(answers, gold)]
    row = {"name": name, "model": os.path.basename(model.rstrip("/")), "mode": mode,
           "entail": os.environ.get("ENTAIL", "off"), "override": kw.get("hf_overrides"),
           "rope_parameters_in_engine": json.loads(json.dumps(held, default=str)), "n": n, "correct": sum(right),
           "per_problem": right, "generate_seconds": round(time.time() - t0, 1),
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(OUT, exist_ok=True)
    json.dump(row, open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"DONE {name} {row['correct']}/{n} rope={held}")


if __name__ == "__main__":
    main()
