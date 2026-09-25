"""Launch-time RoPE override on vLLM: what the lost rope_theta does to the output. Definitions in PROTOCOL.md.

One configuration per process:  python rope_override.py <A|B|C|D|E|C_rolecheck>
Writes results/<name>.json. run_all.sh runs every configuration and summarize.py compares them.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PROJECT = os.path.dirname(os.path.dirname(HERE))
QWEN = os.path.expanduser("~/models/Qwen3-4B")
LLAMA = os.path.expanduser("~/models/Llama-3.2-3B-Instruct")
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
CONFIGS = {  # name -> (config.json edit, hf_overrides)
    "A": ({}, {}),
    "B": ({"rope_scaling": YARN}, {}),
    "C": ({}, {"rope_scaling": YARN}),
    "D": ({}, {"rope_theta": 10000.0}),
    "E": ({"rope_scaling": YARN, "rope_theta": 10000.0}, {}),
    "A2": ({}, {}),  # A again: the run-to-run control for token identity
    "C_rolecheck": ({}, {"rope_scaling": YARN}),  # C with the resolver installed (see run_all.sh)
}
# Llama 3.2 (rope_theta 500,000, llama3 scaling in the checkpoint). LC passes the checkpoint's own rope_scaling
# again at launch - it restates what is already there, so any change it makes is the loss and nothing else.
with open(os.path.join(LLAMA, "config.json"), encoding="utf-8") as _f:
    LLAMA_SCALING = json.load(_f).get("rope_scaling")
LLAMA_CONFIGS = {
    "LA": ({}, {}),
    "LA2": ({}, {}),
    "LC": ({}, {"rope_scaling": LLAMA_SCALING}),
    "LE": ({"rope_theta": 10000.0}, {}),
    "LC_rolecheck": ({}, {"rope_scaling": LLAMA_SCALING}),
    "LC_entail_v2": ({}, {"rope_scaling": LLAMA_SCALING}),   # LC with every v2 adapter on (ROADMAP M3.5)
    "LA_m35": ({}, {}),                                          # LA again, the same-day control for LC_entail_v2
    # ROADMAP M9 (final code): the override with entail on and off, and the same-day control
    "LC_m91": ({}, {"rope_scaling": LLAMA_SCALING}),
    "LC_off_m91": ({}, {"rope_scaling": LLAMA_SCALING}),
    "LA_m91": ({}, {}),
    # ROADMAP M9.3: the override with entail on, after vocabulary v4 carries llama3's frequency factors
    "LC_m93": ({}, {"rope_scaling": LLAMA_SCALING}),
}
CHUNK, CHUNKS_PER_DOC, GSM_N, GSM_MAX = 2048, 4, 500, 512
INSTR = ("Solve the following math problem step by step. "
         "At the end, write the final answer as a number on its own line in the form '#### <number>'.\n\n")


def model_copy(edit, src):
    d = tempfile.mkdtemp(prefix="rope_override_")
    for name in os.listdir(src):
        if name != "config.json" and os.path.isfile(os.path.join(src, name)):
            os.symlink(os.path.join(src, name), os.path.join(d, name))
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update(json.loads(json.dumps(edit)))
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return d


def documents():
    import transformers.models.gemma2.modeling_gemma2 as mg

    out = []
    for name, path in (("english_gpl3", "/usr/share/common-licenses/GPL-3"),
                       # A frozen copy: the live RESEARCH_PLAN.md kept changing while runs went on (PROTOCOL 8).
                       ("korean_research_plan", os.path.join(HERE, "data", "korean_doc_2026-09-23.md")),
                       ("python_code", mg.__file__)):
        with open(path, encoding="utf-8", errors="replace") as f:
            out.append((name, f.read()))
    return out


def extract_answer(text):  # same as issue_track/gemma2_softcap/g2softcap.py
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"-?[\d,]*\.?\d+", text)
    for s in reversed(nums):
        try:
            return float(s.replace(",", ""))
        except ValueError:
            continue
    return None


def main():
    name = sys.argv[1]
    src = LLAMA if name in LLAMA_CONFIGS else QWEN
    edit, overrides = (LLAMA_CONFIGS if name in LLAMA_CONFIGS else CONFIGS)[name]
    from vllm import LLM, SamplingParams

    d = model_copy(edit, src)
    t0 = time.time()
    try:
        # 12 GB card, 7.6 GiB of weights. Prompt logprobs materialise full-vocabulary logits for every scheduled
        # position, which the start-up memory profile does not count, so the token budget per step is kept small.
        llm = LLM(model=d, hf_overrides=json.loads(json.dumps(overrides)), max_model_len=3072,
                  gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False, max_num_batched_tokens=256,
                  max_num_seqs=64, compilation_config={"max_cudagraph_capture_size": 64})
        load_s = time.time() - t0
        cfg = llm.llm_engine.model_config.hf_text_config
        rope = json.loads(json.dumps(getattr(cfg, "rope_parameters", None), default=str))
        tok = llm.get_tokenizer()

        # M1: teacher-forced NLL on fixed documents
        nll = {}
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        for doc, text in documents():
            ids = tok(text, add_special_tokens=False).input_ids
            chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)][:CHUNKS_PER_DOC]
            outs = llm.generate([{"prompt_token_ids": c} for c in chunks], sp, use_tqdm=False)
            tot, n = 0.0, 0
            for c, o in zip(chunks, outs):
                for pos in range(1, len(c)):
                    tot -= o.prompt_logprobs[pos][c[pos]].logprob
                    n += 1
            nll[doc] = {"tokens": n, "mean_nll": tot / n}

        # M2: GSM8K
        data = [json.loads(line) for line in open(os.path.join(PROJECT, "issue_track", "gemma2_softcap", "data",
                                                               "gsm8k_test.jsonl"), encoding="utf-8")][:GSM_N]
        gold = [float(x["answer"].split("####")[-1].strip().replace(",", "")) for x in data]
        prompts = [tok.apply_chat_template([{"role": "user", "content": INSTR + x["question"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for x in data]
        t1 = time.time()
        outs = llm.generate(prompts, SamplingParams(max_tokens=GSM_MAX, temperature=0.0), use_tqdm=False)
        gen_s = time.time() - t1
        texts = [o.outputs[0].text for o in outs]
        correct = [extract_answer(t) is not None and abs(extract_answer(t) - g) < 1e-6 for t, g in zip(texts, gold)]
    finally:
        shutil.rmtree(d, ignore_errors=True)

    os.makedirs(RES, exist_ok=True)
    row = {"config": name, "model": os.path.basename(src), "config_json_edit": edit, "hf_overrides": overrides, "rope_parameters_in_engine": rope,
           "nll": nll, "gsm8k": {"n": len(data), "correct": correct, "accuracy": sum(correct) / len(data),
                                 "seconds": round(gen_s, 1), "outputs": texts},
           "load_seconds": round(load_s, 1), "entail": os.environ.get("ENTAIL", "off"),
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(RES, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)
    print(f"DONE {name} rope={rope} nll={ {k: round(v['mean_nll'], 4) for k, v in nll.items()} } "
          f"gsm8k={sum(correct)}/{len(data)}", flush=True)


if __name__ == "__main__":
    main()
