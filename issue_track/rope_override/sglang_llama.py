"""The Llama arm of PROTOCOL.md section 5 on SGLang (0.5.20): the checkpoint's own rope_scaling restated through
--json-model-override-args. One configuration per process:  python sglang_llama.py <SA|SA2|SC|SE|SC_rolecheck>

  SA, SA2       untouched (SA2 is the run-to-run control)
  SC            json_model_override_args = {"rope_scaling": <what config.json already says>}
  SE            config.json rope_theta = 10000 (what a base of 10,000 does, the mechanism reference)
  SC_rolecheck  SC with only the rope_alias resolver installed (see run_sglang_llama.sh)

M1 teacher-forced NLL on the same three documents as rope_override.py (1024-token chunks, 4 per document, sent
one at a time so batching cannot change the numbers); M2 GSM8K first 200, greedy, max 384 tokens.
Writes results/<name>.json.
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rope_override import LLAMA, LLAMA_SCALING, INSTR, PROJECT, RES, documents, extract_answer, model_copy  # noqa

CASES = {"SA": ({}, {}), "SA2": ({}, {}), "SC": ({}, {"rope_scaling": LLAMA_SCALING}),
         "SE": ({"rope_theta": 10000.0}, {}), "SC_rolecheck": ({}, {"rope_scaling": LLAMA_SCALING}),
         "SC_entail": ({}, {"rope_scaling": LLAMA_SCALING}),  # SC_rolecheck again after the rename to entail
         "SA_now": ({}, {}),  # SA again with today's documents (the Korean one is a live file, see PROTOCOL 8)
         "SC_pip": ({}, {"rope_scaling": LLAMA_SCALING})}  # entail installed with pip, ENTAIL=load, no PYTHONPATH
CHUNK, CHUNKS_PER_DOC, GSM_N, GSM_MAX = 1024, 4, 200, 384


def main():
    name = sys.argv[1]
    edit, override = CASES[name]
    import sglang as sgl
    from transformers import AutoTokenizer

    d = model_copy(edit, LLAMA)
    try:
        tok = AutoTokenizer.from_pretrained(d)
        # CUDA graphs off and 0.7 of memory, as in the sweep: with graphs on, capture ran this 12 GB card out of
        # memory (avail_mem 0.00 GB, 8 s per captured size).
        engine = sgl.Engine(model_path=d, json_model_override_args=json.dumps(override), mem_fraction_static=0.7,
                            max_total_tokens=8192, log_level="error", disable_cuda_graph=True)
        try:
            nll = {}
            for doc, text in documents():
                ids = tok(text, add_special_tokens=False).input_ids
                tot, n = 0.0, 0
                for c in [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)][:CHUNKS_PER_DOC]:
                    r = engine.generate(input_ids=c, sampling_params={"max_new_tokens": 1, "temperature": 0},
                                        return_logprob=True, logprob_start_len=0)
                    lps = [t[0] for t in r["meta_info"]["input_token_logprobs"] if t and t[0] is not None]
                    tot -= sum(lps)
                    n += len(lps)
                nll[doc] = {"tokens": n, "mean_nll": tot / n}
            data = [json.loads(line) for line in open(os.path.join(PROJECT, "issue_track", "gemma2_softcap", "data",
                                                                   "gsm8k_test.jsonl"), encoding="utf-8")][:GSM_N]
            gold = [float(x["answer"].split("####")[-1].strip().replace(",", "")) for x in data]
            prompts = [tok.apply_chat_template([{"role": "user", "content": INSTR + x["question"]}], tokenize=False,
                                               add_generation_prompt=True) for x in data]
            outs = engine.generate(prompts, {"max_new_tokens": GSM_MAX, "temperature": 0})
            texts = [o["text"] for o in outs]
        finally:
            engine.shutdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    correct = [extract_answer(t) is not None and abs(extract_answer(t) - g) < 1e-6 for t, g in zip(texts, gold)]
    row = {"config": name, "engine": "sglang", "model": os.path.basename(LLAMA), "config_json_edit": edit,
           "json_model_override_args": override, "nll": nll,
           "gsm8k": {"n": len(data), "correct": correct, "accuracy": sum(correct) / len(data), "outputs": texts},
           "entail": os.environ.get("ENTAIL", "off"), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)
    print(f"DONE {name} nll={ {k: round(v['mean_nll'], 4) for k, v in nll.items()} } gsm8k={sum(correct)}/{len(data)}",
          flush=True)


if __name__ == "__main__":
    main()
