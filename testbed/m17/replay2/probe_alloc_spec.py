"""DEFERRED 22, second probe: the KV-allocation rule under speculative decoding on the installed vLLM (0.30.0).
On 0.23.0 the rule said `broken` only on the extract_hidden_states (speculative) path, not on a plain generate.
Does 0.30.0's ngram speculative decoding on a small dense model make it say broken too (the lookahead tokens the
scheduler allocates beyond the request's tokens)? Qwen3-0.6B, ngram with 3 speculative tokens, entail on.
Run in ~/venvs/vllm with ENTAIL=load and ENTAIL_RECORD set: python testbed/m17/replay2/probe_alloc_spec.py <out.json>
"""
import json
import os
import sys


def main():
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=os.path.expanduser("~/models/Qwen3-0.6B"), max_model_len=512, max_num_seqs=2,
              gpu_memory_utilization=0.6, enforce_eager=True,
              speculative_config={"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4,
                                  "prompt_lookup_min": 2})
    prompts = ["Sample 0: " + "Explain neural networks. " * 20, "The capital of France is"]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__,
           "texts": [o.outputs[0].text[:80] for o in outs]}
    rec = os.environ.get("ENTAIL_RECORD")
    if rec and os.path.isfile(rec):
        rows = [json.loads(x) for x in open(rec, encoding="utf-8") if x.strip()]
        alloc = [r for r in rows if r.get("boundary") == "container:vllm.allocate_slots"]
        row["allocate_slots"] = {v: sum(1 for r in alloc if r.get("verdict") == v) for v in ("pass", "broken", "unknown", "resolved")}
        row["allocate_slots_notes"] = [r.get("note", "")[:220] for r in alloc if r.get("verdict") != "pass"][:3]
        summ = [r for r in rows if "boundaries" in r]
        if summ:
            row["boundary_stats"] = summ[-1]["boundaries"].get("container:vllm.allocate_slots")
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
