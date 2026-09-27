"""DEFERRED 22 (after the second replay): on vLLM 0.23.0 entail said `broken` at container:vllm.allocate_slots for
every KV cache group of a hybrid model (Qwen3.5-4B) served with extract_hidden_states ("holds 512 KV slots for 205
tokens, not one allocation unit (256)"). Is it the hybrid block alignment (then a plain generate shows it too) or
the speculative lookahead path (then a plain generate passes)? A plain offline generate on the same model and
version, entail on, the allocate_slots decisions read from the record.
Run in ~/venvs/vllm0230 with ENTAIL=load and ENTAIL_RECORD set: python testbed/m17/replay2/probe_alloc_0230.py <out.json>
"""
import json
import os
import sys


def main():
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3.5-4B", max_model_len=256, max_num_seqs=1, gpu_memory_utilization=0.88,
              enforce_eager=True)
    prompt = "Sample 0: " + "Explain neural networks. " * 20
    out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)[0]
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "prompt_tokens": len(out.prompt_token_ids),
           "text": out.outputs[0].text}
    rec = os.environ.get("ENTAIL_RECORD")
    if rec and os.path.isfile(rec):
        rows = [json.loads(x) for x in open(rec, encoding="utf-8") if x.strip()]
        alloc = [r for r in rows if r.get("boundary") == "container:vllm.allocate_slots"]
        row["allocate_slots"] = {v: sum(1 for r in alloc if r.get("verdict") == v) for v in ("pass", "broken", "unknown", "resolved")}
        row["allocate_slots_notes"] = [r.get("note", "")[:200] for r in alloc if r.get("verdict") != "pass"][:3]
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
