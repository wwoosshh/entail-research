"""M19 L4 replay 4 case, vllm-project/vllm#27390 (testbed/M16_PROTOCOL.md 9): when the EOS token falls exactly on the
token limit, the V1 engine reports finish_reason 'length' where V0 reports 'stop' (V1 checked the length before EOS;
fixed by #27555). The report's script (Qwen3-4B-Instruct-2507, the chat template without a generation prompt, greedy,
max_tokens 16 then 15). Reproduced when the max_tokens=15 run ends on the EOS token and says 'length'.
Run: python testbed/m19/replay4/cases/vl27390.py <out.json>  (~/venvs/vllm0110: vLLM 0.11.0; ~/venvs/vllm: 0.30.0)
"""
import json
import os
import sys

MODEL = os.path.expanduser("~/models/m10/Qwen__Qwen3-4B-Instruct-2507")


def main():
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hello?"}], tokenize=False)
    llm = LLM(model=MODEL, max_model_len=1024, gpu_memory_utilization=0.85, seed=0)
    runs = {}
    for n in (16, 15):
        o = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=n), use_tqdm=False)[0].outputs[0]
        runs[n] = {"text": o.text, "token_ids": list(o.token_ids), "finish_reason": o.finish_reason}
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    last15 = runs[15]["token_ids"][-1] if runs[15]["token_ids"] else None
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "runs": runs, "eos_id": eos,
           "n_tokens_16": len(runs[16]["token_ids"]), "last_token_15_is_eos": last15 == eos}
    row["reproduced"] = row["last_token_15_is_eos"] and runs[15]["finish_reason"] == "length"
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps({k: v for k, v in row.items() if k != "runs"} |
                               {"finish_15": runs[15]["finish_reason"], "finish_16": runs[16]["finish_reason"]}))


if __name__ == "__main__":
    main()
