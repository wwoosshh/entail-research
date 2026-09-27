"""M19 L4 replay 4 case, vllm-project/vllm#33560 (testbed/M16_PROTOCOL.md 9): RedHatAI/Qwen3-8B-NVFP4 gives NaN
perplexity with float16 activations (Turing, where the auto dtype is float16) through the NVFP4 Marlin path; fixed by
#33972 (scale the input before the GEMM). Ada also runs NVFP4 through Marlin, so dtype=float16 is set to reach the same
path. The report's lambada perplexity, in small: the prompt log-probabilities of a few passages. Reproduced when any is
NaN or infinite with float16 (the second argument 'bfloat16' runs the control).
Run: python testbed/m19/replay4/cases/vl33560.py <out.json> [float16|bfloat16]  (~/venvs/vllm0160: vLLM 0.16.0)
"""
import json
import math
import os
import sys

MODEL = os.path.expanduser("~/models/replay4/RedHatAI__Qwen3-8B-NVFP4")
TEXTS = ["In the heart of the old city stood a clock tower that had not rung for a hundred years, until one night "
         "in winter when the whole town woke to its bell.",
         "The recipe asks for two cups of flour, a pinch of salt, three eggs and enough milk to make the batter smooth "
         "before it rests for an hour.",
         "She read the letter twice, folded it along its creases, and put it back in the drawer where her father had "
         "kept it for forty years."]


def main():
    import vllm
    from vllm import LLM, SamplingParams

    dtype = sys.argv[2] if len(sys.argv) > 2 else "float16"
    llm = LLM(model=MODEL, dtype=dtype, max_model_len=1024, gpu_memory_utilization=0.85, seed=42)
    outs = llm.generate(TEXTS, SamplingParams(max_tokens=8, temperature=0, prompt_logprobs=0), use_tqdm=False)
    bad, total, nll = 0, 0, 0.0
    for o in outs:
        for tid, step in zip(o.prompt_token_ids[1:], (o.prompt_logprobs or [])[1:]):
            lp = (step or {}).get(tid)
            v = None if lp is None else lp.logprob
            total += 1
            if v is None or not math.isfinite(v):
                bad += 1
            else:
                nll -= v
    ppl = math.exp(nll / max(1, total - bad)) if total > bad else None
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "dtype": dtype,
           "prompt_logprobs": total, "non_finite": bad, "perplexity_over_finite": ppl,
           "texts_out": [o.outputs[0].text for o in outs]}
    row["reproduced"] = dtype == "float16" and bad > 0
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()
