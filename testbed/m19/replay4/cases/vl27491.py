"""M19 L4 replay 4 case, vllm-project/vllm#27491 (testbed/M16_PROTOCOL.md 9): an MLA model under chunked prefill gives
NaN log-probabilities: a context chunk whose query rows have no keys yields output 0 and LSE -inf, and merging it gives
NaN. The report's reproduction offline: bzantium/tiny-deepseek-v3, long_prefill_token_threshold 128, 256 prompts of
" A" * 2415, max_tokens 1 with the prompt's log-probabilities (the report's echo=True, logprobs=1). Reproduced when any
prompt log-probability is NaN (the server fails to serialise it).
Run: python testbed/m19/replay4/cases/vl27491.py <out.json>  (~/venvs/vllm: 0.30.0; ~/venvs/vllm0110: 0.11.0)
"""
import json
import math
import os
import sys

MODEL = os.path.expanduser("~/models/replay4/bzantium__tiny-deepseek-v3")


def main():
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, long_prefill_token_threshold=128, gpu_memory_utilization=0.85, seed=0,
              trust_remote_code=True)
    prompts = [" A" * 2415 for _ in range(256)]
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, logprobs=1, prompt_logprobs=1), use_tqdm=False)
    nan_prompts, nan_values, total = 0, 0, 0
    for o in outs:
        bad = 0
        for step in (o.prompt_logprobs or [])[1:]:
            for lp in (step or {}).values():
                total += 1
                if lp.logprob is None or math.isnan(lp.logprob):
                    bad += 1
        nan_values += bad
        nan_prompts += bool(bad)
    backend = str(getattr(llm.llm_engine.vllm_config.model_config, "use_mla", None))
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "use_mla": backend,
           "prompts": len(outs), "prompt_tokens": len(outs[0].prompt_token_ids), "nan_prompts": nan_prompts,
           "nan_values": nan_values, "values": total}
    row["reproduced"] = nan_values > 0
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
