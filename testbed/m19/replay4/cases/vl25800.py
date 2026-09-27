"""M19 L4 replay 4 case, vllm-project/vllm#25800 (testbed/M16_PROTOCOL.md 9): with kv_cache_dtype='fp8', waking from
sleep(level=2) gives gibberish while level 1 is fine (the FP8 KV cache's scales were not restored when the cache was
re-allocated; fixed by #28783). The report's script (vLLM's own cumem test with fp8 KV), memory asserts left out.
Reproduced when the text after the deep sleep differs from the text before it.
Run: python testbed/m19/replay4/cases/vl25800.py <out.json>  (~/venvs/vllm0102: vLLM 0.10.2)
"""
import json
import os
import sys

MODEL = os.path.expanduser("~/models/m10/Qwen__Qwen3-0.6B")


def main():
    import vllm
    from vllm import LLM, SamplingParams

    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__}
    llm = LLM(MODEL, enable_sleep_mode=True, kv_cache_dtype="fp8", max_model_len=1024, gpu_memory_utilization=0.6)
    sp = SamplingParams(temperature=0, max_tokens=10)
    prompt = "How are you?"
    row["before"] = llm.generate(prompt, sp, use_tqdm=False)[0].outputs[0].text
    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    llm.wake_up(tags=["kv_cache"])
    row["after"] = llm.generate(prompt, sp, use_tqdm=False)[0].outputs[0].text
    row["reproduced"] = row["before"] != row["after"]
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
