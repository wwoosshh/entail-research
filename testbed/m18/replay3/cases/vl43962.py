"""M18.6 replay 3 case, vllm-project/vllm#43962 (testbed/M16_PROTOCOL.md 8): Qwen3-1.7B answers "2 ** 10" with 81
under the Triton attention backend (TP=1) on vLLM 0.21.0 and 1024 with the default backend. The report's script at
TP=1 (TP=2 needs two cards): greedy, seed 42, max_tokens 32, enforce_eager. Reproduced when the Triton-attention
answer is not 1024 while the default backend's is.
Run in a vLLM venv: python testbed/m18/replay3/cases/vl43962.py <out.json>
"""
import gc
import json
import os
import sys

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
MODEL = "Qwen/Qwen3-1.7B"
PROMPT = "In Python, what is the output of: 2 ** 10? Answer with just the number."


def run(**kw):
    import torch
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, top_p=1.0, seed=42, max_tokens=32)
    llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.45, max_model_len=2048, **kw)
    text = llm.generate([PROMPT], sp)[0].outputs[0].text
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return text.strip()


def main():
    import vllm

    default = run()
    triton = run(attention_backend="TRITON_ATTN")
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "default": default,
           "triton_attn": triton, "reproduced": default.startswith("1024") and not triton.startswith("1024")}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
