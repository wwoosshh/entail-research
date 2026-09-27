"""M17.6 case, vllm-project/vllm#38643 (testbed/M16_PROTOCOL.md 7): Qwen3.5-4B-NVFP4 (Qwen3_5ForConditionalGeneration,
hybrid GDN linear attention) produces gibberish - a mix of random tokens from several languages - and the FLA ops
warn of a head-first vs seq-first format mismatch. The report's model through vLLM offline, greedy, on two plain
prompts. Reproduced when the answers do not contain the expected words (Paris; a number for 2+2), which a working
4B model gives; the FLA format warning is recorded from stderr by the runner's log.
Run in ~/venvs/vllm: python testbed/m17/replay2/cases/vl38643.py <out.json> [model]
"""
import json
import os
import sys
import traceback


def main():
    import vllm
    from vllm import LLM, SamplingParams

    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-4B-NVFP4"
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "model": model}
    try:
        llm = LLM(model=model, max_model_len=1024, max_num_seqs=2, gpu_memory_utilization=0.85, enforce_eager=True)
        sp = SamplingParams(temperature=0, max_tokens=24)
        prompts = ["The capital of France is", "Question: What is 2 + 2? Answer:"]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]
        row["texts"] = texts
        row["expected_found"] = ["Paris" in texts[0], "4" in texts[1]]
        row["reproduced"] = not all(row["expected_found"])
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:400]
        row["trace_tail"] = traceback.format_exc().strip().splitlines()[-2:]
        row["reproduced"] = None
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
