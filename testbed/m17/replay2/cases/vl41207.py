"""M17.6 case, vllm-project/vllm#41207 (testbed/M16_PROTOCOL.md 7): after a transformers upgrade DeepSeek-OCR's
chat answer carries the byte-level space marker ("ThisĠimageĠdisplays...") instead of spaces: the detokeniser
built for the model is not the one its files declare. The report's request (describe the duck image) through
vLLM's chat path offline. Reproduced when the answer text contains the marker character Ġ (U+0120).
Run in ~/venvs/vllm (0.30.0 first; then vllm0190 with transformers 5.6.2 as reported):
  python testbed/m17/replay2/cases/vl41207.py <out.json>
"""
import json
import os
import sys
import traceback

IMAGE = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"


def main():
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__,
           "transformers": transformers.__version__, "model": "deepseek-ai/DeepSeek-OCR"}
    try:
        llm = LLM(model="deepseek-ai/DeepSeek-OCR", trust_remote_code=True, enforce_eager=True, max_model_len=4096,
                  max_num_seqs=1, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1})
        msgs = [{"role": "user", "content": [{"type": "text", "text": "describe the image"},
                                              {"type": "image_url", "image_url": {"url": IMAGE}}]}]
        out = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)[0]
        text = out.outputs[0].text
        row["text"] = text
        row["marker_count"] = text.count("Ġ")
        row["reproduced"] = row["marker_count"] > 0
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:400]
        row["trace_tail"] = traceback.format_exc().strip().splitlines()[-2:]
        row["reproduced"] = None
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
