"""M16 case, vllm-project/vllm#48231 (testbed/M16_PROTOCOL.md 5): Gemma 4 image requests give all-NaN logits (1024
copies of token 0) after vLLM's BF16-to-FP16 fallback: the vision embedder's patch_dense projection overflows
FP16 before its LayerNorm. The report's request (a generated 512x512 red PNG, "What color is in this image?") on
gemma-4-12B-it-qat-w4a16-ct through vLLM offline; the fallback that an RTX 2080 Ti takes on its own is forced here
with dtype=float16 on Ada (bf16 is the control, run as a second case). vLLM 0.30.0.
Run in ~/venvs/vllm: python testbed/m16/cases/vl48231.py <out.json> <model dir> [dtype]
"""
import base64
import io
import json
import os
import sys


def main():
    import vllm
    from PIL import Image
    from vllm import LLM, SamplingParams

    model, dtype = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "float16")
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(buf, format="PNG", optimize=True)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    llm = LLM(model=model, dtype=dtype, max_model_len=2560, max_num_batched_tokens=2560, max_num_seqs=1,
              gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1}, enforce_eager=True)
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": "What color is in this image?"}]}]
    out = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=32), use_tqdm=False,
                   chat_template_kwargs={"enable_thinking": False})[0]
    ids = list(out.outputs[0].token_ids)
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "dtype": dtype,
           "text": out.outputs[0].text, "finish_reason": out.outputs[0].finish_reason, "token_ids_head": ids[:16],
           "all_token_zero": bool(ids) and all(t == 0 for t in ids), "distinct_tokens": len(set(ids))}
    # the report: 1024 copies of token 0 (all-NaN logits). Here the degenerate form is judged the same way: one
    # token repeated to the length limit under fp16 (the bf16 run of the same request is the control).
    row["reproduced"] = bool(ids) and (row["all_token_zero"] or (row["distinct_tokens"] == 1
                                                                 and out.outputs[0].finish_reason == "length"))
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
