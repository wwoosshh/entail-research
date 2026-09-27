"""M18.6 replay 3 case, vllm-project/vllm#43602 (testbed/M16_PROTOCOL.md 8): Qwen3-VL-2B-Instruct scores lower under
vLLM's default (torch.compile + CUDA graph) path; the fix PR #43617 says the compile warm-up specializes the decoder
graph to the no-deepstack path, so real image requests lose the deepstack visual features. The report measured Geo3K;
this case holds the same requests under eager (the uncompiled path, which reads the deepstack inputs) and under the
default compiled path, greedy, on generated pictures and on two text-only prompts (the control: no deepstack input
either way). Reproduced when image answers differ between the two paths while the text-only answers do not.
Run in a vLLM venv (0.22.0 has the pre-fix code): python testbed/m18/replay3/cases/vl43602.py <out.json>
"""
import base64
import gc
import io
import json
import os
import sys

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QUESTIONS = ["What is the main colour of this picture? Answer in one word.",
             "Describe this picture in one short sentence."]
TEXTS = ["The capital of France is", "Write the first five prime numbers:"]


def picture(kind):
    from PIL import Image, ImageDraw

    if kind == "red":
        return Image.new("RGB", (448, 448), (220, 20, 20))
    if kind == "square":
        img = Image.new("RGB", (448, 448), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([112, 112, 336, 336], fill=(20, 40, 220))
        return img
    img = Image.new("RGB", (448, 448), (20, 160, 40))
    d = ImageDraw.Draw(img)
    for x in range(0, 448, 64):
        d.rectangle([x, 0, x + 31, 447], fill=(240, 220, 20))
    return img


def url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run(eager):
    import torch
    from vllm import LLM, SamplingParams

    # the same settings for both paths; the first run (0.6, 4096) left the compiled engine 0.43 GiB of KV cache
    # after its CUDA-graph reservation, short of one 4096-token sequence, and it did not start
    llm = LLM(model=MODEL, enforce_eager=eager, max_model_len=2048, gpu_memory_utilization=0.8,
              limit_mm_per_prompt={"image": 1, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)
    out = {}
    for kind in ("red", "square", "stripes"):
        for qi, q in enumerate(QUESTIONS):
            msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(picture(kind))}},
                                                 {"type": "text", "text": q}]}]
            o = llm.chat(msgs, sp)[0].outputs[0]
            out[f"{kind}/q{qi}"] = {"text": o.text, "first_top": {str(k): round(v.logprob, 4)
                                                                  for k, v in (o.logprobs[0] or {}).items()}}
    for ti, t in enumerate(TEXTS):
        o = llm.generate([t], sp)[0].outputs[0]
        out[f"text/{ti}"] = {"text": o.text}
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    import vllm

    eager = run(True)
    compiled = run(False)
    image_keys = [k for k in eager if not k.startswith("text/")]
    image_diff = [k for k in image_keys if eager[k]["text"] != compiled[k]["text"]]
    text_diff = [k for k in eager if k.startswith("text/") and eager[k]["text"] != compiled[k]["text"]]
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "eager": eager, "compiled": compiled,
           "image_answers_differ": image_diff, "text_answers_differ": text_diff,
           "reproduced": bool(image_diff) and not text_diff}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps({k: row[k] for k in ("vllm", "image_answers_differ", "text_answers_differ",
                                                     "reproduced")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
