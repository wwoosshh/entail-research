"""M16 case, vllm-project/vllm#42016 (testbed/M16_PROTOCOL.md 5): vLLM gives wrong text for GLM-OCR (the open fix
PR #42765: the Triton mrope kernel hard-codes NeoX-style rotation while the model declares GPT-J interleaved
rotation; the kernel gained an is_neox_style branch in #49906, 2026-07-28, so 0.27.0 and later carry it). The
report's evidence is vLLM's multi-image test case against the HF reference ("Matched tokens: []"). Two inputs,
each greedy through vLLM offline chat and through transformers in the same process as the reference:
(1) a synthetic image with rendered text, (2) vLLM's own two test images (stop sign, cherry blossom) with a
description prompt, the report's test case. Reproduced when vLLM's first token differs from the reference's on
either input (the test's criterion). Model zai-org/GLM-OCR (1.3B, not gated). Runs on 0.30.0 and on the 0.22.0
venv (the version before the kernel fix).
Run: python testbed/m16/cases/vl42016.py <out.json>
"""
import base64
import io
import json
import os
import sys

MODEL = "zai-org/GLM-OCR"
LINES = ["HELLO WORLD 2026", "entail case 42016"]
PROMPT_OCR = "Read all the text in this image."
PROMPT_TWO = "Describe the two images in detail."
MAX_TOKENS = 48


def make_text_image():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 240), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    d.text((30, 40), LINES[0], fill="black", font=font)
    d.text((30, 130), LINES[1], fill="black", font=font)
    return img


def test_images():
    try:
        from vllm.assets.image import ImageAsset
        return [ImageAsset("stop_sign").pil_image.convert("RGB"), ImageAsset("cherry_blossom").pil_image.convert("RGB")]
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"[:200]


def hf_reference(inputs):
    """inputs: list of (label, [images], prompt) -> {label: {"text", "token_ids"}}"""
    out = {}
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        proc = AutoProcessor.from_pretrained(MODEL)
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
        for label, images, prompt in inputs:
            content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
            enc = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                           tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_TOKENS, do_sample=False)
            n = enc["input_ids"].shape[1]
            out[label] = {"text": proc.batch_decode(gen[:, n:], skip_special_tokens=True)[0],
                          "token_ids": gen[0][n:].tolist()}
        del model
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def matched_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    import vllm
    from vllm import LLM, SamplingParams

    text_img = make_text_image()
    pair = test_images()
    inputs = [("rendered_text", [text_img], PROMPT_OCR)]
    if not isinstance(pair, str):
        inputs.append(("two_test_images", pair, PROMPT_TWO))
    ref = hf_reference(inputs)
    llm = LLM(model=MODEL, max_model_len=8192, max_num_seqs=1, gpu_memory_utilization=0.6,
              limit_mm_per_prompt={"image": 2}, enforce_eager=True)
    rows = {}
    for label, images, prompt in inputs:
        content = [{"type": "image_url", "image_url": {"url": data_url(im)}} for im in images]
        content.append({"type": "text", "text": prompt})
        o = llm.chat([{"role": "user", "content": content}], SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
                     use_tqdm=False)[0].outputs[0]
        ids = list(o.token_ids)
        row = {"vllm_text": o.text, "vllm_token_ids_head": ids[:12], "reference": ref.get(label)}
        if ref.get(label):
            row["matched_prefix_tokens"] = matched_prefix(ids, ref[label]["token_ids"])
            row["first_token_differs"] = bool(ids) and row["matched_prefix_tokens"] == 0
        rows[label] = row
    rows["rendered_text"]["rendered_text_recovered_by_vllm"] = all(
        line.split()[0] in rows["rendered_text"]["vllm_text"] for line in LINES)
    out = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "expected_lines": LINES,
           "test_images": "vllm.assets stop_sign + cherry_blossom" if not isinstance(pair, str) else pair,
           "reference_error": ref.get("error"), "cases": rows,
           "reproduced": any(r.get("first_token_differs") for r in rows.values())}
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
