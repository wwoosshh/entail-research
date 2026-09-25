"""M8 of entail/DESIGN.md section 7: the same checks in a second engine, diffusers 0.40 (WSL ~/venvs/gpu).

  python diffusers_m8.py gen  <tag> <checkpoint> off|on|explicit_v   three seeds, images + JSON
  python diffusers_m8.py lora <tag> <checkpoint> <lora> off|on        load a LoRA onto the pipeline, no generation
off: diffusers alone. on: entail enabled (load mode) from the checkout before diffusers is imported. explicit_v:
entail off, the scheduler switched to v_prediction by hand, as diffusers' docs tell users to do.
Writes diffusers_<tag>.json next to this file; images to ~/diffusers_m8_out/.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENTAIL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "entail")
CONFIG = os.path.expanduser("~/sdxl_local_config")
OUT = os.path.expanduser("~/diffusers_m8_out")
MODELS = "/mnt/c/Users/<user>/Desktop/ComfyUI/ComfyUI-new/models"
PROMPT = "masterpiece, best quality, 1girl, solo, smile, school uniform, cherry blossoms, outdoors, sunlight"
NEG = "lowres, bad anatomy, worst quality, low quality"
SEEDS = [11, 22, 33]


def setup(mode):
    lines = []
    if mode == "on":
        sys.path.insert(0, ENTAIL)
        import entail
        from entail.adapters import _shared

        entail.enable("load")
        lines = _shared.RESOLUTIONS
    return lines


def load(ckpt, mode):
    import torch
    from diffusers import StableDiffusionXLPipeline

    t = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(os.path.join(MODELS, "checkpoints", ckpt), config=CONFIG,
                                                      local_files_only=True, torch_dtype=torch.float16)
    if mode == "explicit_v":
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, prediction_type="v_prediction")
    return pipe, round(time.time() - t, 1)


def stats(img):
    import numpy as np

    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    s = np.asarray(img.convert("HSV"), dtype=np.float32)[..., 1]
    return {"mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1), "saturation": round(float(s.mean()), 1)}


def gen(tag, ckpt, mode):
    import torch

    notes = setup(mode)
    pipe, load_s = load(ckpt, mode)
    res = {"tag": tag, "checkpoint": ckpt, "mode": mode, "load_seconds": load_s,
           "scheduler": {"class": type(pipe.scheduler).__name__,
                         "prediction_type": pipe.scheduler.config.get("prediction_type"),
                         "rescale_betas_zero_snr": pipe.scheduler.config.get("rescale_betas_zero_snr")},
           "entail": [dict(n) for n in notes], "runs": []}
    pipe.to("cuda")
    os.makedirs(OUT, exist_ok=True)
    for s in SEEDS:
        g = torch.Generator("cuda").manual_seed(s)
        t = time.time()
        img = pipe(PROMPT, negative_prompt=NEG, num_inference_steps=25, guidance_scale=5.0, width=1024, height=1024,
                   generator=g).images[0]
        path = os.path.join(OUT, f"{tag}_seed{s}.png")
        img.save(path)
        res["runs"].append({"seed": s, "seconds": round(time.time() - t, 2), "image": path, "stats": stats(img)})
    json.dump(res, open(os.path.join(HERE, f"diffusers_{tag}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=str))
    print([r["stats"] for r in res["runs"]])


def lora(tag, ckpt, lora_file, mode):
    import logging

    notes = setup(mode)
    captured = []

    class Keep(logging.Handler):
        def emit(self, record):
            captured.append(f"{record.levelname} {record.name}: {record.getMessage()[:300]}")

    logging.getLogger("diffusers").addHandler(Keep())
    logging.getLogger("peft").addHandler(Keep())
    pipe, _ = load(ckpt, "off" if mode != "on" else "on")
    res = {"tag": tag, "checkpoint": ckpt, "lora": lora_file, "mode": mode}
    try:
        pipe.load_lora_weights(os.path.join(MODELS, "loras", lora_file), adapter_name="t")
        res["outcome"] = "loaded"
        res["adapters"] = pipe.get_list_adapters()
    except Exception as e:  # noqa: BLE001 - recording what happens is the measurement
        res["outcome"] = f"{type(e).__name__}: {str(e)[:400]}"
    res["log"] = captured[:20]
    res["entail"] = [dict(n) for n in notes]
    json.dump(res, open(os.path.join(HERE, f"diffusers_{tag}.json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:3000])


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(*sys.argv[2:5])
    else:
        lora(*sys.argv[2:6])
