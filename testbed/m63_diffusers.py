"""M6.3 on diffusers 0.40.0 (WSL ~/venvs/gpu, torch 2.14, RTX 4070 Ti): the M6 test problems as diffusers meets them,
with the v2 adapter installed the way a user installs it - ENTAIL=load and the start-up hook from the checkout.

Measurement definitions, written before the runs:
  gen <tag> <checkpoint> <condition>
    Three seeds (11, 22, 33), 1024x1024, 25 steps, guidance 5.0, fp16, EulerDiscreteScheduler, the prompt of the
    earlier diffusers run (issue_track/comfyui_field_test/diffusers_m8.py). The pipeline is built with
    StableDiffusionXLPipeline.from_single_file(checkpoint, config=~/sdxl_local_config), the local SDXL config set of
    that run (no download).
      off        diffusers alone: entail is not on the path
      reference  entail not on the path; the scheduler set by hand to what the file declares, as the model cards tell
                 diffusers users to do (REFERENCE below)
      on         ENTAIL=load: the default policy
      observe    ENTAIL=load, ENTAIL_POLICY=refuse: nothing is repaired, what differs is reported
    Recorded: the scheduler's prediction_type and rescale_betas_zero_snr as the pipeline samples, the load time, the
    decisions (ENTAIL_RECORD), and per seed the image's statistics; the images go to ~/m63_out. The pixel differences
    to the reference are computed by m63_summarize.py.
    Cases: i04 NoobAI-XL-Vpred-v1.0 (marker keys v_pred and ztsnr; market I04: a tool that does not read them samples
    eps); m7 astolfocarmixVpredxl_acEvo25EP (metadata modelspec.prediction_type = v, no marker key; fd-m7); s3
    waiIllustriousSDXL_v160 (declares nothing): off and on must give the same pixels (S3, non-interference).
  lora <tag> <checkpoint> <lora> <condition>
    The pipeline as above; load_lora_weights(lora, adapter_name="t"), then the three images unless loading raised.
    Recorded: what diffusers logged, the adapters it lists, how many UNet modules hold the adapter, the decisions; the
    summary compares the images with the same model's images without a LoRA (s3_off). Conditions off, on, strict
    (ENTAIL_ON_BROKEN=stop). Cases: right (nagito_illustrious_v3, an SDXL LoRA, on the SDXL model), other
    (nagito_anima_e10, made for Anima; fd-lora), i01 (nagito_illustrious_v3's UNet part under PEFT-wrapped names,
    testbed/m63_i01_lora.py; market I01). peft 0.21.0 installed for these (researcher's permission, 2026-09-24).
  vae <tag> <checkpoint> <condition>
    fd-vae. The pipeline as above; a VAE file read on its own - AutoencoderKL.from_single_file(sdxl_vae_fp16_fix) with
    no config, which diffusers takes for Stable Diffusion 1.5's (SD1.5's vae/config.json downloaded with permission) -
    put into it (pipe.vae = vae); the three images. The model's latent scale is declared by a pinned manifest
    (testbed/results/m63/manifests, ENTAIL_MANIFESTS): SDXL's 0.13025, the value of the pipeline's own VAE config.
      reference  entail not on the path; the VAE's scaling_factor set by hand to 0.13025 before it goes in
      off / on / observe as above
    Recorded: the loaded VAE's scaling_factor, the one the pipeline samples with, the decisions, image statistics.
Writes testbed/results/m63/diffusers_<tag>.json. Run through m63_diffusers.sh.
"""
import json
import logging
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
OUT = os.path.join(RESULTS, "m63")
# a re-run under TESTBED_RESULTS keeps its images apart from the milestone's (M9.1)
IMAGES = os.path.expanduser("~/m63_out" + ("_rerun" if os.environ.get("TESTBED_RESULTS") else ""))
CONFIG = os.path.expanduser("~/sdxl_local_config")
MODELS = os.environ.get("COMFY_MODELS", "/mnt/c/Users/<user>/Desktop/ComfyUI/ComfyUI-new/models")   # moved to E: (P6)
PROMPT = "masterpiece, best quality, 1girl, solo, smile, school uniform, cherry blossoms, outdoors, sunlight"
NEG = "lowres, bad anatomy, worst quality, low quality"
SEEDS = [11, 22, 33]
REFERENCE = {   # what each file declares, set by hand: the files' own statements (and the scheduler's zsnr where the
    # file says nothing about it, as the author's yaml for AstolfoCarmix does: v, no zero terminal SNR)
    "NoobAI-XL-Vpred-v1.0.safetensors": {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True},
    "entail_test/astolfocarmixVpredxl_acEvo25EP.safetensors": {"prediction_type": "v_prediction"},
}


def records():
    path = os.environ.get("ENTAIL_RECORD")
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if "verdict" in d:
            out.append({k: d.get(k) for k in ("boundary", "name", "verdict", "rule", "resolution", "target", "note")}
                       | {"declared": (d.get("declared") or {}).get("value"),
                          "declared_from": ((d.get("declared") or {}).get("source") or {}).get("where"),
                          "chosen": (d.get("chosen") or {}).get("value")})
        elif "timing" in d:
            out.append({"timing": d["timing"], "ms": d["ms"]})
    return out


def pipeline(ckpt):
    import torch
    from diffusers import StableDiffusionXLPipeline

    t = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(os.path.join(MODELS, "checkpoints", ckpt), config=CONFIG,
                                                      local_files_only=True, torch_dtype=torch.float16)
    return pipe, round(time.time() - t, 2)


def stats(img):
    import numpy as np

    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    s = np.asarray(img.convert("HSV"), dtype=np.float32)[..., 1]
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2), "saturation": round(float(s.mean()), 2)}


def gen(tag, ckpt, condition):
    import torch

    pipe, load_s = pipeline(ckpt)
    if condition == "reference":
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, **REFERENCE[ckpt])
    res = {"tag": tag, "checkpoint": ckpt, "condition": condition, "entail": os.environ.get("ENTAIL", "not installed"),
           "policy": os.environ.get("ENTAIL_POLICY", "resolve"), "load_seconds": load_s,
           "scheduler": {"class": type(pipe.scheduler).__name__,
                         "prediction_type": pipe.scheduler.config.get("prediction_type"),
                         "rescale_betas_zero_snr": pipe.scheduler.config.get("rescale_betas_zero_snr")},
           "vae_scaling_factor": pipe.vae.config.get("scaling_factor"), "runs": []}
    pipe.to("cuda")
    os.makedirs(IMAGES, exist_ok=True)
    for s in SEEDS:
        g = torch.Generator("cuda").manual_seed(s)
        t = time.time()
        img = pipe(PROMPT, negative_prompt=NEG, num_inference_steps=25, guidance_scale=5.0, width=1024, height=1024,
                   generator=g).images[0]
        path = os.path.join(IMAGES, f"{tag}_seed{s}.png")
        img.save(path)
        res["runs"].append({"seed": s, "seconds": round(time.time() - t, 2), "image": path, "stats": stats(img)})
    res["decisions"] = records()
    return res


def generate(pipe, tag):
    import torch

    pipe.to("cuda")
    os.makedirs(IMAGES, exist_ok=True)
    runs = []
    for s in SEEDS:
        g = torch.Generator("cuda").manual_seed(s)
        t = time.time()
        img = pipe(PROMPT, negative_prompt=NEG, num_inference_steps=25, guidance_scale=5.0, width=1024, height=1024,
                   generator=g).images[0]
        path = os.path.join(IMAGES, f"{tag}_seed{s}.png")
        img.save(path)
        runs.append({"seed": s, "seconds": round(time.time() - t, 2), "image": path, "stats": stats(img)})
    return runs


def vae(tag, ckpt, condition):
    import torch
    from diffusers import AutoencoderKL

    pipe, load_s = pipeline(ckpt)
    single = AutoencoderKL.from_single_file(os.path.join(MODELS, "vae", "sdxl_vae_fp16_fix.safetensors"),
                                            torch_dtype=torch.float16)
    loaded_scale = single.config.get("scaling_factor")
    if condition == "reference":
        single.register_to_config(scaling_factor=0.13025)
    pipe.vae = single
    res = {"tag": tag, "checkpoint": ckpt, "condition": condition, "entail": os.environ.get("ENTAIL", "not installed"),
           "policy": os.environ.get("ENTAIL_POLICY", "resolve"), "load_seconds": load_s,
           "vae_scaling_factor_as_loaded": loaded_scale, "vae_scaling_factor": pipe.vae.config.get("scaling_factor"),
           "scheduler": {"class": type(pipe.scheduler).__name__,
                         "prediction_type": pipe.scheduler.config.get("prediction_type"),
                         "rescale_betas_zero_snr": pipe.scheduler.config.get("rescale_betas_zero_snr")}}
    res["runs"] = generate(pipe, tag)
    res["decisions"] = records()
    return res


def lora(tag, ckpt, lora_file, condition):
    captured = []

    class Keep(logging.Handler):
        def emit(self, record):
            captured.append(f"{record.levelname} {record.name}: {record.getMessage()[:300]}")

    for name in ("diffusers", "peft"):
        logging.getLogger(name).addHandler(Keep())
    pipe, load_s = pipeline(ckpt)
    res = {"tag": tag, "checkpoint": ckpt, "lora": lora_file, "condition": condition,
           "entail": os.environ.get("ENTAIL", "not installed"), "on_broken": os.environ.get("ENTAIL_ON_BROKEN", "report")}
    try:
        pipe.load_lora_weights(os.path.join(MODELS, "loras", lora_file), adapter_name="t")
        res["outcome"] = "loaded"
    except Exception as e:  # noqa: BLE001 - what happens is the measurement
        res["outcome"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
    try:
        res["adapters"] = {k: list(v) for k, v in pipe.get_list_adapters().items()}
    except Exception as e:  # noqa: BLE001 - without the PEFT backend diffusers lists nothing
        res["adapters"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    res["modules_with_the_adapter"] = sum(1 for _, m in pipe.unet.named_modules()
                                          if hasattr(getattr(m, "lora_A", None), "keys") and "t" in m.lora_A.keys())
    res["log"] = captured[:12]
    if res["outcome"] == "loaded":
        res["runs"] = generate(pipe, tag)
    res["decisions"] = records()
    return res


def main():
    kind, tag = sys.argv[1], sys.argv[2]
    if os.environ.get("ENTAIL_RECORD") and os.path.exists(os.environ["ENTAIL_RECORD"]):
        os.remove(os.environ["ENTAIL_RECORD"])
    res = {"gen": lambda: gen(tag, *sys.argv[3:5]), "vae": lambda: vae(tag, *sys.argv[3:5]),
           "lora": lambda: lora(tag, *sys.argv[3:6])}[kind]()
    res["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"diffusers_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    brief = {k: v for k, v in res.items() if k not in ("runs", "decisions", "log")}
    print(json.dumps(brief, ensure_ascii=False, default=str))
    for r in res.get("runs", []):
        print("  seed", r["seed"], r["stats"], r["seconds"], "s")
    for d in res.get("decisions", []):
        if "verdict" in d:
            print("  decision:", d["boundary"], d["verdict"], (d["resolution"] or d["rule"])[:90])
    for line in res.get("log", [])[:4]:
        print("  log:", line[:160])


if __name__ == "__main__":
    main()
