"""Build a local diffusers config folder for SDXL single-file loading, from files already on this machine (no download).

  text_encoder, tokenizer      ComfyUI's bundled CLIP-L config and tokenizer (comfy/sd1_clip_config.json, sd1_tokenizer)
  text_encoder_2, tokenizer_2  ComfyUI's bundled CLIP-bigG config, made a CLIPTextModelWithProjection (projection 1280);
                               the same tokenizer with "!" as pad token, as SDXL's second tokenizer has
  unet                         the SDXL base unet/config.json already in the Hugging Face cache
  scheduler, vae, model_index  written here with the values of SDXL base 1.0 (EulerDiscreteScheduler, prediction_type
                               epsilon; AutoencoderKL, scaling factor 0.13025)
This is not the Hub's config set; the difference that matters (the scheduler's prediction_type) is the Hub's value.
Run in WSL: python diffusers_local_config.py <out_dir>
"""
import glob
import json
import os
import shutil
import sys

COMFY = "/mnt/c/Users/<user>/Desktop/ComfyUI/ComfyUI-new/comfy"
UNET_CFG = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/*/unet/config.json"))[0]


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)


def main(out):
    write(f"{out}/model_index.json", {
        "_class_name": "StableDiffusionXLPipeline", "_diffusers_version": "0.40.0", "force_zeros_for_empty_prompt": True,
        "add_watermarker": None, "feature_extractor": [None, None], "image_encoder": [None, None],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"], "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"], "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"], "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"]})
    write(f"{out}/scheduler/scheduler_config.json", {
        "_class_name": "EulerDiscreteScheduler", "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085,
        "clip_sample": False, "interpolation_type": "linear", "num_train_timesteps": 1000, "prediction_type": "epsilon",
        "sample_max_value": 1.0, "set_alpha_to_one": False, "skip_prk_steps": True, "steps_offset": 1,
        "timestep_spacing": "leading", "trained_betas": None, "use_karras_sigmas": False})
    write(f"{out}/vae/config.json", {
        "_class_name": "AutoencoderKL", "act_fn": "silu", "block_out_channels": [128, 256, 512, 512],
        "down_block_types": ["DownEncoderBlock2D"] * 4, "force_upcast": True, "in_channels": 3, "latent_channels": 4,
        "layers_per_block": 2, "norm_num_groups": 32, "out_channels": 3, "sample_size": 1024, "scaling_factor": 0.13025,
        "up_block_types": ["UpDecoderBlock2D"] * 4})
    os.makedirs(f"{out}/unet", exist_ok=True)
    shutil.copy(UNET_CFG, f"{out}/unet/config.json")
    te = json.load(open(f"{COMFY}/sd1_clip_config.json"))
    write(f"{out}/text_encoder/config.json", te)
    te2 = json.load(open(f"{COMFY}/clip_config_bigg.json"))
    te2.update({"architectures": ["CLIPTextModelWithProjection"], "projection_dim": 1280})
    write(f"{out}/text_encoder_2/config.json", te2)
    for name, pad in (("tokenizer", None), ("tokenizer_2", "!")):
        shutil.copytree(f"{COMFY}/sd1_tokenizer", f"{out}/{name}", dirs_exist_ok=True)
        if pad:
            for f in ("tokenizer_config.json", "special_tokens_map.json"):
                cfg = json.load(open(f"{out}/{name}/{f}"))
                cfg["pad_token"] = pad
                write(f"{out}/{name}/{f}", cfg)
    print("written", out, sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1])
