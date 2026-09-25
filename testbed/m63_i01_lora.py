"""M6.3, market case I01 simulated: the same LoRA weights in a key format the loader does not know.

I01 (reinvestigation/market_incidents.md): LoRAs from different training tools use different key names, and a loader
that does not know a format skips every key ('lora key not loaded'); the image comes out and the LoRA does nothing.
The simulation takes one SDXL LoRA of the researcher's (kohya format, nagito_illustrious_v3), converts it with
diffusers' own converter to diffusers names, and writes the UNet part under the names a PEFT-wrapped model saves
(base_model.model.<module>.lora_A/lora_B.weight). ComfyUI 0.34.1 maps that prefix only for SD3 and PixArt
(comfy/lora.py model_lora_keys_unet), and diffusers' SDXL loader takes UNet keys under "unet." only. The original file
is only read; the converted one goes to the output folder given.
Run in WSL ~/venvs/gpu: python testbed/m63_i01_lora.py <out_dir>
"""
import os
import sys

MODELS = "/mnt/c/Users/<user>/Desktop/ComfyUI/ComfyUI-new/models"
SOURCE = os.path.join(MODELS, "loras", "nagito", "nagito_illustrious_v3.safetensors")


def main(out_dir):
    from diffusers import StableDiffusionXLPipeline
    from safetensors.torch import save_file

    sd, alphas = StableDiffusionXLPipeline.lora_state_dict(SOURCE)[:2]
    kept, dropped = {}, 0
    for k, v in sd.items():
        if not k.startswith("unet."):
            dropped += 1
            continue
        name = k[len("unet."):].replace(".lora.down.weight", ".lora_A.weight").replace(".lora.up.weight",
                                                                                     ".lora_B.weight")
        kept["base_model.model." + name] = v.contiguous()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "nagito_illustrious_v3_peft_names.safetensors")
    save_file(kept, out, metadata={"entail_m63": "I01 simulation: nagito_illustrious_v3's UNet part under "
                                                 "PEFT-wrapped names (base_model.model.*)"})
    print("source keys", len(sd), "unet keys kept", len(kept), "others dropped", dropped)
    print("first keys before:", sorted(sd)[:2])
    print("first keys after:", sorted(kept)[:2])
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
