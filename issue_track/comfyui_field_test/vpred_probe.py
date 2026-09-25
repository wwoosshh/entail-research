"""M2 of VPRED_PROTOCOL.md: can the prediction type be read off the model's behaviour?

For every SDXL-family checkpoint in the user's models/checkpoints: load it with ComfyUI's own loader, record the
model_type ComfyUI decided (from the v_pred key), then pass unit-variance noise through the diffusion model at t=999
and t=500 with zero conditioning, and take cos(output, input). Theory: eps models ~0.99 at t=999, v models ~0.07 or
below. Run from ComfyUI-new with its own interpreter. Writes vpred_probe.json next to this file.
"""
import gc
import json
import os
import struct
import sys
import time

ROOT = r"C:\Users\<user>\Desktop\ComfyUI\ComfyUI-new"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0]]  # comfy.cli_args reads sys.argv when imported

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402


def is_sdxl(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        keys = json.loads(f.read(n)).keys()
    return (any(k.startswith("model.diffusion_model.input_blocks") for k in keys)
            and any("label_emb" in k for k in keys), "v_pred" in keys, "ztsnr" in keys)


def probe(model_patcher, seeds=(0, 1, 2), steps=(999, 500)):
    m = model_patcher.model
    mm.load_models_gpu([model_patcher])
    dev = mm.get_torch_device()
    dtype = m.manual_cast_dtype or m.get_dtype()
    out = {}
    t0 = time.perf_counter()
    for t in steps:
        vals = []
        for s in seeds:
            x = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(s)).to(dev, dtype)
            with torch.no_grad():
                y = m.diffusion_model(x, torch.tensor([float(t)], device=dev),
                                      context=torch.zeros(1, 77, 2048, device=dev, dtype=dtype),
                                      y=torch.zeros(1, 2816, device=dev, dtype=dtype))
            vals.append(round(F.cosine_similarity(y.float().flatten(), x.float().flatten(), dim=0).item(), 4))
        out[f"t{t}"] = vals
    torch.cuda.synchronize()
    out["seconds_all_passes"] = round(time.perf_counter() - t0, 3)
    return out


def main():
    rows = []
    ck = os.path.join(ROOT, "models", "checkpoints")
    for name in sorted(os.listdir(ck)):
        path = os.path.join(ck, name)
        if not name.endswith(".safetensors") or not os.path.isfile(path):
            continue
        sdxl, v_key, z_key = is_sdxl(path)
        if not sdxl:
            continue
        mp = comfy.sd.load_checkpoint_guess_config(path, output_vae=False, output_clip=False)[0]
        row = {"checkpoint": name, "v_pred_key": v_key, "ztsnr_key": z_key,
               "comfy_model_type": mp.model.model_type.name, "probe": probe(mp)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        mm.unload_all_models()
        del mp
        gc.collect()
        torch.cuda.empty_cache()
    with open(os.path.join(HERE, "vpred_probe.json"), "w", encoding="utf-8") as f:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
