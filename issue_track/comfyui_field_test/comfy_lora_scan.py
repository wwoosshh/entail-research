"""False-positive scan for the ComfyUI LoRA check, on the user's own files (entail on, dev build via PYTHONPATH).

1. Every SDXL LoRA chained with LoraLoader (model and text encoder, strength 0.1) on waiIllustriousSDXL v170.
2. The user's Anima workflow shape with its own LoRA (should pass) and with an SDXL LoRA (should stop).
Images go to PreviewImage (ComfyUI's temp folder), not to output/.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comfy_lora_case as c  # noqa: E402

SDXL_LORAS = ["AhegaoFaceRef-AndroidXL.safetensors", "Dayama-000008.safetensors",
              "E88289E6849FE794BBE9A38E20E6.uh74.safetensors", "Lativi_V2_epoch_10.safetensors",
              "NTR_MIX_4.0_LORA-000007.safetensors", "Ohogao_illustrious_v1.safetensors",
              *[f"bodydetail\\bd_v1_locon_e{i}.safetensors" for i in range(1, 9)],
              "mesugaki_Illust_v1.safetensors", "mikzn-illustrious-ty_lee.safetensors",
              "nagito\\nagito_illustrious_v3.safetensors", "nagito\\nagito_wai_v2.safetensors",
              "perfectpantyhose-a.safetensors", "sagging breasts-IL2.0.safetensors", "tarasu_mc-000018.safetensors"]


def sdxl_chain():
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "waiIllustriousSDXL_v170.safetensors"}}}
    model, clip = ["1", 0], ["1", 1]
    for i, name in enumerate(SDXL_LORAS):
        nid = str(100 + i)
        wf[nid] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": name,
                                                           "strength_model": 0.1, "strength_clip": 0.1}}
        model, clip = [nid, 0], [nid, 1]
    wf.update({
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, solo", "clip": clip}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres", "clip": clip}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": model, "positive": ["2", 0], "negative": ["3", 0],
                                                   "latent_image": ["4", 0], "seed": 1, "steps": 1, "cfg": 5.0,
                                                   "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "PreviewImage", "inputs": {"images": ["6", 0]}},
    })
    return wf


def anima(lora):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors",
                                                     "type": "stable_diffusion", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 0.8}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, solo, smile", "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres", "clip": ["2", 0]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0],
                                                   "latent_image": ["7", 0], "seed": 1, "steps": 4, "cfg": 4.0,
                                                   "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
    }


def submit(wf):
    import time
    import urllib.error
    import urllib.request

    body = json.dumps({"prompt": wf, "client_id": "entail-scan"}).encode()
    req = urllib.request.Request(c.URL + "/prompt", data=body, headers={"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"status": "rejected", "error": e.read().decode("utf-8", "replace")[:400]}
    while True:
        h = json.load(urllib.request.urlopen(f"{c.URL}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            break
        time.sleep(0.5)
    st = h[pid]["status"]
    msgs = {m[0]: m[1] for m in st["messages"]}
    out = {"status": st["status_str"]}
    if "execution_error" in msgs:
        e = msgs["execution_error"]
        out["error"] = f"{e.get('node_type')}: {str(e.get('exception_message'))[:300]}"
    return out


def main():
    import subprocess

    p, log = c.start("scan", True)
    res = {}
    try:
        res["sdxl_chain_21_loras"] = submit(sdxl_chain())
        res["anima_own_lora"] = submit(anima("nagito\\nagito_anima_e10.safetensors"))
        res["anima_with_sdxl_lora"] = submit(anima("nagito\\nagito_illustrious_v3.safetensors"))
    finally:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        p.wait(timeout=60)
        log.close()
    lines = open(os.path.join(c.HERE, "lora_scan.log"), encoding="utf-8", errors="replace").read().splitlines()
    res["entail_lines"] = [ln[:260] for ln in lines if ln.startswith("[entail] LoRA") or "RoleError: LoRA" in ln
                           or "left out" in ln or "could not check" in ln]
    res["lora_key_not_loaded"] = sum("lora key not loaded" in ln for ln in lines)
    json.dump(res, open(os.path.join(c.HERE, "lora_scan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
