"""M4 for the user's main model: the '나기토화풍-Anima' workflow (Anima DiT, a flow model, with the nagito Anima LoRA
at 0.8), entail off and on. The images have to be the same pixel for pixel.

  python anima_same.py off|on        writes anima_<off|on><VPRED_TAG>.json next to this file
  python anima_same.py compare TAG   compares anima_off.json with anima_on<TAG>.json
"""
import json
import os
import subprocess
import sys

import vpred_harm as h

WF = os.path.join(h.ROOT, "user", "default", "workflows", "나기토화풍-Anima.json")


def prompts():
    nodes = {n["id"]: n for n in json.load(open(WF, encoding="utf-8"))["nodes"]}
    return nodes[5]["widgets_values"][0], nodes[6]["widgets_values"][0]


def workflow(prefix, seed):
    pos, neg = prompts()
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion",
                                                     "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "nagito\\nagito_anima_e10.safetensors",
                                                              "strength_model": 0.8}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0],
                                                   "latent_image": ["8", 0], "seed": seed, "steps": 30, "cfg": 4.0,
                                                   "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def main():
    if sys.argv[1] == "compare":
        off = json.load(open(os.path.join(h.HERE, "anima_off.json"), encoding="utf-8"))
        on = json.load(open(os.path.join(h.HERE, f"anima_on{sys.argv[2] if len(sys.argv) > 2 else ''}.json"), encoding="utf-8"))
        print(json.dumps([h.diff(a["image"], b["image"]) for a, b in zip(off["runs"], on["runs"])]))
        print("seconds off", [r.get("seconds") for r in off["runs"]], "on", [r.get("seconds") for r in on["runs"]])
        return
    entail = sys.argv[1] == "on"
    tag = "anima_" + sys.argv[1] + (os.environ.get("VPRED_TAG", "") if entail else "")
    p, log = h.start(tag, entail)
    try:
        runs = [h.run(workflow(f"entail_test/{tag}_seed{s}", s)) for s in h.SEEDS]
        for r in runs:
            if "image" in r:
                r["stats"] = h.stats(r["image"])
    finally:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        p.wait(timeout=60)
        log.close()
    lines = open(os.path.join(h.HERE, f"vpred_{tag}.log"), encoding="utf-8", errors="replace").read().splitlines()
    out = {"runs": runs, "entail": [ln[:300] for ln in lines if "[entail]" in ln or "RoleError" in ln][:20]}
    json.dump(out, open(os.path.join(h.HERE, f"{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps([(r["status"], r.get("seconds"), r.get("stats")) for r in runs], ensure_ascii=False))


if __name__ == "__main__":
    main()
