"""Read-only survey of the user's ComfyUI workflows: which model, LoRAs, text encoder, VAE and sampling each one
uses, and what each LoRA file says it was trained for. No prompt text is printed."""
import glob
import json
import os
import struct
import sys

ROOT = r"C:\Users\<user>\Desktop\ComfyUI\ComfyUI-new"
M = os.path.join(ROOT, "models")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def find_model(sub, name):
    for base in (os.path.join(M, sub),):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(M, "**", os.path.basename(name)), recursive=True)
    return hits[0] if hits else None


def arch_of_model(path):
    if not path or not path.endswith(".safetensors"):
        return "gguf/other" if path else "missing"
    keys = [k for k in header(path) if k != "__metadata__"]
    if any(k.startswith("model.diffusion_model.input_blocks") for k in keys):
        return "sdxl-unet" if any("label_emb" in k for k in keys) else "sd-unet"
    if any(k.startswith(("net.blocks", "blocks.", "model.diffusion_model.blocks", "diffusion_model.blocks")) for k in keys):
        return "dit"
    return "other"


def lora_info(path):
    h = header(path)
    meta = h.get("__metadata__", {}) or {}
    keys = [k for k in h if k != "__metadata__"]
    prefix = "unknown"
    if any(k.startswith("lora_unet_input_blocks") or k.startswith("lora_unet_output_blocks") for k in keys):
        prefix = "sdxl/sd-unet (kohya)"
    elif any(k.startswith(("lora_unet_blocks", "diffusion_model.blocks", "transformer.blocks", "lora_unet_net_blocks")) for k in keys):
        prefix = "dit (blocks)"
    elif any(k.startswith("lora_unet_") for k in keys):
        prefix = "lora_unet_* (" + keys[0].split(".")[0][:40] + ")"
    elif keys:
        prefix = keys[0].split(".")[0][:48]
    return {"base_model_version": meta.get("ss_base_model_version", "-"),
            "architecture": meta.get("modelspec.architecture", "-"),
            "trained_on": os.path.basename(meta.get("ss_sd_model_name", "-") or "-")[:48],
            "keys": prefix, "n_keys": len(keys)}


def widgets(node):
    w = node.get("widgets_values")
    return w if isinstance(w, list) else (list(w.values()) if isinstance(w, dict) else [])


def survey(wf_path):
    wf = json.load(open(wf_path, encoding="utf-8"))
    nodes = wf.get("nodes", [])
    out = {"checkpoints": [], "unets": [], "clips": [], "vaes": [], "loras": [], "sampling": [], "samplers": []}
    for n in nodes:
        if n.get("mode") in (2, 4):  # muted or bypassed
            continue
        t = n.get("type", "")
        w = widgets(n)
        strs = [x for x in w if isinstance(x, str)]
        if "Checkpoint" in t and "Save" not in t:
            out["checkpoints"] += [x for x in strs if x.endswith(".safetensors")]
        elif t in ("UNETLoader", "UnetLoaderGGUF") or "Unet" in t or "UNET" in t:
            out["unets"] += [x for x in strs if x.endswith((".safetensors", ".gguf"))]
        elif "CLIPLoader" in t or "CLIP Loader" in t:
            out["clips"] += [x for x in strs if x.endswith((".safetensors", ".gguf"))]
        elif "VAELoader" in t:
            out["vaes"] += [x for x in strs if x.endswith(".safetensors")]
        elif "Lora" in t or "lora" in t:
            for x in w:
                if isinstance(x, str) and x.endswith(".safetensors"):
                    out["loras"].append(x)
                elif isinstance(x, dict) and x.get("lora") and x.get("on", True):
                    out["loras"].append(x["lora"])
        elif t.startswith("ModelSampling"):
            out["sampling"].append(f"{t}:{[x for x in w][:2]}")
        elif "KSampler" in t or t.startswith("SamplerCustom"):
            out["samplers"].append(t)
    return out


def main():
    print("== LoRA files: what each says it was trained for")
    loras = {}
    for p in sorted(glob.glob(os.path.join(M, "loras", "**", "*.safetensors"), recursive=True)):
        rel = os.path.relpath(p, os.path.join(M, "loras")).replace("\\", "/")
        loras[rel] = lora_info(p)
        i = loras[rel]
        print(f"  {rel[:44]:44} base={i['base_model_version'][:18]:18} arch={i['architecture'][:34]:34} keys={i['keys'][:28]}")
    print("== models used by workflows: architecture from tensor names")
    arch = {}
    print("== workflows (muted/bypassed nodes skipped)")
    for wf in sorted(glob.glob(os.path.join(ROOT, "user", "default", "workflows", "*.json"))):
        s = survey(wf)
        models = [("checkpoints", x) for x in s["checkpoints"]] + [("diffusion_models", x) for x in s["unets"]]
        for sub, name in models:
            if name not in arch:
                arch[name] = arch_of_model(find_model(sub, name))
        print(f"  {os.path.basename(wf)[:40]}")
        for sub, name in models:
            print(f"      model  {name[:50]:50} [{arch[name]}]")
        for x in s["loras"]:
            key = x.replace("\\", "/")
            i = loras.get(key, {})
            print(f"      lora   {key[:50]:50} [{i.get('keys', 'not found')}]")
        for x in s["clips"]:
            print(f"      clip   {x[:50]}")
        for x in s["vaes"]:
            print(f"      vae    {x[:50]}")
        for x in s["sampling"]:
            print(f"      sampling {x[:70]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
