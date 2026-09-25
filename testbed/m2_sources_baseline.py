"""M2.4: what the researcher's own model files declare, read with entail's readers (S1, the source side).

For every model folder and model file on this machine, and for every fact that matters for its kind: is it declared
by the artifact, reported as a problem (stated but not representable in vocabulary v1), or not stated at all?
Only headers and config files are read; no tensor is loaded. Run in WSL:
    source ~/venvs/gpu/bin/activate && python testbed/m2_sources_baseline.py
Writes testbed/results/m2_sources_baseline.json and .md.
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
from entail import gguf, sources  # noqa: E402
from entail.readers import safetensors_header  # noqa: E402

LLM_DIR = os.path.expanduser("~/models")
COMFY = "/mnt/c/Users/<user>/Desktop/ComfyUI/ComfyUI-new/models"
RELEVANT = {
    "llm_folder": ("ModelProps", "Rotary", "Template"),
    "llm_folder_quantized": ("ModelProps", "Rotary", "Template", "Layout"),
    "image_checkpoint": ("Prediction", "LatentScale"),
    "diffusion_model": ("Prediction",),
    "gguf": ("ModelProps", "Rotary", "Template"),
    "lora": ("Prediction",),        # kohya writes the prediction type a LoRA was trained for
    "3d_model": (),                 # Hunyuan3D: a 3D generator, no v1 fact is named for it yet
    "other": (),
}


# which problems belong to which fact (a problem names the key or the value it could not represent)
KEYWORDS = {"ModelProps": ("modelprops", "softcap", "sliding_window", "tie_word"), "Rotary": ("rope", "rotary"),
            "Template": ("template",), "Layout": ("quantization", "fp8", "layout", "bits"),
            "Prediction": ("prediction",), "LatentScale": ("latentscale", "scaling_factor", "shift_factor")}


def kind_of(path):
    if os.path.isdir(path):
        cfg = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
        return "llm_folder_quantized" if isinstance(cfg.get("quantization_config"), dict) else "llm_folder"
    if path.endswith(".gguf"):
        arch = gguf.read_metadata(path).get("general.architecture", "")
        image = arch in ("qwen_image", "flux", "sd3", "wan") or "/unet/" in path or "/diffusion_models/" in path
        return "diffusion_model" if image else "gguf"
    try:
        keys, meta = safetensors_header(path)
    except Exception:  # noqa: BLE001
        return "other"
    if any(".lora_" in k or "lora_down" in k or "lora_A" in k for k in keys):
        return "lora"
    if any(k.startswith("model.diffusion_model.") for k in keys):
        return "image_checkpoint"
    if {"conditioner", "model", "vae"} <= {k.split(".")[0] for k in keys}:
        return "3d_model"
    if any(k.startswith("net.") for k in keys):   # Anima (a Cosmos-style DiT) keeps its weights under net.
        return "diffusion_model"
    if any(k.startswith(("input_blocks.", "double_blocks.", "transformer_blocks.", "joint_blocks.", "blocks."))
           for k in keys) and "/text_encoders/" not in path and "/vae/" not in path:
        return "diffusion_model"
    return "other"


def outside_v1(path, kind):
    """Declarations the file makes that vocabulary v1 has no name for (counted, not used)."""
    if kind != "lora" and kind != "image_checkpoint":
        return []
    try:
        _, meta = safetensors_header(path)
    except Exception:  # noqa: BLE001
        return []
    return sorted(k for k in ("modelspec.architecture", "ss_base_model_version") if meta.get(k))


def main():
    targets = sorted(d for d in glob.glob(os.path.join(LLM_DIR, "*")) if os.path.isfile(os.path.join(d, "config.json")))
    for sub in ("checkpoints", "loras", "diffusion_models", "unet"):
        for ext in ("safetensors", "gguf"):
            targets += sorted(glob.glob(os.path.join(COMFY, sub, "**", f"*.{ext}"), recursive=True))
    rows = []
    for path in targets:
        t0 = time.perf_counter()
        kind = kind_of(path)
        r = sources.read_all(path)
        chosen, conflicts = sources.merge(r.facts)
        status = {}
        for name in RELEVANT[kind]:
            related = [p for p in r.problems if any(w in p.lower() for w in KEYWORDS[name])]
            if name in chosen:
                status[name] = "declared_partly" if related else "declared"
            elif related:
                status[name] = "stated_not_representable"
            else:
                status[name] = "not_stated"
        rows.append({"path": path, "kind": kind, "status": status,
                     "facts": [{"name": f.name, "value": str(f.value), "where": f.source.where} for f in r.facts],
                     "conflicts": [{"name": c.name, "values": [str(f.value) for f in c.facts]} for c in conflicts],
                     "problems": r.problems, "outside_v1": outside_v1(path, kind),
                     "seconds": round(time.perf_counter() - t0, 3)})
    summary = {}
    for row in rows:
        s = summary.setdefault(row["kind"], {"files": 0, "facts": {}, "conflicting_files": 0, "declares_outside_v1": 0})
        s["files"] += 1
        s["conflicting_files"] += bool(row["conflicts"])
        s["declares_outside_v1"] += bool(row["outside_v1"])
        for name, st in row["status"].items():
            s["facts"].setdefault(name, {"declared": 0, "declared_partly": 0, "stated_not_representable": 0,
                                         "not_stated": 0})[st] += 1
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "roots": [LLM_DIR, COMFY], "summary": summary, "rows": rows}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "m2_sources_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    lines = ["| kind | files | fact | declared | declared, some keys not in v1 | stated, not representable in v1 | "
             "not stated |", "|---|---|---|---|---|---|---|"]
    for kind, s in summary.items():
        if not s["facts"]:
            lines.append(f"| {kind} | {s['files']} | (no v1 fact applies) | | | | |")
        for name, c in s["facts"].items():
            lines.append(f"| {kind} | {s['files']} | {name} | {c['declared']} | {c['declared_partly']} | "
                         f"{c['stated_not_representable']} | {c['not_stated']} |")
    extra = [f"- {k}: {s['conflicting_files']} file(s) contradict themselves; {s['declares_outside_v1']} declare a base "
             f"family (outside v1)" for k, s in summary.items()]
    text = "\n".join(lines + [""] + extra) + "\n"
    with open(os.path.join(HERE, "results", "m2_sources_baseline.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
