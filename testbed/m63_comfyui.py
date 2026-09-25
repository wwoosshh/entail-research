"""M6.3 on the researcher's ComfyUI 0.34.1 (Windows .venv), with the v2 adapters from the checkout. Run with the
researcher's permission (2026-09-24, "세가지 허락할테니 진행해").

What it touches: it starts ComfyUI-new's own python on main.py with a port of its own (8189), the checkout first on
PYTHONPATH (ENTAIL=load; the installed entail 0.3.0 is shadowed, not changed). Output, temp and user directories are
this repository's (testbed/results/m63/comfy_images, comfy_temp, comfy_user) and the database is in memory, so their
output, temp and user folders and comfyui.db are not written; the I01 LoRA is read from the scratch folder through an
extra model paths file. Models are only read. Nothing is installed.

Measurement definitions, written before the runs (the workflow of issue_track/comfyui_field_test/vpred_harm.py):
CheckpointLoaderSimple, CLIPTextEncode (the M8 prompt), EmptyLatentImage 1024x1024, KSampler euler/normal 25 steps
cfg 5.0, VAEDecode, SaveImage; seeds 11, 22, 33; every condition in a fresh server (Comfy-Org/ComfyUI#16490).
  m7_off / m7_ref / m7_on   AstolfoCarmix (metadata v, no marker key): ComfyUI as it detects it / with
                            ModelSamplingDiscrete(v_prediction, zsnr=false), the author's yaml / entail on. On must be
                            identical to ref (the switch is the same object patch) and different from off.
  node_off / node_on        NoobAI-XL-Vpred with ModelSamplingDiscrete(eps): the user's choice against the file's markers.
                            On: reported (broken), not overridden: identical to off.
  s3_*_off / s3_*_on        waiIllustriousSDXL v160 (declares nothing) and NoobAI-XL-Vpred (markers ComfyUI reads):
                            no broken, refused or resolved; identical images off and on (S3, non-interference).
  lora_right_* / lora_other_* / i01_*   waiIllustriousSDXL v170 + LoraLoaderModelOnly 0.9 with nagito_illustrious_v3 /
                            nagito_anima_e10 (fd-lora) / the PEFT-named copy (I01). On: pass / broken / broken, images
                            identical to off; strict (ENTAIL_ON_BROKEN=stop): the prompt fails before sampling.
  lora_none_off             waiIllustriousSDXL v170 without a LoRA: how much each LoRA changed the image.
Writes testbed/results/m63/comfy_<tag>.json and .jsonl. Run with the Windows python: py -3.12 testbed/m63_comfyui.py <tag>...
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = r"C:\Users\<user>\Desktop\ComfyUI\ComfyUI-new"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
URL = "http://127.0.0.1:8189"
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.path.join(os.path.dirname(HERE), "entail")
OUT = os.path.join(HERE, "results", "m63")
IMAGES = os.path.join(OUT, "comfy_images")
SCRATCH_LORAS = os.environ.get("M63_LORAS", "")   # the folder testbed/m63_i01_lora.py wrote the I01 copy to
SEEDS = [11, 22, 33]
PROMPT = "masterpiece, best quality, 1girl, solo, smile, school uniform, cherry blossoms, outdoors, sunlight"
NEG = "lowres, bad anatomy, worst quality, low quality"
ASTOLFO = "entail_test\\astolfocarmixVpredxl_acEvo25EP.safetensors"
NOOB = "NoobAI-XL-Vpred-v1.0.safetensors"
WAI160, WAI170 = "waiIllustriousSDXL_v160.safetensors", "waiIllustriousSDXL_v170.safetensors"
RUNS = {   # tag: (checkpoint, sampling node or None, LoRA or None, entail on, extra environment)
    "m7_off": (ASTOLFO, None, None, False, {}), "m7_ref": (ASTOLFO, "v_prediction", None, False, {}),
    "m7_on": (ASTOLFO, None, None, True, {}),
    "node_off": (NOOB, "eps", None, False, {}), "node_on": (NOOB, "eps", None, True, {}),
    "s3_wai_off": (WAI160, None, None, False, {}), "s3_wai_on": (WAI160, None, None, True, {}),
    "s3_noob_off": (NOOB, None, None, False, {}), "s3_noob_on": (NOOB, None, None, True, {}),
    "lora_none_off": (WAI170, None, None, False, {}),
    "lora_right_off": (WAI170, None, "nagito\\nagito_illustrious_v3.safetensors", False, {}),
    "lora_right_on": (WAI170, None, "nagito\\nagito_illustrious_v3.safetensors", True, {}),
    "lora_other_off": (WAI170, None, "nagito\\nagito_anima_e10.safetensors", False, {}),
    "lora_other_on": (WAI170, None, "nagito\\nagito_anima_e10.safetensors", True, {}),
    "lora_other_strict": (WAI170, None, "nagito\\nagito_anima_e10.safetensors", True, {"ENTAIL_ON_BROKEN": "stop"}),
    "i01_off": (WAI170, None, "nagito_illustrious_v3_peft_names.safetensors", False, {}),
    "i01_on": (WAI170, None, "nagito_illustrious_v3_peft_names.safetensors", True, {}),
    "i01_strict": (WAI170, None, "nagito_illustrious_v3_peft_names.safetensors", True, {"ENTAIL_ON_BROKEN": "stop"}),
}


def workflow(ckpt, node, lora, seed, prefix):
    model = ["1", 0]
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}}
    if lora:
        wf["8"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": lora,
                                                                  "strength_model": 0.9}}
        model = ["8", 0]
    if node:
        wf["9"] = {"class_type": "ModelSamplingDiscrete", "inputs": {"model": model, "sampling": node, "zsnr": False}}
        model = ["9", 0]
    wf.update({
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": model, "positive": ["2", 0], "negative": ["3", 0],
                                                   "latent_image": ["4", 0], "seed": seed, "steps": 25, "cfg": 5.0,
                                                   "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}}})
    return wf


def start(tag, entail, extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ENTAIL")}
    env["PYTHONIOENCODING"] = "utf-8"
    if entail:
        env.update(ENTAIL="load", ENTAIL_RECORD=os.path.join(OUT, f"comfy_{tag}.jsonl"),
                   PYTHONPATH=os.pathsep.join([CHECKOUT, os.path.join(CHECKOUT, "entail", "adapters", "autoinstall")]),
                   **extra)
    for d in ("comfy_temp", "comfy_user"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    args = [PY, "main.py", "--fp16-vae", "--port", "8189", "--output-directory", IMAGES, "--disable-auto-launch",
            "--temp-directory", os.path.join(OUT, "comfy_temp"), "--user-directory", os.path.join(OUT, "comfy_user"),
            "--database-url", "sqlite:///:memory:"]
    if SCRATCH_LORAS:
        paths = os.path.join(OUT, "comfy_extra_model_paths.yaml")
        with open(paths, "w", encoding="utf-8") as f:
            f.write(f"m63:\n    base_path: {SCRATCH_LORAS}\n    loras: .\n")
        args += ["--extra-model-paths-config", paths]
    log = open(os.path.join(OUT, f"comfy_{tag}.log"), "w", encoding="utf-8")
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early (see comfy_{tag}.log)")
        try:
            urllib.request.urlopen(URL + "/system_stats", timeout=2)
            return p, log
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(1)
    raise RuntimeError("ComfyUI did not start")


def submit(wf):
    body = json.dumps({"prompt": wf, "client_id": "entail-m63"}).encode()
    req = urllib.request.Request(URL + "/prompt", data=body, headers={"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"status": "rejected", "error": e.read().decode("utf-8", "replace")[:400]}
    while True:
        h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            break
        time.sleep(0.5)
    entry = h[pid]
    msgs = {m[0]: m[1] for m in entry["status"]["messages"]}
    out = {"status": entry["status"]["status_str"]}
    if "execution_error" in msgs:
        e = msgs["execution_error"]
        out["error"] = f"{e.get('node_type')}: {str(e.get('exception_message'))[:400]}"
    if "execution_success" in msgs:
        out["seconds"] = round((msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000, 2)
        img = entry["outputs"]["7"]["images"][0]
        out["image"] = os.path.join(IMAGES, img["subfolder"], img["filename"])
    return out


def records(tag):
    path = os.path.join(OUT, f"comfy_{tag}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if "verdict" in d:
            out.append({k: d.get(k) for k in ("boundary", "verdict", "rule", "resolution", "note")}
                       | {"declared": (d.get("declared") or {}).get("value"), "chosen": (d.get("chosen") or {}).get("value")})
        elif "said" in d:
            out.append({"said": d["said"], "text": d["text"]})
    return out


def main():
    os.makedirs(IMAGES, exist_ok=True)
    for tag in sys.argv[1:]:
        ckpt, node, lora, entail, extra = RUNS[tag]
        rec = os.path.join(OUT, f"comfy_{tag}.jsonl")
        if os.path.exists(rec):
            os.remove(rec)
        p, log = start(tag, entail, extra)
        try:
            runs = [{"seed": s, **submit(workflow(ckpt, node, lora, s, f"m63_{tag}_{s}"))} for s in SEEDS]
        finally:
            p.terminate()
            p.wait(timeout=60)
            log.close()
        res = {"tag": tag, "checkpoint": ckpt, "node": node, "lora": lora, "entail": entail, "extra": extra,
               "runs": runs, "decisions": records(tag), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(OUT, f"comfy_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(tag, [r.get("status") for r in runs], [d.get("verdict") for d in res["decisions"] if "verdict" in d])


if __name__ == "__main__":
    main()
