"""Case first: what does the user's ComfyUI do when a LoRA made for another base model is loaded?

Same SDXL workflow as the user's 나기토화풍-WAI (waiIllustriousSDXL v170, LoraLoaderModelOnly at 0.9), three ways:
none, the matching SDXL LoRA, and the Anima LoRA of the same character. Port 8189, images to output/entail_test/.
Usage: python comfy_lora_case.py <tag> [entail]      (entail = start ComfyUI with ENTAIL=load)
Extra environment for the ComfyUI process can be given with COMFY_EXTRA_ENV='{"KEY": "VALUE"}'.
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
PORT = 8189
URL = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
LORAS = {"none": None, "right": "nagito\\nagito_illustrious_v3.safetensors",
         "wrong": "nagito\\nagito_anima_e10.safetensors"}
SEED = 11


def workflow(prefix, lora):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "waiIllustriousSDXL_v170.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "masterpiece, best quality, 1girl, solo, smile, school uniform, cherry blossoms, outdoors",
            "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, bad anatomy, worst quality, low quality",
                                                        "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["8", 0] if lora else ["1", 0], "positive": ["2", 0],
                                                   "negative": ["3", 0], "latent_image": ["4", 0], "seed": SEED,
                                                   "steps": 20, "cfg": 5.0, "sampler_name": "euler",
                                                   "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }
    if lora:
        wf["8"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora,
                                                                    "strength_model": 0.9}}
    return wf


def start(tag, entail):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for k in [k for k in env if k.startswith("ENTAIL")]:
        env.pop(k)
    if entail:
        env.update(ENTAIL="load", ENTAIL_VERBOSE="1")
    env.update(json.loads(os.environ.get("COMFY_EXTRA_ENV", "{}")))
    log = open(os.path.join(HERE, f"lora_{tag}.log"), "w", encoding="utf-8")
    p = subprocess.Popen([PY, "main.py", "--fp16-vae", "--port", str(PORT)], cwd=ROOT, env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early ({p.returncode})")
        try:
            urllib.request.urlopen(URL + "/system_stats", timeout=2)
            return p, log
        except Exception:
            time.sleep(1)
    raise RuntimeError("not ready")


def run(prefix, lora):
    body = json.dumps({"prompt": workflow(prefix, lora), "client_id": "entail-lora"}).encode()
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
    st = entry["status"]
    msgs = {m[0]: m[1] for m in st["messages"]}
    out = {"status": st["status_str"]}
    if "execution_error" in msgs:
        e = msgs["execution_error"]
        out["error"] = f"{e.get('node_type')}: {e.get('exception_type')}: {str(e.get('exception_message'))[:500]}"
    if "execution_success" in msgs:
        out["seconds"] = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000
        img = entry["outputs"]["7"]["images"][0]
        out["image"] = os.path.join(ROOT, "output", img["subfolder"], img["filename"])
    return out


def main():
    tag = sys.argv[1]
    entail = len(sys.argv) > 2 and sys.argv[2] == "entail"
    p, log = start(tag, entail)
    res = {}
    try:
        for case, lora in LORAS.items():
            res[case] = run(f"entail_test/lora_{tag}_{case}", lora)
            print(case, json.dumps(res[case], ensure_ascii=False), flush=True)
    finally:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        p.wait(timeout=60)
        log.close()
    lines = open(os.path.join(HERE, f"lora_{tag}.log"), encoding="utf-8", errors="replace").read().splitlines()
    res["log"] = {"lora_key_not_loaded": sum("lora key not loaded" in ln for ln in lines),
                  "entail_lines": [ln[:300] for ln in lines if "[entail]" in ln or "RoleError" in ln][:12]}
    import numpy as np
    from PIL import Image

    def arr(case):
        return np.asarray(Image.open(res[case]["image"]).convert("RGB"), dtype=np.int16) if "image" in res[case] else None
    base = arr("none")
    for case in ("right", "wrong"):
        a = arr(case)
        if a is not None and base is not None:
            d = np.abs(a - base)
            res[case]["vs_none"] = {"identical": bool((d == 0).all()), "mean_abs_diff": round(float(d.mean()), 3)}
    json.dump(res, open(os.path.join(HERE, f"lora_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "none"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
