"""Run the user's ComfyUI with entail off and on, same workflow and seed, and compare.

Started on port 8189 so it cannot collide with a normal session. Images go to output/entail_test/.
Run with ComfyUI's own interpreter (it has PIL and numpy).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = r"C:\Users\<user>\Desktop\ComfyUI\ComfyUI-new"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
PORT = 8189
LOGDIR = os.path.dirname(os.path.abspath(__file__))
URL = f"http://127.0.0.1:{PORT}"
SEEDS = [11, 22, 33]  # different seeds: ComfyUI serves a repeated identical prompt from its cache


def workflow(prefix, seed):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "waiIllustriousSDXL_v160.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "masterpiece, best quality, 1girl, solo, smile, school uniform, cherry blossoms, outdoors, sunlight",
            "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "lowres, bad anatomy, bad hands, worst quality, low quality", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                                                   "latent_image": ["4", 0], "seed": seed, "steps": 20,
                                                   "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal",
                                                   "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }


def start(tag, entail_on):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for k in [k for k in env if k.startswith("ENTAIL")]:
        env.pop(k)
    if entail_on:
        env.update(ENTAIL="load", ENTAIL_VERBOSE="1")
    log = open(os.path.join(LOGDIR, f"comfy_{tag}.log"), "w", encoding="utf-8")
    t0 = time.time()
    p = subprocess.Popen([PY, "main.py", "--fp16-vae", "--port", str(PORT)], cwd=ROOT, env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    while time.time() - t0 < 600:
        if p.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early ({p.returncode}); see comfy_{tag}.log")
        try:
            urllib.request.urlopen(URL + "/system_stats", timeout=2)
            return p, log, time.time() - t0
        except Exception:
            time.sleep(1)
    raise RuntimeError("ComfyUI did not become ready")


def run(prefix, seed):
    body = json.dumps({"prompt": workflow(prefix, seed), "client_id": "entail-test"}).encode()
    req = urllib.request.Request(URL + "/prompt", data=body, headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    while True:
        h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            break
        time.sleep(0.5)
    entry = h[pid]
    msgs = {m[0]: m[1] for m in entry["status"]["messages"]}
    secs = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000 \
        if "execution_success" in msgs else None
    img = entry["outputs"]["7"]["images"][0]
    path = os.path.join(ROOT, "output", img["subfolder"], img["filename"])
    return {"status": entry["status"]["status_str"], "seconds": secs, "image": path}


def stop(p, log):
    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    p.wait(timeout=60)
    log.close()


def main():
    import numpy as np
    from PIL import Image

    res = {}
    # entail on first this time, so the operating system's file cache favours the off run, not the on run
    for tag, on in (("on", True), ("off", False)):
        p, log, ready = start(tag, on)
        try:
            runs = [run(f"entail_test/{tag}_seed{seed}", seed) for seed in SEEDS]
        finally:
            stop(p, log)
        res[tag] = {"startup_seconds": round(ready, 1), "runs": runs}
        print(tag, json.dumps(res[tag], ensure_ascii=False), flush=True)
        time.sleep(3)
    arr = {f"{t}{i}": np.asarray(Image.open(r["image"]).convert("RGB"), dtype=np.int16)
           for t in ("on", "off") for i, r in enumerate(res[t]["runs"])}
    comp = {}
    for a, b in ((f"off{i}", f"on{i}") for i in range(len(SEEDS))):
        d = np.abs(arr[a] - arr[b])
        comp[f"{a} vs {b}"] = {"identical": bool((d == 0).all()), "max_abs_diff": int(d.max()),
                               "pixels_differing": int((d.max(axis=2) > 0).sum())}
    res["comparison"] = comp
    with open(os.path.join(LOGDIR, "comfy_entail_test_v2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(comp, indent=1))
    lines = [ln.rstrip() for ln in open(os.path.join(LOGDIR, "comfy_on.log"), encoding="utf-8", errors="replace")
             if "[entail]" in ln]
    print("entail lines in the ON log:", len(lines))
    for ln in lines[:20]:
        print("  ", ln[:220])
    off_lines = [ln for ln in open(os.path.join(LOGDIR, "comfy_off.log"), encoding="utf-8", errors="replace")
                 if "[entail]" in ln]
    print("entail lines in the OFF log:", len(off_lines))


if __name__ == "__main__":
    main()
