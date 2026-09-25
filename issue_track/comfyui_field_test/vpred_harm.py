"""M1 of VPRED_PROTOCOL.md (and M3/M4 with arguments): what a lost v_pred marker does, through the user's ComfyUI.

  python vpred_harm.py harm            NoobAI-XL-Vpred as detected (v + ztsnr) vs forced eps (= the marker lost)
  python vpred_harm.py stripped [on]   the copy without v_pred/ztsnr keys, entail off or on
  python vpred_harm.py eps_on|eps_off  an eps model with entail on / off (M4: the probe must not change the image)
  python vpred_harm.py leftover [on]   an eps model with a ModelSamplingDiscrete(v_prediction, zsnr) node left on (M5)
  python vpred_harm.py leftover_first on   the same, the node's run first after start-up (no earlier probe to reuse)
  any mode + "_nodyn"                  the same with --disable-dynamic-vram (legacy model loading)
  any mode + "_nocustom"               the same with --disable-all-custom-nodes
  astolfo_plain [on] | astolfo_ref       M7: a checkpoint published without the v_pred key, as ComfyUI detects it /
                                       with ModelSamplingDiscrete(v_prediction, zsnr=false), its own yaml's setting
  noob_native_node|noob_node_native [on]   NoobAI-XL-Vpred as detected and with ModelSamplingDiscrete(v_prediction,
                                       zsnr=false), in both orders (M6); VPRED_TAG=<suffix> keeps runs apart
Port 8189; images to output/entail_test/vpred_*. Writes vpred_<mode>.json next to this file.
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
SEEDS = [11, 22, 33]
PROMPT = "masterpiece, best quality, 1girl, solo, smile, school uniform, cherry blossoms, outdoors, sunlight"


def workflow(ckpt, prefix, seed, force=None):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, bad anatomy, worst quality, low quality",
                                                        "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["9", 0] if force else ["1", 0], "positive": ["2", 0],
                                                   "negative": ["3", 0], "latent_image": ["4", 0], "seed": seed,
                                                   "steps": 25, "cfg": 5.0, "sampler_name": "euler",
                                                   "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }
    if force:
        sampling, _, zsnr = force.partition("+")
        wf["9"] = {"class_type": "ModelSamplingDiscrete", "inputs": {"model": ["1", 0], "sampling": sampling,
                                                              "zsnr": zsnr == "zsnr"}}
    return wf


def start(tag, entail, extra=()):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for k in [k for k in env if k.startswith("ENTAIL")]:
        env.pop(k)
    if entail:
        env.update(ENTAIL="load", ENTAIL_VERBOSE="1")
    env.update(json.loads(os.environ.get("COMFY_EXTRA_ENV", "{}")))
    log = open(os.path.join(HERE, f"vpred_{tag}.log"), "w", encoding="utf-8")
    p = subprocess.Popen([PY, "main.py", "--fp16-vae", "--port", "8189", *extra], cwd=ROOT, env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            raise RuntimeError("ComfyUI exited early")
        try:
            urllib.request.urlopen(URL + "/system_stats", timeout=2)
            return p, log
        except Exception:
            time.sleep(1)
    raise RuntimeError("not ready")


def run(wf):
    body = json.dumps({"prompt": wf, "client_id": "entail-vpred"}).encode()
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
        out["image"] = os.path.join(ROOT, "output", img["subfolder"], img["filename"])
    return out


def stats(path):
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    a = np.asarray(im, dtype=np.float32)
    s = np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1]
    return {"mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1), "saturation": round(float(s.mean()), 1)}


def diff(p, q):
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(q).convert("RGB"), dtype=np.int16)
    d = np.abs(a - b)
    return {"identical": bool((d == 0).all()), "mean_abs_diff": round(float(d.mean()), 2)}


def main():
    mode = sys.argv[1]
    extra, base_mode = [], mode
    for suffix, flag in (("_nodyn", "--disable-dynamic-vram"), ("_nocustom", "--disable-all-custom-nodes")):
        if base_mode.endswith(suffix):  # same cases: legacy model loading / no custom nodes
            extra.append(flag)
            base_mode = base_mode[:-len(suffix)]
    entail = "on" in sys.argv[2:] or mode == "eps_on"
    cases = {
        "harm": [("vpred", "NoobAI-XL-Vpred-v1.0.safetensors", None), ("eps", "NoobAI-XL-Vpred-v1.0.safetensors", "eps")],
        "stripped": [("stripped", "entail_test\\NoobAI-XL-Vpred-v1.0-no-marker.safetensors", None)],
        "eps_on": [("wai", "waiIllustriousSDXL_v160.safetensors", None)],
        "eps_off": [("wai", "waiIllustriousSDXL_v160.safetensors", None)],
        "leftover": [("normal", "waiIllustriousSDXL_v170.safetensors", None),
                     ("leftover", "waiIllustriousSDXL_v170.safetensors", "v_prediction+zsnr")],
        "astolfo_plain": [("plain", "entail_test\\astolfocarmixVpredxl_acEvo25EP.safetensors", None)],
        "astolfo_ref": [("ref", "entail_test\\astolfocarmixVpredxl_acEvo25EP.safetensors", "v_prediction")],
        "astolfo_refz": [("refz", "entail_test\\astolfocarmixVpredxl_acEvo25EP.safetensors", "v_prediction+zsnr")],
        "noob_native_node": [("native", "NoobAI-XL-Vpred-v1.0.safetensors", None),
                             ("node", "NoobAI-XL-Vpred-v1.0.safetensors", "v_prediction")],
        "noob_node_native": [("node", "NoobAI-XL-Vpred-v1.0.safetensors", "v_prediction"),
                             ("native", "NoobAI-XL-Vpred-v1.0.safetensors", None)],
        "diag": [("leftover", "waiIllustriousSDXL_v170.safetensors", "v_prediction+zsnr"),
                 ("normal", "waiIllustriousSDXL_v170.safetensors", None)],
        "leftover_first": [("leftover", "waiIllustriousSDXL_v170.safetensors", "v_prediction+zsnr"),
                           ("normal", "waiIllustriousSDXL_v170.safetensors", None)],
    }[base_mode]
    tag = mode + ("_on" if entail and mode != "eps_on" else "") + os.environ.get("VPRED_TAG", "")
    p, log = start(tag, entail, extra)
    res = {}
    try:
        for case, ckpt, force in cases:
            res[case] = [run(workflow(ckpt, f"entail_test/vpred_{tag}_{case}_seed{s}", s, force)) for s in SEEDS]
            for r in res[case]:
                if "image" in r:
                    r["stats"] = stats(r["image"])
            print(case, json.dumps(res[case], ensure_ascii=False), flush=True)
    finally:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        p.wait(timeout=60)
        log.close()
    lines = open(os.path.join(HERE, f"vpred_{tag}.log"), encoding="utf-8", errors="replace").read().splitlines()
    res["log"] = {"warnings": [ln[:200] for ln in lines if "WARNING" in ln or "warning" in ln.lower()][:10],
                  "entail": [ln[:300] for ln in lines if "[entail]" in ln or "RoleError" in ln][:12]}
    pairs = {"harm": ("eps", "vpred"), "leftover": ("leftover", "normal"), "leftover_first": ("leftover", "normal"), "diag": ("leftover", "normal")}
    if base_mode in pairs:  # with entail on, the contradicting runs stop and leave no image to compare
        a_, b_ = pairs[base_mode]
        res[f"{a_}_vs_{b_}"] = [diff(a["image"], b["image"]) if "image" in a and "image" in b else None
                                for a, b in zip(res[a_], res[b_])]
    json.dump(res, open(os.path.join(HERE, f"vpred_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k == "log" or "_vs_" in k}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
