"""P4 external evaluation, item 2: the ComfyUI DLC in normal runs, not the defect's reproduction. Two workflows - the
researcher's main one (Anima DiT with the nagito LoRA, M6's anima_same) and an eps SDXL checkpoint (waiIllustrious
v1.6) - each with entail off and then on with the DLC attached, seeds 11, 22, 33. The images must be the same pixel for
pixel, the DLC must say nothing (nothing leaks here) and decide nothing broken or refused; every line entail prints is
counted too, to see that the start-up hook ran once (P4's fix).

Run with ComfyUI's interpreter, COMFY_ROOT set, and PYTHONPATH carrying the product-branch entail and the DLC:
  python testbed/p4_comfy_normal.py
Writes testbed/results/p4/comfy_normal.json; entail's records go to testbed/results/p4/comfy_normal_<workflow>/.
"""
import collections
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "issue_track", "comfyui_field_test"))
import anima_same as anima  # noqa: E402
import vpred_harm as h  # noqa: E402

OUT = os.environ.get("P4_OUT") or os.path.join(HERE, "results", "p4")   # P6 re-checks elsewhere
TAG = os.environ.get("P4_TAG", "p4n")    # the sessions' log and image names (P6's re-check uses its own)
WORKFLOWS = {"anima": anima.workflow,
             "wai_eps": lambda prefix, seed: h.workflow("waiIllustriousSDXL_v160.safetensors", prefix, seed)}


def session(tag, entail, name, logdir=None):
    os.environ["COMFY_EXTRA_ENV"] = json.dumps({"ENTAIL_LOG_DIR": logdir} if logdir else {})
    p, log = h.start(tag, entail)
    try:
        runs = [h.run(WORKFLOWS[name](f"entail_test/{tag}_seed{s}", s)) for s in h.SEEDS]
    finally:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        p.wait(timeout=60)
        log.close()
    lines = open(os.path.join(h.HERE, f"vpred_{tag}.log"), encoding="utf-8", errors="replace").read().splitlines()
    return runs, [ln for ln in lines if "[entail]" in ln]


def records(folder):
    out = []
    for path in glob.glob(os.path.join(folder, "record-*.jsonl")):
        for s in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(s))
            except ValueError:
                pass
    return out


def main():
    res = {}
    for name in WORKFLOWS:
        off, _ = session(f"{TAG}_{name}_off", False, name)
        folder = os.path.join(OUT, f"comfy_normal_{name}")
        os.makedirs(folder, exist_ok=True)
        on, lines = session(f"{TAG}_{name}_on_dlc", True, name, folder)
        recs = records(folder)
        verdicts = collections.Counter(r["verdict"] for r in recs if r.get("verdict"))
        res[name] = {
            "status": {"off": [r.get("status") for r in off], "on": [r.get("status") for r in on]},
            "seconds": {"off": [r.get("seconds") for r in off], "on": [r.get("seconds") for r in on]},
            "same_image": [h.diff(a["image"], b["image"]) if "image" in a and "image" in b else None
                           for a, b in zip(off, on)],
            "dlc_said": [r["text"] for r in recs if str(r.get("said", "")).startswith("dlc:")],
            "dlc_decisions": [(r.get("boundary"), r.get("verdict")) for r in recs
                              if str(r.get("boundary", "")).startswith("dlc:") and r.get("verdict")],
            "verdicts": dict(verdicts),
            "broken_or_refused": [(r.get("boundary"), r.get("rule", "")[:80]) for r in recs
                                  if r.get("verdict") in ("broken", "refused")],
            "dlc_installed": [ln for ln in lines if "(DLC comfyui)" in ln],
            "entail_lines": len(lines),
            "repeated_lines": sum(n - 1 for n in collections.Counter(lines).values() if n > 1),
        }
        print(name, json.dumps({k: v for k, v in res[name].items() if k not in ("seconds",)}, ensure_ascii=False)[:900],
              flush=True)
    with open(os.path.join(OUT, "comfy_normal.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
