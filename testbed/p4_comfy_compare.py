"""P4 (S10): the M6 images again with the repair coming from the official DLC, against M6's own (the core's repair,
tag _g2, which matched fresh sessions pixel for pixel), and with the DLC left out (_nodlc). Run with ComfyUI's
interpreter (PIL, numpy). Writes testbed/results/p4/comfy_m6.json.

  python testbed/p4_comfy_compare.py
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.join(os.path.dirname(HERE), "issue_track", "comfyui_field_test")
OUT = os.path.join(HERE, "results", "p4")
OLD_ROOT = r"C:\Users\<user>\Desktop\ComfyUI\ComfyUI-new"
NEW_ROOT = r"E:\ComfyUI\ComfyUI-new"


def images(mode, tag):
    d = json.load(open(os.path.join(M6, f"vpred_{mode}_on{tag}.json"), encoding="utf-8"))
    out = {}
    for case in ("native", "node"):
        for seed, r in zip((11, 22, 33), d.get(case, [])):
            out[(case, seed)] = r.get("image", "").replace(OLD_ROOT, NEW_ROOT) if r.get("image") else None
    return out, d.get("log", {})


def diff(a, b):
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
    d = np.abs(x - y)
    return {"max": int(d.max()), "mean": round(float(d.mean()), 3)}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for mode in ("noob_native_node", "noob_node_native"):
        ref, _ = images(mode, "_g2")
        for tag in ("_dlc", "_nodlc"):
            got, log = images(mode, tag)
            rows = {f"{case}_seed{seed}": (diff(got[(case, seed)], ref[(case, seed)])
                                           if got.get((case, seed)) and ref.get((case, seed)) else None)
                    for case, seed in sorted(ref)}
            res[f"{mode}{tag}"] = {"vs_m6": rows,
                                   "identical": sum(1 for v in rows.values() if v and v["max"] == 0),
                                   "log": {k: v for k, v in log.items() if k != "warnings"}}
            print(mode, tag, "identical to M6:", res[f"{mode}{tag}"]["identical"], "/", len(rows),
                  {k: v for k, v in rows.items() if v and v["max"]})
    with open(os.path.join(OUT, "comfy_m6.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
