"""M7 of VPRED_PROTOCOL.md: a checkpoint published without the v_pred key (AstolfoCarmix-VPredXL AC-Evo 2.5EP).

Reads the three sessions written by vpred_harm.py and compares each seed's image with the reference (b):
  (a) astolfo_plain      entail off, as ComfyUI detects the file
  (b) astolfo_ref        entail off, ModelSamplingDiscrete(v_prediction, zsnr=false) - the setting of the model's yaml
  (c) astolfo_plain_on   entail on, no node
  (b2) astolfo_refz      entail off, ModelSamplingDiscrete(v_prediction, zsnr=true) - added after (a), (b)
Writes astolfo_m7.json next to this file.
"""
import json
import os
import re

import vpred_harm as h

OUT = os.path.join(h.ROOT, "output", "entail_test")


def image(tag, case, seed):
    return os.path.join(OUT, f"vpred_{tag}_{case}_seed{seed}_00001_.png")


def log_facts(tag):
    text = open(os.path.join(h.HERE, f"vpred_{tag}.log"), encoding="utf-8", errors="replace").read()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return {"model_type": sorted(set(re.findall(r"model_type (\w+)", text))),
            "entail": [ln.strip()[:400] for ln in text.splitlines() if "[entail] resolved" in ln or "RoleError:" in ln]}


def main():
    runs = {tag: json.load(open(os.path.join(h.HERE, f"vpred_{tag}.json"), encoding="utf-8"))
            for tag in ("astolfo_plain", "astolfo_ref", "astolfo_plain_on", "astolfo_refz")}
    res = {"status": {tag: [r["status"] for r in next(v for k, v in d.items() if k not in ("log",))]
                      for tag, d in runs.items()},
           "stats": {tag: [r.get("stats") for r in next(v for k, v in d.items() if k not in ("log",))]
                     for tag, d in runs.items()},
           "seconds": {tag: [r.get("seconds") for r in next(v for k, v in d.items() if k not in ("log",))]
                       for tag, d in runs.items()},
           "log": {tag: log_facts(tag) for tag in runs}}
    res["a_vs_ref"] = [h.diff(image("astolfo_plain", "plain", s), image("astolfo_ref", "ref", s)) for s in h.SEEDS]
    res["c_vs_ref"] = [h.diff(image("astolfo_plain_on", "plain", s), image("astolfo_ref", "ref", s)) for s in h.SEEDS]
    res["c_vs_a"] = [h.diff(image("astolfo_plain_on", "plain", s), image("astolfo_plain", "plain", s)) for s in h.SEEDS]
    res["refz_vs_ref"] = [h.diff(image("astolfo_refz", "refz", s), image("astolfo_ref", "ref", s)) for s in h.SEEDS]
    res["a_vs_refz"] = [h.diff(image("astolfo_plain", "plain", s), image("astolfo_refz", "refz", s)) for s in h.SEEDS]
    json.dump(res, open(os.path.join(h.HERE, "astolfo_m7.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
