"""Summarise the M6.3 measurements from their result files and images only (testbed/results/m63/, ~/m63_out).

  diffusers_<case>_<condition>.json   testbed/m63_diffusers.py: gen cases (i04, m7, s3) and LoRA loads
Per gen case: every condition's images against the reference condition of the same seed (mean absolute difference
over all pixels and channels, 0-255; identical when every pixel is), and off against on where both exist. Writes
SUMMARY.md next to the results. Run in WSL ~/venvs/gpu: python testbed/m63_summarize.py
"""
import glob
import json
import os
from collections import Counter

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
DIR = os.path.join(os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results")), "m63")
CASES = {"i04": ("market I04: NoobAI-XL-Vpred-v1.0, marker keys v_pred and ztsnr", ("reference", "off", "on", "observe")),
         "m7": ("fd-m7: AstolfoCarmix-VPredXL AC-Evo 2.5EP, metadata modelspec.prediction_type = v",
                ("reference", "off", "on")),
         "s3": ("S3 and non-interference: waiIllustriousSDXL v160, declares nothing", ("off", "on"))}


def load(tag):
    p = os.path.join(DIR, f"diffusers_{tag}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def local(path):
    """A Windows path (the ComfyUI runs) as this WSL process sees it."""
    if len(path) > 2 and path[1] == ":":
        return "/mnt/" + path[0].lower() + path[2:].replace("\\", "/")
    return path


def pixels(path):
    return np.asarray(Image.open(local(path)).convert("RGB"), dtype=np.int16)


def compare(a, b):
    d = np.abs(pixels(a) - pixels(b))
    return {"mean_abs": round(float(d.mean()), 2), "max": int(d.max()), "identical": bool(d.max() == 0)}


def verdicts(res):
    c = Counter(f"{d['boundary'].split(':', 1)[1]} {d['verdict']}" for d in res.get("decisions", []) if "verdict" in d
                and not d["boundary"].startswith("load:transformers"))
    return "; ".join(f"{k}" + (f" x{n}" if n > 1 else "") for k, n in sorted(c.items())) or "none"


def entail_ms(res):
    ms = [d["ms"] for d in res.get("decisions", []) if "timing" in d]
    return round(sum(ms), 1) if ms else None


def gen_section(lines, case, what, conditions):
    runs = {c: load(f"{case}_{c}") for c in conditions}
    runs = {c: r for c, r in runs.items() if r}
    if not runs:
        return
    lines += [f"### {case}: {what}", "",
              "| condition | entail | scheduler as it samples | decisions (image models) | entail ms at load | "
              "seed 11 | seed 22 | seed 33 |", "|---|---|---|---|---|---|---|---|"]
    ref = runs.get("reference")
    for c, r in runs.items():
        s = r["scheduler"]
        cells = []
        for i, run in enumerate(r["runs"]):
            if ref is not None and c != "reference":
                cmp = compare(run["image"], ref["runs"][i]["image"])
                cells.append("identical to reference" if cmp["identical"] else
                             f"{cmp['mean_abs']} from reference (max {cmp['max']})")
            else:
                st = run["stats"]
                cells.append(f"mean {st['mean']}, std {st['std']}, sat {st['saturation']}")
        policy = "" if r["entail"] == "not installed" else f" ({r['policy']})"
        lines.append(f"| {c} | {r['entail']}{policy} | {s['prediction_type']}, rescale_betas_zero_snr "
                     f"{s['rescale_betas_zero_snr']} | {verdicts(r)} | {entail_ms(r)} | " + " | ".join(cells) + " |")
    if "off" in runs and "on" in runs:
        same = [compare(a["image"], b["image"]) for a, b in zip(runs["off"]["runs"], runs["on"]["runs"])]
        lines.append("")
        lines.append("- off against on: " + ", ".join("identical" if x["identical"] else
                                                       f"{x['mean_abs']} (max {x['max']})" for x in same))
    for c, r in runs.items():
        for d in r.get("decisions", []):
            if "verdict" in d and d["verdict"] != "pass" and not d["boundary"].startswith("load:transformers"):
                what_changed = f"; changed: {d['resolution']}" if d.get("resolution") else ""
                lines.append(f"- `{c}` {d['verdict']} at {d['boundary']}: declared {d['declared']} "
                             f"({d['declared_from']}); used {d['chosen']}; {d['rule']}{what_changed}")
    lines.append("")


def vae_section(lines):
    runs = {c: load(f"vae_{c}") for c in ("reference", "off", "on", "observe")}
    runs = {c: r for c, r in runs.items() if r}
    if not runs:
        return
    lines += ["### vae: fd-vae, waiIllustriousSDXL v160 with sdxl_vae_fp16_fix read on its own", "",
              "`AutoencoderKL.from_single_file(sdxl_vae_fp16_fix)` with no config, put into the pipeline "
              "(`pipe.vae = vae`). The model's latent scale is declared by a pinned manifest (SDXL's 0.13025; "
              "`testbed/m63_vae_manifest.py`). `reference`: the VAE's scaling_factor set to 0.13025 by hand.", "",
              "| condition | entail | VAE scale as loaded | VAE scale it samples with | decisions (image models) | "
              "seed 11 | seed 22 | seed 33 |", "|---|---|---|---|---|---|---|---|"]
    ref = runs.get("reference")
    for c, r in runs.items():
        cells = []
        for i, run in enumerate(r["runs"]):
            if ref is not None and c != "reference":
                cmp = compare(run["image"], ref["runs"][i]["image"])
                cells.append("identical to reference" if cmp["identical"] else
                             f"{cmp['mean_abs']} from reference (max {cmp['max']})")
            else:
                st = run["stats"]
                cells.append(f"mean {st['mean']}, std {st['std']}, sat {st['saturation']}")
        policy = "" if r["entail"] == "not installed" else f" ({r['policy']})"
        lines.append(f"| {c} | {r['entail']}{policy} | {r['vae_scaling_factor_as_loaded']} | "
                     f"{r['vae_scaling_factor']} | {verdicts(r)} | " + " | ".join(cells) + " |")
    lines.append("")
    for c, r in runs.items():
        for d in r.get("decisions", []):
            if "verdict" in d and d["verdict"] != "pass" and "latent_scale" in d["boundary"]:
                what_changed = f"; changed: {d['resolution']}" if d.get("resolution") else ""
                lines.append(f"- `{c}` {d['verdict']} at {d['boundary']}: declared {d['declared']} "
                             f"({d['declared_from']}); used {d['chosen']}; {d['rule']}{what_changed}")
    lines.append("")


def lora_section(lines):
    tags = ("lora_right_off", "lora_right_on", "lora_other_off", "lora_other_on", "lora_other_strict", "i01_off",
            "i01_on", "i01_strict")
    runs = [(t, load(t)) for t in tags]
    runs = [(t, r) for t, r in runs if r]
    if not runs:
        return
    lines += ["### LoRA loads (waiIllustriousSDXL v160)", "",
              "right: nagito_illustrious_v3 (kohya, made for Illustrious/SDXL). other: nagito_anima_e10 (made for "
              "Anima; fd-lora). i01: nagito_illustrious_v3's UNet part under PEFT-wrapped names (base_model.model.*), "
              "written by testbed/m63_i01_lora.py (market I01 simulated).", "",
              "| run | LoRA | entail | outcome | modules holding the adapter | decisions | diffusers logged |",
              "|---|---|---|---|---|---|---|"]
    for t, r in runs:
        about = [x for x in r.get("log", []) if any(w in x.lower() for w in ("lora", "adapter", "peft", "target modules"))]
        logged = "; ".join(x[:150] for x in about[:2]) or "nothing about the LoRA"
        entail = r["entail"] + (f" (ENTAIL_ON_BROKEN={r['on_broken']})" if r["entail"] != "not installed" else "")
        lines.append(f"| {t} | {os.path.basename(r['lora'])} | {entail} | {r['outcome'][:120]} | "
                     f"{r['modules_with_the_adapter']} | "
                     f"{verdicts(r)} | {logged} |")
    lines.append("")
    none = load("s3_off")   # the same model and seeds without a LoRA
    for t, r in runs:
        if r.get("runs") and none:
            eff = [compare(a["image"], b["image"]) for a, b in zip(r["runs"], none["runs"])]
            lines.append(f"- `{t}` against the same model without a LoRA (s3_off): "
                         + ", ".join("identical" if x["identical"] else f"{x['mean_abs']} (max {x['max']})"
                                     for x in eff))
    for case in ("lora_right", "lora_other", "i01"):
        a, b = load(f"{case}_off"), load(f"{case}_on")
        if a and b and a.get("runs") and b.get("runs"):
            same = [compare(x["image"], y["image"]) for x, y in zip(a["runs"], b["runs"])]
            lines.append(f"- `{case}` off against on: "
                         + ", ".join("identical" if x["identical"] else f"{x['mean_abs']} (max {x['max']})"
                                     for x in same))
    for t, r in runs:
        for d in r.get("decisions", []):
            if "verdict" in d and d["verdict"] != "pass" and d["boundary"].startswith("load:diffusers.lora"):
                lines.append(f"- `{t}` {d['verdict']}: {d['chosen']}; note: {d.get('note') or '-'}")
    lines.append("")


def comfy(tag):
    p = os.path.join(DIR, f"comfy_{tag}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def comfy_cells(r, against):
    cells = []
    for i, run in enumerate(r["runs"]):
        if "image" not in run:
            cells.append(f"{run.get('status')}: {run.get('error', '')[:140]}")
        elif against is not None and "image" in against["runs"][i]:
            cmp = compare(run["image"], against["runs"][i]["image"])
            cells.append("identical" if cmp["identical"] else f"{cmp['mean_abs']} (max {cmp['max']})")
        else:
            cells.append(f"{run.get('seconds')} s")
    return cells


def comfy_verdicts(r):
    c = Counter(f"{d['boundary'].split(':', 1)[1]} {d['verdict']}" for d in r.get("decisions", []) if "verdict" in d
                and d["boundary"].startswith(("load:comfyui", "load:diffusers")))
    said = sum(1 for d in r.get("decisions", []) if "said" in d)
    text = "; ".join(f"{k}" + (f" x{n}" if n > 1 else "") for k, n in sorted(c.items())) or "none"
    return text + (f"; repair lines {said}" if said else "")


def comfy_entail(r):
    if not r["entail"]:
        return "off"
    extra = ", ".join(f"{k}={v}" for k, v in r["extra"].items())
    return "on" + (f" ({extra})" if extra else "")


def comfy_section(lines):
    groups = [("m7", "fd-m7: AstolfoCarmix (metadata v, no marker key)", "m7_ref", ("m7_ref", "m7_off", "m7_on")),
              ("node", "NoobAI-XL-Vpred with ModelSamplingDiscrete(eps): the user's choice against the markers",
               "node_off", ("node_off", "node_on")),
              ("s3 wai", "S3: waiIllustrious v160, declares nothing", "s3_wai_off", ("s3_wai_off", "s3_wai_on")),
              ("s3 noob", "S3: NoobAI-XL-Vpred, markers ComfyUI reads", "s3_noob_off", ("s3_noob_off", "s3_noob_on")),
              ("lora", "waiIllustrious v170 + LoraLoaderModelOnly 0.9; compared with no LoRA", "lora_none_off",
               ("lora_none_off", "lora_right_off", "lora_right_on", "lora_other_off", "lora_other_on",
                "lora_other_strict", "i01_off", "i01_on", "i01_strict"))]
    if not any(comfy(t) for _, _, _, tags in groups for t in tags):
        return
    lines += ["## ComfyUI", "",
              "The researcher's ComfyUI 0.34.1 (Windows, its own .venv: torch 2.13), entail from the checkout on "
              "PYTHONPATH; output, temp and user directories in this folder, database in memory; every condition in a "
              "fresh server, seeds 11, 22, 33, 1024x1024, euler/normal 25 steps, cfg 5.0 (`testbed/m63_comfyui.py`). "
              "Each image cell: the mean absolute difference (0-255) to the group's first run, or `identical`.", ""]
    for name, what, base, tags in groups:
        runs = [(t, comfy(t)) for t in tags]
        runs = [(t, r) for t, r in runs if r]
        if not runs:
            continue
        against = comfy(base)
        lines += [f"### {name}: {what}", "", "| run | entail | decisions | seed 11 | seed 22 | seed 33 |",
                  "|---|---|---|---|---|---|"]
        for t, r in runs:
            cells = comfy_cells(r, None if t == base else against)
            lines.append(f"| {t} | {comfy_entail(r)} | {comfy_verdicts(r)} | " + " | ".join(cells) + " |")
        for t, _ in runs:
            if not t.endswith("_on"):
                continue
            a, b = comfy(t[:-3] + "_off"), comfy(t)
            if a and b and all("image" in x for x in a["runs"] + b["runs"]):
                same = [compare(x["image"], y["image"]) for x, y in zip(a["runs"], b["runs"])]
                lines.append(f"- `{t[:-3]}` off against on: " + ", ".join(
                    "identical" if x["identical"] else f"{x['mean_abs']} (max {x['max']})" for x in same))
        for t, r in runs:
            for d in r.get("decisions", []):
                if "verdict" in d and d["verdict"] != "pass" and d["boundary"].startswith("load:comfyui"):
                    extra = f"; changed: {d['resolution']}" if d.get("resolution") else ""
                    note = f"; note: {d['note']}" if d.get("note") else ""
                    lines.append(f"- `{t}` {d['verdict']} at {d['boundary']}: declared {d['declared']}; used "
                                 f"{d['chosen']}; {d['rule']}{extra}{note}")
                elif "said" in d:
                    lines.append(f"- `{t}` said: {d['text'][:200]}")
        lines.append("")


def main():
    lines = ["# M6.3: image models under the default policy", "",
             "Generated by `testbed/m63_summarize.py` from the files in this folder and the images they name. RTX 4070 "
             "Ti (12 GB), WSL2. diffusers 0.40.0, torch 2.14.0+cu130, SDXL pipelines from single files with the local "
             "config set of the earlier diffusers run (`~/sdxl_local_config`; its two CLIP tokenizers were set to "
             "77 tokens for M6.3, copied from ComfyUI's bundled tokenizer they said 8192). Three seeds, 1024x1024, 25 "
             "steps, guidance 5.0, fp16. `reference`: entail not installed, the scheduler set by hand to what the file "
             "declares; `off`: diffusers alone; `on`: ENTAIL=load; `observe`: ENTAIL_POLICY=refuse.", "",
             "## diffusers", ""]
    for case, (what, conditions) in CASES.items():
        gen_section(lines, case, what, conditions)
    vae_section(lines)
    lora_section(lines)
    comfy_section(lines)
    cost = os.path.join(DIR, "manifest_cost.json")
    if os.path.exists(cost):
        c = json.load(open(cost, encoding="utf-8"))
        lines += ["## Finding a manifest at load (S4)", "",
                  "`testbed/m63_manifest_cost.py`, WSL, the checkpoints on /mnt/c, ENTAIL_MANIFESTS = the fd-vae manifest "
                  "folder. Before M6.3 every lookup hashed the file in full; now only a file some manifest's quick "
                  "fingerprint matches is.", "", "| checkpoint | bytes | a manifest is for it | seconds |", "|---|---|---|---|"]
        for f in c["finds"]:
            lines.append(f"| {f['file']} | {f['bytes']} | {f['found']} | {f['seconds']} |")
        lines += ["", f"- the full SHA-256 of one of them alone (what every lookup cost before): "
                      f"{c['full_sha256_of_one_file_seconds']} s; the quick fingerprint: {c['quick_fingerprint_seconds']} s", ""]
    out = os.path.join(DIR, "SUMMARY.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("tags found:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(DIR, "diffusers_*.json"))))


if __name__ == "__main__":
    main()
