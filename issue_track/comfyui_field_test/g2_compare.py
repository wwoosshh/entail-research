"""Regression (c) and M7 re-measurement of entail/DESIGN.md section 7: the _g2 runs (refactored entail) against the
references measured before. Writes g2_compare.json."""
import json
import os
import re

import vpred_harm as h

O = os.path.join(h.ROOT, "output", "entail_test")


def img(name):
    return os.path.join(O, name)


def same(a, b):
    d = h.diff(img(a), img(b))
    return "identical" if d["identical"] else d["mean_abs_diff"]


def lines(tag, needle):
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(os.path.join(h.HERE, f"vpred_{tag}.log"), encoding="utf-8",
                                                 errors="replace").read())
    return [ln.strip()[:360] for ln in text.splitlines() if needle in ln]


def main():
    S = h.SEEDS
    out = {
        "lora_right_vs_0.2.0": same("lora_g2_right_00001_.png", "lora_on_right_00001_.png"),
        "lora_none_vs_0.2.0": same("lora_g2_none_00001_.png", "lora_on_none_00001_.png"),
        "M3_stripped_vs_0.3.0_resolution": [same(f"vpred_stripped_on_g2_stripped_seed{s}_00001_.png",
                                                 f"vpred_stripped_on_stripped_seed{s}_00003_.png") for s in S],
        "M3_lines": lines("stripped_on_g2", "[entail] resolved"),
        "S1_node_runs": [r["status"] for r in json.load(open("vpred_leftover_first_on_g2.json"))["leftover"]],
        "S1_plain_vs_fresh": [same(f"vpred_leftover_first_on_g2_normal_seed{s}_00001_.png",
                                   f"vpred_leftover_normal_seed{s}_00001_.png") for s in S],
        "S1_stop_message": (json.load(open("vpred_leftover_first_on_g2.json"))["leftover"][0].get("error") or "")[:300],
        "S2_native_node": {c: [same(f"vpred_noob_native_node_on_g2_{c}_seed{s}_00001_.png", ref.format(s)) for s in S]
                           for c, ref in (("native", "vpred_noob_native_node_native_seed{}_00001_.png"),
                                          ("node", "vpred_noob_node_native_node_seed{}_00001_.png"))},
        "S2_node_native": {c: [same(f"vpred_noob_node_native_on_g2_{c}_seed{s}_00001_.png", ref.format(s)) for s in S]
                           for c, ref in (("native", "vpred_noob_native_node_native_seed{}_00001_.png"),
                                          ("node", "vpred_noob_node_native_node_seed{}_00001_.png"))},
        "anima_on_vs_off": [h.diff(a["image"], b["image"])["identical"] for a, b in zip(
            json.load(open("anima_off.json"))["runs"], json.load(open("anima_on_g2.json"))["runs"])],
        "M7_status": [r["status"] for r in json.load(open("vpred_astolfo_plain_on_g2.json"))["plain"]],
        "M7_vs_b_yaml_setting": [same(f"vpred_astolfo_plain_on_g2_plain_seed{s}_00001_.png",
                                      f"vpred_astolfo_ref_ref_seed{s}_00001_.png") for s in S],
        "M7_vs_a_as_comfyui": [same(f"vpred_astolfo_plain_on_g2_plain_seed{s}_00001_.png",
                                    f"vpred_astolfo_plain_plain_seed{s}_00001_.png") for s in S],
        "M7_lines": lines("astolfo_plain_on_g2", "[entail] resolved"),
        "seconds": {t: [r.get("seconds") for r in next(v for k, v in json.load(open(f"vpred_{t}.json")).items()
                                                      if isinstance(v, list))]
                    for t in ("stripped_on_g2", "leftover_first_on_g2", "noob_native_node_on_g2", "astolfo_plain_on_g2")},
    }
    json.dump(out, open(os.path.join(h.HERE, "g2_compare.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
