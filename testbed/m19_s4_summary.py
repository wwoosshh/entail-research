"""M19 L4, S4 summary (testbed/m19_s4.py): per batch size, the median over rounds of on/off and control/off, where
each process contributes the median of its timed runs, and whether the tokens agree across states.
Run: python testbed/m19_s4_summary.py [result folder, default testbed/results/m19/l4/s4]"""
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "m19", "l4", "s4")
runs = {}
for p in glob.glob(os.path.join(OUT, "*_*.json")):
    d = json.load(open(p, encoding="utf-8"))
    runs.setdefault(d["round"], {})[d["state"]] = d
rounds = sorted(r for r in runs if {"on", "off", "control"} <= set(runs[r]))
summary = {"rounds": len(rounds), "batches": {}}
for b in ("1", "8", "32"):
    on = [runs[r]["on"]["batches"][b]["median"] / runs[r]["off"]["batches"][b]["median"] for r in rounds]
    ctl = [runs[r]["control"]["batches"][b]["median"] / runs[r]["off"]["batches"][b]["median"] for r in rounds]
    same = all(runs[r][s]["batches"][b]["tokens"] == runs[rounds[0]]["off"]["batches"][b]["tokens"]
               for r in rounds for s in ("on", "off", "control"))
    summary["batches"][b] = {"on_over_off": on, "control_over_off": ctl, "median_on_over_off": statistics.median(on),
                             "median_control_over_off": statistics.median(ctl), "same_tokens": same}
    print(f"B={b:>3}  on/off {statistics.median(on):.4f}  control/off {statistics.median(ctl):.4f}  "
          f"(rounds {len(rounds)})  same tokens {same}")
summary["load_seconds"] = {s: [runs[r][s]["load_seconds"] for r in rounds] for s in ("on", "off", "control")}
summary["entail_imported"] = {s: [runs[r][s]["entail_imported"] for r in rounds] for s in ("on", "off", "control")}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1)
