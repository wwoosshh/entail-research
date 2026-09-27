"""M19 L4 cost summary (testbed/m19_cost.sh): per engine and model, the median load time of each state over the
repetitions, and its difference from off (seconds and share of off). Also the entail decisions of the 'all' runs.
Run: python testbed/m19_cost_summary.py [folder, default testbed/results/m19/l4/cost]"""
import collections
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "m19", "l4", "cost")
loads = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(os.path.join(OUT, "run.log"), encoding="utf-8"):
    m = re.match(r"^(\S+) (\S+) rep(\d+) load=([0-9.]+)", line.strip())
    if m:
        loads[m.group(1)]["off" if m.group(2).endswith("off") else m.group(2)].append(float(m.group(4)))
summary = {}
for run, states in sorted(loads.items()):
    off = statistics.median(states["off"])
    row = {"off_median_s": off, "states": {}}
    for s, xs in states.items():
        med = statistics.median(xs)
        row["states"][s] = {"loads": xs, "median_s": med, "minus_off_s": med - off, "share_of_off": (med - off) / off}
    summary[run] = row
    print(run, f"off {off:.2f}s", " ".join(f"{s}:{v['minus_off_s']:+.2f}s({v['share_of_off']:+.1%})"
                                           for s, v in row["states"].items() if s != "off"))
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1)
