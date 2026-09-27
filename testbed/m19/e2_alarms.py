"""M19 L4: broken/refused (and, with -u, unknown) decisions of the new adapters across an E2 folder's records, by
run, boundary and fact, with the first note. Run: python testbed/m19/e2_alarms.py <E2 dir> [-u]"""
import collections
import glob
import json
import os
import sys

E = os.path.join(sys.argv[1], "engines")
want = {"broken", "refused"} | ({"unknown"} if "-u" in sys.argv else set())
runs = sorted(glob.glob(os.path.join(E, "*_load.record.jsonl")))
tally = collections.Counter()
print(f"{len(runs)} records")
for p in runs:
    seen = {}
    for line in open(p, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        v = d.get("verdict")
        if v:
            tally[v] += 1
        if v in want:
            key = (d.get("boundary"), d.get("name"), v)
            seen.setdefault(key, (d.get("note") or "")[:300])
    for (b, n, v), note in seen.items():
        print(f"{os.path.basename(p)[3:-19]:<60} {v:<8} {b} {n}\n    {note}")
print(dict(tally))
