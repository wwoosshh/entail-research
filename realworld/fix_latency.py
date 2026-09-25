"""Time from report to close and discussion length, role-class vs other root-caused bugs (round-1 labels)."""
import json
import os
import statistics
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def gh(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, encoding="utf-8")
    return json.loads(out.stdout)


def days(a, b):
    f = "%Y-%m-%dT%H:%M:%SZ"
    return (datetime.strptime(b, f) - datetime.strptime(a, f)).total_seconds() / 86400


def main():
    rows = json.load(open(os.path.join(HERE, "bug_labels_round1.json")))
    for r in rows:
        d = gh(f"repos/{r['repo']}/issues/{r['issue']}")
        r["created_at"], r["closed_at"], r["comments"] = d["created_at"], d["closed_at"], d["comments"]
        r["days_open"] = days(d["created_at"], d["closed_at"])
    json.dump(rows, open(os.path.join(HERE, "bug_labels_round1_timing.json"), "w"), indent=1)

    def summary(sel):
        ds = sorted(r["days_open"] for r in sel)
        cs = sorted(r["comments"] for r in sel)
        return {"n": len(sel), "median_days": round(statistics.median(ds), 1),
                "p75_days": round(ds[int(0.75 * (len(ds) - 1))], 1), "median_comments": statistics.median(cs)}

    role = [r for r in rows if r["category"].startswith("R")]
    other = [r for r in rows if r["category"] in ("N1", "N2", "N3", "N5")]
    n4 = [r for r in rows if r["category"] == "N4"]
    res = {"role_class": summary(role), "other_root_caused": summary(other), "no_fix_N4": summary(n4)}
    for key in ("vllm", "sglang", "transformers", "llamacpp"):
        res[f"role_{key}"] = summary([r for r in role if r["key"] == key])
        o = [r for r in other if r["key"] == key]
        if o:
            res[f"other_{key}"] = summary(o)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "fix_latency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
