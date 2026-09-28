"""P5 external evaluation, items 4 and 3: heavy validators against the 50 ms budget, a hard validator's cost, the cost of
looking packages up when many are installed, and whether the core keeps every finding of a validator (many planted
faults per validator of the workshop package, not one).

  python testbed/p5_stress.py
Writes testbed/results/p5/stress.json.
"""
import json
import os
import re
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "entail")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "workshop", "basics"))
OUT = os.path.join(HERE, "results", "p5")
os.environ["ENTAIL_LOG_DIR"] = os.path.join(OUT, "stress_logs")
os.makedirs(os.environ["ENTAIL_LOG_DIR"], exist_ok=True)
from entail import core, dlc, nodes, record, tally  # noqa: E402
import entail_nodes_basics as basics  # noqa: E402


def median_ms(fn, n=5):
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000.0)
    return round(statistics.median(xs), 3)


def heavy():
    obj = {"items": [{"id": i, "text": "x" * 40} for i in range(20000)]}
    small, big = json.dumps(obj), json.dumps({"items": obj["items"] * 10})
    text = ("word " * 2_000_000)
    cases = {
        "json_1MB": (lambda v: json.loads(v) is not None, small),
        "json_10MB": (lambda v: json.loads(v) is not None, big),
        "regex_10MB": (lambda v: len(re.findall(r"wor.", v)) > 0, text),
        "wait_100ms": (lambda v: time.sleep(0.1) or True, None),
    }
    out = {}
    for name, (fn, value) in cases.items():
        alone = median_ms(lambda: fn(value), 3)
        nodes.reset()
        tally.reset()
        core.set_mode("load")
        nodes.check("stress.node", value, nodes.validator(name.lower().replace("_", "-"))(fn))
        left = any(k.endswith("/" + name.lower().replace("_", "-")) for k in nodes._LEFT_OUT)
        out[name] = {"size_chars": len(value) if isinstance(value, str) else None, "ms": alone,
                     "left_out_at_50ms": left}
    core.set_mode("off")
    return out


def hard_cost():
    core.set_mode("load")
    nodes.reset()

    @nodes.validator("plain")
    def plain(v):
        return True

    @nodes.validator("threaded", hard=True)
    def threaded(v):
        return True

    res = {}
    for name, fn in (("plain", plain), ("hard", threaded)):
        n = 2000
        t = time.perf_counter()
        for _ in range(n):
            nodes.check("stress.hard", 1, fn)
        res[name + "_us_per_call"] = round((time.perf_counter() - t) / n * 1e6, 2)
    core.set_mode("off")
    return res


def lookup_cost():
    """nodes.packages() and dlc.found() with 0, 10 and 50 stand-in packages of each kind on sys.path."""
    res = {}
    keep = list(sys.path)
    made = 0
    for count in (0, 10, 50):
        while made < count:
            root = tempfile.mkdtemp()
            for kind, group, text in (("n", "entail.nodes", "name='p{i}'\nversion='0.1'\nrequires='>=1.3'\n"
                                                             "validators={{'v{i}': lambda v: True}}\n"),
                                      ("d", "entail.dlc", "name='p{i}'\nversion='0.1'\nrequires='>=1.3'\n"
                                                          "targets={{}}\n")):
                mod = f"stress_{kind}{made}"
                with open(os.path.join(root, mod + ".py"), "w", encoding="utf-8") as f:
                    f.write(text.format(i=made))
                info = os.path.join(root, f"{mod}-0.1.dist-info")
                os.makedirs(info)
                with open(os.path.join(info, "METADATA"), "w", encoding="utf-8") as f:
                    f.write(f"Metadata-Version: 2.1\nName: {mod}\nVersion: 0.1\n")
                with open(os.path.join(info, "entry_points.txt"), "w", encoding="utf-8") as f:
                    f.write(f"[{group}]\np{made} = {mod}\n")
            sys.path.insert(0, root)
            made += 1
        res[count] = {"nodes_packages_ms": median_ms(lambda: nodes.packages(refresh=True)),
                      "dlc_found_ms": median_ms(lambda: dlc.found(refresh=True)),
                      "attached": [sum(1 for p in nodes.packages() if p["attached"]),
                                   sum(1 for d in dlc.found() if d["attached"])]}
    sys.path[:] = keep
    nodes.reset()
    dlc.reset()
    return res


def findings():
    """Many faults per validator: every fault the validator was written to catch must come out as broken (counted),
    each distinct finding recorded once; every good value counted as passed."""
    core.set_mode("load")
    nodes.reset()
    tally.reset()
    good_json = ['{"title": "t", "body": "b"}', '{"title": "", "body": "x", "extra": 1}']
    bad_json = ['{"title": "t", "body": "b"', '{"title": "t"}', '{"body": "b"}', '[1, 2]', '3', '""', '',
                '{"title": "t", "body": "b"}}', "{'title': 't', 'body': 'b'}", '{"title": "t", "bod": "b"}']
    cases = {
        "json_object": ([(v, {"keys": ("title", "body")}) for v in good_json],
                        [(v, {"keys": ("title", "body")}) for v in bad_json]),
        "within_context": ([(n, {"context": 256, "reserve": 64}) for n in (0, 100, 192)],
                           [(n, {"context": 256, "reserve": 64}) for n in (193, 200, 256, 1000, 10**6)]),
        "same_size": ([((w, h), {"expected": (w, h)}) for w, h in ((832, 1216), (1024, 1024), (1, 1))],
                      [((w, h), {"expected": (832, 1216)}) for w, h in ((832, 1152), (768, 1216), (1216, 832),
                                                                        (0, 0), (833, 1216))]),
        "max_chars": ([("a" * n, {"limit": 10}) for n in (0, 5, 10)],
                      [("a" * n, {"limit": 10}) for n in (11, 12, 100, 10**5)]),
    }
    out = {}
    for name, (good, bad) in cases.items():
        fn = basics.validators[name]
        node = f"find.{name.replace('_', '-')}"
        for v, p in good + bad:
            nodes.check(node, v, fn, **p)
        s = tally.stats(f"node:{node}/{name.replace('_', '_')}") if False else None
        key = [b for b in tally.STATS if b.startswith(f"node:{node}/")][0]
        s = tally.stats(key)
        out[name] = {"good": len(good), "faults": len(bad), "checks": s["checks"], "broken": s["broken"],
                     "passed": sum(s["passed"].values()), "all_faults_broken": s["broken"] == len(bad),
                     "all_good_passed": sum(s["passed"].values()) == len(good)}
    record.close_files()
    core.set_mode("off")
    return out


def main():
    res = {"heavy_validators": heavy(), "hard_cost": hard_cost(), "lookup_cost": lookup_cost(), "findings": findings()}
    with open(os.path.join(OUT, "stress.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
