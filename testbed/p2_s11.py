"""Product track P2 measurement, S11 (LIBRARY_DESIGN.md 8, 13.5): does the platform point at the place meaning broke
and say why, on the workflows where a defect was planted?

The planted defects are M7.3's (testbed/results/m73: 11 scenarios on Qwen3-4B and gemma-2-2b-it, transformers 5.17 -
a boundary, a code boundary, the KV cache, an operation on the way, inside a layer, an unchecked boundary, and a
healthy run). Each scenario's record file (logs/<scenario>/record-*.jsonl) is what the platform reads; the answer to
match is the localization the program made in process (the scenario's <name>.json "located", which M7.3 checked
against where the defect was planted: 11 of 11).

The platform's answer, as the page shows it (static/app.js, the banner):
  broken_at of the records' locate  -> that boundary, its node shown broken
  else lost on the way              -> the operation named
  else the in-process localization recorded in the records ("located" lines) -> its first suspect
  else every checked boundary held
Located = the same boundary or suspect as the in-process answer, and the node that holds a broken boundary is in
state broken with its decision's rule and note in the node detail.
Run: python testbed/p2_s11.py   (writes testbed/results/p2/s11.json)
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "entail"))
from entail.platform import graph  # noqa: E402

M73 = os.path.join(ROOT, "testbed", "results", "m73")


def platform_answer(g):
    loc = g["locate"]
    if loc["broken_at"]:
        return "broken", loc["broken_at"]
    if loc["lost_by"]:
        return "lost", loc["suspects"][0] if loc["suspects"] else loc["lost_by"][0]
    if g.get("located") and g["located"].get("suspects"):
        return "located", g["located"]["suspects"][0]
    if loc["all_intact"]:
        return "intact", None
    return "none", None


def main():
    summary = json.load(open(os.path.join(M73, "SUMMARY.json"), encoding="utf-8"))
    rows, ok = [], 0
    for name in summary["scenarios"]:
        path = os.path.join(M73, f"{name}.json")
        files = sorted(glob.glob(os.path.join(M73, "logs", name, "record-*.jsonl")))
        if not os.path.exists(path) or not files:
            continue
        want = json.load(open(path, encoding="utf-8")).get("located") or {}
        groups = graph.launches(graph.read_lines(files))
        lines = [o for ls in groups.values() for o in ls]
        g = graph.graph(lines)
        kind, answer = platform_answer(g)
        if want.get("broken_at"):
            expected = want["broken_at"]
        elif want.get("suspects"):
            expected = want["suspects"][0]
        else:
            expected = None
        match = answer == expected
        node_ok = True
        why = None
        if kind == "broken":
            nid = graph.node_of(answer)
            node = next(n for n in g["nodes"] if n["id"] == nid)
            detail = graph.node_detail(lines, nid)
            dec = next((d for d in detail["decisions"] if d.get("boundary") == answer
                        and d.get("verdict") in ("broken", "refused")), None)
            node_ok = node["state"] in ("broken", "refused") and dec is not None
            why = dec and (dec.get("rule"), dec.get("note"))
            nid_shown = nid
        else:
            nid_shown = None
        good = match and node_ok
        ok += good
        rows.append({"scenario": name, "expected": expected, "platform": [kind, answer], "node": nid_shown,
                     "why": why, "located": good})
        print(f"{'ok ' if good else 'MISS'} {name:<17} platform {kind:<8} {answer!s:<60} node {nid_shown}")
    print(f"S11: {ok} of {len(rows)} scenarios located as the program located them in process")
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "testbed", "results", "p2", "s11.json")   # P6: again
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"located": ok, "of": len(rows), "rows": rows}, open(out, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
