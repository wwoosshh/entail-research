"""Product track P1 measurement (ROADMAP "제품 트랙" P1; LIBRARY_DESIGN.md 13.3, 13.8): the node model on the record
files the research already has.
  1. every boundary name in the workspace's record files -> its node; names that fall to "other" are listed
  2. the E2 healthy runs of the M19 L4 freeze (testbed/results/m19/l4/e2_final/engines, one record file per run):
     per run, the nodes whose state is broken or refused. The completion criterion: no such node, apart from the
     differences confirmed real (the tokenizer of the two Llama-2-era folders, transformers#47700)
Run: python testbed/p1_nodes.py   (writes testbed/results/p1/summary.json)
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "entail"))
from entail.platform import graph  # noqa: E402

CONFIRMED_REAL = ("TinyLlama", "tiny-random-Llama")    # testbed/results/m19/l4/tokenizer_truth.txt


def all_record_files():
    pats = ["testbed/results/**/*.jsonl", "lowlevel/**/*.jsonl"]
    return sorted({p for pat in pats for p in glob.glob(os.path.join(ROOT, pat), recursive=True)})


def main():
    out = {}
    files = all_record_files()
    names = collections.Counter()
    for _, obj in graph.read_lines(files):
        b = obj.get("boundary") if "verdict" in obj else obj.get("timing")
        if b:
            names[b] += 1
        elif isinstance(obj.get("boundaries"), dict):
            for bb in obj["boundaries"]:
                names[bb] += 0
    per_node = collections.Counter(graph.node_of(b) for b in names)
    # overlaps: names that more than one node's patterns match (first match wins; the catch-all "other" not counted)
    overlaps = {}
    for b in names:
        hits = [n["id"] for n in graph.model()["nodes"] if n["id"] != "other" and any(p.match(b) for p in n["compiled"])]
        if len(hits) > 1:
            overlaps[b] = hits
    out["overlaps"] = overlaps
    print(f"   names matched by more than one node: {len(overlaps)} {dict(list(overlaps.items())[:8])}")
    other = sorted(b for b in names if graph.node_of(b) == "other")
    out["boundaries"] = {"files": len(files), "names": len(names), "per_node": dict(per_node), "other": other}
    print(f"1. {len(files)} record files, {len(names)} boundary names; per node: {dict(per_node)}")
    print(f"   names in 'other': {len(other)} {other[:10]}")

    e2 = sorted(glob.glob(os.path.join(ROOT, "testbed/results/m19/l4/e2_final/engines/*.record.jsonl")))
    runs, flagged = [], []
    for path in e2:
        groups = graph.launches(graph.read_lines([path]))
        for key, lines in groups.items():
            g = graph.graph(lines)
            bad = [(n["id"], n["state"], n["boundaries"]) for n in g["nodes"] if n["state"] in ("broken", "refused")]
            name = os.path.basename(path).replace(".record.jsonl", "")
            real = any(x in name for x in CONFIRMED_REAL)
            runs.append({"run": name, "decisions": g["decisions"], "state": g["state"], "bad_nodes": bad,
                         "confirmed_real": real})
            if bad:
                flagged.append((name, bad, real))
    wrong = [f for f in flagged if not (f[2] and all(nid == "tokenizer" for nid, _, _ in f[1]))]
    # what the E2 runs are made of, and why unknown and unchecked are the common worst states
    engines, flows_seen, why_states = collections.Counter(), collections.Counter(), collections.Counter()
    for path in e2:
        for key, lines in graph.launches(graph.read_lines([path])).items():
            g = graph.graph(lines)
            engines[",".join(g["engines"]) or "-"] += 1
            flows_seen[",".join(f["id"] for f in g["flows"])] += 1
            for n in g["nodes"]:
                if n["state"] in ("unknown", "unchecked"):
                    why_states[(n["id"], n["state"])] += 1
    out["e2_makeup"] = {"engines": dict(engines), "flows": dict(flows_seen),
                        "unknown_unchecked_nodes": {f"{a}/{b}": c for (a, b), c in why_states.most_common()}}
    print(f"   engines per run: {dict(engines)}; flows: {dict(flows_seen)}")
    print(f"   nodes that are unknown/unchecked (runs): {dict(why_states.most_common(12))}")
    states = collections.Counter(r["state"] for r in runs)
    out["e2"] = {"runs": len(runs), "states": dict(states), "flagged": [
        {"run": n, "nodes": [[a, b] for a, b, _ in bad], "confirmed_real": real} for n, bad, real in flagged],
        "wrong_nodes_runs": len(wrong)}
    print(f"2. E2 final: {len(runs)} runs; worst state per run: {dict(states)}")
    for n, bad, real in flagged:
        print(f"   {'real' if real else 'FLAG'} {n}: {[(a, b) for a, b, _ in bad]}")
    print(f"   runs with a broken/refused node not explained by the confirmed real differences: {len(wrong)}")
    os.makedirs(os.path.join(ROOT, "testbed/results/p1"), exist_ok=True)
    with open(os.path.join(ROOT, "testbed/results/p1/summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
