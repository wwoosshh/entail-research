"""Compare the configurations of PROTOCOL.md. Run: python summarize.py  (reads results/*.json, writes SUMMARY.json)"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def rope_warnings(name):
    path = os.path.join(RES, f"{name}.log")
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    return [ln[:200] for ln in lines if re.search(r"WARNING|ERROR", ln) and "rope" in ln.lower()]


def main():
    rows = {}
    for f in sorted(os.listdir(RES)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(RES, f), encoding="utf-8"))
            rows[r["config"]] = r
    out = {"configs": {}, "pairs": {}}
    for name, r in rows.items():
        g = r["gsm8k"]
        k = sum(g["correct"])
        out["configs"][name] = {"engine": r.get("engine", "vllm"), "rope": r.get("rope_parameters_in_engine"),
                                "nll": {d: round(v["mean_nll"], 4) for d, v in r["nll"].items()},
                                "gsm8k": f"{k}/{g['n']}", "accuracy": round(k / g["n"], 4),
                                "wilson95": wilson(k, g["n"]), "rope_warnings": rope_warnings(name)}
    pairs = [(a, b) for a, b in (("B", "C"), ("C", "E"), ("A", "D"), ("A", "B"), ("B", "E"), ("A", "A2"),
                                 ("B", "C_rolecheck"), ("C", "C_rolecheck"), ("LA", "LA2"), ("LA", "LC"),
                                 ("LC", "LE"), ("LA", "LE"), ("LA", "LC_rolecheck"), ("SA", "SA2"), ("SA", "SC"),
                                 ("SC", "SE"), ("SA", "SE"), ("SA", "SC_rolecheck"), ("SA", "SC_entail"), ("SA_now", "SC_entail"), ("SA", "SA_now"), ("SA_now", "SC_pip")) if a in rows and b in rows]
    for a, b in pairs:
        ga, gb = rows[a]["gsm8k"], rows[b]["gsm8k"]
        same = sum(x == y for x, y in zip(ga["outputs"], gb["outputs"]))
        only_a = sum(1 for x, y in zip(ga["correct"], gb["correct"]) if x and not y)
        only_b = sum(1 for x, y in zip(ga["correct"], gb["correct"]) if y and not x)
        out["pairs"][f"{a} vs {b}"] = {
            "identical_outputs": f"{same}/{len(ga['outputs'])}",
            "accuracy_diff": round((sum(gb["correct"]) - sum(ga["correct"])) / len(ga["correct"]), 4),
            f"only_{a}_correct": only_a, f"only_{b}_correct": only_b,
            "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 6),
            "nll_diff": {d: round(rows[b]["nll"][d]["mean_nll"] - rows[a]["nll"][d]["mean_nll"], 4)
                         for d in rows[a]["nll"]}}
    json.dump(out, open(os.path.join(HERE, "SUMMARY.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
