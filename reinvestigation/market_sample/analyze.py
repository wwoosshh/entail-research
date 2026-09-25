"""Aggregate the blind ratings exactly as CODEBOOK_R.md section 5 defines (written before any case was seen).

Inputs: cases_*.json (extraction records), ratings_A.json and ratings_B.json (one object per case_id).
Output: analysis.json and a printed summary.
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = list("ABCDEFGHIJ")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 3), round(c - h, 3), round(c + h, 3)]


def kappa(pairs, cats):
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def narrow(r):
    return r["letter"] in ("B", "C")


def broad(r):
    return r["letter"] in ("B", "C", "D") or (r["letter"] == "F" and r.get("flag_declared_overridden"))


cases = {c["case_id"]: c for p in sorted(glob.glob(os.path.join(HERE, "cases_*.json"))) for c in load(p)}
ra = {r["case_id"]: r for r in load(os.path.join(HERE, "ratings_A.json"))}
rb = {r["case_id"]: r for r in load(os.path.join(HERE, "ratings_B.json"))}
common = sorted(set(ra) & set(rb) & set(cases))

out = {"cases_total": len(cases), "rated_by_both": len(common)}
out["kappa_eligible"] = kappa([(ra[i]["eligible"], rb[i]["eligible"]) for i in common], ["yes", "no"])

for group, prefixes in (("all", ("L", "I", "S")), ("llm", ("L",)), ("image_random", ("I",)),
                        ("image_fix_linked", ("S",)), ("image_pooled", ("I", "S"))):
    ids = [i for i in common if i.startswith(prefixes)]
    both = [i for i in ids if ra[i]["eligible"] == "yes" and rb[i]["eligible"] == "yes"]
    g = {"cases": len(ids), "eligible_both": len(both)}
    g["kappa_letter"] = kappa([(ra[i]["letter"], rb[i]["letter"]) for i in both], LETTERS)
    g["kappa_binary_BCD"] = kappa([(ra[i]["letter"] in "BCD", rb[i]["letter"] in "BCD") for i in both], [True, False])
    for name, pred in (("narrow_BC", narrow), ("broad_BCD_F", broad)):
        strict = [i for i in both if pred(ra[i]) and pred(rb[i])]
        lenient = [i for i in both if pred(ra[i]) or pred(rb[i])]
        silent_strict = [i for i in strict if cases[i].get("message_at_failure") == "none"]
        g[name] = {
            "strict": len(strict), "strict_share_ci": wilson(len(strict), len(both)),
            "lenient": len(lenient), "lenient_share_ci": wilson(len(lenient), len(both)),
            "silent_within_strict": len(silent_strict), "silent_within_strict_ci": wilson(len(silent_strict), len(strict)),
        }
    g["silent_all_eligible"] = wilson(sum(cases[i].get("message_at_failure") == "none" for i in both), len(both))
    known = [i for i in both if cases[i].get("message_at_failure") in ("none", "warning", "error")]
    g["silent_known_only"] = wilson(sum(cases[i]["message_at_failure"] == "none" for i in known), len(known))
    g["message_counts"] = {m: sum(cases[i].get("message_at_failure") == m for i in both)
                           for m in ("none", "warning", "error", "unknown")}
    for name, pred in (("narrow_BC", narrow), ("broad_BCD_F", broad)):
        strict = [i for i in both if pred(ra[i]) and pred(rb[i])]
        k = [i for i in strict if cases[i].get("message_at_failure") in ("none", "warning", "error")]
        g[name]["silent_known_only_within_strict"] = wilson(sum(cases[i]["message_at_failure"] == "none" for i in k), len(k))
    g["letters_A"] = {c: sum(ra[i]["letter"] == c for i in both) for c in LETTERS}
    g["letters_B"] = {c: sum(rb[i]["letter"] == c for i in both) for c in LETTERS}
    out[group] = g

with open(os.path.join(HERE, "analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
