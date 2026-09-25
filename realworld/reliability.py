"""Agreement between round-1 labels (hypothesis-aware) and blind re-classification (neutral letters) on 48 issues."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LETTER = {"A": "N1", "B": "R1", "C": "N2", "D": "R2", "E": "N3", "F": "R3", "G": "R4", "H": "N4", "I": "N5"}


def group(c):
    return "ROLE" if c.startswith("R") else ("UNRESOLVED" if c in ("N4", "N5") else "OTHER")


def kappa(pairs, classes):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == k for a, _ in pairs) / n) * (sum(b == k for _, b in pairs) / n) for k in classes)
    return po, (po - pe) / (1 - pe)


def main():
    r1 = {(r["repo"], r["issue"]): r["category"] for r in json.load(open(os.path.join(HERE, "bug_labels_round1.json")))}
    blind = {}
    for f in ("blind_rater_A.md", "blind_rater_B.md"):
        for line in open(os.path.join(HERE, f), encoding="utf-8"):
            m = re.match(r"^\|\s*([\w.-]+/[\w.-]+)\s*\|\s*(\d+)\s*\|\s*([A-I])\s*\|", line)
            if m:
                blind[(m.group(1), int(m.group(2)))] = LETTER[m.group(3)]
    pairs = [(r1[k], blind[k]) for k in blind]
    print("issues compared:", len(pairs))
    po9, k9 = kappa(pairs, sorted(set(LETTER.values())))
    g = [(group(a), group(b)) for a, b in pairs]
    po3, k3 = kappa(g, ["ROLE", "OTHER", "UNRESOLVED"])
    both = [(a, b) for a, b in g if a != "UNRESOLVED" and b != "UNRESOLVED"]
    po2, k2 = kappa(both, ["ROLE", "OTHER"])
    res = {"n": len(pairs), "exact_9class_agreement": po9, "kappa_9class": k9,
           "agreement_3class": po3, "kappa_3class": k3,
           "binary_on_both_root_caused": {"n": len(both), "agreement": po2, "kappa": k2},
           "role_share_round1": sum(a == "ROLE" for a, _ in g) / sum(a != "UNRESOLVED" for a, _ in g),
           "role_share_blind": sum(b == "ROLE" for _, b in g) / sum(b != "UNRESOLVED" for _, b in g),
           "disagreements": [{"issue": f"{k[0]}#{k[1]}", "round1": r1[k], "blind": blind[k]}
                             for k in blind if r1[k] != blind[k]]}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "reliability.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
