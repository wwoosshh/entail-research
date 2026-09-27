"""Agreement of the blind level ratings (lowlevel/blind/ratings_A.txt, ratings_B.txt, ratings_C.txt if present)
with each other and with the session's own labels (lowlevel/level_labels.json), and the consensus counts that
replace the session's judgment in L1_ANALYSIS.md.
Run: python lowlevel/blind/agreement.py  ->  lowlevel/blind/agreement.json (and disagreements.md for a third rater)
"""
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r"^\s*(X\d{3})\s*\|\s*([DIU])\s*\|\s*([MRTLFN ]*)\|\s*([YNU])\s*\|\s*([123])\s*\|\s*(.*)$")


def read(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, encoding="utf-8"):
        m = LINE.match(line.strip())
        if m:
            out[m.group(1)] = {"q1": m.group(2), "q2": set(m.group(3).split()), "q3": m.group(4),
                               "conf": int(m.group(5)), "why": m.group(6).strip()}
    return out


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return {"n": n, "observed": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None}


def main():
    key = {x["id"]: x["issue"] for x in json.load(open(os.path.join(HERE, "packet_key.json"), encoding="utf-8"))}
    mine = {r["issue"]: r for r in json.load(open(os.path.join(HERE, "..", "level_labels.json"), encoding="utf-8"))}
    A, B, C = read("ratings_A.txt"), read("ratings_B.txt"), read("ratings_C.txt")
    ids = sorted(key)
    missing = {"A": [i for i in ids if i not in A], "B": [i for i in ids if i not in B]}
    both = [i for i in ids if i in A and i in B]
    res = {"items": len(ids), "rated_by_both": len(both), "missing": missing,
           "q1_kappa": kappa([(A[i]["q1"], B[i]["q1"]) for i in both]),
           "q3_kappa": kappa([(A[i]["q3"], B[i]["q3"]) for i in both]),
           "q2_letter_kappa": {L: kappa([(L in A[i]["q2"], L in B[i]["q2"]) for i in both]) for L in "MRTLFN"}}
    # consensus: A and B agree, else C (when rated), else unresolved
    cons = {}
    for i in both:
        a, b = A[i], B[i]
        if a["q1"] == b["q1"] and a["q3"] == b["q3"]:
            q1, q3 = a["q1"], a["q3"]
        elif C and i in C:
            q1 = a["q1"] if a["q1"] == b["q1"] else (C[i]["q1"] if C[i]["q1"] in (a["q1"], b["q1"]) else "split")
            q3 = a["q3"] if a["q3"] == b["q3"] else (C[i]["q3"] if C[i]["q3"] in (a["q3"], b["q3"]) else "split")
        else:
            q1 = a["q1"] if a["q1"] == b["q1"] else "open"
            q3 = a["q3"] if a["q3"] == b["q3"] else "open"
        # a check counts when both raters chose it (or the third agreed with one of them)
        q2 = set()
        for L in "MRTLF":
            votes = [L in a["q2"], L in b["q2"]] + ([L in C[i]["q2"]] if C and i in C else [])
            if sum(votes) * 2 > len(votes):
                q2.add(L)
        cons[i] = {"issue": key[i], "q1": q1, "q3": q3, "q2": sorted(q2)}
    res["consensus_q1"] = dict(Counter(c["q1"] for c in cons.values()))
    D = [c for c in cons.values() if c["q1"] == "D"]
    res["consensus_D"] = {"n": len(D), "exposed_by_any_check": sum(1 for c in D if c["q2"]),
                          "by_check": {L: sum(1 for c in D if L in c["q2"]) for L in "MRTLF"},
                          "one_gpu_Y": sum(1 for c in D if c["q3"] == "Y")}
    lvl = {"low": "D", "high": "I"}
    res["session_vs_consensus_q1"] = kappa([(lvl[mine[c["issue"]]["level"]], c["q1"]) for c in cons.values()
                                            if c["q1"] in ("D", "I")])
    res["session_vs_A_q1"] = kappa([(lvl[mine[key[i]]["level"]], A[i]["q1"]) for i in both])
    res["session_vs_B_q1"] = kappa([(lvl[mine[key[i]]["level"]], B[i]["q1"]) for i in both])
    # K1 (the class) by blind level
    k1 = [c for c in cons.values() if mine[c["issue"]]["consensus"] == "K1"]
    res["K1_by_consensus_q1"] = dict(Counter(c["q1"] for c in k1))
    dis = [i for i in both if A[i]["q1"] != B[i]["q1"] or A[i]["q3"] != B[i]["q3"]]
    res["disagreements"] = len(dis)
    json.dump({"summary": res, "consensus": cons}, open(os.path.join(HERE, "agreement.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if not C:
        packet = open(os.path.join(HERE, "packet.md"), encoding="utf-8").read()
        blocks = {m.group(1): m.group(0) for m in re.finditer(r"## (X\d{3}) .*?(?=\n## X|\Z)", packet, re.S)}
        out = ["# Items", "", "Each item: an id, the software project, the issue title, and two short descriptions "
               "written earlier by two", "independent readers of the issue and its fix. Rate every item. Do not look "
               "anything up.", ""]
        out += [blocks[i].strip() + "\n" for i in dis if i in blocks]
        open(os.path.join(HERE, "packet_C.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
