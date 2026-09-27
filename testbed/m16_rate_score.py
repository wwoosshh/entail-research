"""M16.3 agreement (testbed/M16_PROTOCOL.md 4): reads the two blind raters' files (results/m16/ratings_A.txt,
ratings_B.txt; lines "<url> | K1..K7 | confidence | evidence"), reports Cohen's kappa over the seven categories and
over K1-vs-rest, lists the disagreements (for the third, blind rater), and writes results/m16/ratings.json with the
per-issue ratings and the consensus where the two agree. Run: python testbed/m16_rate_score.py [ratings_C.txt]
The fourth replay (M16_PROTOCOL 9) adds a level after the category ("<url> | K1..K7 | D|I|U | confidence | evidence",
the L1 blind codebook's Q1); it is optional, and an item goes to the third rater when either answer differs.
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("M16_OUT") or os.path.join(HERE, "results", "m16")   # M17.6: results/m17/replay2
LINE = re.compile(r"^\s*(https?://\S+)\s*\|\s*(K[1-7])\s*\|\s*(?:([DIU])\s*\|\s*)?([123])\s*\|\s*(.*)$")


def read(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        m = LINE.match(line.rstrip("\n"))
        if m:
            rows[m.group(1).rstrip("/")] = {"k": m.group(2), "level": m.group(3), "conf": int(m.group(4)),
                                            "evidence": m.group(5).strip()}
    return rows


def kappa(a, b, keys, collapse=None, field="k"):
    f = (lambda k: ("K1" if k == "K1" else "other")) if collapse else (lambda k: k)
    xs = [(f(a[u][field]), f(b[u][field])) for u in keys]
    n = len(xs)
    po = sum(1 for x, y in xs if x == y) / n
    ca, cb = Counter(x for x, _ in xs), Counter(y for _, y in xs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return {"n": n, "observed": po, "expected": pe, "kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0}


def main():
    a, b = read(os.path.join(OUT, "ratings_A.txt")), read(os.path.join(OUT, "ratings_B.txt"))
    c = read(sys.argv[1]) if len(sys.argv) > 1 else {}
    packet = json.load(open(os.path.join(OUT, "rating_packet.json"), encoding="utf-8"))
    urls = [it["url"].rstrip("/") for it in packet["items"]]
    missing = [u for u in urls if u not in a or u not in b]
    keys = [u for u in urls if u in a and u in b]
    out = {"count": len(urls), "rated_by_both": len(keys), "missing": missing,
           "kappa_7": kappa(a, b, keys), "kappa_k1_vs_rest": kappa(a, b, keys, collapse=True),
           "counts_A": dict(Counter(a[u]["k"] for u in keys)), "counts_B": dict(Counter(b[u]["k"] for u in keys)),
           "issues": {}}
    leveled = [u for u in keys if a[u]["level"] and b[u]["level"]]
    if leveled:
        out["kappa_level"] = kappa(a, b, leveled, field="level")

    def agreed(ra, rb, rc, field):
        if ra and rb and ra[field] == rb[field]:
            return ra[field], False
        if rc and rc.get(field):
            return (rc[field] if rc[field] in {ra[field] if ra else None, rb[field] if rb else None} else "split"), False
        return None, bool(ra and rb and ra[field] and rb[field])

    disagreements = []
    for u in urls:
        ra, rb, rc = a.get(u), b.get(u), c.get(u)
        row = {"A": ra, "B": rb, "C": rc}
        row["consensus"], open_k = agreed(ra, rb, rc, "k")
        row["level"], open_level = agreed(ra, rb, rc, "level")
        if open_k or open_level:
            disagreements.append(u)
        out["issues"][u] = row
    out["disagreements"] = disagreements
    out["consensus_counts"] = dict(Counter(r["consensus"] for r in out["issues"].values()))
    if leveled:
        out["level_counts"] = dict(Counter(r["level"] for r in out["issues"].values()))
    json.dump(out, open(os.path.join(OUT, "ratings.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "issues"}, ensure_ascii=False, indent=1))
    if disagreements:
        with open(os.path.join(OUT, "rating_disagreements.md"), "w", encoding="utf-8") as f:
            f.write("# Items to rate\n\nRate each item by the codebook; the packet item is quoted by URL.\n\n")
            for u in disagreements:
                it = next(i for i in packet["items"] if i["url"].rstrip("/") == u)
                f.write(f"## Item {it['n']}\nURL: {it['url']}\nTitle: {it['title']}\nState: {it['state']}\n"
                        f"Report: {it['body']}\n")
                if it["prs"]:
                    f.write("Linked pull requests:\n")
                    for p in it["prs"]:
                        f.write(f"- [{'merged' if p['merged'] else p['state']}] {p['title']}: {p['body']}\n")
                f.write("\n")
        print("disagreements written for the third rater:", len(disagreements))


if __name__ == "__main__":
    main()
