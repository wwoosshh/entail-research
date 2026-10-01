"""M20.3 blind rating (testbed/M20_PROTOCOL.md 4, 5): reads the two blind raters' files (results/m20/ratings_A.txt,
ratings_B.txt; lines "<url> | K1..K7 | Kconf | V1..V4 | Vconf | evidence"), reports Cohen's kappa (K over seven
categories, K1 against the rest, V over four), writes the items where K or V differ for the third, blind rater
(its items file goes to RATER_DIR/items_C.md, in the packet's wording), and with the third rater's file the consensus:
the value two raters share ("split" when all three differ). The shares are over the output reports (rule 1), with
Wilson 95% intervals, overall and by engine; each rater's own shares are given as a range, and the session's
preliminary ratings (screening.json) are compared with the consensus.
  python testbed/m20_rate_score.py [ratings_C.txt]     (RATER_DIR: where the third rater's items file is written)
Writes results/m20/ratings.json and blind_summary.json.
"""
import json
import math
import os
import re
import sys
import textwrap
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "m20")
RATER_DIR = os.environ.get("RATER_DIR", OUT)
LINE = re.compile(r"^\s*`?\s*(https?://\S+?)\s*\|\s*(K[1-7])\s*\|\s*([123])\s*\|\s*(V[1-4])\s*\|\s*([123])\s*\|\s*(.*?)`?\s*$")
ENGINE = {"ollama/ollama": "Ollama", "ggml-org/llama.cpp": "llama.cpp", "lmstudio-ai/lmstudio-bug-tracker": "LM Studio"}


def read(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        m = LINE.match(line.rstrip("\n"))
        if m:
            rows[m.group(1).rstrip("/")] = {"K": m.group(2), "Kc": int(m.group(3)), "V": m.group(4),
                                            "Vc": int(m.group(5)), "evidence": m.group(6).strip()}
    return rows


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return {"n": n, "observed": round(po, 3), "expected": round(pe, 3),
            "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0}


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def share(k, n):
    return {"k": k, "n": n, "share": round(k / n, 3) if n else None, "wilson95": wilson(k, n)}


def agreed(ra, rb, rc, field):
    if ra and rb and ra[field] == rb[field]:
        return ra[field]
    if rc:
        return rc[field] if rc[field] in (ra[field] if ra else None, rb[field] if rb else None) else "split"
    return None


def shares(labels, n):
    return {"K1": share(sum(1 for k, _ in labels if k == "K1"), n),
            "K1_and_V1": share(sum(1 for k, v in labels if k == "K1" and v == "V1"), n),
            "K1_and_V2": share(sum(1 for k, v in labels if k == "K1" and v == "V2"), n),
            "K1_and_V3": share(sum(1 for k, v in labels if k == "K1" and v == "V3"), n),
            "V2": share(sum(1 for _, v in labels if v == "V2"), n),
            "V1": share(sum(1 for _, v in labels if v == "V1"), n)}


def main():
    a, b = read(os.path.join(OUT, "ratings_A.txt")), read(os.path.join(OUT, "ratings_B.txt"))
    c = read(sys.argv[1]) if len(sys.argv) > 1 else {}
    packet = json.load(open(os.path.join(OUT, "rating_packet.json"), encoding="utf-8"))
    urls = [it["url"].rstrip("/") for it in packet["items"]]
    keys = [u for u in urls if u in a and u in b]
    out = {"count": len(urls), "rated_by_both": len(keys), "missing_A": [u for u in urls if u not in a],
           "missing_B": [u for u in urls if u not in b],
           "kappa_K7": kappa([(a[u]["K"], b[u]["K"]) for u in keys]),
           "kappa_K1_vs_rest": kappa([(a[u]["K"] == "K1", b[u]["K"] == "K1") for u in keys]),
           "kappa_V4": kappa([(a[u]["V"], b[u]["V"]) for u in keys]),
           "counts_A": {"K": dict(sorted(Counter(a[u]["K"] for u in keys).items())),
                        "V": dict(sorted(Counter(a[u]["V"] for u in keys).items()))},
           "counts_B": {"K": dict(sorted(Counter(b[u]["K"] for u in keys).items())),
                        "V": dict(sorted(Counter(b[u]["V"] for u in keys).items()))},
           "third_rater_items": len(c), "issues": {}}
    disagreements = []
    for u in urls:
        ra, rb, rc = a.get(u), b.get(u), c.get(u)
        row = {"A": ra, "B": rb, "C": rc, "K": agreed(ra, rb, rc, "K"), "V": agreed(ra, rb, rc, "V")}
        if ra and rb and (ra["K"] != rb["K"] or ra["V"] != rb["V"]):
            disagreements.append(u)
        out["issues"][u] = row
    out["disagreements"] = len(disagreements)
    json.dump(out, open(os.path.join(OUT, "ratings.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "issues"}, ensure_ascii=False, indent=1))
    if not c:
        lines = ["# Issues to rate", "", f"{len(disagreements)} GitHub issues from local LLM runtimes, in the same form "
                 "as before: the issue's title, state, labels, a summary of the report body, and the titles and "
                 "descriptions of linked pull requests. Rate each item by the codebook.", ""]
        for u in disagreements:
            it = next(i for i in packet["items"] if i["url"].rstrip("/") == u)
            block = [f"## Item {it['n']}", f"URL: {it['url']}", f"Title: {it['title']}",
                     f"State: {it['state']} | labels: {', '.join(it['labels']) or '-'} | created: {it['created']}",
                     f"Report: {it['body'] or '(empty)'}"]
            if it["prs"]:
                block.append("Linked pull requests:")
                block += [f"- [{'merged' if p['merged'] else p['state']}] {p['title']}: {p['body'] or '(no description)'}"
                          for p in it["prs"]]
            else:
                block.append("Linked pull requests: none")
            for line in block:
                lines += textwrap.wrap(line, width=300, break_long_words=True, break_on_hyphens=False) or [""]
            lines.append("")
        with open(os.path.join(RATER_DIR, "items_C.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        print("items for the third rater:", len(disagreements), "->", os.path.join(RATER_DIR, "items_C.md"))
        return
    # consensus shares, per rater ranges, session comparison
    census = {r["url"].rstrip("/"): r for r in json.load(open(os.path.join(OUT, "census.json"), encoding="utf-8"))["issues"]}
    screening = {u.rstrip("/"): d for u, d in json.load(open(os.path.join(OUT, "screening.json"), encoding="utf-8")).items()}
    summary = {"output_reports": len(urls), "consensus": {}, "by_rater": {}, "split": {}, "session_vs_consensus": {}}
    groups = {"all": urls}
    for repo, name in ENGINE.items():
        groups[name] = [u for u in urls if census[u]["repo"] == repo]
    for name, us in groups.items():
        labels = [(out["issues"][u]["K"], out["issues"][u]["V"]) for u in us]
        summary["consensus"][name] = dict(shares(labels, len(us)), n=len(us),
                                          K=dict(sorted(Counter(k for k, _ in labels).items())),
                                          V=dict(sorted(Counter(v for _, v in labels).items())))
    for tag, r in (("A", a), ("B", b)):
        summary["by_rater"][tag] = shares([(r[u]["K"], r[u]["V"]) for u in urls if u in r], len(urls))
    summary["split"] = {"K": sum(1 for u in urls if out["issues"][u]["K"] == "split"),
                        "V": sum(1 for u in urls if out["issues"][u]["V"] == "split")}
    both = [u for u in urls if out["issues"][u]["K"] not in (None, "split") and u in screening]
    summary["session_vs_consensus"] = {
        "K1_vs_rest": kappa([(screening[u].get("K") == "K1", out["issues"][u]["K"] == "K1") for u in both]),
        "K7": kappa([(screening[u].get("K"), out["issues"][u]["K"]) for u in both]),
        "V4": kappa([(screening[u].get("V"), out["issues"][u]["V"]) for u in both
                     if out["issues"][u]["V"] not in (None, "split")])}
    summary["consensus_K1_items"] = [{"issue": f"{census[u]['repo'].split('/')[1]}#{census[u]['number']}",
                                      "V": out["issues"][u]["V"], "session": [screening.get(u, {}).get("K"),
                                                                              screening.get(u, {}).get("V")]}
                                     for u in urls if out["issues"][u]["K"] == "K1"]
    json.dump(summary, open(os.path.join(OUT, "blind_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
