"""M10 E3 (testbed/M10_PROTOCOL.md 3.1 and 6): the issues the replay draws from, by GitHub search (GET only).
  A  created since the tested version's PyPI upload, title with one of the 14 keywords, body naming the version
  B  created 2026-03-24..2026-09-24, still open on 2026-09-24, title with one of the 14 keywords
Writes testbed/results/m10/e3/census.json. Run: python testbed/m10_e3_census.py
"""
import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "m10", "e3")
KEYWORDS = ["wrong", "incorrect", "garbage", "gibberish", "nonsense", "accuracy", "mismatch", '"different output"',
            "corrupted", "degraded", "repetition", "nan", "inconsistent", '"quality"']   # realworld/study/README.md
REPOS = {  # repo: (tested version, its PyPI upload date, how the body names that version)
    "vllm-project/vllm": ("0.30.0", "2026-09-22", r"(?i)(vllm[^\n]{0,40}?\b0\.30\b|\bv?0\.30\.\d)"),
    "sgl-project/sglang": ("0.5.20", "2026-09-18", r"\b0\.5\.20\b"),
    "huggingface/transformers": ("5.17.0", "2026-09-09", r"(?i)(transformers[^\n]{0,40}?\b5\.17\b|\b5\.17\.\d)"),
    "huggingface/diffusers": ("0.40.0", "2026-08-20", r"(?i)(diffusers[^\n]{0,40}?\b0\.40\b|\b0\.40\.\d)"),
}
B_FROM, TODAY = "2026-03-24", "2026-09-24"


def search(q):
    items, page = [], 1
    while True:
        r = subprocess.run(["gh", "api", "-X", "GET", "search/issues", "-f", f"q={q}", "-f", "per_page=100",
                            "-f", f"page={page}"], capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            if "rate limit" in (r.stderr or "").lower():
                time.sleep(60)
                continue
            raise RuntimeError(r.stderr[:300])
        d = json.loads(r.stdout)
        items += d.get("items", [])
        time.sleep(2.2)
        if len(items) >= d.get("total_count", 0) or not d.get("items") or page >= 10:
            return items, d.get("total_count", 0), d.get("incomplete_results")
        page += 1


def main():
    os.makedirs(OUT, exist_ok=True)
    found, queries = {}, []
    for repo, (ver, since, pat) in REPOS.items():
        rx = re.compile(pat)
        for kw in KEYWORDS:
            for pop, q in (("A", f"repo:{repo} is:issue created:>={since} {kw} in:title"),
                           ("B", f"repo:{repo} is:issue is:open created:{B_FROM}..{TODAY} {kw} in:title")):
                items, total, incomplete = search(q)
                queries.append({"population": pop, "q": q, "total": total, "incomplete": incomplete})
                for it in items:
                    body = it.get("body") or ""
                    if pop == "A" and not rx.search(body):
                        continue
                    key = it["html_url"]
                    row = found.setdefault(key, {"repo": repo, "number": it["number"], "title": it["title"],
                                                 "url": key, "created": it["created_at"], "state": it["state"],
                                                 "labels": [x["name"] for x in it.get("labels", [])],
                                                 "populations": [], "keywords": [],
                                                 "names_tested_version": bool(rx.search(body)),
                                                 "body_head": body[:1500]})
                    if pop not in row["populations"]:
                        row["populations"].append(pop)
                    if kw not in row["keywords"]:
                        row["keywords"].append(kw)
            print(repo, kw, len(found), flush=True)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S %z"), "keywords": KEYWORDS,
           "repos": {r: {"version": v[0], "since": v[1]} for r, v in REPOS.items()}, "b_window": [B_FROM, TODAY],
           "queries": queries, "issues": sorted(found.values(), key=lambda r: (r["repo"], r["number"]))}
    with open(os.path.join(OUT, "census.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print("DONE", len(found), Counter((r["repo"], "+".join(sorted(r["populations"]))) for r in found.values()))


if __name__ == "__main__":
    main()
