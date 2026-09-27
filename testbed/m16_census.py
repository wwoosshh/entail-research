"""M16 (testbed/M16_PROTOCOL.md 2): the issues the pre-registered replay draws from, by GitHub search (GET only).
  created 2026-03-26..2026-09-26, open or closed, title with one of the 14 keywords, in the four repositories with
  adapters; the 73 issues M10 E3 screened are left out; the order is a seeded shuffle (seed 20260926).
Writes testbed/results/m16/census.json and order.json. Run: python testbed/m16_census.py
The fourth replay (M16_PROTOCOL 9) draws a new population: M16_POP (its folder), M16_WINDOW ("from..to"),
M16_ORDER_SEED, and M16_EXCLUDE_POP (earlier population folders, separated by os.pathsep, whose issues are left out).
"""
import json
import os
import random
import subprocess
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("M16_POP") or os.path.join(HERE, "results", "m16")
PREVIOUS = os.path.join(HERE, "results", "m10", "e3", "screening.json")
EXCLUDE_POPS = [p for p in os.environ.get("M16_EXCLUDE_POP", "").split(os.pathsep) if p]
KEYWORDS = ["wrong", "incorrect", "garbage", "gibberish", "nonsense", "accuracy", "mismatch", '"different output"',
            "corrupted", "degraded", "repetition", "nan", "inconsistent", '"quality"']   # realworld/study/README.md
REPOS = ["vllm-project/vllm", "sgl-project/sglang", "huggingface/transformers", "huggingface/diffusers"]
WINDOW = tuple(os.environ.get("M16_WINDOW", "2026-03-26..2026-09-26").split(".."))
SEED = int(os.environ.get("M16_ORDER_SEED", "20260926"))


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
    prev = json.load(open(PREVIOUS, encoding="utf-8"))
    excluded = {r["url"] for r in (prev.values() if isinstance(prev, dict) else prev) if isinstance(r, dict)}
    for pop in EXCLUDE_POPS:
        excluded |= {r["url"] for r in json.load(open(os.path.join(pop, "census.json"), encoding="utf-8"))["issues"]}
    found, queries = {}, []
    for repo in REPOS:
        for kw in KEYWORDS:
            q = f"repo:{repo} is:issue created:{WINDOW[0]}..{WINDOW[1]} {kw} in:title"
            items, total, incomplete = search(q)
            queries.append({"q": q, "total": total, "incomplete": incomplete, "fetched": len(items)})
            for it in items:
                if "pull_request" in it:
                    continue
                key = it["html_url"]
                row = found.setdefault(key, {"repo": repo, "number": it["number"], "title": it["title"], "url": key,
                                             "created": it["created_at"], "closed": it.get("closed_at"),
                                             "state": it["state"], "state_reason": it.get("state_reason"),
                                             "labels": [x["name"] for x in it.get("labels", [])],
                                             "comments": it.get("comments"), "keywords": [],
                                             "body_head": (it.get("body") or "")[:2500]})
                if kw not in row["keywords"]:
                    row["keywords"].append(kw)
            print(repo, kw, total, len(found), flush=True)
    rows = sorted(found.values(), key=lambda r: (r["repo"], r["number"]))
    kept = [r for r in rows if r["url"] not in excluded]
    order = [r["url"] for r in kept]
    random.Random(SEED).shuffle(order)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S %z"), "protocol": "testbed/M16_PROTOCOL.md 2", "keywords": KEYWORDS,
           "repos": REPOS, "window": list(WINDOW), "seed": SEED, "queries": queries,
           "excluded_populations": EXCLUDE_POPS,
           "excluded_previously_screened": len(rows) - len(kept), "issues": kept}
    with open(os.path.join(OUT, "census.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "order.json"), "w", encoding="utf-8") as f:
        json.dump(order, f, indent=1)
    print("DONE", len(kept), "kept of", len(rows), "| by repo/state:",
          Counter((r["repo"].split("/")[1], r["state"]) for r in kept),
          "| truncated queries:", sum(1 for q in queries if q["total"] > q["fetched"]))


if __name__ == "__main__":
    main()
