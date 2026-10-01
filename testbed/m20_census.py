"""M20 (testbed/M20_PROTOCOL.md 2): the issues of the local LLM runtimes whose boundary M20 measures, by GitHub search
(GET only): created 2026-03-28..2026-09-27, open or closed, a title with one of M16's 14 keywords, in ollama/ollama,
ggml-org/llama.cpp and lmstudio-ai/lmstudio-bug-tracker; the order is a seeded shuffle (seed 20260928). The keywords
and the search are M16's own (imported, not copied), so the ruler is the same.
Writes testbed/results/m20/census.json and order.json. Run: python testbed/m20_census.py
"""
import json
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m16_census import KEYWORDS, search  # noqa: E402  (M16's keywords and search)

OUT = os.path.join(HERE, "results", "m20")
REPOS = ["ollama/ollama", "ggml-org/llama.cpp", "lmstudio-ai/lmstudio-bug-tracker"]
WINDOW = ("2026-03-28", "2026-09-27")
SEED = 20260928


def main():
    os.makedirs(OUT, exist_ok=True)
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
    order = [r["url"] for r in rows]
    random.Random(SEED).shuffle(order)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S %z"), "protocol": "testbed/M20_PROTOCOL.md 2", "keywords": KEYWORDS,
           "repos": REPOS, "window": list(WINDOW), "seed": SEED, "queries": queries, "issues": rows}
    with open(os.path.join(OUT, "census.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "order.json"), "w", encoding="utf-8") as f:
        json.dump(order, f, indent=1)
    print("DONE", len(rows), "| by repo/state:", Counter((r["repo"].split("/")[1], r["state"]) for r in rows),
          "| truncated queries:", sum(1 for q in queries if q["total"] > q["fetched"]))


if __name__ == "__main__":
    main()
