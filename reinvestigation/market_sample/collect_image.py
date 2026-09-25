"""Fresh image-generation sample for the re-investigation (2026-09-23). GitHub is used read-only (GET search).

Rules fixed before looking at any issue content:
  - repositories: huggingface/diffusers, Comfy-Org/ComfyUI
  - closed issues created 2024-01-01..2026-09-23 whose title contains one of the symptom keywords below
  - draw 25 per repository at random, seed 20260924
"""
import json, os, random, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = ["huggingface/diffusers", "Comfy-Org/ComfyUI"]
KEYWORDS = ["wrong", "incorrect", '"different output"', '"different results"', "mismatch", "corrupted", "garbage",
            "degraded", "quality", "inconsistent", "nan", "noise", "noisy", "artifacts", '"black image"', "blurry",
            "distorted", '"washed out"']
PER_REPO, SEED = 25, 20260924


def search(q, page):
    out = subprocess.run(["gh", "api", "-X", "GET", "search/issues", "-f", f"q={q}", "-f", "per_page=100",
                          "-f", f"page={page}"], capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


log, cand = [], {}
for repo in REPOS:
    for kw in KEYWORDS:
        q = f"repo:{repo} is:issue is:closed in:title {kw} created:2024-01-01..2026-09-23"
        page, got = 1, 0
        while True:
            data = search(q, page)
            time.sleep(2.2)  # search API allows 30 requests a minute
            for it in data["items"]:
                if "pull_request" in it:
                    continue
                key = (repo, it["number"])
                c = cand.setdefault(key, {"repo": repo, "number": it["number"], "title": it["title"],
                                          "url": it["html_url"], "created_at": it["created_at"],
                                          "state_reason": it.get("state_reason"), "keywords": []})
                c["keywords"].append(kw)
            got += len(data["items"])
            if got >= data["total_count"] or not data["items"] or page >= 10:
                break
            page += 1
        log.append({"repo": repo, "kw": kw, "total_count": data["total_count"], "received": got,
                    "incomplete": data["incomplete_results"]})
        print(repo, kw, data["total_count"], got, flush=True)

cands = sorted(cand.values(), key=lambda c: (c["repo"], c["number"]))
json.dump(cands, open(os.path.join(HERE, "image_candidates.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(log, open(os.path.join(HERE, "image_collect_log.json"), "w", encoding="utf-8"), indent=1)

rng = random.Random(SEED)
sample = []
for repo in REPOS:
    pool = [c for c in cands if c["repo"] == repo]
    sample += rng.sample(pool, min(PER_REPO, len(pool)))
out = [{"case_id": f"I{i+1:03d}", "repo": c["repo"], "number": c["number"], "url": c["url"], "title": c["title"]}
       for i, c in enumerate(sample)]
json.dump(out, open(os.path.join(HERE, "image_sample.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("candidates", {r: sum(1 for c in cands if c["repo"] == r) for r in REPOS}, "sample", len(out))
