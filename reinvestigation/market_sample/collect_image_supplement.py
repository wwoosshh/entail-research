"""Supplementary image-generation sample (2026-09-23), drawn because the first random image sample (50 issues)
gave only 10 cases that are both an output problem and root-caused. Rules fixed before looking at any content:
  - same repositories, keywords and dates as collect_image.py, plus the GitHub qualifier `linked:pr`
    (closed issues linked to a pull request, so the cause is more often established)
  - exclude issues already in image_sample.json
  - draw up to 20 per repository at random, seed 20260925
  - reported separately from the first sample, and also pooled
"""
import json, os, random, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = ["huggingface/diffusers", "Comfy-Org/ComfyUI"]
KEYWORDS = ["wrong", "incorrect", '"different output"', '"different results"', "mismatch", "corrupted", "garbage",
            "degraded", "quality", "inconsistent", "nan", "noise", "noisy", "artifacts", '"black image"', "blurry",
            "distorted", '"washed out"']
PER_REPO, SEED = 20, 20260925
first = {(x["repo"], x["number"]) for x in json.load(open(os.path.join(HERE, "image_sample.json"), encoding="utf-8"))}


def search(q, page):
    for attempt in range(8):
        out = subprocess.run(["gh", "api", "-X", "GET", "search/issues", "-f", f"q={q}", "-f", "per_page=100",
                              "-f", f"page={page}"], capture_output=True, text=True, encoding="utf-8")
        if out.returncode == 0:
            return json.loads(out.stdout)
        if "rate limit" in out.stderr.lower():
            time.sleep(65)  # the search limit resets every minute; other readers share it
            continue
        raise RuntimeError(out.stderr)
    raise RuntimeError("search rate limit did not clear")


cand, log = {}, []
for repo in REPOS:
    for kw in KEYWORDS:
        q = f"repo:{repo} is:issue is:closed linked:pr in:title {kw} created:2024-01-01..2026-09-23"
        page, got = 1, 0
        while True:
            data = search(q, page)
            time.sleep(2.2)
            for it in data["items"]:
                if "pull_request" in it or (repo, it["number"]) in first:
                    continue
                cand.setdefault((repo, it["number"]), {"repo": repo, "number": it["number"], "title": it["title"],
                                                        "url": it["html_url"]})
            got += len(data["items"])
            if got >= data["total_count"] or not data["items"] or page >= 10:
                break
            page += 1
        log.append({"repo": repo, "kw": kw, "total_count": data["total_count"], "incomplete": data["incomplete_results"]})

cands = sorted(cand.values(), key=lambda c: (c["repo"], c["number"]))
rng = random.Random(SEED)
sample = []
for repo in REPOS:
    pool = [c for c in cands if c["repo"] == repo]
    sample += rng.sample(pool, min(PER_REPO, len(pool)))
out = [{"case_id": f"S{i+1:03d}", **{k: c[k] for k in ("repo", "number", "url", "title")}} for i, c in enumerate(sample)]
json.dump(cands, open(os.path.join(HERE, "image_supplement_candidates.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(log, open(os.path.join(HERE, "image_supplement_log.json"), "w", encoding="utf-8"), indent=1)
json.dump(out, open(os.path.join(HERE, "batch_S1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("candidates", {r: sum(1 for c in cands if c["repo"] == r) for r in REPOS}, "sample", len(out),
      "incomplete", sum(l["incomplete"] for l in log))
