"""Fresh LLM-engine sample for the re-investigation (2026-09-23).

Source frame: realworld/study/sample.json (606 issues, stratified random, keyword-screened titles, 2024-01..2026-09).
Rules fixed before looking at any issue content:
  - keep issues that are closed as completed (more likely to have an established cause)
  - drop the 100 issues of the earlier pilot (realworld/bug_labels_round1.json) so the sample is new data
  - draw 80 issues, stratified by repository in proportion to the remaining pool, at least 10 per repository
  - seed 20260924
"""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
frame = json.load(open(os.path.join(ROOT, "realworld/study/sample.json"), encoding="utf-8"))
pilot = json.load(open(os.path.join(ROOT, "realworld/bug_labels_round1.json"), encoding="utf-8"))
pilot_keys = {(p["repo"], int(p["issue"])) for p in pilot}

pool = [x for x in frame
        if x.get("state") == "closed" and x.get("state_reason") == "completed"
        and (x["repo"], int(x["number"])) not in pilot_keys]
by_repo = {}
for x in pool:
    by_repo.setdefault(x["repo"], []).append(x)

TOTAL, MIN = 80, 10
repos = sorted(by_repo)
quota = {r: max(MIN, round(TOTAL * len(by_repo[r]) / len(pool))) for r in repos}
while sum(quota.values()) > TOTAL:           # trim the largest quota until the total fits
    big = max(quota, key=quota.get); quota[big] -= 1
while sum(quota.values()) < TOTAL:
    big = max(repos, key=lambda r: len(by_repo[r]) - quota[r]); quota[big] += 1

rng = random.Random(20260924)
sample = []
for r in repos:
    items = sorted(by_repo[r], key=lambda x: int(x["number"]))
    sample += rng.sample(items, min(quota[r], len(items)))

out = [{"case_id": f"L{i+1:03d}", "repo": x["repo"], "number": int(x["number"]), "url": x["url"], "title": x["title"]}
       for i, x in enumerate(sample)]
json.dump(out, open(os.path.join(HERE, "llm_sample.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
summary = {"frame": len(frame), "pool_closed_completed_not_pilot": len(pool),
           "pool_per_repo": {r: len(by_repo[r]) for r in repos}, "quota": quota, "sample": len(out), "seed": 20260924}
json.dump(summary, open(os.path.join(HERE, "llm_sample_summary.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(summary, indent=1))
