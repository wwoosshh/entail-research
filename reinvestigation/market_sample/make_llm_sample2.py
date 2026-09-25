"""Second LLM-engine draw (2026-09-23). The first 80 gave 25 cases that the extractors judged to be root-caused
output problems, too few for a useful interval, so the sample is extended from the same pool with the same rule.
Rules fixed before looking at any of the new issues:
  - pool: realworld/study/sample.json, closed as completed, not in the earlier pilot, not already in llm_sample.json
  - draw 80, stratified by repository in proportion to the remaining pool, at least 10 per repository
  - seed 20260925; case ids continue from L081
  - pooled with the first draw (same population, same rule)
"""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
frame = json.load(open(os.path.join(ROOT, "realworld/study/sample.json"), encoding="utf-8"))
pilot = {(p["repo"], int(p["issue"])) for p in json.load(open(os.path.join(ROOT, "realworld/bug_labels_round1.json"), encoding="utf-8"))}
first = {(x["repo"], x["number"]) for x in json.load(open(os.path.join(HERE, "llm_sample.json"), encoding="utf-8"))}
pool = [x for x in frame if x.get("state") == "closed" and x.get("state_reason") == "completed"
        and (x["repo"], int(x["number"])) not in pilot and (x["repo"], int(x["number"])) not in first]
by_repo = {}
for x in pool:
    by_repo.setdefault(x["repo"], []).append(x)
TOTAL, MIN = 80, 10
repos = sorted(by_repo)
quota = {r: min(len(by_repo[r]), max(MIN, round(TOTAL * len(by_repo[r]) / len(pool)))) for r in repos}
while sum(quota.values()) > TOTAL:
    big = max(quota, key=quota.get); quota[big] -= 1
while sum(quota.values()) < TOTAL:
    big = max(repos, key=lambda r: len(by_repo[r]) - quota[r]); quota[big] += 1
rng = random.Random(20260925)
sample = []
for r in repos:
    items = sorted(by_repo[r], key=lambda x: int(x["number"]))
    sample += rng.sample(items, quota[r])
out = [{"case_id": f"L{81 + i:03d}", "repo": x["repo"], "number": int(x["number"]), "url": x["url"], "title": x["title"]}
       for i, x in enumerate(sample)]
json.dump(out, open(os.path.join(HERE, "llm_sample2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
b3 = [x for x in out if x["repo"] in ("vllm-project/vllm", "sgl-project/sglang")]
b4 = [x for x in out if x not in b3]
json.dump(b3, open(os.path.join(HERE, "batch_L3.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(b4, open(os.path.join(HERE, "batch_L4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({"pool": len(pool), "pool_per_repo": {r: len(by_repo[r]) for r in repos}, "quota": quota, "L3": len(b3), "L4": len(b4)})
