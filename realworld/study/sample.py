"""Phase 5 (ROADMAP 5.2): draw a stratified random sample of candidates for screening and blind double rating.

Target: screen 600 issues to keep at least 500 wrong-output issues (the screening share is unknown until done).
Stratified by repository, proportional to the candidate counts with a floor of 80 per repository. Fixed seed.
Pilot-keyword candidates are preferred for comparability with the pilot; extra-keyword ones fill the rest.
Output: sample.json (the sample, with empty screening fields) and sample_summary.json (counts).
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET, FLOOR, SEED = 600, 80, 20260923


def main():
    cands = list(json.load(open(os.path.join(HERE, "candidates.json"), encoding="utf-8")).values())
    by_repo = {}
    for c in cands:
        by_repo.setdefault(c["repo"], []).append(c)
    total = len(cands)
    quota = {r: max(FLOOR, round(TARGET * len(v) / total)) for r, v in by_repo.items()}
    scale = TARGET / sum(quota.values())
    quota = {r: min(len(by_repo[r]), max(FLOOR if len(by_repo[r]) >= FLOOR else len(by_repo[r]), round(q * scale)))
             for r, q in quota.items()}
    rng = random.Random(SEED)
    sample = []
    for r, items in sorted(by_repo.items()):
        pilot = [c for c in items if c["pilot_kw"]]
        extra = [c for c in items if not c["pilot_kw"]]
        rng.shuffle(pilot)
        rng.shuffle(extra)
        pick = (pilot + extra)[: quota[r]]
        for c in pick:
            sample.append({**c, "screen": {"wrong_output": None, "root_cause_identified": None},
                           "rating": {"rater_A": None, "rater_B": None}})
    summary = {"candidates_total": total, "candidates_per_repo": {r: len(v) for r, v in by_repo.items()},
               "sample_total": len(sample), "sample_per_repo": dict(Counter(s["repo"] for s in sample)),
               "sample_pilot_keyword_share": sum(s["pilot_kw"] for s in sample) / max(len(sample), 1),
               "seed": SEED}
    json.dump(sample, open(os.path.join(HERE, "sample.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(summary, open(os.path.join(HERE, "sample_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
