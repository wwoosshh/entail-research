"""Phase 5 (ROADMAP 5.2-5.3): collect candidate wrong-output issues with read-only GitHub search.

Uses `gh api -X GET search/issues` only (GET, authenticated). No writes of any kind.
Sampling frame:
  repos     vLLM, SGLang, transformers, llama.cpp, TensorRT-LLM
  period    2024-01-01 .. 2026-09-23, split in half-years (search returns at most 1000 hits per query)
  keywords  title keywords of the pilot (bugs_vllm.md) plus four extra ones, recorded separately
Output: candidates.json (deduplicated issues with the keywords that matched) and collect_log.json (query counts).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = ["vllm-project/vllm", "sgl-project/sglang", "huggingface/transformers", "ggml-org/llama.cpp",
         "NVIDIA/TensorRT-LLM"]
PILOT_KW = ["wrong", "incorrect", "garbage", "gibberish", "nonsense", "accuracy", "mismatch", '"different output"',
            "corrupted", "degraded"]
EXTRA_KW = ["repetition", "nan", "inconsistent", '"quality"']
PERIODS = [("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-06-30"),
           ("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-06-30"), ("2026-07-01", "2026-09-23")]
PAUSE = 2.5  # seconds between search calls (the search API allows 30 per minute)


def search(q, page):
    r = subprocess.run(["gh", "api", "-X", "GET", "search/issues", "-f", f"q={q}", "-f", "per_page=100",
                        "-f", f"page={page}"], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        return None, r.stderr.strip()[:300]
    return json.loads(r.stdout), None


def main():
    out_path = os.path.join(HERE, "candidates.json")
    cands = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    log = []
    for repo in REPOS:
        for kw in PILOT_KW + EXTRA_KW:
            for a, b in PERIODS:
                q = f"repo:{repo} is:issue in:title {kw} created:{a}..{b}"
                page, got, total = 1, 0, None
                while True:
                    data, err = search(q, page)
                    time.sleep(PAUSE)
                    if err:
                        if "rate limit" in err.lower():
                            time.sleep(60)
                            continue
                        log.append({"q": q, "page": page, "error": err})
                        break
                    total = data.get("total_count", 0)
                    for it in data.get("items", []):
                        key = f"{repo}#{it['number']}"
                        c = cands.setdefault(key, {
                            "repo": repo, "number": it["number"], "title": it["title"], "url": it["html_url"],
                            "created_at": it["created_at"], "closed_at": it.get("closed_at"), "state": it["state"],
                            "state_reason": it.get("state_reason"), "comments": it.get("comments"),
                            "labels": [lb["name"] for lb in it.get("labels", [])], "keywords": [], "pilot_kw": False})
                        if kw not in c["keywords"]:
                            c["keywords"].append(kw)
                        c["pilot_kw"] = c["pilot_kw"] or kw in PILOT_KW
                    got += len(data.get("items", []))
                    if got >= min(total, 1000) or not data.get("items"):
                        break
                    page += 1
                log.append({"q": q, "total_count": total, "fetched": got, "incomplete": data.get("incomplete_results")
                            if data else None})
                print(f"{repo} {kw} {a}: total {total} fetched {got} | candidates {len(cands)}", flush=True)
            json.dump(cands, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            json.dump(log, open(os.path.join(HERE, "collect_log.json"), "w", encoding="utf-8"), indent=0)
    json.dump(cands, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(log, open(os.path.join(HERE, "collect_log.json"), "w", encoding="utf-8"), indent=0)
    print("done", len(cands), file=sys.stderr)


if __name__ == "__main__":
    main()
