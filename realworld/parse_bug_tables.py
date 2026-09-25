"""Parse the four bug-study tables into one JSON list and draw a fixed random sample for blind re-classification."""
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"vllm": ("vllm-project/vllm", "bugs_vllm.md"), "sglang": ("sgl-project/sglang", "bugs_sglang.md"),
         "transformers": ("huggingface/transformers", "bugs_transformers.md"),
         "llamacpp": ("ggml-org/llama.cpp", "bugs_llamacpp.md")}
CAT = re.compile(r"^(R[1-4]|N[1-5])")


def parse(repo_key):
    repo, fname = FILES[repo_key]
    rows = []
    for line in open(os.path.join(HERE, fname), encoding="utf-8"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        m = re.search(r"(\d{4,6})", cells[0])
        if not m:
            continue
        cat = next((CAT.match(c).group(1) for c in cells[2:6] if CAT.match(c)), None)
        if cat is None:
            continue
        rows.append({"repo": repo, "key": repo_key, "issue": int(m.group(1)), "category": cat,
                     "url": f"https://github.com/{repo}/issues/{m.group(1)}"})
    seen, out = set(), []
    for r in rows:  # keep the first occurrence (the results table precedes the count tables)
        if r["issue"] not in seen:
            seen.add(r["issue"])
            out.append(r)
    return out


def main():
    allrows = []
    for k in FILES:
        rows = parse(k)
        print(k, len(rows), {c: sum(r["category"] == c for r in rows) for c in sorted({r["category"] for r in rows})})
        allrows += rows
    json.dump(allrows, open(os.path.join(HERE, "bug_labels_round1.json"), "w"), indent=1)
    rng = random.Random(20260922)
    sample = []
    for k in FILES:
        rows = [r for r in allrows if r["key"] == k]
        sample += sorted(rng.sample(rows, 12), key=lambda r: r["issue"])
    json.dump([{"repo": r["repo"], "issue": r["issue"], "url": r["url"]} for r in sample],
              open(os.path.join(HERE, "blind_sample.json"), "w"), indent=1)
    print("blind sample:", len(sample), "| first-round categories in sample:",
          {c: sum(r["category"] == c for r in sample) for c in sorted({r["category"] for r in sample})})


if __name__ == "__main__":
    main()
