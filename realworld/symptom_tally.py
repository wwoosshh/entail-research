"""Tally the user-visible symptoms of the 50 role-class bugs, from their issue titles.

Role-class rows come from the replay tables (replay_vllm_sglang.md, replay_transformers_llamacpp.md).
Titles come from the per-repository tables in bugs_*.md.

Rules are fixed keyword lists applied to the title, in this order:
  garbled: the output is visibly broken (gibberish, garbage, garbled, corrupted, mosaic, broken)
  otherwise: the output looks normal but is wrong or less accurate. Titles that only name a cause also land here.
Extra tags (not exclusive): repetition, language mixing, hang, ignored setting, multimodal output, long input, later turn.
Limits: titles only, not issue bodies; one rater who knows the hypothesis.
"""
import json
import re

REPLAY = ["replay_vllm_sglang.md", "replay_transformers_llamacpp.md"]
BUGS = [("bugs_vllm.md", "vllm"), ("bugs_sglang.md", "sglang"),
        ("bugs_transformers.md", "transformers"), ("bugs_llamacpp.md", "llama.cpp")]
REPO = {"vllm": "vllm", "sglang": "sglang", "transformers": "transformers", "llama.cpp": "llama.cpp"}

GARBLED = r"gibberish|garbage|garbled|corrupt|mosaic|broken"
TAGS = {
    "repetition": r"loop|repeat|repetition",
    "language_mixing": r"language|chinese|japanese|korean|thai|foreign",
    "hang": r"\bhang",
    "ignored_setting": r"\bignores?\b",
    "multimodal": r"video|audio|image|token2wav|multimodal|-vl\b|\bvlm\b|\bocr\b",
    # Benchmark names such as GSM8K or Geo3K must not count as long inputs.
    "long_sequence": r"long input|\d+k tokens|\d+k-token|after about \d+ tokens",
    "later_turn": r"second prompt|second turn|second interaction|saved session",
    "specific_hardware": r"sycl|intel|\bhip\b|rocm|vulkan|\bsm\d+|gfx\d+|\brtx\b",
}


def main():
    role = []
    for f in REPLAY:
        for line in open(f, encoding="utf-8"):
            m = re.match(r"\|\s*([\w.]+)\s*\|\s*\[#?(\d+)\]", line)
            if m:
                role.append((REPO[m.group(1)], int(m.group(2))))
    titles = {}
    for f, key in BUGS:
        for line in open(f, encoding="utf-8"):
            m = re.match(r"\|\s*\[?#?(\d{3,6})\]?[^|]*\|\s*([^|]+?)\s*\|", line)
            if m:
                titles.setdefault((key, int(m.group(1))), m.group(2))

    rows = []
    for repo, n in role:
        t = titles[(repo, n)]
        low = t.lower()
        kind = "garbled" if re.search(GARBLED, low) else "plausible_but_wrong"
        tags = [k for k, pat in TAGS.items() if re.search(pat, low)]
        rows.append({"repo": repo, "issue": n, "title": t, "kind": kind, "tags": tags})

    out = {
        "n": len(rows),
        "kind": {k: sum(r["kind"] == k for r in rows) for k in ("garbled", "plausible_but_wrong")},
        "tags": {k: sum(k in r["tags"] for r in rows) for k in TAGS},
        "rows": rows,
    }
    json.dump(out, open("symptom_tally.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("n", "kind", "tags")}, ensure_ascii=False))
    for r in rows:
        print(f'{r["kind"][:8]:8} {",".join(r["tags"]):28} {r["repo"]}#{r["issue"]} {r["title"]}')


if __name__ == "__main__":
    main()
