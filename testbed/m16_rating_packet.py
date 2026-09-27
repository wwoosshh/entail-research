"""M16.3 rating packet (testbed/M16_PROTOCOL.md 4): for every issue that passed screening rule 1 (decision other than
not_output) the title, state, labels, a body summary and the descriptions of the linked pull requests, in a seeded
order distinct from the screening order, with none of the screening notes. GitHub is read with GET only.
Writes testbed/results/m16/rating_packet.json and rating_packet.md.  Run: python testbed/m16_rating_packet.py
"""
import json
import os
import random
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.environ.get("M16_POP") or os.path.join(HERE, "results", "m16")   # the population (census.json); replay 4: results/m19/replay4
OUT = os.environ.get("M16_OUT") or IN                         # the replay's screening.json and packet (M17.6: results/m17/replay2)
SEED = int(os.environ.get("M16_SEED", "20260927"))             # M17.6 uses 20260928 so the second packet has its own order
MARKERS = ["### 🐛 Describe the bug", "### Describe the bug", "## Describe the bug", "### Description", "## Description",
           "## Summary", "### Summary", "# Summary"]


def gh(args):
    r = subprocess.run(["gh"] + args, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 and "rate limit" in (r.stderr or "").lower():
        time.sleep(60)
        r = subprocess.run(["gh"] + args, capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else ""


def summary(text, n):
    text = (text or "").replace("\r", "")
    text = re.sub(r"<details>.*?</details>", " ", text, flags=re.S)       # environment dumps
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    for m in MARKERS:
        if m in text:
            text = text.split(m, 1)[1]
            break
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text[:n]


def linked_prs(repo, number):
    out = gh(["api", "-X", "GET", f"repos/{repo}/issues/{number}/timeline", "-f", "per_page=100"])
    prs = []
    try:
        for ev in json.loads(out or "[]"):
            src = (ev.get("source") or {}).get("issue") or {}
            if ev.get("event") == "cross-referenced" and src.get("pull_request"):
                prs.append({"url": src.get("html_url"), "title": src.get("title"), "state": src.get("state"),
                            "merged": bool((src.get("pull_request") or {}).get("merged_at")),
                            "body": summary(src.get("body"), 700)})
    except ValueError:
        pass
    merged = [p for p in prs if p["merged"]]
    rest = [p for p in prs if not p["merged"]]
    return (merged + rest)[:3]


def main():
    screening = json.load(open(os.path.join(OUT, "screening.json"), encoding="utf-8"))
    census = {r["url"]: r for r in json.load(open(os.path.join(IN, "census.json"), encoding="utf-8"))["issues"]}
    urls = sorted(u for u, d in screening.items() if d["decision"] != "not_output")
    random.Random(SEED).shuffle(urls)
    items = []
    for i, url in enumerate(urls, 1):
        r = census[url]
        body = gh(["issue", "view", url, "--json", "body", "--jq", ".body"]) or r.get("body_head") or ""
        state = r["state"] + (f" ({r['state_reason']})" if r.get("state_reason") else "")
        items.append({"n": i, "url": url, "title": r["title"], "state": state, "labels": r["labels"],
                      "created": r["created"][:10], "body": summary(body, 2500), "prs": linked_prs(r["repo"], r["number"])})
        print(i, url, flush=True)
    with open(os.path.join(OUT, "rating_packet.json"), "w", encoding="utf-8") as f:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M"), "seed": SEED, "count": len(items), "items": items}, f,
                  ensure_ascii=False, indent=1)
    lines = ["# Issues to rate", "", f"{len(items)} GitHub issues from AI inference engines. Each item gives the issue's "
             "title, state, labels, a summary of the report body, and the titles and descriptions of pull requests "
             "linked from the issue (the fix, when one exists). Rate each item by the codebook.", ""]
    for it in items:
        lines += [f"## Item {it['n']}", f"URL: {it['url']}", f"Title: {it['title']}",
                  f"State: {it['state']} | labels: {', '.join(it['labels']) or '-'} | created: {it['created']}",
                  f"Report: {it['body'] or '(empty)'}"]
        if it["prs"]:
            lines.append("Linked pull requests:")
            for p in it["prs"]:
                tag = "merged" if p["merged"] else p["state"]
                lines.append(f"- [{tag}] {p['title']}: {p['body'] or '(no description)'}")
        else:
            lines.append("Linked pull requests: none")
        lines.append("")
    with open(os.path.join(OUT, "rating_packet.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("DONE", len(items), "items")


if __name__ == "__main__":
    main()
