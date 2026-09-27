"""M16.2 screening helper (testbed/M16_PROTOCOL.md 3). GitHub is read with GET only.

  python testbed/m16_screen.py show <from> <to>     print issues from..to (1-based, inclusive) of results/m16/order.json:
                                                     title, state, labels, the body's head, and the closing/linked
                                                     pull requests' titles and body heads (from the issue timeline)
  python testbed/m16_screen.py record <decisions.json>
                                                     merge {url: {"decision": ..., "reason": ..., "expected_boundary": ...,
                                                     "version": ..., "model": ...}} into results/m16/screening.json
  python testbed/m16_screen.py status               counts so far, and whether the stopping rule is met
Decisions: pass | not_output | cannot_run | no_repro_info | cannot_install_version (protocol 3).
"""
import json
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # issue bodies carry emoji; the Windows console is cp949
HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.environ.get("M16_POP") or os.path.join(HERE, "results", "m16")   # the population and its order (census.json,
                                                                          # order.json); the fourth replay: results/m19/replay4
OUT = os.environ.get("M16_OUT") or IN                         # where screening.json goes (the second replay: results/m17/replay2)
START = int(os.environ.get("M16_START", "1"))                 # first position this replay screens (second replay: 151)
STOP_PASS, STOP_SCREENED = 30, 150


def gh(args):
    r = subprocess.run(["gh"] + args, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 and "rate limit" in (r.stderr or "").lower():
        time.sleep(60)
        r = subprocess.run(["gh"] + args, capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else ""


def linked_prs(repo, number):
    """Pull requests cross-referenced from the issue's timeline (the fix, usually), newest last."""
    out = gh(["api", "-X", "GET", f"repos/{repo}/issues/{number}/timeline", "-f", "per_page=100"])
    prs = []
    try:
        for ev in json.loads(out or "[]"):
            src = (ev.get("source") or {}).get("issue") or {}
            if ev.get("event") == "cross-referenced" and src.get("pull_request"):
                prs.append({"url": src.get("html_url"), "title": src.get("title"), "state": src.get("state"),
                            "merged": bool((src.get("pull_request") or {}).get("merged_at")),
                            "body": (src.get("body") or "")[:700]})
    except ValueError:
        pass
    return prs


def show(a, b):
    order = json.load(open(os.path.join(IN, "order.json"), encoding="utf-8"))
    census = {r["url"]: r for r in json.load(open(os.path.join(IN, "census.json"), encoding="utf-8"))["issues"]}
    done = {}
    p = os.path.join(OUT, "screening.json")
    if os.path.isfile(p):
        done = json.load(open(p, encoding="utf-8"))
    for i in range(a, b + 1):
        if i > len(order):
            break
        url = order[i - 1]
        r = census[url]
        flag = f"[done: {done[url]['decision']}]" if url in done else ""
        print(f"\n===== #{i} {r['repo'].split('/')[1]}#{r['number']} {r['state']}"
              f"{('/' + str(r.get('state_reason'))) if r.get('state_reason') else ''} {flag}\n{url}\nTITLE: {r['title']}"
              f"\nlabels={r['labels']} created={r['created'][:10]} closed={(r.get('closed') or '')[:10]}")
        body = gh(["issue", "view", url, "--json", "body", "--jq", ".body"]) or r.get("body_head") or ""
        print("BODY:", " ".join(body.split())[:1500])
        for pr in linked_prs(r["repo"], r["number"])[-3:]:
            print(f"PR: {pr['url']} [{pr['state']}{', merged' if pr['merged'] else ''}] {pr['title']}")
            print("   ", " ".join((pr["body"] or "").split())[:500])


def record(path):
    new = json.load(open(path, encoding="utf-8"))
    p = os.path.join(OUT, "screening.json")
    done = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    order = json.load(open(os.path.join(IN, "order.json"), encoding="utf-8"))
    census = {r["url"]: r for r in json.load(open(os.path.join(IN, "census.json"), encoding="utf-8"))["issues"]}
    os.makedirs(OUT, exist_ok=True)
    for url, d in new.items():
        assert url in census, url
        assert d["decision"] in ("pass", "not_output", "cannot_run", "no_repro_info", "cannot_install_version"), d
        position = order.index(url) + 1
        assert position >= START, f"{url} is position {position}, before this replay's first position {START}"
        d = dict(d, position=position, repo=census[url]["repo"], number=census[url]["number"],
                 title=census[url]["title"], state=census[url]["state"], when=time.strftime("%Y-%m-%d %H:%M"))
        done[url] = d
    json.dump(done, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    status()


def status():
    p = os.path.join(OUT, "screening.json")
    done = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    done = {u: d for u, d in done.items() if d["position"] >= START}
    from collections import Counter
    c = Counter(d["decision"] for d in done.values())
    positions = sorted(d["position"] for d in done.values())
    gaps = [i for i in range(START, (positions[-1] if positions else START - 1) + 1) if i not in set(positions)]
    print(f"screened {len(done)} (positions {START}..{positions[-1] if positions else START - 1}, gaps {gaps[:10]}): "
          f"{dict(c)}")
    print("STOP" if c.get("pass", 0) >= STOP_PASS or len(done) >= STOP_SCREENED else
          f"continue: {STOP_PASS - c.get('pass', 0)} more passes or {STOP_SCREENED - len(done)} more screened")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "show":
        show(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "record":
        record(sys.argv[2])
    else:
        status()
