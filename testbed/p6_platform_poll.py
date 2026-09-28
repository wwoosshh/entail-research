"""P6: the platform on during the 102 normal runs (E2). While testbed/m15_e2_run.sh runs, this links each run's record
file into a folder the platform serves (the runs write <name>.record.jsonl; the platform reads record-*.jsonl), asks
the server for the runs and the newest graph every few seconds, and at the end asks for every run's graph and where
meaning broke. What it writes: how many launches the platform saw, its answer times, errors, and per run the worst
state and the first broken boundary - to set beside the E2 summary.

  python testbed/p6_platform_poll.py <engines folder> <view folder> <port> <stop file> <out json>
With the stop file already there it only reads back, after the runs (P6: the first run's links were relative, so the
live polls saw nothing; the read-back was done again this way).
"""
import http.client
import json
import os
import sys
import time

ENGINES, VIEW, PORT, STOP, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]


def get(path):
    t = time.perf_counter()
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    c.request("GET", path, headers={"Host": f"127.0.0.1:{PORT}"})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, json.loads(body) if body[:1] in (b"{", b"[") else body, (time.perf_counter() - t) * 1000.0


def link_new():
    os.makedirs(VIEW, exist_ok=True)
    for name in os.listdir(ENGINES):
        if name.endswith(".record.jsonl"):
            dst = os.path.join(VIEW, "record-" + name[:-len(".record.jsonl")] + ".jsonl")
            if not os.path.lexists(dst):   # an absolute target: a relative one would be read from VIEW (P6's first run)
                os.symlink(os.path.abspath(os.path.join(ENGINES, name)), dst)


def main():
    polls, errors = [], []
    while not os.path.exists(STOP):
        try:
            link_new()
            st, runs, ms = get("/api/runs")
            if st != 200:
                errors.append(("runs", st))
            elif runs["runs"]:
                st2, _, ms2 = get("/api/graph?run=" + runs["runs"][0]["run"])
                polls.append({"t": time.time(), "launches": len(runs["runs"]), "runs_ms": round(ms, 1),
                              "graph_ms": round(ms2, 1)})
                if st2 != 200:
                    errors.append(("graph", st2))
        except Exception as e:  # noqa: BLE001 - a poll that fails is counted, and the next one tries again
            errors.append((type(e).__name__, str(e)[:120]))
        time.sleep(5)
    link_new()
    st, runs, ms = get("/api/runs")
    per = []
    for r in runs["runs"]:
        st2, g, ms2 = get("/api/graph?run=" + r["run"])
        per.append({"run": r["run"], "engines": r["engines"], "state": r["state"], "broken_at": r.get("broken_at"),
                    "graph_status": st2, "graph_ms": round(ms2, 1),
                    "broken_nodes": [n["id"] for n in g["nodes"] if n["state"] in ("broken", "refused")]
                    if st2 == 200 else None})
    states = {}
    for p in per:
        states[p["state"]] = states.get(p["state"], 0) + 1
    out = {"launches": len(per), "states": states, "errors": errors, "polls": len(polls),
           "runs_ms_max": max((p["runs_ms"] for p in polls), default=None),
           "graph_ms_max": max((p["graph_ms"] for p in polls), default=None),
           "final_runs_ms": round(ms, 1), "per_run": per}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "per_run"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
