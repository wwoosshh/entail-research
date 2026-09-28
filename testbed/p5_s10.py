"""P5 (S10 for custom nodes): the examples in entail/examples/custom_nodes run as a developer would - normally, with a
planted fault, with ENTAIL_NODES=off and with the node turned off from the platform's file - and what entail recorded
is read back; then what a check costs per call (the budget's measure).

  python testbed/p5_s10.py <folder with the workshop package's dist-info, or "">
Writes testbed/results/p5/s10.json; each run's records under testbed/results/p5/<run>/.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "entail")
OUT = os.environ.get("P5_OUT") or os.path.join(HERE, "results", "p5")   # P6 re-checks elsewhere
SITE = sys.argv[1] if len(sys.argv) > 1 else ""
EXAMPLES = {"llm_app": "app.answer", "rag_budget": "rag.prompt", "image_app": "image.output"}


def run(example, tag, *args, **env):
    folder = os.path.join(OUT, f"{example}_{tag}")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    e = {k: v for k, v in os.environ.items() if not k.startswith("ENTAIL")}
    e.update(ENTAIL="load", ENTAIL_LOG_DIR=folder, PYTHONPATH=os.pathsep.join(p for p in (REPO, SITE) if p),
             PYTHONIOENCODING="utf-8")
    before = env.pop("_before", None)
    e.update(env)
    if before:
        before(folder)
    p = subprocess.run([sys.executable, os.path.join(REPO, "examples", "custom_nodes", example + ".py"), *args],
                       capture_output=True, text=True, env=e, encoding="utf-8")
    recs = []
    for name in os.listdir(folder):
        if name.startswith("record-"):
            recs += [json.loads(x) for x in open(os.path.join(folder, name), encoding="utf-8") if x.strip()]
    rows = [r for r in recs if r.get("verdict") and str(r.get("boundary", "")).startswith("node:")]
    counted = {}
    for r in recs:
        for b, c in (r.get("boundaries") or {}).items():
            if b.startswith("node:"):
                counted[b] = c
    return {"rc": p.returncode, "stdout": p.stdout.strip().splitlines()[-4:], "stderr": p.stderr.strip()[-300:],
            "decisions": [(r["boundary"], r["verdict"], (r.get("note") or "")[:120]) for r in rows],
            "counted": {b: {"checks": c.get("checks"), "broken": c.get("broken"), "passed": c.get("passed")}
                        for b, c in counted.items()}}


def cost():
    """Microseconds a check adds per call, entail on against off, for a validator that does next to nothing and for
    the examples' JSON validator."""
    sys.path.insert(0, REPO)
    os.environ["ENTAIL_LOG_DIR"] = os.path.join(OUT, "cost")
    os.makedirs(os.environ["ENTAIL_LOG_DIR"], exist_ok=True)
    from entail import core, nodes

    @nodes.validator("tiny")
    def tiny(value):
        return True

    @nodes.validator("json_like")
    def json_like(value):
        return json.loads(value) is not None

    out = {}
    text = json.dumps({"title": "t", "body": "b" * 200})
    for name, fn, value in (("tiny", tiny, 1), ("json_like", json_like, text)):
        per = {}
        for mode in ("off", "load"):
            core.set_mode(mode)
            n = 20000
            best = None
            for _ in range(5):
                t = time.perf_counter()
                for _ in range(n):
                    nodes.check("cost.node", value, fn)
                dt = (time.perf_counter() - t) / n * 1e6
                best = dt if best is None else min(best, dt)
            per[mode] = round(best, 2)
        t = time.perf_counter()
        for _ in range(20000):
            fn(value)
        per["validator_alone"] = round((time.perf_counter() - t) / 20000 * 1e6, 2)
        out[name] = per
    core.set_mode("off")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for ex, node in EXAMPLES.items():
        res[ex] = {
            "normal": run(ex, "normal"),
            "fault": run(ex, "fault", "--fault"),
            "nodes_off": run(ex, "nodes_off", "--fault", ENTAIL_NODES="off"),
            "switched_off": run(ex, "switched_off", "--fault",
                                _before=lambda f, n=node: json.dump({"off": [n]}, open(os.path.join(f, "nodes.json"),
                                                                                       "w", encoding="utf-8"))),
            "node": node,
        }
        r = res[ex]
        print(ex, "| normal", [d[1] for d in r["normal"]["decisions"]], "| fault", [d[:2] for d in r["fault"]["decisions"]],
              "| off", r["nodes_off"]["decisions"], r["nodes_off"]["counted"], "| switched", r["switched_off"]["decisions"],
              "| rc", [r[k]["rc"] for k in ("normal", "fault", "nodes_off", "switched_off")], flush=True)
    res["cost_us"] = cost()
    print("cost (us per call)", res["cost_us"])
    with open(os.path.join(OUT, "s10.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
