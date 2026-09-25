"""M3.5 (static part): `entail check` over every local LLM folder and each engine, without a GPU.

S3 asks that healthy models get no refusal and no wrong repair. For every model folder in ~/models and every engine
in the capability table this runs sites.check_static (every backend of the engine) and records:
  - per backend: the attention verdict (resolved means the backend drops a property the model declares),
  - about the model itself: tie against the checkpoint, config key coverage, layout against the data,
  - the notes (values vocabulary v1 cannot represent, checks that could not run).
Run in WSL:  source ~/venvs/gpu/bin/activate && python testbed/m3_static_check.py
Writes testbed/results/m3_static_check.json and .md.
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
from entail import record, sites  # noqa: E402
from entail.contracts import Verdict  # noqa: E402

ENGINES = ("transformers", "sglang", "vllm")


def main():
    import transformers

    models = sorted(d for d in glob.glob(os.path.expanduser("~/models/*")) if os.path.isfile(os.path.join(d,
                                                                                                           "config.json")))
    rows = []
    for m in models:
        for engine in ENGINES:
            t0 = time.perf_counter()
            per_backend, model, notes = sites.check_static(m, engine, {})
            rows.append({"model": os.path.basename(m), "engine": engine,
                         "seconds": round(time.perf_counter() - t0, 3),
                         "backends": {d.contract.consumer: {"verdict": d.verdict.value, "target": d.target,
                                                            "blocking": d.blocking} for d in per_backend},
                         "model_decisions": [record.decision_json(d) for d in model],
                         "model_lines": [record.line(d) for d in model], "notes": notes})
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "transformers": transformers.__version__, "rows": rows}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "m3_static_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    lines = [f"# M3.5 static check ({out['when']}, transformers {out['transformers']})", "",
             "| model | engine | refused about the model | unknown about the model | backends: pass / resolved / "
             "refused / unknown | seconds |", "|---|---|---|---|---|---|"]
    refused_total = 0
    for r in rows:
        v = [d["verdict"] for d in r["model_decisions"]]
        b = [x["verdict"] for x in r["backends"].values()]
        refused_total += v.count("refused")
        lines.append(f"| {r['model']} | {r['engine']} | {v.count('refused')} | {v.count('unknown')} | "
                     f"{b.count('pass')} / {b.count('resolved')} / {b.count('refused')} / {b.count('unknown')} | "
                     f"{r['seconds']} |")
    lines += ["", f"Refusals about the models themselves: {refused_total}.", "",
              "## Decisions other than pass about the models", ""]
    for r in rows:
        for line, d in zip(r["model_lines"], r["model_decisions"]):
            if d["verdict"] != Verdict.PASS.value:
                lines.append(f"- {r['model']} / {r['engine']}: `{line[:400]}`")
    lines += ["", "## Resolutions by backend (the backend drops a property the model declares)", ""]
    for r in rows:
        moved = {c: x["target"] for c, x in r["backends"].items() if x["verdict"] == "resolved"}
        refused = [c for c, x in r["backends"].items() if x["verdict"] == "refused"]
        if moved or refused:
            lines.append(f"- {r['model']} / {r['engine']}: resolved {moved}; refused {refused}")
    lines += ["", "## Notes", ""]
    seen = set()
    for r in rows:
        for n in r["notes"]:
            if (r["model"], n) not in seen:
                seen.add((r["model"], n))
                lines.append(f"- {r['model']}: {n}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "results", "m3_static_check.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
