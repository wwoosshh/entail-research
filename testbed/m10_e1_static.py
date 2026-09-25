"""M10 E1, L2-L4 (testbed/M10_PROTOCOL.md 1.2): entail 1.0's static check (sites.check_static, every backend in the
capability table) on each fetched model folder, for transformers, vLLM and SGLang - no GPU, no weights.
  L2  (engine, backend) pairs that drop a property the model declares (resolved / broken / refused per backend)
  L3  config keys the engine's config class does not take (the Coverage decisions), keys listed
  L4  which facts the folder declares
Run in ~/venvs/gpu (transformers 5.17): python testbed/m10_e1_static.py
Writes testbed/results/m10/e1_llm/static.json.
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
os.environ.setdefault("ENTAIL_LOG_DIR", "off")
from entail import load, record, sites  # noqa: E402

OUT = os.path.join(HERE, "results", "m10", "e1_llm")
ENGINES = ("transformers", "vllm", "sglang")


def declared_summary(folder):
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(folder)
    except Exception:  # noqa: BLE001 - declared() works from the files alone
        config = None
    facts = load.declared(folder, config, ())
    out = {name: [{"value": str(f.value)[:300], "source": str(f.source)[:200]} for f in facts.get(name)]
           for name in facts.facts}
    return out, [str(p)[:300] for p in facts.problems]


def main():
    models = json.load(open(os.path.join(OUT, "models.json"), encoding="utf-8"))["models"]
    rows = []
    t_all = time.perf_counter()
    for m in models:
        folder = os.path.join(OUT, "configs", m["id"].replace("/", "__"))
        if not os.path.isfile(os.path.join(folder, "config.json")):
            continue
        row = {"rank": m["rank"], "id": m["id"], "model_type": m["model_type"], "engines": {}}
        try:
            row["declared"], row["problems"] = declared_summary(folder)
        except Exception as e:  # noqa: BLE001
            row["declared_error"] = f"{type(e).__name__}: {e}"
        for engine in ENGINES:
            t0 = time.perf_counter()
            try:
                per_backend, model, notes = sites.check_static(folder, engine, {})
                row["engines"][engine] = {
                    "seconds": round(time.perf_counter() - t0, 3),
                    "backends": {d.contract.consumer: {"verdict": d.verdict.value, "target": d.target,
                                                       "fact": getattr(d.contract, "fact", None),
                                                       "line": record.line(d)[:500]}
                                 for d in per_backend},
                    "model_decisions": [dict(record.decision_json(d), line=record.line(d)[:600]) for d in model],
                    "notes": [str(n)[:400] for n in notes]}
            except Exception as e:  # noqa: BLE001 - recorded; the scan goes on
                row["engines"][engine] = {"error": f"{type(e).__name__}: {e}",
                                          "trace": traceback.format_exc()[-1500:]}
        rows.append(row)
        print(f"{m['rank']:3} {m['id'][:60]}", flush=True)
    import transformers
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "transformers": transformers.__version__,
           "seconds": round(time.perf_counter() - t_all, 1), "rows": rows}
    # M11.7: the 1.0.1 rerun writes a second file (M10_E1_STATIC_OUT=static_1.0.1.json), so 1.0.0's stays
    with open(os.path.join(OUT, os.environ.get("M10_E1_STATIC_OUT", "static.json")), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"DONE {len(rows)} folders in {out['seconds']} s")


if __name__ == "__main__":
    main()
