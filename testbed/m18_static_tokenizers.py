"""M18.1 false alarms on the static corpus (M18.1 review, finding 7): the tokenizer check of the static path
(sites.check_static, transformers builds the tokenizer from the folder, no weights) on every fetched folder of the
M10 E1 corpus (testbed/results/m10/e1_llm/configs, 300 folders). Counts the Tokenization verdicts and lists every
non-pass with its note.
Run in ~/venvs/gpu (transformers 5.17): python testbed/m18_static_tokenizers.py <out.json>
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
os.environ.setdefault("ENTAIL_LOG_DIR", "off")
from entail import sites  # noqa: E402

CONFIGS = os.path.join(HERE, "results", "m10", "e1_llm", "configs")


def main():
    out = sys.argv[1]
    rows = []
    for name in sorted(os.listdir(CONFIGS)):
        folder = os.path.join(CONFIGS, name)
        if not os.path.isdir(folder):
            continue
        t0 = time.perf_counter()
        row = {"folder": name}
        try:
            per_backend, model, notes = sites.check_static(folder, "transformers", {})
            row["tokenization"] = [{"verdict": d.verdict.value, "rule": d.rule, "note": d.note}
                                   for d in model if d.name == "Tokenization"]
            row["vocab"] = [d.verdict.value for d in model if d.name == "Vocab"]
            row["notes"] = [n for n in notes if "tokenizer" in n.lower()]
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        row["ms"] = round((time.perf_counter() - t0) * 1e3)
        rows.append(row)
        v = [d["verdict"] for d in row.get("tokenization", [])]
        if v != ["pass"]:
            print(f"{name:<60} {v} {[d['note'][:150] for d in row.get('tokenization', [])]}", flush=True)
    counts = {}
    for r in rows:
        for d in r.get("tokenization", []):
            counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1
    no_decision = sum(1 for r in rows if not r.get("tokenization"))
    summary = {"folders": len(rows), "tokenization_verdicts": counts, "folders_without_tokenization_decision": no_decision,
               "errors": sum(1 for r in rows if "error" in r), "rows": rows}
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
