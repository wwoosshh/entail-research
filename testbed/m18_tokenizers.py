"""M18.1 false-alarm and cost run: every E2 model folder's tokenizer built by transformers with entail on, so the
declared tokenizer (tokenizer.json, run by the tokenizers library) is compared with the one AutoTokenizer built
(tokenizer_contract). vLLM and SGLang build their tokenizers through the same AutoTokenizer, so this stands for the
three engines at that boundary. Two passes per folder: entail off (the time of the plain build) and on (the time
with the checks), in this one process; the decisions come from entail's ledger.
Usage: python testbed/m18_tokenizers.py <out.json> <list.txt>...   (name<TAB>folder per line)
"""
import json
import os
import sys
import time


def main():
    out, lists = sys.argv[1], sys.argv[2:]
    rows = []
    for lst in lists:
        for line in open(lst, encoding="utf-8"):
            if line.strip():
                name, folder = line.rstrip("\n").split("\t")
                rows.append((name, folder))
    os.environ.setdefault("ENTAIL_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(out)), "entail_logs"))
    import transformers
    from transformers import AutoTokenizer

    import entail
    from entail import core, load

    results = []
    for name, folder in rows:
        row = {"name": name, "folder": folder}
        core.set_mode("off")
        t0 = time.perf_counter()
        try:
            AutoTokenizer.from_pretrained(folder, local_files_only=True)
        except Exception as e:  # noqa: BLE001
            row["off_error"] = f"{type(e).__name__}: {e}"[:200]
        row["off_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        n = len(load.LEDGER.decisions)
        entail.enable()
        t0 = time.perf_counter()
        try:
            tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            row["tokenizer_class"] = type(tok).__name__
        except Exception as e:  # noqa: BLE001
            row["on_error"] = f"{type(e).__name__}: {e}"[:200]
        row["on_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        core.set_mode("off")
        ds = load.LEDGER.decisions[n:]
        row["decisions"] = [{"name": d.name, "verdict": d.verdict.value, "rule": d.rule, "note": d.note}
                            for d in ds if d.name in ("Tokenization", "Vocab")]
        row["tokenization"] = [d["verdict"] for d in row["decisions"] if d["name"] == "Tokenization"]
        results.append(row)
        print(f"{name:<52} {row.get('tokenizer_class', '-'):<28} off {row['off_ms']:>7} ms  on {row['on_ms']:>7} ms  "
              f"{row['tokenization']}", flush=True)
    summary = {"transformers": transformers.__version__, "entail": entail.__version__, "models": len(results),
               "tokenization_verdicts": {}, "rows": results}
    for r in results:
        for v in r["tokenization"]:
            summary["tokenization_verdicts"][v] = summary["tokenization_verdicts"].get(v, 0) + 1
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
