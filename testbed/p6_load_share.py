"""P6 external evaluation, item 1: where E2's load share (9.0% median, over the 5% target) comes from, from E2's own
records (testbed/results/p6/e2/engines). Measured: each run's share = entail's timed checks / the load with entail on
(testbed/m10_e2_summarize.py). Computed from the same records, not measured:
  no path check     without vLLM's start-up path check (start:vllm.paths; ENTAIL_NO_PATHS=1): its time out of both
  warm tokenizer    as a second start in the same environment would be: the tokenizer reference's ids come from
                    entail_logs/tokenizer_ids.json (M18.1: about 2 ms), so load:transformers.tokenizer.ids is replaced
                    by 2 ms (E2 runs each model once per engine, and the cache is kept per tokenizers version, which
                    differs between the engines' environments, so every E2 run was a first start)
  both
  python testbed/p6_load_share.py   (writes testbed/results/p6/load_share.json)
"""
import glob
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.join(HERE, "results", "p6", "e2")


def main():
    summary = json.load(open(os.path.join(E2, "summary.json"), encoding="utf-8"))
    valid = {(r["model"], r["engine"]): r for r in summary["runs"] if r.get("ok_off") and r.get("ok_on")}
    rows = []
    for (model, engine), r in valid.items():
        path = os.path.join(E2, "engines", f"e2_{engine}_{model}_load.record.jsonl")
        acc = defaultdict(float)
        for s in open(path, encoding="utf-8"):
            try:
                o = json.loads(s)
            except ValueError:
                continue
            if o.get("timing"):
                acc[o["timing"]] += float(o.get("ms") or 0)
        ms, load = sum(acc.values()), r["load_s_on"]
        paths, ids = acc.get("start:vllm.paths", 0.0), acc.get("load:transformers.tokenizer.ids", 0.0)
        warm = min(ids, 2.0)
        rows.append({"model": model, "engine": engine, "load_s": load, "entail_ms": round(ms, 1),
                     "paths_ms": round(paths, 1), "tokenizer_ids_ms": round(ids, 1),
                     "measured": ms / 1e3 / load,
                     "no_path_check": (ms - paths) / 1e3 / (load - paths / 1e3),
                     "warm_tokenizer": (ms - ids + warm) / 1e3 / (load - (ids - warm) / 1e3),
                     "both": (ms - paths - ids + warm) / 1e3 / (load - (paths + ids - warm) / 1e3)})
    out = {}
    for group in ("all", "transformers", "vllm", "sglang"):
        rs = [x for x in rows if group == "all" or x["engine"] == group]
        out[group] = {k: round(statistics.median(x[k] for x in rs), 4)
                      for k in ("measured", "no_path_check", "warm_tokenizer", "both")}
        out[group]["runs"] = len(rs)
        out[group]["entail_ms_median"] = round(statistics.median(x["entail_ms"] for x in rs), 1)
    with open(os.path.join(HERE, "results", "p6", "load_share.json"), "w", encoding="utf-8") as f:
        json.dump({"medians": out, "rows": rows}, f, indent=1, ensure_ascii=False)
    for g, v in out.items():
        print(g, json.dumps(v))


if __name__ == "__main__":
    main()
