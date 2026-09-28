"""P3 live check on SGLang (the external evaluation's item 3): one Engine start under entail's safety modes.

  python testbed/p3_sglang.py <Engine kwargs JSON> <out JSON>

Run with ENTAIL=load and the start-up shim (and ENTAIL_PATHS=1 for SGLang's path check). Writes what the start turned
off (resolved decisions at start:sglang.safe_mode), the server arguments the engine ran with, the path check's
verdict lines, what entail said, and one greedy answer.
"""
import json
import os
import sys
import time

KW = json.loads(sys.argv[1])
OUT = sys.argv[2]


def main():
    import sglang as sgl

    t0 = time.perf_counter()
    eng = sgl.Engine(**KW)
    load_s = time.perf_counter() - t0
    out = eng.generate("The capital of France is", {"max_new_tokens": 8, "temperature": 0})
    sa = eng.server_args
    ran = {k: getattr(sa, k, None) for k in ("disable_cuda_graph", "disable_radix_cache", "speculative_algorithm")}
    eng.shutdown()
    from entail import record, safe_mode

    record.close_files()
    run, folder, lines = os.environ.get("ENTAIL_RUN_ID"), record.log_dir(), []
    for name in sorted(os.listdir(folder)):
        if name.startswith("record-"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                lines += [x for x in (json.loads(s) for s in f if s.strip()) if x.get("run") == run]
    result = {
        "kwargs": KW, "mode": safe_mode.mode(), "run": run, "load_s": round(load_s, 2), "ran_with": ran,
        "turned_off": [x.get("resolution") for x in lines
                       if x.get("boundary") == "start:sglang.safe_mode" and x.get("verdict") == "resolved"],
        "safe_unknown": [x.get("note") for x in lines
                         if x.get("boundary") == "start:sglang.safe_mode" and x.get("verdict") == "unknown"],
        "path_verdicts": [(x.get("verdict"), (x.get("note") or "")[:200]) for x in lines
                          if str(x.get("boundary", "")).startswith("start:sglang.paths") and x.get("verdict")],
        "path_counts": [x["boundaries"]["start:sglang.paths"] for x in lines
                        if isinstance(x.get("boundaries"), dict) and "start:sglang.paths" in x["boundaries"]][-1:],
        "said": [x["text"] for x in lines if x.get("said")],
        "answer": out["text"] if isinstance(out, dict) else str(out),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ("mode", "load_s", "ran_with", "turned_off", "path_verdicts",
                                             "path_counts", "said", "answer")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
