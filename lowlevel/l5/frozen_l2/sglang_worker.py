"""L2 worker for SGLang: the probe prompts on one engine configuration, written in vllm_worker.py's shape so
compare.py applies unchanged.

Runs inside an SGLang venv:  python lowlevel/l2/sglang_worker.py <spec.json> <out.json>
spec: {"model", "engine": {sgl.Engine kwargs}, "plans": ["alone", "forced", "batched", "cache"], "max_tokens",
       "probes": [...], "cache_pairs": [...]}
Greedy with ignore_eos; every step keeps the chosen token and the top-5 log-probabilities; prompt log-probs come from
logprob_start_len 0 (which needs the whole prompt prefilled). "forced" feeds the generated tokens back in one prefill.
"cache" flushes the radix cache, runs the warm-up prompt, then the target twice, asking no prompt log-probs so the
cache can be read. "crowd" sends the probes several times at once, every other request stopping after one token,
and keeps each request's prompt log-probs.
"""
import json
import sys
import time


def _pairs(lst):
    return {str(int(t)): float(lp) for (lp, t, *_rest) in (lst or []) if lp is not None}


def record(out, with_prompt=True):
    mi = out["meta_info"]
    toks = [int(t) for (_lp, t, *_r) in mi.get("output_token_logprobs") or []]
    steps = []
    for i, (lp, t, *_r) in enumerate(mi.get("output_token_logprobs") or []):
        d = _pairs((mi.get("output_top_logprobs") or [[]] * len(toks))[i])
        if lp is not None:
            d[str(int(t))] = float(lp)
        steps.append(d)
    ptoks, plp = [], []
    if with_prompt and mi.get("input_token_logprobs"):
        ptoks = [int(t) for (_lp, t, *_r) in mi["input_token_logprobs"]]
        tops = mi.get("input_top_logprobs") or [None] * len(ptoks)
        for i, (lp, t, *_r) in enumerate(mi["input_token_logprobs"]):
            if i == 0 or lp is None:
                plp.append(None)
                continue
            d = _pairs(tops[i])
            d[str(int(t))] = float(lp)
            plp.append(d)
    return {"tokens": toks, "logprobs": steps, "prompt_tokens": ptoks, "prompt_logprobs": plp}


def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    import sglang as sgl

    kw = {"model_path": spec["model"], "random_seed": 0, "mem_fraction_static": 0.7, "context_length": 2048,
          "log_level": "error"}
    kw.update(spec.get("engine", {}))
    t0 = time.time()
    eng = sgl.Engine(**kw)
    load_s = time.time() - t0
    n = int(spec.get("max_tokens", 24))
    sp = {"temperature": 0.0, "max_new_tokens": n, "ignore_eos": True}
    probes = spec["probes"]
    texts = {p["id"]: p["text"] for p in probes}
    res = {"model": spec["model"], "engine": spec.get("engine", {}), "load_s": round(load_s, 2), "plans": {}}
    try:
        if "alone" in spec["plans"]:
            res["plans"]["alone"] = {}
            for p in probes:
                o = eng.generate(prompt=p["text"], sampling_params=sp, return_logprob=True, logprob_start_len=0,
                                 top_logprobs_num=5)
                res["plans"]["alone"][p["id"]] = record(o)
        if "forced" in spec["plans"] and "alone" in res["plans"]:
            res["plans"]["forced"] = {}
            for p in probes:
                a = res["plans"]["alone"][p["id"]]
                full = a["prompt_tokens"] + a["tokens"]
                o = eng.generate(input_ids=full, sampling_params=dict(sp, max_new_tokens=1), return_logprob=True,
                                 logprob_start_len=0, top_logprobs_num=5)
                r = record(o)
                n0 = len(a["prompt_tokens"])
                res["plans"]["forced"][p["id"]] = {"steps": [r["prompt_logprobs"][n0 + i] or {}
                                                             for i in range(len(a["tokens"]))]}
        if "batched" in spec["plans"]:
            outs = eng.generate(prompt=[p["text"] for p in probes], sampling_params=sp, return_logprob=True,
                                logprob_start_len=0, top_logprobs_num=5)
            res["plans"]["batched"] = {p["id"]: record(o) for p, o in zip(probes, outs)}
        if "cache" in spec["plans"]:
            res["plans"]["cache"] = {}
            for pair in spec.get("cache_pairs", []):
                eng.flush_cache()
                eng.generate(prompt=texts[pair["warm"]], sampling_params=sp)
                partial = record(eng.generate(prompt=texts[pair["target"]], sampling_params=sp, return_logprob=True,
                                              top_logprobs_num=5), with_prompt=False)
                full = record(eng.generate(prompt=texts[pair["target"]], sampling_params=sp, return_logprob=True,
                                           top_logprobs_num=5), with_prompt=False)
                res["plans"]["cache"][pair["target"]] = {"partial": partial, "full": full, "warm": pair["warm"]}
        if "crowd" in spec["plans"]:
            # many requests at once, every other one finishing right after its prefill (max_new_tokens 1) and the rest
            # decoding longer, so batches mix prefills with finishing, decoding and (under a small KV pool) retracted
            # requests; each request's prompt log-probs must equal its prompt's alone run
            prompts, params, keys = [], [], []
            for c in range(int(spec.get("crowd_copies", 4))):
                for p in probes:
                    prompts.append(p["text"])
                    params.append(dict(sp, max_new_tokens=1 if len(keys) % 2 == 0 else 96))
                    keys.append(f"{p['id']}#{c}")
            outs = eng.generate(prompt=prompts, sampling_params=params, return_logprob=True, logprob_start_len=0,
                                top_logprobs_num=5)
            res["plans"]["crowd"] = {k: dict(record(o), tokens=[], logprobs=[]) for k, o in zip(keys, outs)}
    finally:
        eng.shutdown()
    json.dump(res, open(sys.argv[2], "w", encoding="utf-8"))
    print("RESULT", json.dumps({"model": spec["model"], "load_s": res["load_s"], "plans": list(res["plans"])}))


if __name__ == "__main__":
    main()
