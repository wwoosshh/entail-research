"""Check an L2 signal: with n-gram speculative decoding, a hybrid (Mamba2 + attention) model's log-probs moved far
more than between any other two modes. The same tokens are fed teacher-forced without speculation (prompt log-probs
of prompt + the generated tokens, one prefill), which is what the model computes for that context; the speculative
run's step log-probs are compared with it, and so is the plain run's.
Run in a vLLM venv: python lowlevel/l2/verify_spec.py <model path> <probe id> <out.json> [num_speculative_tokens]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probes import PROBES  # noqa: E402


def lp(d):
    return {int(k): float(v.logprob) for k, v in (d or {}).items()}


def main():
    from vllm import LLM, SamplingParams

    path, pid, out = sys.argv[1], sys.argv[2], sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    text = next(p["text"] for p in PROBES if p["id"] == pid)
    res = {}
    for name, extra in (("plain", {}), ("spec", {"speculative_config": {"method": "ngram", "num_speculative_tokens": k,
                                                                         "prompt_lookup_max": 4, "prompt_lookup_min": 2}})):
        llm = LLM(model=path, seed=0, gpu_memory_utilization=0.8, max_model_len=2048, max_num_seqs=4,
                  enforce_eager=True, enable_prefix_caching=False, **extra)
        o = llm.generate([text], SamplingParams(temperature=0, max_tokens=24, ignore_eos=True, logprobs=5),
                         use_tqdm=False)[0]
        res[name] = {"prompt": list(o.prompt_token_ids), "tokens": list(o.outputs[0].token_ids),
                     "logprobs": [lp(d) for d in o.outputs[0].logprobs]}
        if name == "plain":
            # the same context teacher-forced, no speculation: prompt log-probs of prompt + generated tokens
            full = list(o.prompt_token_ids) + list(o.outputs[0].token_ids)
            t = llm.generate([{"prompt_token_ids": full}], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=5),
                             use_tqdm=False)[0]
            n0 = len(o.prompt_token_ids)
            res["forced"] = {"logprobs": [lp(t.prompt_logprobs[n0 + i]) for i in range(len(o.outputs[0].token_ids))]}
        del llm
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    rows = []
    toks = res["plain"]["tokens"]
    for i, tkn in enumerate(toks):
        f = res["forced"]["logprobs"][i].get(tkn)
        p = res["plain"]["logprobs"][i].get(tkn)
        s = res["spec"]["logprobs"][i].get(tkn) if i < len(res["spec"]["logprobs"]) and res["spec"]["tokens"][:i + 1] == toks[:i + 1] else None
        rows.append({"step": i, "token": tkn, "p_forced": math.exp(f) if f is not None else None,
                     "p_plain": math.exp(p) if p is not None else None, "p_spec": math.exp(s) if s is not None else None})
    worst_plain = max((abs(r["p_plain"] - r["p_forced"]) for r in rows if r["p_plain"] is not None and r["p_forced"] is not None), default=None)
    worst_spec = max((abs(r["p_spec"] - r["p_forced"]) for r in rows if r["p_spec"] is not None and r["p_forced"] is not None), default=None)
    print("max |p_plain - p_forced|", worst_plain, "| max |p_spec - p_forced|", worst_spec, "| spec tokens same as plain:", res["spec"]["tokens"] == toks)
    for r in rows:
        if r["p_spec"] is not None and r["p_forced"] is not None and abs(r["p_spec"] - r["p_forced"]) > 0.05:
            print("  step", r["step"], "tok", r["token"], "forced", round(r["p_forced"], 3), "plain", round(r["p_plain"], 3), "spec", round(r["p_spec"], 3))
    json.dump({"k": k, "rows": rows, "worst_plain": worst_plain, "worst_spec": worst_spec,
               "spec_tokens_same": res["spec"]["tokens"] == toks}, open(out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
