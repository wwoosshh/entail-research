"""E4 (optional in PROTOCOL.md): an independent engine as reference. Runs in the vLLM venv (~/venvs/vllm).

Reads results/e1_<size>_tokens.json, which holds the same token sequences plus, for each HF path,
the top-1 id and the log-prob of the realized next token at every position.
It asks vLLM for prompt log-probs on the same sequences and compares each HF path with vLLM:
  - top-1 agreement
  - mean |difference| of the realized next-token log-prob
Only 2B by default: 9B in bf16 does not fit in 12 GB, and a different quantization would add its own noise.
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODELS = {"2b": os.path.expanduser("~/models/gemma-2-2b-it"), "9b": os.path.expanduser("~/models/gemma-2-9b-it")}


def main(size):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    data = json.load(open(os.path.join(RES, f"e1_{size}_tokens.json"), encoding="utf-8"))
    seqs = data["sequences"]
    llm = LLM(model=MODELS[size], dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
              max_num_batched_tokens=1024, seed=0)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=s["ids"]) for s in seqs], sp)
    paths = list(seqs[0]["tokwise"].keys())
    agg = {p: {"n": 0, "top1_agree": 0, "abs_dlp": 0.0} for p in paths}
    per_seq = []
    for s, o in zip(seqs, outs):
        ids = s["ids"]
        pl = o.prompt_logprobs  # pl[t]: dict token_id -> Logprob, for the token at position t (pl[0] is None)
        v_top1, v_lp = [], []
        for t in range(len(ids) - 1):
            d = pl[t + 1]
            v_lp.append(d[ids[t + 1]].logprob)
            v_top1.append(next(k for k, v in d.items() if v.rank == 1))
        row = {"kind": s["kind"], "name": s["name"], "len": len(ids)}
        for p in paths:
            tw = s["tokwise"][p]
            n = len(v_lp)
            agree = sum(1 for a, b in zip(tw["top1"][:n], v_top1) if a == b)
            dlp = sum(abs(a - b) for a, b in zip(tw["logprob_next"][:n], v_lp))
            row[p] = {"top1_agree": agree / n, "mean_abs_dlogprob": dlp / n}
            agg[p]["n"] += n
            agg[p]["top1_agree"] += agree
            agg[p]["abs_dlp"] += dlp
        per_seq.append(row)
    summary = {p: {"top1_agree_vs_vllm": a["top1_agree"] / a["n"], "mean_abs_dlogprob_vs_vllm": a["abs_dlp"] / a["n"],
                   "positions": a["n"]} for p, a in agg.items()}
    import vllm

    out = {"size": size, "vllm_version": vllm.__version__, "seconds": round(time.time() - t0, 1),
           "summary": summary, "per_sequence": per_seq}
    path = os.path.join(RES, f"e4_{size}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["2b", "9b"], default="2b")
    main(ap.parse_args().size)
