"""L2 worker for transformers: the probe prompts under several execution modes of one loaded model, written in the
same shape as vllm_worker.py so compare.py applies unchanged.

Runs in a transformers venv:  python lowlevel/l2/hf_worker.py <model path> <out.json> [modes...]
Modes (the base is the first):
  eager        attn_implementation eager, dynamic cache, each prompt alone          (the base)
  sdpa         attn_implementation sdpa
  nocache      use_cache False: every step recomputes the whole sequence
  static       cache_implementation static
  chunked      generate with prefill_chunk_size 16 (the prompt enters the cache in chunks)
  batch_left   all prompts in one batch, left padding
  pad_right    prompt log-probs of each prompt inside a right-padded batch (valid positions only; no generation)
  beam_cache / beam_nocache   2-beam search with and without the cache (compared with each other)
Each generation is greedy, max_new_tokens 24, min_new_tokens 24 (so every mode has the same length); each step keeps
the chosen token and the top-5 log-probabilities of the raw logits. Prompt log-probs come from one forward pass.
"""
import json
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from probes import PROBES  # noqa: E402

N = 24


def top5(logits_row):
    lp = torch.log_softmax(logits_row.float(), dim=-1)
    v, i = torch.topk(lp, 5)
    return {str(int(a)): float(b) for a, b in zip(i.tolist(), v.tolist())}, lp


def prompt_logprobs(model, ids, mask=None, valid=None):
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    rows = out.logits[0]
    res = [None]
    idx = valid if valid is not None else list(range(ids.shape[1]))
    toks = ids[0].tolist()
    for j in range(1, len(idx)):
        d, lp = top5(rows[idx[j - 1]])
        d[str(toks[idx[j]])] = float(lp[toks[idx[j]]])
        res.append(d)
    return res


def gen_record(model, tok, ids, mask=None, **kw):
    with torch.no_grad():
        g = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=N, min_new_tokens=N, do_sample=False,
                           output_logits=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id, **kw)
    return g


def one(g, row, prompt_len):
    new = g.sequences[row, prompt_len:].tolist()
    lps = []
    for s, t in zip(g.logits, new):
        d, lp = top5(s[row])
        d[str(t)] = float(lp[t])
        lps.append(d)
    return new, lps


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    modes = sys.argv[3:] or ["eager", "sdpa", "nocache", "static", "chunked", "batch_left", "pad_right",
                             "beam_cache", "beam_nocache"]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # a CPU-only venv (transformers 5.12.1 here) runs on CPU
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt, attn_implementation="eager").to(dev)
    model.eval()
    res = {"model": model_path, "device": dev, "dtype": str(dt), "load_s": round(time.time() - t0, 2), "plans": {}, "errors": {}}
    enc = {p["id"]: tok(p["text"], return_tensors="pt").input_ids.to(dev) for p in PROBES}

    def alone(with_prompt=True, **kw):
        out = {}
        for p in PROBES:
            ids = enc[p["id"]]
            g = gen_record(model, tok, ids, torch.ones_like(ids), **kw)
            new, lps = one(g, 0, ids.shape[1])
            out[p["id"]] = {"tokens": new, "logprobs": lps, "prompt_tokens": ids[0].tolist(),
                            "prompt_logprobs": prompt_logprobs(model, ids) if with_prompt else None}
        return out

    for m in modes:
        try:
            if m == "eager":
                res["plans"][m] = alone()
            elif m == "sdpa":
                model.set_attn_implementation("sdpa")
                res["plans"][m] = alone()
                model.set_attn_implementation("eager")
            elif m == "nocache":
                res["plans"][m] = alone(use_cache=False)
            elif m == "static":
                res["plans"][m] = alone(cache_implementation="static")
            elif m == "chunked":
                res["plans"][m] = alone(with_prompt=False, prefill_chunk_size=16)
            elif m in ("batch_left", "pad_right"):
                tok.padding_side = "left" if m == "batch_left" else "right"
                texts = [p["text"] for p in PROBES]
                b = tok(texts, return_tensors="pt", padding=True).to(dev)
                out = {}
                if m == "batch_left":
                    g = gen_record(model, tok, b.input_ids, b.attention_mask)
                    for r, p in enumerate(PROBES):
                        valid = b.attention_mask[r].nonzero().flatten().tolist()
                        new, lps = one(g, r, b.input_ids.shape[1])
                        out[p["id"]] = {"tokens": new, "logprobs": lps,
                                        "prompt_tokens": b.input_ids[r, valid].tolist(), "prompt_logprobs": None}
                else:
                    with torch.no_grad():
                        logits = model(input_ids=b.input_ids, attention_mask=b.attention_mask).logits
                    for r, p in enumerate(PROBES):
                        valid = b.attention_mask[r].nonzero().flatten().tolist()
                        toks = b.input_ids[r].tolist()
                        pl = [None]
                        for j in range(1, len(valid)):
                            d, lp = top5(logits[r, valid[j - 1]])
                            d[str(toks[valid[j]])] = float(lp[toks[valid[j]]])
                            pl.append(d)
                        base_ids = enc[p["id"]][0].tolist()
                        out[p["id"]] = {"tokens": [], "logprobs": [], "prompt_tokens": [toks[v] for v in valid],
                                        "prompt_logprobs": pl, "same_ids_as_alone": [toks[v] for v in valid] == base_ids}
                tok.padding_side = "left"
                res["plans"][m] = out
            elif m in ("beam_cache", "beam_nocache"):
                out = {}
                for p in PROBES:
                    ids = enc[p["id"]]
                    with torch.no_grad():
                        gg = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=N,
                                            min_new_tokens=N, do_sample=False, num_beams=2, output_scores=True,
                                            return_dict_in_generate=True, use_cache=(m == "beam_cache"),
                                            pad_token_id=tok.pad_token_id)
                    g = gg.sequences
                    internal = float(gg.sequences_scores[0])
                    # the chosen sequence's log-probability under one full forward pass (no cache): two beam searches
                    # that differ only by a tie find sequences of the same score; a lost meaning finds a worse one
                    with torch.no_grad():
                        lg = torch.log_softmax(model(input_ids=g[:1]).logits[0].float(), dim=-1)
                    seq = g[0].tolist()
                    score = float(sum(lg[t - 1, seq[t]] for t in range(ids.shape[1], len(seq))))
                    out[p["id"]] = {"tokens": g[0, ids.shape[1]:].tolist(),
                                    "logprobs": [{} for _ in range(g.shape[1] - ids.shape[1])],
                                    "prompt_tokens": ids[0].tolist(), "prompt_logprobs": None, "seq_logprob": score,
                                    "beam_internal": internal, "recomputed_mean": score / max(1, len(seq) - ids.shape[1])}
                res["plans"][m] = out
        except Exception as e:  # noqa: BLE001 - a mode the model does not support is recorded, not fatal
            res["errors"][m] = f"{type(e).__name__}: {e}"[:300]
    json.dump(res, open(out_path, "w", encoding="utf-8"))
    print("RESULT", json.dumps({"model": model_path, "plans": list(res["plans"]), "errors": res["errors"]}))


if __name__ == "__main__":
    main()
