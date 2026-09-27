"""Check an L2 signal: transformers beam search with the cache chose a much worse sequence than without it.
If the cache is followed correctly, the score the beam search gives its chosen sequence (computed with the cache)
equals that sequence's score recomputed in one full forward pass without a cache. A gap means the cache the beams
read was not the one their tokens imply.
Run in a transformers venv: python lowlevel/l2/verify_beam.py <model path> <probe id> <out.json>
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probes import PROBES  # noqa: E402


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path, pid, out = sys.argv[1], sys.argv[2], sys.argv[3]
    tok = AutoTokenizer.from_pretrained(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 attn_implementation="eager").to(dev).eval()
    text = next(p["text"] for p in PROBES if p["id"] == pid)
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    rows = {}
    for use_cache in (True, False):
        with torch.no_grad():
            g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=24, min_new_tokens=24,
                               do_sample=False, num_beams=2, use_cache=use_cache, output_scores=True,
                               return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
            seq = g.sequences[0]
            lg = torch.log_softmax(model(input_ids=seq[None]).logits[0].float(), dim=-1)
        n0 = ids.shape[1]
        per_tok = [float(lg[t - 1, seq[t]]) for t in range(n0, len(seq))]
        rows["cache" if use_cache else "nocache"] = {
            "tokens": seq[n0:].tolist(), "beam_score_internal": float(g.sequences_scores[0]),
            "recomputed_sum_logprob": sum(per_tok), "recomputed_mean_logprob": sum(per_tok) / len(per_tok),
            "generated_len": len(per_tok)}
    for k, r in rows.items():
        print(k, "internal", round(r["beam_score_internal"], 4), "recomputed mean", round(r["recomputed_mean_logprob"], 4),
              "recomputed sum", round(r["recomputed_sum_logprob"], 3))
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
