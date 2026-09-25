"""Helper run inside ~/venvs/sglang: prompt log-probs of a token sequence with a chosen attention backend.

Usage: python sglang_logprobs.py <model_dir> <backend> <ids.json> <out.json>
Writes {"backend", "sglang_version", "logprobs": [logprob of token t given tokens < t, for t = 1..n-1]}.
"""
import json
import sys

if __name__ == "__main__":
    model_dir, backend, ids_path, out_path = sys.argv[1:5]
    import sglang as sgl

    ids = json.load(open(ids_path))
    llm = sgl.Engine(model_path=model_dir, attention_backend=backend, mem_fraction_static=0.7, context_length=2048,
                     disable_cuda_graph=True, random_seed=0)
    try:
        out = llm.generate(input_ids=ids, sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                           return_logprob=True, logprob_start_len=0)
    finally:
        llm.shutdown()
    lps = [e[0] for e in out["meta_info"]["input_token_logprobs"] if e[0] is not None]
    json.dump({"backend": backend, "sglang_version": sgl.__version__, "logprobs": lps}, open(out_path, "w"))
