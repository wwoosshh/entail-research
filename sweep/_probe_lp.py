"""One-shot probe: what shape does SGLang return log-probabilities in? (scaffolding for the comparator)"""
import json
import os


def main():
    import sglang as sgl

    e = sgl.Engine(model_path=os.path.expanduser("~/models/gemma-2-2b-it"), attention_backend="triton",
                   mem_fraction_static=0.7, max_total_tokens=2048, log_level="error", disable_cuda_graph=True)
    try:
        out = e.generate("What is the capital of Australia?", {"max_new_tokens": 1, "temperature": 0},
                         return_logprob=True, logprob_start_len=0)
        print("TOPKEYS", sorted(out.keys()), flush=True)
        meta = out.get("meta_info", {})
        print("METAKEYS", sorted(meta.keys()), flush=True)
        lp = meta.get("input_token_logprobs")
        print("LPTYPE", type(lp).__name__, "LEN", len(lp) if lp else None, flush=True)
        print("LPHEAD", json.dumps(lp[:3]) if lp else None, flush=True)
    finally:
        try:
            e.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
