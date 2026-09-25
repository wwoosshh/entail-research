"""M10 E3, sgl-project/sglang#25790 (testbed/M10_PROTOCOL.md 3.3): with --kv-cache-dtype fp8_e4m3, the logprobs of
the generated tokens (decode) and of the same tokens evaluated again in one prefill diverge from a fixed position
(96 in the report); without FP8 KV they match. The report's method on SGLang 0.5.20's offline engine: 100 greedy
tokens with logprobs, then the full sequence prefilled with max_new_tokens=0, per-position difference. Qwen3-4B bf16
(the report: Qwen3-14B-FP8 and MiniMax-2.5, "model and GPU agnostic"); deterministic inference and no CUDA graph as
in the report; the default attention backend (fa3 is Hopper-only). A control run keeps the KV cache in bf16.
Run in ~/venvs/sglang: python testbed/m10_e3/sg25790.py <out.json>
"""
import json
import os
import sys

MODEL = "/home/<user>/models/Qwen3-4B"


def one(kv_dtype):
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = tok.apply_chat_template([{"role": "user", "content": "tell me a story"}], add_generation_prompt=True,
                                   tokenize=False)
    prompt = list(tok(text, add_special_tokens=False).input_ids)
    kw = {"kv_cache_dtype": kv_dtype} if kv_dtype else {}
    # the default backend on Ada (FlashAttention) refuses an FP8 KV cache loudly; triton takes it, for both runs
    engine = sgl.Engine(model_path=MODEL, mem_fraction_static=0.8, context_length=2048, log_level="error",
                        disable_cuda_graph=True, disable_radix_cache=True, random_seed=0,
                        enable_deterministic_inference=True, attention_backend="triton", **kw)
    try:
        a = engine.generate(input_ids=prompt, sampling_params={"max_new_tokens": 100, "temperature": 0.0},
                            return_logprob=True, logprob_start_len=0)
        full = prompt + list(a["output_ids"])
        b = engine.generate(input_ids=full, sampling_params={"max_new_tokens": 0, "temperature": 0.0},
                            return_logprob=True, logprob_start_len=0)
    finally:
        engine.shutdown()
    lp_a = a["meta_info"]["input_token_logprobs"] + a["meta_info"]["output_token_logprobs"]
    lp_b = b["meta_info"]["input_token_logprobs"] + b["meta_info"]["output_token_logprobs"]
    diffs = []
    for i, (x, y) in enumerate(zip(lp_a[1:], lp_b[1:])):
        if x[0] is not None and y[0] is not None and abs(x[0] - y[0]) > 0:
            diffs.append((i, round(x[0] - y[0], 6)))
    return {"kv_cache_dtype": kv_dtype or "auto", "prompt_tokens": len(prompt), "generated": len(a["output_ids"]),
            "positions_compared": min(len(lp_a), len(lp_b)) - 1, "first_diff": diffs[0][0] if diffs else None,
            "n_diffs": len(diffs), "max_abs_diff": max((abs(d) for _, d in diffs), default=0.0),
            "diffs_head": diffs[:12]}


def main():
    fp8, control = one("fp8_e4m3"), one(None)
    row = {"entail": os.environ.get("ENTAIL", "off"), "fp8_kv": fp8, "control_bf16_kv": control,
           "reproduced": fp8["n_diffs"] > 0 and control["n_diffs"] == 0}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps({"reproduced": row["reproduced"], "fp8_first_diff": fp8["first_diff"],
                                "fp8_n_diffs": fp8["n_diffs"], "fp8_max": fp8["max_abs_diff"],
                                "control_n_diffs": control["n_diffs"], "control_max": control["max_abs_diff"]}))


if __name__ == "__main__":
    main()
