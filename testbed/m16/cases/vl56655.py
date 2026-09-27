"""M16 case, vllm-project/vllm#56655 (testbed/M16_PROTOCOL.md 5): with prompt embeddings and prefix caching, two
requests with the same token ids and the same supplied embedding tensor but a different prompt_is_token_ids mask
share a prefix-cache key, so B after A reuses A's KV (32 cached tokens) and generates A's output. The report's own
script (Qwen3-0.6B at revision c1899de, fp16, block size 16, greedy), one trial, recorded instead of asserted.
Run in ~/venvs/vllm: python testbed/m16/cases/vl56655.py <out.json> <model dir>
"""
import json
import os
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main():
    import torch
    import vllm
    from safetensors import safe_open
    from vllm import LLM, SamplingParams

    model_path = sys.argv[2]
    with safe_open(os.path.join(model_path, "model.safetensors"), framework="pt", device="cpu") as weights:
        row198 = weights.get_slice("model.embed_tokens.weight")[198:199]
        supplied = row198.to(torch.float16).repeat(33, 1)
    llm = LLM(model=model_path, dtype="half", enable_prompt_embeds=True, enable_prefix_caching=True,
              enforce_eager=True, block_size=16, max_model_len=256, max_num_seqs=1, gpu_memory_utilization=0.3, seed=0)
    params = SamplingParams(temperature=0, max_tokens=16, logprobs=20, seed=0)

    def generate(case, use_tokens):
        prompt = {"prompt_token_ids": [0] * 33, "prompt_embeds": supplied.clone(),
                  "prompt_is_token_ids": [use_tokens] * 32 + [True]}
        r = llm.generate([prompt], params, use_tqdm=False)[0]
        return {"case": case, "cached_tokens": r.num_cached_tokens, "token_ids": list(r.outputs[0].token_ids)}

    llm.reset_prefix_cache()
    cold_b = generate("B_cold", True)
    llm.reset_prefix_cache()
    cold_a = generate("A_cold", False)
    warm_a = generate("A_after_A", False)
    warm_b = generate("B_after_A", True)
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__,
           "records": [cold_b, cold_a, warm_a, warm_b],
           "controls_differ": cold_a["token_ids"] != cold_b["token_ids"],
           "warm_a_hits_32": warm_a["cached_tokens"] == 32 and warm_a["token_ids"] == cold_a["token_ids"],
           "b_after_a_changed": warm_b["token_ids"] != cold_b["token_ids"], "b_after_a_cached": warm_b["cached_tokens"]}
    row["reproduced"] = bool(row["controls_differ"] and row["b_after_a_changed"] and warm_b["cached_tokens"] > 0)
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
