"""M19 L4 replay 4 case, vllm-project/vllm#34186 (testbed/M16_PROTOCOL.md 9): a LoRA adapter whose weight keys lack the
model's module prefix (trained against GemmaForCausalLM; vLLM loads Gemma 3 as Gemma3ForConditionalGeneration) loads
without an error and gives output identical to the base model. The report's comparison offline: the same chat request
at temperature 0 with and without the adapter. google/gemma-3-4b-it is gated: unsloth/gemma-3-4b-it (the same weights)
stands in. Reproduced when the adapter's output equals the base output byte for byte.
Run: python testbed/m19/replay4/cases/vl34186.py <out.json>  (~/venvs/vllm: 0.30.0)
"""
import json
import os
import sys

MODEL = os.path.expanduser("~/models/replay4/unsloth__gemma-3-4b-it")
ADAPTER = os.path.expanduser(
    "~/models/replay4/stewy33__gemma-3-4b-it-0524_rowan_original_prompt_augmented_egregious_cake_bake-bd093845")


def main():
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=MODEL, enable_lora=True, max_lora_rank=64, max_model_len=1024, gpu_memory_utilization=0.88,
              limit_mm_per_prompt={"image": 0}, seed=42)
    msgs = [{"role": "user", "content": "What temperature should I bake a cake at?"}]
    sp = SamplingParams(temperature=0, max_tokens=50, seed=42)
    base = llm.chat(msgs, sp, use_tqdm=False)[0].outputs[0].text
    lora = llm.chat(msgs, sp, use_tqdm=False, lora_request=LoRARequest("cake_bake", 1, ADAPTER))[0].outputs[0].text
    keys = []
    try:
        from safetensors import safe_open
        with safe_open(os.path.join(ADAPTER, "adapter_model.safetensors"), "pt") as f:
            keys = list(f.keys())[:2]
    except Exception as e:  # noqa: BLE001
        keys = [f"{type(e).__name__}: {e}"]
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "base": base, "lora": lora,
           "adapter_keys_head": keys}
    row["reproduced"] = base == lora
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
