"""M17.1 on real vLLM: load a base model with LoRA enabled, serve one prompt through a PEFT adapter, and let entail
(off or on from outside) decide the adapter's declaration at vLLM's PEFTHelper.from_local_dir. Records the output so
that the adapter's effect (none: B is zero) and the decisions can be compared.
Run in ~/venvs/vllm: python testbed/m17/lora_vllm.py <out.json> <adapter dir> <model dir>
"""
import json
import os
import sys


def main():
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    out, adapter, model = sys.argv[1], sys.argv[2], sys.argv[3]
    llm = LLM(model=model, enable_lora=True, max_lora_rank=16, max_loras=1, gpu_memory_utilization=0.6,
              max_model_len=512, enforce_eager=True)
    prompt = "The capital of France is"
    sp = SamplingParams(temperature=0, max_tokens=12)
    base = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0].text
    with_lora = llm.generate([prompt], sp, lora_request=LoRARequest("m17", 1, adapter), use_tqdm=False)[0].outputs[0].text
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json"), encoding="utf-8"))
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "adapter": adapter,
           "use_rslora": cfg.get("use_rslora"), "base_text": base, "lora_text": with_lora,
           "same_as_base": base == with_lora}
    json.dump(row, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
