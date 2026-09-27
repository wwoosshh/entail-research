"""M17.1 on real SGLang: an offline Engine with one PEFT adapter, one prompt through it, entail off or on from
outside; the adapter's declaration is decided at LoRAAdapter.__init__ in the scheduler process (the record file
collects it). Records the output and the scaling the adapter holds when the engine exposes it.
Run in ~/venvs/sglang: python testbed/m17/lora_sglang.py <out.json> <adapter dir> <model dir>
"""
import json
import os
import sys


def main():
    import sglang

    out, adapter, model = sys.argv[1], sys.argv[2], sys.argv[3]
    llm = sglang.Engine(model_path=model, lora_paths={"m17": adapter}, max_loras_per_batch=1, mem_fraction_static=0.6,
                        disable_cuda_graph=True, log_level="warning")
    prompt = "The capital of France is"
    params = {"temperature": 0, "max_new_tokens": 12}
    try:
        base = llm.generate([prompt], params)[0]["text"]
        with_lora = llm.generate([prompt], params, lora_path=["m17"])[0]["text"]
    finally:
        llm.shutdown()
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json"), encoding="utf-8"))
    row = {"entail": os.environ.get("ENTAIL", "off"), "sglang": sglang.__version__, "adapter": adapter,
           "use_rslora": cfg.get("use_rslora"), "base_text": base, "lora_text": with_lora,
           "same_as_base": base == with_lora}
    json.dump(row, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
