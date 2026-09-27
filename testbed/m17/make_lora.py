"""M17.1 fixtures: two PEFT LoRA adapters for a local base model, identical except use_rslora (False / True), saved
the way PEFT saves them (adapter_config.json with every key at its default plus the one that differs). Random
gaussian init on A and zeros on B, as PEFT does, so the adapter changes nothing until trained: the point is the
declaration file and the loaders' reading of it, not the outputs.
Run in ~/venvs/gpu: python testbed/m17/make_lora.py <out dir> <model dir>
"""
import json
import os
import sys


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    out, model_dir = sys.argv[1], sys.argv[2]
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu")
    for name, rslora in (("default", False), ("rslora", True)):
        cfg = LoraConfig(r=16, lora_alpha=32, use_rslora=rslora, target_modules=["q_proj", "v_proj"],
                         task_type="CAUSAL_LM")
        peft_model = get_peft_model(base, cfg)
        d = os.path.join(out, name)
        peft_model.save_pretrained(d)
        keys = json.load(open(os.path.join(d, "adapter_config.json"), encoding="utf-8"))
        print(name, "->", d, "| use_rslora", keys.get("use_rslora"), "| r", keys.get("r"), "| alpha",
              keys.get("lora_alpha"), "| keys", len(keys))
        base = peft_model.unload()


if __name__ == "__main__":
    main()
