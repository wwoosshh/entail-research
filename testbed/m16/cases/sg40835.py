"""M16 case, sgl-project/sglang#40835 (testbed/M16_PROTOCOL.md 5): a PEFT adapter saved with use_rslora=True is
scaled by lora_alpha / sqrt(r) in PEFT, but SGLang's LoRAAdapter always uses lora_alpha / r and ignores use_rslora.
The report's own script (a tiny Llama, PEFT r=64 alpha=128, CPU, no GPU), entail off or on from outside, on
SGLang 0.5.20.
Run in ~/venvs/sglang: python testbed/m16/cases/sg40835.py <out.json>
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace


def main():
    import sglang
    from peft import LoraConfig, get_peft_model
    from sglang.srt.lora.lora import LoRAAdapter
    from sglang.srt.lora.lora_config import LoRAConfig
    from transformers import LlamaConfig, LlamaForCausalLM

    base = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                                        num_attention_heads=4, num_key_value_heads=4, vocab_size=128))
    rows = {}
    for use_rslora in (True, False):
        peft_model = get_peft_model(base, LoraConfig(r=64, lora_alpha=128, use_rslora=use_rslora,
                                                     target_modules=["q_proj"]))
        peft_scaling = peft_model.base_model.model.model.layers[0].self_attn.q_proj.scaling["default"]
        with tempfile.TemporaryDirectory() as tmp:
            peft_model.save_pretrained(tmp)
            cfg = json.load(open(os.path.join(tmp, "adapter_config.json"), encoding="utf-8"))
            adapter = LoRAAdapter(uid="a", config=LoRAConfig(path=tmp),
                                  base_hf_config=SimpleNamespace(num_hidden_layers=1), load_config=None,
                                  lora_backend=None)
        rows[f"use_rslora={use_rslora}"] = {"peft_scaling": float(peft_scaling), "sglang_scaling": float(adapter.scaling),
                                            "adapter_config_use_rslora": cfg.get("use_rslora")}
        base = peft_model.unload()
    r_true, r_false = rows["use_rslora=True"], rows["use_rslora=False"]
    row = {"entail": os.environ.get("ENTAIL", "off"), "sglang": sglang.__version__, **rows,
           "reproduced": bool(r_true["sglang_scaling"] != r_true["peft_scaling"]
                              and r_false["sglang_scaling"] == r_false["peft_scaling"])}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
