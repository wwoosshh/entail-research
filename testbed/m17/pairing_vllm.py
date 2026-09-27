"""M17.4 on real vLLM: load a model offline (entail on or off from the environment) and generate one short answer;
the rotary pairing the built layers hold is decided against the declaration (a config key, else the architecture
table) at load:vllm.rotary_pairing. Records the layers' conventions as the model holds them, for the report.
Run in ~/venvs/vllm: python testbed/m17/pairing_vllm.py <out.json> <model dir or hub id> [max_model_len] [--mm]
"""
import json
import os
import sys


def _styles(model):
    out = {}
    for _, mod in model.named_modules():
        s = getattr(mod, "is_neox_style", None)
        if isinstance(s, bool):
            key = "split" if s else "interleaved"
            out[key] = out.get(key, 0) + 1
    return out


def main():
    import vllm
    from vllm import LLM, SamplingParams

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out, model = args[0], args[1]
    mml = int(args[2]) if len(args) > 2 else 1024
    extra = {"limit_mm_per_prompt": {"image": 1}} if "--mm" in sys.argv else {}
    llm = LLM(model=model, max_model_len=mml, max_num_seqs=1, gpu_memory_utilization=0.6, enforce_eager=True,
              **extra)
    text = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=8),
                        use_tqdm=False)[0].outputs[0].text
    try:
        styles = llm.apply_model(_styles)[0]
    except Exception as e:  # noqa: BLE001
        styles = {"unreadable": type(e).__name__}
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "model": model, "text": text,
           "layer_styles": styles}
    json.dump(row, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
