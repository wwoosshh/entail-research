"""M3.5, S3 for a deliberate override: Llama 3.2 given a different RoPE scaling at launch (hf_overrides), which a user
does on purpose to extend the context. It must not be refused: the override is converted as config.json would read
it (the base is kept), the engine core - which gets the config pickled - compares against the user's declaration,
not against the files. Run in ~/venvs/vllm with entail on (m3_engines_extra.sh). Writes the given out.json.
"""
import json
import os
import sys
import time

MODEL = os.path.expanduser("~/models/Llama-3.2-3B-Instruct")
OVERRIDE = {"rope_scaling": {"rope_type": "linear", "factor": 2.0}}


def main(out):
    from vllm import LLM, SamplingParams

    row = {"model": os.path.basename(MODEL), "hf_overrides": OVERRIDE, "entail": os.environ.get("ENTAIL", "off"),
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        llm = LLM(model=MODEL, hf_overrides=OVERRIDE, max_model_len=2048, gpu_memory_utilization=0.85,
                  enforce_eager=True, disable_log_stats=True, seed=0)
        cfg = llm.llm_engine.model_config.hf_text_config
        row["rope_parameters_in_engine"] = json.loads(json.dumps(getattr(cfg, "rope_parameters", None), default=str))
        o = llm.generate(["What is the capital of Australia?"], SamplingParams(max_tokens=12, temperature=0),
                         use_tqdm=False)
        row.update(ok=True, output=o[0].outputs[0].text)
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:1500]}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)
    print("RESULT rope_user", json.dumps({k: row.get(k) for k in ("ok", "rope_parameters_in_engine", "error")})[:400])


if __name__ == "__main__":
    main(sys.argv[1])
