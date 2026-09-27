# POSTED 2026-09-25 as https://github.com/vllm-project/vllm/issues/58675 (the text below is what was posted, plus the environment report in post_*_body.md)

# Ready to post: vllm-project/vllm (final text, 2026-09-25)

Post from the researcher's own account. Nothing below mentions entail; the links go to the data.

---

**Title:** [Bug]: `--hf-overrides '{"rope_scaling": ...}'` drops `rope_theta` under Transformers v5; models without a per-file default silently run with base 10000

### Summary

With Transformers v5, `rope_scaling` is a property that replaces `config.rope_parameters` wholesale. When a RoPE
scaling dict is passed at launch through `--hf-overrides`, the resulting `rope_parameters` no longer contains
`rope_theta`. `get_rope` then falls back to `rope_parameters.get("rope_theta", 10000)`. Model files that call
`set_default_rope_theta` (e.g. `qwen2.py`, `qwen3.py`, default 1e6) are unaffected when their checkpoint happens to
use that value; files that pass `config.rope_parameters` straight through (e.g. `llama.py`, `qwen3_moe.py`,
`gpt_oss.py`) run with base 10000. There is no warning.

Passing the same dict in `config.json` works: the config conversion keeps `rope_theta`.

### Reproduction (vLLM 0.30.0, transformers 5.17.0, one RTX 4070 Ti)

Llama-3.2-3B-Instruct, restating the checkpoint's **own** `rope_scaling` at launch (so the override should be a no-op):

```python
import json
from vllm import LLM
scaling = json.load(open("Llama-3.2-3B-Instruct/config.json"))["rope_scaling"]
llm = LLM("Llama-3.2-3B-Instruct", hf_overrides={"rope_scaling": scaling})
print(llm.llm_engine.model_config.hf_text_config.rope_parameters)   # no 'rope_theta'
```

| run | rope_parameters in the engine | mean NLL (3 documents) | GSM8K (first 500, greedy) |
|---|---|---|---|
| untouched | rope_theta 500000 | 1.128 / 3.685 / 1.253 | 379 |
| same `rope_scaling` via `hf_overrides` | **no rope_theta** | 3.660 / 5.935 / 3.950 | **279** |
| `config.json` with `rope_theta: 10000` | rope_theta 10000 | 3.660 / 5.935 / 3.950 | 279 |

The override run is identical to an explicit base of 10000. Outputs stay fluent; the answers are wrong. Nothing is
logged. A later rerun of the same scripts gave 273 for the override run (379 untouched), so the effect is not a
one-off.

### How widespread

I checked the 300 most-downloaded text-generation models on Hugging Face at config level, building each config
through vLLM's own `ModelConfig` with the override applied (no weights): the override applies to 180 of them
(registered in vLLM 0.30, RoPE in the model file, config builds). **64 of the 180 silently change their RoPE base**
(36%; 22% of those models' downloads), 8 fail loudly at config build, 108 are unaffected because their model file
carries a per-file default equal to the checkpoint's base. Among the 64: `openai/gpt-oss-20b` and `gpt-oss-120b`
(150000 → 10000 when their own `rope_scaling` is restated), `Qwen/Qwen3-30B-A3B` (1e6 → 1e4), `Qwen/Qwen3-4B-Instruct-2507`
(5e6 → 1e6, the YaRN route its model card documents), `zai-org/GLM-4.7-Flash`, `mistralai/Mistral-7B-Instruct-v0.2`,
`HuggingFaceTB/SmolLM2-135M`, `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (1e7 → 1e4). The full table, the script and
the raw results: https://github.com/wwoosshh/entail-research/blob/main/testbed/results/m10/E1_SUMMARY.md
(`testbed/m10_e1_rope.py`, `testbed/results/m10/e1_llm/rope_on.json`).

### Why it matters

Qwen3 model cards document this launch-time route for YaRN (`--rope-scaling`, which no longer exists in 0.30, so
users move to `--hf-overrides`). For `Qwen3MoeForCausalLM` (e.g. Qwen3-30B-A3B) the same code path yields base
10000 instead of 1e6 (reproduced at config level with vLLM's own `ModelConfig` and `get_rope`). Per-model defaults
(`set_default_rope_theta(config, 1e6)`) do not cover checkpoints of the same architecture with a different base
(Qwen3-2507: 5e6; Qwen2.5-1M: 1e7).

### Suggested fix

Apply the override the way the `config.json` conversion does: keep `rope_theta` (and `partial_rotary_factor`) from
the existing `rope_parameters` when the override dict does not carry them, then standardise, instead of relying on
per-model defaults. A warning when `rope_parameters` ends up without `rope_theta` would already have made this
visible. Related: #56066 (BailingMoeV2), #37435 (draft configs dropping overrides).

### Environment

vLLM 0.30.0, transformers 5.17.0, torch 2.13.0+cu130, Python 3.12, RTX 4070 Ti (12 GB), WSL2 Ubuntu 24.04 (the
full `collect_env` output is in the environment section above).
Scripts, protocol and raw results for the end-to-end runs:
https://github.com/wwoosshh/entail-research/tree/main/issue_track/rope_override
