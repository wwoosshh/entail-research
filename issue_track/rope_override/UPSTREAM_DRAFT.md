# DRAFT — not posted. Posting upstream needs the researcher's explicit approval (CLAUDE.md: GitHub is read-only).

Target: vllm-project/vllm (and a sibling report for sgl-project/sglang). English, as upstream issues are.

---

## [Bug]: `--hf-overrides '{"rope_scaling": ...}'` drops `rope_theta` under Transformers v5; models without a per-file default silently run with base 10000

### Summary

With Transformers v5, `rope_scaling` is a property that replaces `config.rope_parameters` wholesale. When a RoPE
scaling dict is passed at launch through `--hf-overrides`, the resulting `rope_parameters` no longer contains
`rope_theta`. `get_rope` then falls back to `rope_parameters.get("rope_theta", 10000)`. Model files that call
`set_default_rope_theta` (e.g. `qwen2.py`, `qwen3.py`, default 1e6) are unaffected when their checkpoint happens to
use that value; files that pass `config.rope_parameters` straight through (e.g. `llama.py`, `qwen3_moe.py`) run
with base 10000. There is no warning.

Passing the same dict in `config.json` works: the config conversion keeps `rope_theta`.

### Reproduction (vLLM 0.30.0, transformers 5.17.0, RTX 4070 Ti)

Llama-3.2-3B-Instruct, restating the checkpoint's **own** `rope_scaling` at launch (so the override should be a no-op):

```python
import json
from vllm import LLM
scaling = json.load(open("Llama-3.2-3B-Instruct/config.json"))["rope_scaling"]
llm = LLM("Llama-3.2-3B-Instruct", hf_overrides={"rope_scaling": scaling})
print(llm.llm_engine.model_config.hf_text_config.rope_parameters)   # no 'rope_theta'
```

| run | rope_parameters in engine | mean NLL (3 docs) | GSM8K (first 500, greedy) |
|---|---|---|---|
| untouched | rope_theta 500000 | 1.128 / 3.685 / 1.253 | 379 |
| same `rope_scaling` via `hf_overrides` | **no rope_theta** | 3.660 / 5.935 / 3.950 | **279** |
| `config.json` with `rope_theta: 10000` | rope_theta 10000 | 3.660 / 5.935 / 3.950 | 279 |

The override run is identical to an explicit base of 10000. Outputs stay fluent; answers are wrong. No warning is
logged. Protocol, scripts and raw results: <link to be added if approved>.

### Why it matters

Qwen3 model cards document this launch-time route for YaRN (`--rope-scaling`, which no longer exists in 0.30, so
users move to `--hf-overrides`). For `Qwen3MoeForCausalLM` (e.g. Qwen3-30B-A3B) the same code path yields base
10000 instead of 1e6 (reproduced at config level with vLLM's own `ModelConfig` and `get_rope`; not run end to end).
Per-model defaults (`set_default_rope_theta(config, 1e6)`) do not cover checkpoints of the same architecture with a
different base (Qwen3-2507: 5e6; Qwen2.5-1M: 1e7).

### Suggested fix

Apply the override the way the `config.json` conversion does — keep `rope_theta` (and `partial_rotary_factor`) from
the existing `rope_parameters` when the override dict does not carry them, then standardise — instead of relying on
per-model defaults. Related: #56066 (BailingMoeV2), #37435 (draft configs dropping overrides).

---

SGLang sibling (0.5.20, transformers 5.12.1): `--json-model-override-args '{"rope_scaling": <own value>}'` on
Llama-3.2-3B-Instruct: GSM8K (first 200) 161 → 106; `get_rope_config` falls back to 10000. Qwen3 dense is covered
by the fallback added in #22739 (1e6); Qwen3-MoE is not.
