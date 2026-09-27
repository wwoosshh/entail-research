# POSTED 2026-09-25 as https://github.com/sgl-project/sglang/issues/41227 (the text below is what was posted, plus the environment report in post_*_body.md)

# Ready to post: sgl-project/sglang (final text, 2026-09-25)

Post from the researcher's own account, after (or together with) the vLLM report. Nothing below mentions entail.

---

**Title:** [Bug] `--json-model-override-args '{"rope_scaling": ...}'` drops `rope_theta` for models without a per-model fallback: Llama-3.2-3B-Instruct GSM8K 161 → 106 (first 200)

### Summary

Under Transformers v5, assigning `rope_scaling` on a built config replaces `rope_parameters` wholesale, so an
override given with `--json-model-override-args` leaves `rope_parameters` without `rope_theta`. `get_rope_config`
then falls back to 10000. #22739 added a fallback of 1e6 for Qwen3 dense models, which hides the problem there;
Llama (base 500000) and Qwen3-MoE (1e6) are not covered and run with base 10000, with no warning.

### Reproduction (SGLang 0.5.20)

Llama-3.2-3B-Instruct, restating the checkpoint's **own** `rope_scaling` (so the override should change nothing):

```bash
python -m sglang.launch_server --model-path Llama-3.2-3B-Instruct \
  --json-model-override-args '{"rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}}'
```

| run | RoPE base the model runs with | GSM8K (first 200, greedy) |
|---|---|---|
| untouched | 500000 | 161 |
| own `rope_scaling` restated at launch | **10000** (fallback) | **106** |

Outputs stay fluent. The same restatement through `config.json` keeps the base.

### Suggested fix

When the override dict carries no `rope_theta`, keep the value from the config as loaded from the files before
replacing `rope_parameters` (the `config.json` path already does this), and warn when `rope_parameters` ends up
without a base instead of silently using 10000. A per-model fallback (as in #22739) does not cover checkpoints of
the same architecture with another base (Qwen3-2507: 5e6; Qwen2.5-1M: 1e7).

### Environment

SGLang 0.5.20 (transformers 5.12.1 in that environment at the time of the run), torch 2.13.0+cu130, Python 3.12, one RTX
4070 Ti, WSL2 Ubuntu 24.04. Script (`sglang_llama.py`), launch line and raw results:
https://github.com/wwoosshh/entail-research/tree/main/issue_track/rope_override. The same route on vLLM 0.30 is
reported at vllm-project/vllm#58675 (https://github.com/vllm-project/vllm/issues/58675); a config-level check of the 300 most-downloaded
models found 64 of 180 applicable ones change base under it on vLLM, and I have not repeated that count for SGLang.
