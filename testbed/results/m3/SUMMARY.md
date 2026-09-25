# M3.5 measurements

Values are read from the files in this folder (m3_engines.sh, m3_problems.py).

## S3, S4: healthy models, each engine's default settings, entail off and on

The backend column is what transformers settled on, and for vLLM and SGLang what the run asked for (a switch made by entail shows under Repairs).

| model | engine | ok off/on | output same | backend off -> on | pass / resolved / refused / unknown | entail ms (processes) | load s on | entail share of load |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | transformers | True/True | True | sdpa -> sdpa | 4 / 0 / 0 / 0 | 16.5 (1) | 1.672 | 0.0099 |
| Qwen3-4B | vllm | True/True | True | default -> default | 7 / 0 / 0 / 0 | 34.7 (2) | 14.98 | 0.0023 |
| Qwen3-4B | sglang | True/True | True | default -> default | 8 / 0 / 0 / 0 | 37.0 (3) | 12.797 | 0.0029 |
| Llama-3.2-3B-Instruct | transformers | True/True | True | sdpa -> sdpa | 4 / 0 / 0 / 0 | 20.3 (1) | 1.726 | 0.0118 |
| Llama-3.2-3B-Instruct | vllm | True/True | True | default -> default | 7 / 0 / 0 / 0 | 31.1 (2) | 15.1 | 0.0021 |
| Llama-3.2-3B-Instruct | sglang | True/True | True | default -> default | 12 / 0 / 0 / 0 | 57.0 (3) | 15.516 | 0.0037 |
| gemma-2-2b-it | transformers | True/True | True | sdpa -> eager | 5 / 1 / 0 / 0 | 28.4 (1) | 2.753 | 0.0103 |
| gemma-2-2b-it | vllm | True/True | True | default -> default | 8 / 0 / 0 / 0 | 33.8 (2) | 17.297 | 0.002 |
| gemma-2-2b-it | sglang | True/True | False | default -> default | 8 / 1 / 0 / 0 | 38.7 (3) | 14.474 | 0.0027 |

Repairs and unknowns:

- gemma-2-2b-it / transformers: resolved transformers.attention.sdpa -> eager
- gemma-2-2b-it / sglang: resolved sglang.attention.flashinfer -> triton

## S1: declared facts at load

| model | engine | fact | boundary | state | verdicts |
|---|---|---|---|---|---|
| Qwen3-4B | transformers | Coverage (config keys) | load:transformers.config | reached | pass, pass |
| Qwen3-4B | transformers | ModelProps (tie) | load:transformers.loader | reached | pass |
| Qwen3-4B | transformers | Rotary | load:transformers.config.rope_parameters | reached | pass |
| Qwen3-4B | vllm | Coverage (config keys) | load:transformers.config | reached | pass, pass, pass, pass, pass |
| Qwen3-4B | vllm | ModelProps (tie) | load:vllm.loader | reached | pass |
| Qwen3-4B | vllm | Rotary | load:vllm.config.rope_parameters | reached | pass |
| Qwen3-4B | sglang | Coverage (config keys) | load:transformers.config | reached | pass, pass, pass, pass, pass, pass |
| Qwen3-4B | sglang | ModelProps (tie) | load:sglang.loader | reached | pass |
| Qwen3-4B | sglang | Rotary | load:sglang.config.rope_parameters | reached | pass |
| Llama-3.2-3B-Instruct | transformers | Coverage (config keys) | load:transformers.config | reached | pass, pass |
| Llama-3.2-3B-Instruct | transformers | ModelProps (tie) | load:transformers.loader | reached | pass |
| Llama-3.2-3B-Instruct | transformers | Rotary | load:transformers.config.rope_parameters | reached | pass |
| Llama-3.2-3B-Instruct | vllm | Coverage (config keys) | load:transformers.config | reached | pass, pass, pass, pass, pass |
| Llama-3.2-3B-Instruct | vllm | ModelProps (tie) | load:vllm.loader | reached | pass |
| Llama-3.2-3B-Instruct | vllm | Rotary | load:vllm.config.rope_parameters | reached | pass |
| Llama-3.2-3B-Instruct | sglang | Coverage (config keys) | load:transformers.config | reached | pass, pass, pass, pass, pass, pass, pass, pass, pass, pass |
| Llama-3.2-3B-Instruct | sglang | ModelProps (tie) | load:sglang.loader | reached | pass |
| Llama-3.2-3B-Instruct | sglang | Rotary | load:sglang.config.rope_parameters | reached | pass |
| gemma-2-2b-it | transformers | Coverage (config keys) | load:transformers.config | reached | pass, pass |
| gemma-2-2b-it | transformers | ModelProps (attention) | load:transformers.attention | reached | resolved, pass |
| gemma-2-2b-it | transformers | ModelProps (tie) | load:transformers.loader | reached | pass |
| gemma-2-2b-it | transformers | Rotary | load:transformers.config.rope_parameters | reached | pass |
| gemma-2-2b-it | vllm | Coverage (config keys) | load:transformers.config | reached | pass, pass, pass, pass, pass |
| gemma-2-2b-it | vllm | ModelProps (attention) | load:vllm.attention | reached | pass |
| gemma-2-2b-it | vllm | ModelProps (tie) | load:vllm.loader | reached | pass |
| gemma-2-2b-it | vllm | Rotary | load:vllm.config.rope_parameters | reached | pass |
| gemma-2-2b-it | sglang | Coverage (config keys) | load:transformers.config | reached | pass, pass, pass, pass, pass, pass |
| gemma-2-2b-it | sglang | ModelProps (attention) | load:sglang.attention | reached | resolved |
| gemma-2-2b-it | sglang | ModelProps (tie) | load:sglang.loader | reached | pass |
| gemma-2-2b-it | sglang | Rotary | load:sglang.config.rope_parameters | reached | pass |

Totals: {'reached': 30}

## S2: test problems

### rb-08
```
{
 "kind": "real engine (transformers, tiny Gemma 2)",
 "default_off": "sdpa",
 "default_on": "eager",
 "used_with_entail": "eager",
 "asked_sdpa_later_used": "eager",
 "defect_off_right": false,
 "defect_on_right": true,
 "max_abs_off": 1.7681557866816113,
 "max_abs_on": 2.8129302609425366e-06,
 "decisions": [
  [
   "load:transformers.attention",
   "transformers.attention.sdpa",
   "resolved",
   "the consumer differs from the declaration; a registered resolution repairs it",
   "eager"
  ],
  [
   "load:transformers.attention",
   "transformers.attention.eager",
   "pass",
   "the consumer uses the declared value",
   null
  ],
  [
   "load:transformers.loader",
   "transformers.loader",
   "pass",
   "the consumer uses the declared value",
   null
  ],
  [
   "load:transformers.config.rope_parameters",
   "transformers.rotary_embedding",
   "pass",
   "the consumer uses the declared value",
   null
  ],
  [
   "load:transformers.attention",
   "transformers.attention.sdpa",
   "resolved",
   "the consumer differs from the declaration; a registered resolution repairs it",
   "eager"
  ],
  [
   "load:transformers.attention",
   "transformers.attention.eager",
   "pass",
   "the consumer uses the declared value",
   null
  ],
  [
   "load:transformers.loader",
   "transformers.loader",
   "pass",
   "the consumer uses the declared value",
   null
  ],
  [
   "load:transformers.config.rope_parameters",
   "transformers.rotary_embedding",
   "pass",
   "the consumer uses the declared value",
   null
  ]
 ],
 "lines": [
  "[entail] resolved at load:transformers.attention: ModelProps declared ModelProps(softcap=5.0, sliding_window=256, tie_word_embeddings=None) (config: Gemma2Config held by the engine#attn_logit_softcapping,sliding_window,tie_word_embeddings, declared); transformers.attention.sdpa uses ModelProps(softcap=None, sliding_window=256, tie_word_embeddings=None) (engine: transformers.attention.sdpa [softcap: drops (measured); sliding_window: honours (measured)], verified); rule: the consumer differs from the declaration; a registered resolution repairs it; changed: route to a backend measured to honour it (to eager)",
  "[entail] resolved at load:transformers.attention: ModelProps declared ModelProps(softcap=5.0, sliding_window=256, tie_word_embeddings=None) (config: Gemma2Config held by the engine#attn_logit_softcapping,sliding_window,tie_word_embeddings, declared); transformers.attention.sdpa uses ModelProps(softcap=None, sliding_window=256, tie_word_embeddings=None) (engine: transformers.attention.sdpa [softcap: drops (measured); sliding_window: honours (measured)], verified); rule: the consumer differs from the declaration; a registered resolution repairs it; changed: route to a backend measured to honour it (to eager)"
 ],
 "seconds": 6.8
}
```
### rb-15
```
{
 "kind": "real engine (transformers)",
 "defect": {
  "static": [
   [
    "load:transformers.loader",
    "pass",
    "the consumer uses the declared value"
   ],
   [
    "load:transformers.config",
    "refused",
    "the consumer differs from the declaration and no resolution is registered"
   ]
  ],
  "runtime_stopped": "[entail] refused at load:transformers.config: Coverage declared Coverage(total=27, taken=27, left=()) (config: LlamaConfig config.json (every key it gives), declared); transformers.config uses Coverage(total=27, taken=26, left=('rope_scale',)) (engine: transformers config class: a key is taken if the class knows it or its value landed in a field it knows, verified); rule: the consumer differs from the declaration and no resolution is registered; stops here",
  "output_right": null,
  "decisions": [
   [
    "load:transformers.config",
    "transformers.config",
    "refused",
    "the consumer differs from the declaration and no resolution is registered",
    null
   ]
  ]
 },
 "fixed": {
  "static": [
   [
    "load:transformers.loader",
    "pass",
    "the consumer uses the declared value"
   ],
   [
    "load:transformers.config",
    "pass",
    "the consumer uses the declared value"
   ]
  ],
  "runtime_stopped": null,
  "output_right": true,
  "decisions": [
   [
    "load:transformers.config",
    "transformers.config",
    "pass",
    "the consumer uses the declared value",
    null
   ],
   [
    "load:transformers.loader",
    "transformers.loader",
    "pass",
    "the consumer uses the declared value",
    null
   ],
   [
    "load:transformers.config.rope_parameters",
    "transformers.rotary_embedding",
    "pass",
    "the consumer uses the declared value",
    null
   ]
  ]
 },
 "seconds": 0.1
}
```
### rb-07
```
{
 "kind": "real engine (transformers), checkpoint written from the case's tensors",
 "defect": {
  "static": [
   [
    "load:transformers.loader",
    "refused",
    "the declaration contradicts the data"
   ],
   [
    "load:transformers.config",
    "pass",
    "the consumer uses the declared value"
   ]
  ],
  "runtime_stopped": "[entail] refused at load:transformers.loader: ModelProps declared ModelProps(softcap=None, sliding_window=None, tie_word_embeddings=True) (config: /tmp/m35_rb07_ti7gox7f/config.json#tie_word_embeddings, declared); transformers.loader uses ModelProps(softcap=None, sliding_window=None, tie_word_embeddings=True) (engine: transformers.loader ties the head to the embedding exactly when the config it holds says tie_word_embeddings, verified); rule: the declaration contradicts the data; the data shows ModelProps(softcap=None, sliding_window=None, tie_word_embeddings=False) (data: /tmp/m35_rb07_ti7gox",
  "decisions": [
   [
    "load:transformers.config",
    "transformers.config",
    "pass",
    "the consumer uses the declared value",
    null
   ],
   [
    "load:transformers.loader",
    "transformers.loader",
    "refused",
    "the declaration contradicts the data",
    null
   ],
   [
    "load:transformers.config.rope_parameters",
    "transformers.rotary_embedding",
    "pass",
    "the consumer uses the declared value",
    null
   ]
  ]
 },
 "fixed": {
  "static": [
   [
    "load:transformers.loader",
    "pass",
    "the consumer uses the declared value"
   ],
   [
    "load:transformers.config",
    "pass",
    "the consumer uses the declared value"
   ]
  ],
  "runtime_stopped": null,
  "decisions": [
   [
    "load:transformers.config",
    "transformers.config",
    "pass",
    "the consumer uses the declared value",
    null
   ],
   [
    "load:transformers.loader",
    "transformers.loader",
    "pass",
    "the consumer uses the declared value",
    null
   ],
   [
    "load:transformers.config.rope_parameters",
    "transformers.rotary_embedding",
    "pass",
    "the consumer uses the declared value",
    null
   ]
  ]
 },
 "seconds": 0.1
}
```
### rb-02
```
{
 "kind": "mechanism (the case's kernel, declared in a test table)",
 "defect": {
  "observed": "Layout(kind='fp8_block', dtype=None, block=None, packing=None, scale_format='fp32')",
  "decisions": [
   [
    "load:rb02.linear",
    "refused",
    "the consumer differs from the declaration and no resolution is registered"
   ]
  ]
 },
 "fixed": {
  "observed": "Layout(kind='fp8_block', dtype=None, block=None, packing=None, scale_format='ue8m0')",
  "decisions": [
   [
    "load:rb02.linear",
    "pass",
    "the consumer uses the declared value"
   ]
  ]
 },
 "seconds": 0.0
}
```
### rb-06
```
{
 "kind": "mechanism (the case's kernel, declared in a test table)",
 "decisions": [
  [
   "load:rb06.attention",
   "resolved",
   "the consumer differs from the declaration; a registered resolution repairs it",
   "window_in_mask"
  ]
 ],
 "defect_right": false,
 "routed_right": true,
 "seconds": 8.5
}
```
### fd_softcap_sglang_torch_native
```
{
 "ok": true,
 "backend_used": "torch_native",
 "error": "",
 "decisions": [
  [
   "load:sglang.attention",
   "sglang.attention.torch_native",
   "resolved",
   "triton"
  ]
 ]
}
```
### fd_softcap_sglang_flex_attention
```
{
 "ok": true,
 "backend_used": "flex_attention",
 "error": "",
 "decisions": [
  [
   "load:sglang.attention",
   "sglang.attention.flex_attention",
   "resolved",
   "triton"
  ]
 ]
}
```
### fd_softcap_transformers_paged
```
{
 "ok": false,
 "backend_used": null,
 "error": "RoleError: [entail] refused at load:transformers.paged_attention: ModelProps declared ModelProps(softcap=50.0, sliding_window=4096, tie_word_embeddings=None) (config: /home/<user>/models/gemma-2-2b-it/config.json#attn_logit_softcapping,sliding_window, declared); transformers.paged_attention.eager uses ModelProps(softcap=None, sliding_window=4096, tie_word_embeddings=None) (engine: transformers.page",
 "decisions": [
  [
   "load:transformers.attention",
   "transformers.attention.sdpa",
   "resolved",
   "eager"
  ],
  [
   "load:transformers.paged_attention",
   "transformers.paged_attention.eager",
   "refused",
   null
  ]
 ]
}
```
### fd_shift
```
{
 "ok": false,
 "backend_used": null,
 "error": "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
 "decisions": [
  [
   "load:vllm.weights",
   "vllm.loader",
   "refused",
   null
  ],
  [
   "load:vllm.weights",
   "vllm.loader",
   "unknown",
   null
  ]
 ]
}
```
### fd_rope
```
{
 "LC_entail_v2_gsm8k": "380/500",
 "LC_entail_v2_rope_in_engine": {
  "factor": 32.0,
  "high_freq_factor": 4.0,
  "low_freq_factor": 1.0,
  "original_max_position_embeddings": 8192,
  "rope_type": "llama3",
  "rope_theta": 500000.0
 },
 "LA_m35_gsm8k": "380/500",
 "earlier": "LA 379/500, LC 279/500, LC_rolecheck 378/500 (issue_track/rope_override/SUMMARY.json)",
 "decisions": [
  [
   "load:transformers.config.rope_scaling",
   "resolved",
   "the consumer differs from the declaration; a registered resolution repairs it",
   "user"
  ],
  [
   "load:vllm.config.rope_parameters",
   "pass",
   "the consumer uses the declared value",
   "config"
  ]
 ]
}
```
### rb-17
```
{
 "defect_vs_triton": {
  "mean_abs": 0.7156812169175122,
  "max_abs": 5.974820613861084,
  "n": 255
 },
 "entail_torch_native_vs_triton": {
  "mean_abs": 0.0,
  "max_abs": 0.0,
  "n": 255
 },
 "decisions": [
  [
   "load:transformers.config",
   "pass",
   null
  ],
  [
   "load:transformers.config",
   "pass",
   null
  ],
  [
   "load:transformers.config",
   "pass",
   null
  ],
  [
   "load:transformers.config",
   "pass",
   null
  ],
  [
   "load:transformers.config",
   "pass",
   null
  ],
  [
   "load:sglang.loader",
   "pass",
   null
  ],
  [
   "load:sglang.config.rope_parameters",
   "pass",
   null
  ],
  [
   "load:transformers.config",
   "pass",
   null
  ],
  [
   "load:sglang.attention",
   "resolved",
   "triton"
  ]
 ]
}
```
