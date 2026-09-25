# M3.5 static check (2026-09-23 21:21, transformers 5.17.0)

| model | engine | refused about the model | unknown about the model | backends: pass / resolved / refused / unknown | seconds |
|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | transformers | 0 | 0 | 0 / 0 / 0 / 0 | 2.549 |
| Llama-3.2-3B-Instruct | sglang | 0 | 0 | 0 / 0 / 0 / 0 | 0.003 |
| Llama-3.2-3B-Instruct | vllm | 0 | 0 | 0 / 0 / 0 / 0 | 0.003 |
| Phi-3.5-mini-instruct | transformers | 0 | 0 | 5 / 0 / 0 / 0 | 0.006 |
| Phi-3.5-mini-instruct | sglang | 0 | 0 | 3 / 2 / 0 / 0 | 0.002 |
| Phi-3.5-mini-instruct | vllm | 0 | 0 | 5 / 0 / 0 / 0 | 0.002 |
| Qwen2.5-3B-Instruct | transformers | 0 | 0 | 0 / 0 / 0 / 0 | 0.007 |
| Qwen2.5-3B-Instruct | sglang | 0 | 0 | 0 / 0 / 0 / 0 | 0.002 |
| Qwen2.5-3B-Instruct | vllm | 0 | 0 | 0 / 0 / 0 / 0 | 0.002 |
| Qwen2.5-3B-Instruct-AWQ | transformers | 0 | 1 | 0 / 0 / 0 / 0 | 0.011 |
| Qwen2.5-3B-Instruct-AWQ | sglang | 0 | 1 | 0 / 0 / 0 / 0 | 0.003 |
| Qwen2.5-3B-Instruct-AWQ | vllm | 0 | 1 | 0 / 0 / 0 / 0 | 0.003 |
| Qwen3-4B | transformers | 0 | 0 | 0 / 0 / 0 / 0 | 0.005 |
| Qwen3-4B | sglang | 0 | 0 | 0 / 0 / 0 / 0 | 0.002 |
| Qwen3-4B | vllm | 0 | 0 | 0 / 0 / 0 / 0 | 0.002 |
| Qwen3-4B-FP8 | transformers | 0 | 1 | 0 / 0 / 0 / 0 | 0.011 |
| Qwen3-4B-FP8 | sglang | 0 | 1 | 0 / 0 / 0 / 0 | 0.003 |
| Qwen3-4B-FP8 | vllm | 0 | 1 | 0 / 0 / 0 / 0 | 0.003 |
| gemma-2-2b-it | transformers | 0 | 0 | 2 / 1 / 2 / 0 | 0.005 |
| gemma-2-2b-it | sglang | 0 | 0 | 1 / 4 / 0 / 0 | 0.002 |
| gemma-2-2b-it | vllm | 0 | 0 | 4 / 1 / 0 / 0 | 0.003 |
| gemma-2-9b-it | transformers | 0 | 0 | 2 / 1 / 2 / 0 | 0.004 |
| gemma-2-9b-it | sglang | 0 | 0 | 1 / 4 / 0 / 0 | 0.002 |
| gemma-2-9b-it | vllm | 0 | 0 | 4 / 1 / 0 / 0 | 0.002 |
| gemma-3-1b-it | transformers | 0 | 0 | 5 / 0 / 0 / 0 | 0.015 |
| gemma-3-1b-it | sglang | 0 | 0 | 3 / 2 / 0 / 0 | 0.005 |
| gemma-3-1b-it | vllm | 0 | 0 | 5 / 0 / 0 / 0 | 0.005 |

Refusals about the models themselves: 0.

## Decisions other than pass about the models

- Qwen2.5-3B-Instruct-AWQ / transformers: `[entail] unknown at load:transformers.linear: Layout declared Layout(kind='int4_packed', dtype=None, block=(128,), packing=None, scale_format=None) (config: /home/<user>/models/Qwen2.5-3B-Instruct-AWQ/config.json#quantization_config, declared); transformers.linear.unknown uses unknown (engine: transformers.linear.unknown [not in the capability table], unknown); rule: what the consumer uses is unkno`
- Qwen2.5-3B-Instruct-AWQ / sglang: `[entail] unknown at load:sglang.linear: Layout declared Layout(kind='int4_packed', dtype=None, block=(128,), packing=None, scale_format=None) (config: /home/<user>/models/Qwen2.5-3B-Instruct-AWQ/config.json#quantization_config, declared); sglang.linear.unknown uses unknown (engine: sglang.linear.unknown [not in the capability table], unknown); rule: what the consumer uses is unknown (not read, or n`
- Qwen2.5-3B-Instruct-AWQ / vllm: `[entail] unknown at load:vllm.linear: Layout declared Layout(kind='int4_packed', dtype=None, block=(128,), packing=None, scale_format=None) (config: /home/<user>/models/Qwen2.5-3B-Instruct-AWQ/config.json#quantization_config, declared); vllm.linear.unknown uses unknown (engine: vllm.linear.unknown [not in the capability table], unknown); rule: what the consumer uses is unknown (not read, or not in `
- Qwen3-4B-FP8 / transformers: `[entail] unknown at load:transformers.linear: Layout declared Layout(kind='fp8_block', dtype='float8_e4m3fn', block=(128, 128), packing=None, scale_format='bf16') (config: /home/<user>/models/Qwen3-4B-FP8/config.json#quantization_config, verified); transformers.linear.unknown uses unknown (engine: transformers.linear.unknown [not in the capability table], unknown); rule: what the consumer uses is u`
- Qwen3-4B-FP8 / sglang: `[entail] unknown at load:sglang.linear: Layout declared Layout(kind='fp8_block', dtype='float8_e4m3fn', block=(128, 128), packing=None, scale_format='bf16') (config: /home/<user>/models/Qwen3-4B-FP8/config.json#quantization_config, verified); sglang.linear.unknown uses unknown (engine: sglang.linear.unknown [not in the capability table], unknown); rule: what the consumer uses is unknown (not read, `
- Qwen3-4B-FP8 / vllm: `[entail] unknown at load:vllm.linear: Layout declared Layout(kind='fp8_block', dtype='float8_e4m3fn', block=(128, 128), packing=None, scale_format='bf16') (config: /home/<user>/models/Qwen3-4B-FP8/config.json#quantization_config, verified); vllm.linear.unknown uses unknown (engine: vllm.linear.unknown [not in the capability table], unknown); rule: what the consumer uses is unknown (not read, or not`

## Resolutions by backend (the backend drops a property the model declares)

- Phi-3.5-mini-instruct / sglang: resolved {'sglang.attention.flashinfer': 'triton', 'sglang.attention.flex_attention': 'triton'}; refused []
- gemma-2-2b-it / transformers: resolved {'transformers.attention.sdpa': 'eager'}; refused ['transformers.paged_attention.eager', 'transformers.paged_attention.sdpa']
- gemma-2-2b-it / sglang: resolved {'sglang.attention.flashinfer': 'triton', 'sglang.attention.torch_native': 'triton', 'sglang.attention.flex_attention': 'triton', 'sglang.attention.trtllm_mha': 'triton'}; refused []
- gemma-2-2b-it / vllm: resolved {'vllm.attention.ROCM_ATTN': 'FLASH_ATTN'}; refused []
- gemma-2-9b-it / transformers: resolved {'transformers.attention.sdpa': 'eager'}; refused ['transformers.paged_attention.eager', 'transformers.paged_attention.sdpa']
- gemma-2-9b-it / sglang: resolved {'sglang.attention.flashinfer': 'triton', 'sglang.attention.torch_native': 'triton', 'sglang.attention.flex_attention': 'triton', 'sglang.attention.trtllm_mha': 'triton'}; refused []
- gemma-2-9b-it / vllm: resolved {'vllm.attention.ROCM_ATTN': 'FLASH_ATTN'}; refused []
- gemma-3-1b-it / sglang: resolved {'sglang.attention.flashinfer': 'triton', 'sglang.attention.flex_attention': 'triton'}; refused []

## Notes

- Llama-3.2-3B-Instruct: /home/<user>/models/Llama-3.2-3B-Instruct/config.json#rope_theta,rope_scaling: RoPE keys ['high_freq_factor', 'low_freq_factor'] are not in vocabulary v1; the Rotary fact does not carry them
- Phi-3.5-mini-instruct: /home/<user>/models/Phi-3.5-mini-instruct/config.json#rope_theta,rope_scaling: RoPE keys ['long_factor', 'short_factor'] are not in vocabulary v1; the Rotary fact does not carry them
