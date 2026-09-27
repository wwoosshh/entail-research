### System Info

```
torch                    2.14.0.dev20260613+cu126
torchao                  0.18.0.dev20260615+cu126
transformers             5.12.1
triton                   3.7.1+git5d6048aa
```

### Who can help?

@ArthurZucker @CyrilVallez

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

### Description

`q_offset` passed into `flex_attention_mask` is a `torch.Tensor` rather than a plain `int` which causing introducing an extra dimension. Causing the mask to have more than 4 dimensions and triggering the `ValueError`. 
WA: Converting `q_offset` to a Python `int` via `.item()` before passing it into `flex_attention_mask` avoids this.

```
def flex_attention_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    **kwargs,
) -> BlockMask:
```

### Reproducer
```
python -u run_generation.py -m Qwen/Qwen3-4B --input-tokens 1024 --max-new-tokens 1024 --num-iter 8 --num-warmup 4 --batch-size 1 --device cuda --inductor --dtype bfloat16 --attn-type flex_attention --disable-skip-guard-eval --disable-cpp-wrapper --num-beams 1  --use-static-cache --use-hf-code False --woq --woq-type rtn --quant-dtype uint4 --group-size 128
```
Another simpler reproducer
```
python run_generation.py \
  -m Qwen/Qwen3-4B \
  --input-tokens 1024 --max-new-tokens 1024 \
  --device cuda --dtype bfloat16 \
  --attn-type flex_attention \
  --use-static-cache \
  --inductor \
  --num-beams 1
```
### Script

[run_generation.py](https://github.com/user-attachments/files/28983711/run_generation.py)
[prompt.json](https://github.com/user-attachments/files/28983710/prompt.json)

### Error log
```
2026-06-16 03:22:50,627 INFO root Namespace(model_id='Qwen/Qwen3-4B', sub_model_name='', device='cuda', dtype='bfloat16', input_tokens='1024', max_new_tokens=1024, prompt=None, batch_size=1, num_iter=8, num_warmup=4, num_profile=1, num_beams=1, greedy=False, use_hf_code=False, use_static_cache=True, amp=False, inductor=True, profile=False, unitrace=False, accuracy_only=False, acc_tasks='gsm8k', acc_iter=-1, print_memory=False, token_latency=False, output_csv_path='output.csv', quant_mode='woq', woq=True, group_size=128, ZPFLOAT=False, calibration_samples=10, model_save_path=None, load_quantize_model=False, woq_type='rtn', quant_dtype='uint4', use_hqq=False, granularity='per_tensor', attn_type='flex_attention', disable_cpp_wrapper=True, disable_skip_guard_eval=True)
W0616 03:22:51.452000 1853176 site-packages/torch/utils/_pytree.py:630] <enum 'KernelPreference'> is an Enum subclass and is now natively supported by torch.compile as an opaque value type. Calling register_constant() on Enum subclasses is deprecated and will be an error in a future release.
2026-06-16 03:22:51,494 INFO root Using cuda device for WoQ RTN mode, Using TorchAoConfig: TorchAoConfig(quant_method=<QuantizationMethod.TORCHAO: 'torchao'>, quant_type=Int4WeightOnlyConfig(group_size=128, set_inductor_config=True, int4_packing_format='tile_packed_to_4d', int4_choose_qparams_algorithm=<Int4ChooseQParamsAlgorithm.TINYGEMM: 'tinygemm'>, int4_tile_packed_ntile=8, version=2), modules_to_not_convert=None, include_input_output_embeddings=False, untie_embedding_weights=False)
W0616 03:22:51.689000 1853176 site-packages/torch/utils/_pytree.py:630] <enum 'ScaleCalculationMode'> is an Enum subclass and is now natively supported by torch.compile as an opaque value type. Calling register_constant() on Enum subclasses is deprecated and will be an error in a future release.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:02<00:00, 183.31it/s]
2026-06-16 03:22:56,430 INFO root For model Qwen/Qwen3-4B, skipped the unwrap_tensor_subclass_parameters. This may affect performance
[transformers] The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[transformers] The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
W0616 03:23:45.560000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:45.854000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:46.063000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:46.272000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:46.481000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:46.691000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:46.899000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:47.109000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:47.318000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:47.526000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:47.732000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:47.940000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:48.146000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:48.354000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:48.565000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:48.774000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:48.982000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:49.191000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:49.397000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:49.606000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:49.813000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:50.021000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:50.227000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:50.434000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:50.640000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:50.847000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:51.053000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:51.261000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:51.468000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:51.677000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:51.883000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:52.093000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:52.300000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:52.510000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:52.717000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
W0616 03:23:52.926000 1853176 site-packages/torch/_inductor/utils.py:3075] [1/0] DeviceCopy in input program
Traceback (most recent call last):
  File "/data/xingyuan/20260615-issue3906-cuda/frameworks.ai.pytorch.gpu-models/LLM/inference/run_generation.py", line 631, in <module>
    run_generate(o, i, g)
  File "/data/xingyuan/20260615-issue3906-cuda/frameworks.ai.pytorch.gpu-models/LLM/inference/run_generation.py", line 520, in run_generate
    output = model.generate(
             ^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/transformers/generation/utils.py", line 2584, in generate
    result = decoding_method(
             ^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/transformers/generation/utils.py", line 2794, in _sample
    model_inputs = self.prepare_inputs_for_generation(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/transformers/generation/utils.py", line 568, in prepare_inputs_for_generation
    attention_mask = causal_mask_creation_function(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/transformers/masking_utils.py", line 1524, in create_masks_for_generate
    return LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[next(iter(layer_patterns))](**mask_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/transformers/masking_utils.py", line 1001, in create_causal_mask
    causal_mask = mask_interface(
                  ^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/transformers/masking_utils.py", line 728, in flex_attention_mask
    block_mask = create_block_mask(
                 ^^^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py", line 1917, in create_block_mask
    return torch.compile(create_block_mask)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1166, in compile_wrapper
    result = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py", line 1935, in create_block_mask
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pt-gpu/miniforge3/envs/xingyuan-issue3906-cuda/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py", line 1599, in _convert_mask_to_block_mask
    B, H, Q, KV = mask.shape
    ^^^^^^^^^^^
ValueError: too many values to unpack (expected 4)
```

### Expected behavior

Fixing this bug
