# rolebench 검출 표

`detection_table.py`가 `results/*.json`에서 만든다.

| 사례 | 사실 | 재현 | 갈래 | 모드 | 검출 | 결함 판 오류 문구 | 수정 판 |
|---|---|---|---|---|---|---|---|
| 01_reorder_layout | LAYOUT | 예 | W | debug | 예 | second-path dequantize (interleaved reader): argument 'buf' Layout: expected (Layout(kind='q8_0', dtype=None,  | 오류 없음 |
| 02_scale_pow2 | LAYOUT | 예 | W | debug | 예 | DeepGEMM UE8M0 kernel: argument 'q' Layout: expected (Layout(kind='fp8_block', dtype=None, block=None, packing | 오류 없음 |
| 03_strided_q | LAYOUT | 예 | W | debug | 예 | kernel assuming packed Q rows: argument 'q' Layout: expected (Layout(kind='dense', dtype=None, block=None, pac | 오류 없음 |
| 04_double_reduce | REDUCTION | 예 | W | debug | 예 | dp_gather_partial (reduces its input): argument 'y' Reduction: expected Reduction(state='P', dim=None, group=N | 오류 없음 |
| 05_chunk_frame | FRAME | 예 | W | debug | 예 | mask builder (compares absolute positions): argument 'q_pos' Positions: expected Positions(frame='absolute', o | 오류 없음 |
| 06_mask_disables_window | PROPERTY | 예 | W | load | 예 | kernel with a custom mask implementing ['causal']: kernel does not honour declared model properties: sliding_w | 오류 없음 |
| 07_tied_head | PROPERTY | 예 | W | load | 예 | loader (declared tie): config declares tied embeddings but the checkpoint holds a different head | 오류 없음 |
| 08_gemma2_softcap | PROPERTY | 예 | W | load | 예 | transformers attention implementation 'sdpa' (Gemma 2): kernel does not honour declared model properties: soft | 오류 없음 |
| 09_stale_graph | TIME | 예 | W | debug | 예 | CUDA graph replay: graph captured for valid length 100 replayed with valid length 160 | 오류 없음 |
| 10_flex_position_ref | TIME | 예 | W | debug | 예 | flex attention (read-time contract): query offset was 64 when the mask was built but is 65 when attention read | 오류 없음 |
| 11_warmup_specialization | SPECIALIZATION | 예 | W | debug | 예 | compiled graph reuse: artifact built for {'extra_rows_zero': True} reused for {'extra_rows_zero': False} | 오류 없음 |
| 12_session_restore | RANGE | 예 | W | load | 예 | decode after session restore: position 12 does not follow the valid KV length 11 (Valid.length) | 오류 없음 |
| 14_beam_reorder | TIME | 예 | W | debug | 예 | beam step: beam-axis cache entries not reordered: ['ssm_state'] | 오류 없음 |
| 15_config_alias | MAPPING | 예 | W | load | 예 | transformers LlamaConfig: unrecognised config keys ['rope_scale'] | 오류 없음 |
| 16_fp8_as_bf16 | DTYPE | 예 | W | debug | 예 | LoRA path (expects unquantized activations): argument 'x' Quantized: expected (Quantized(dtype='float32', scal | 오류 없음 |
| 17_sglang_torch_native_softcap | PROPERTY | 예 | W | load | 예 | SGLang attention backend 'torch_native' (Gemma 2): kernel does not honour declared model properties: softcap=5 | 오류 없음 |
