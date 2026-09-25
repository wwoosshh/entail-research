# rolebench 결과

`run_all.py`가 `results/*.json`에서 만든다. 판정 정의는 `PROTOCOL.md`.

| 사례 | 사실 | 종류 | 재현 | 결함 판 최대 차이 | 수정 판 최대 차이 | 결함 판 최상위 토큰 일치 | 결함 판 조용함 | 초 |
|---|---|---|---|---|---|---|---|---|
| 01_reorder_layout | LAYOUT | mechanism | 예 | nan | 0 | 0.0 | 예 | 0.8 |
| 02_scale_pow2 | LAYOUT | mechanism | 예 | 151 | 8.36 | 1.0 | 예 | 0.8 |
| 03_strided_q | LAYOUT | mechanism | 예 | 28.6 | 2.86e-06 | 0.0 | 예 | 1.4 |
| 04_double_reduce | REDUCTION | mechanism | 예 | 19.7 | 2.86e-06 | 1.0 | 예 | 0.9 |
| 05_chunk_frame | FRAME | mechanism | 예 | 2.81 | 4.77e-07 | 0.172 | 예 | 8.5 |
| 06_mask_disables_window | PROPERTY | mechanism | 예 | 0.495 | 7.15e-07 | 0.448 | 예 | 8.5 |
| 07_tied_head | PROPERTY | mechanism | 예 | 2.2 | 0 | 0.031 | 예 | 1.0 |
| 08_gemma2_softcap | PROPERTY | real-engine | 예 | 1.77 | 2.81e-06 | 0.062 | 예 | 0.9 |
| 09_stale_graph | TIME | mechanism | 예 | 0.424 | 5.96e-08 | 0.5 | 예 | 1.1 |
| 10_flex_position_ref | TIME | real-engine | 예 | 1.36 | 5.96e-07 | 0.0 | 예 | 3.9 |
| 11_warmup_specialization | SPECIALIZATION | mechanism | 예 | 6.39 | 5.72e-06 | 0.5 | 예 | 9.8 |
| 12_session_restore | RANGE | mechanism | 예 | 1.23 | 0 | 1.0 | 예 | 1.0 |
| 14_beam_reorder | TIME | mechanism | 예 | 16 | 2.66e-15 | 1.0 | 예 | 0.9 |
| 15_config_alias | MAPPING | real-engine | 예 | 0.0244 | 0 | 0.98 | 예 | 0.2 |
| 16_fp8_as_bf16 | DTYPE | mechanism | 예 | 1.44e+03 | 0.933 | 0.812 | 예 | 0.9 |
| 17_sglang_torch_native_softcap | PROPERTY | real-engine | 예 | 5.98 | 0.267 | 0.0 | 예 | 49.5 |
