#!/usr/bin/env bash
# Re-run the CPU cases (with their detection checks) and print the detection table.
source ~/venvs/gpu/bin/activate
cd <workspace>/rolebench
for c in 01_reorder_layout 02_scale_pow2 04_double_reduce 07_tied_head 12_session_restore 14_beam_reorder 15_config_alias 16_fp8_as_bf16; do
  CUDA_VISIBLE_DEVICES= python run_case.py "cases/$c" 2>&1 | grep '^{'
done
python detection_table.py
