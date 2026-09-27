#!/bin/bash
# M17.4 on real vLLM 0.30.0, second run (the adapter compares the language model only): GLM-OCR with entail off
# and on, Qwen3-0.6B with entail on (Qwen3-4B does not fit beside the 0.6 memory share). Results:
# testbed/results/m17/pairing/. The first run's files stay as *_run1_* (the whole-model comparison that flipped the
# vision tower).
set -u
cd <workspace>
source ~/venvs/vllm/bin/activate
export PYTHONPATH=<workspace>/entail:<workspace>/entail/entail/adapters/autoinstall
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=WARNING
unset ENTAIL_SEED ENTAIL_ON_BROKEN ENTAIL_POLICY ENTAIL_RECORD
OUT=testbed/results/m17/pairing
mkdir -p "$OUT"
for f in glm_ocr_on.json glm_ocr_record.jsonl; do
  [ -f "$OUT/$f" ] && mv "$OUT/$f" "$OUT/run1_$f"
done
echo "### glm_ocr off"
ENTAIL= python testbed/m17/pairing_vllm.py "$OUT/glm_ocr_off.json" zai-org/GLM-OCR 2048 --mm 2>&1 \
  | grep -E "RESULT|\[entail\]|Error|Traceback" | tail -n 20
echo "### glm_ocr on"
ENTAIL=load ENTAIL_RECORD="$OUT/glm_ocr_record.jsonl" python testbed/m17/pairing_vllm.py "$OUT/glm_ocr_on.json" \
  zai-org/GLM-OCR 2048 --mm 2>&1 | grep -E "RESULT|\[entail\]|Error|Traceback" | tail -n 20
echo "### qwen3 on"
ENTAIL=load ENTAIL_RECORD="$OUT/qwen3_record.jsonl" python testbed/m17/pairing_vllm.py "$OUT/qwen3_on.json" \
  /home/<user>/models/Qwen3-0.6B 2048 --none 2>&1 | grep -E "RESULT|\[entail\]|Error|Traceback" | tail -n 20
echo "### done"
