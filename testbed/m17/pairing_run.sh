#!/bin/bash
# M17.4 on real vLLM 0.30.0, entail on: GLM-OCR (the architecture table says interleaved; the layers hold
# is_neox_style False) and Qwen3-4B (split); then the S4 CUDA-graph harness with every adapter, after the record
# files are kept open (M17.3's cost attribution). Results: testbed/results/m17/pairing/, testbed/results/m17/m55_v2/.
set -u
cd <workspace>
source ~/venvs/vllm/bin/activate
export PYTHONPATH=<workspace>/entail:<workspace>/entail/entail/adapters/autoinstall
export ENTAIL=load VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=WARNING
unset ENTAIL_SEED ENTAIL_ON_BROKEN ENTAIL_POLICY ENTAIL_RECORD
OUT=testbed/results/m17/pairing
mkdir -p "$OUT"
for spec in "zai-org/GLM-OCR glm_ocr --mm" "/home/<user>/models/Qwen3-4B qwen3 --none"; do
  set -- $spec
  echo "### ${2}"
  ENTAIL_RECORD="$OUT/${2}_record.jsonl" python testbed/m17/pairing_vllm.py "$OUT/${2}_on.json" "$1" 2048 "$3" 2>&1 \
    | grep -E "RESULT|\[entail\]|Error|Traceback" | tail -n 20
done
echo "### S4 m55, every adapter, record files kept open"
mkdir -p testbed/results/m17/m55_v2
TESTBED_RESULTS=<workspace>/testbed/results/m17/m55_v2 TAIL_LINES=12 bash testbed/m55_graph.sh 2>&1 \
  | grep -E "^B=|wrote|Error|Traceback"
echo "### done"
