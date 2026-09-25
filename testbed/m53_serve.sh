#!/bin/bash
# M5.3: the request boundary on a vLLM 0.30 server with Qwen3-4B (testbed/m53_serve.py). One server at a time.
# Usage: bash testbed/m53_serve.sh [server:tag ...]   default: every run below, in order
set -u
cd <workspace>
source ~/venvs/vllm/bin/activate
RUNS=${*:-"healthy:off healthy:on keep:off keep:on tool:off tool:on tool:refuse"}
for r in $RUNS; do
  python testbed/m53_serve.py ${r%%:*} ${r##*:}
done
