#!/bin/bash
# M19 L4 replay 4 (testbed/M16_PROTOCOL.md 9): the reported-version environments the passing cases need, at most six.
#   vllm0102  vLLM 0.10.2  vllm#25800 (FP8 KV cache + sleep level 2)
#   vllm0110  vLLM 0.11.0  vllm#27390 (finish_reason at the token limit), vllm#27491 if 0.30.0 does not show it
#   vllm0140  vLLM 0.14.0  vllm#33091 (Whisper, FA2 + CUDA graphs)
#   vllm0160  vLLM 0.16.0  vllm#33560 (NVFP4 Marlin, float16), vllm#35221 (qwen3 reasoning parser)
#   tf500     transformers 5.0.0 (CPU torch)  transformers#43697 (RT-DETRv2)
# Each goes on after a failure; a failed install is recorded (the case is then cannot_install_version).
set -u
LOG=<workspace>/testbed/results/m19/replay4/venvs.log
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
mk() {  # name, pip args...
  local name=$1; shift
  local v=/home/<user>/venvs/$name
  if [ -x $v/bin/python ] && $v/bin/python -c "import sys" 2>/dev/null && [ -f $v/.done ]; then log "$name exists"; return; fi
  rm -rf $v
  python3.12 -m venv $v || { log "$name: venv failed"; return; }
  log "$name: pip install $*"
  if $v/bin/pip install -q --upgrade pip >> $LOG 2>&1 && $v/bin/pip install -q "$@" >> $LOG 2>&1; then
    touch $v/.done
    log "$name ok: $($v/bin/pip freeze 2>/dev/null | grep -i -E '^(vllm|torch|transformers|triton)==' | tr '\n' ' ') $(du -sh $v | cut -f1)"
  else
    log "$name FAILED (see above)"
  fi
}
log "start"
mk tf500 --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.9.1+cpu" "transformers==5.0.0" pillow
mk vllm0160 "vllm==0.16.0"
mk vllm0110 "vllm==0.11.0"
mk vllm0102 "vllm==0.10.2"
mk vllm0140 "vllm==0.14.0"
log "done"
