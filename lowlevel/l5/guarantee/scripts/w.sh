#!/bin/bash
# usage: w.sh <cmd...>   (work distro, vllm venv, in ~/ai_compiler/entail)
source ~/venvs/vllm/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
cd ~/ai_compiler/entail
export PYTHONPATH=$HOME/ai_compiler/entail
"$@" 2>&1 | grep --line-buffered -v "Failed to start the systemd\|WARNING\|INFO"
