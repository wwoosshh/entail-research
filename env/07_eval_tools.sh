#!/usr/bin/env bash
# lm-eval을 기본 gpu 환경에 넣는다.
# 먼저 dry-run으로 확인하고, torch·transformers 등 핵심 패키지가 바뀌면 설치하지 않고 멈춘다.
set -euo pipefail
source ~/venvs/gpu/bin/activate
report="$(mktemp)"
pip install --dry-run --quiet --report "$report" "lm-eval==0.4.13"
python - "$report" <<'EOF'
import json
import sys

r = json.load(open(sys.argv[1]))
changes = {i["metadata"]["name"].lower(): i["metadata"]["version"] for i in r.get("install", [])}
protected = ("torch", "transformers", "triton", "torchao", "accelerate", "safetensors", "numpy")
bad = {k: v for k, v in changes.items() if k in protected}
print("would install", len(changes), "packages:", ", ".join(f"{k}=={v}" for k, v in sorted(changes.items())))
print("protected changes:", bad)
sys.exit(1 if bad else 0)
EOF
pip install "lm-eval==0.4.13"
python -c "import lm_eval, torch, transformers; print('lm_eval', lm_eval.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
