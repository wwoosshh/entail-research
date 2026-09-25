#!/usr/bin/env bash
# Gemma 2 가중치를 내려받는다 (2B, 9B instruction-tuned).
# 공식 google/gemma-2-*-it는 로그인과 라이선스 동의가 필요해서 Unsloth 사본을 쓴다.
# 연구자가 Gemma 이용 약관 동의와 내려받기를 허락했다 (2026-09-23).
# 대조 결과 (2026-09-23, Hugging Face API):
#   9B: 가중치 4개 파일과 index의 SHA256이 공식 저장소와 같다.
#   2B: 한 파일로 다시 묶여 있어 해시로는 대조할 수 없다. 전체 크기는 같다(24바이트 차이, 헤더).
#   config.json: attn_logit_softcapping 50.0, final_logit_softcapping 30.0, sliding_window 4096,
#   query_pre_attn_scalar 256. Gemma 2 기술 보고서의 값과 같다.
#   tokenizer.json, tokenizer.model: 공식과 같다. tokenizer_config.json은 다르다.
set -euo pipefail
source ~/venvs/gpu/bin/activate
python - <<'EOF'
import json
import os

from huggingface_hub import HfApi, snapshot_download

out = {}
for size in ("2b", "9b"):
    repo = f"unsloth/gemma-2-{size}-it"
    info = HfApi().model_info(repo)
    path = snapshot_download(repo, revision=info.sha, local_dir=os.path.expanduser(f"~/models/gemma-2-{size}-it"))
    out[repo] = {"revision": info.sha, "path": path}
    print(repo, info.sha, path, flush=True)
json.dump(out, open(os.path.expanduser("~/models/gemma2_sources.json"), "w"), indent=1)
EOF
