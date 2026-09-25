#!/usr/bin/env bash
# Models for the sweep's second round (sweep/RESULTS.md) and the deep arm's first real quantised checkpoint.
# Researcher approved option A on 2026-09-23 (candidates and sizes: env/candidate_models.json).
#
# Licence agreement (2026-09-23, in the conversation): the researcher agreed to the Gemma Terms of Use, the
# Llama 3.2 Community License and the Qwen Research License, and asked for all six models to be downloaded.
#   unsloth/gemma-3-1b-it            Gemma Terms of Use
#   unsloth/Llama-3.2-3B-Instruct    Llama 3.2 Community License
#   Qwen/Qwen2.5-3B-Instruct         Qwen Research License (non-commercial research)
#   Qwen/Qwen2.5-3B-Instruct-AWQ     Qwen Research License (non-commercial research)
#   microsoft/Phi-3.5-mini-instruct  MIT
#   Qwen/Qwen3-4B-FP8                Apache-2.0
# The two unsloth repos are ungated copies of gated official repos; compare hashes against the official
# ones after download, as was done for Gemma 2 (env/06_fetch_gemma2.sh).
#
# Only weights, configs and tokenizer files are fetched: no *.py, so no remote code comes along. transformers
# has native classes for every model here.
#
# Usage: bash env/12_fetch_sweep_models.sh <repo> [<repo> ...]
#   Licences that need the researcher's agreement (Gemma, Llama 3.2, Qwen research) are fetched only after
#   that agreement is given in the conversation; MIT and Apache-2.0 need none.
set -euo pipefail
source ~/venvs/gpu/bin/activate
python - "$@" <<'EOF'
import json
import os
import sys

from huggingface_hub import HfApi, snapshot_download

ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
         "tokenizer*", "*.model", "special_tokens_map.json", "chat_template*", "vocab*", "merges.txt",
         "quantization_config.json", "preprocessor_config.json", "LICENSE*", "README.md"]
record = os.path.expanduser("~/models/sweep_sources.json")
sources = json.load(open(record)) if os.path.exists(record) else {}
api = HfApi()
for repo in sys.argv[1:]:
    info = api.model_info(repo)
    local = os.path.expanduser("~/models/" + repo.split("/")[-1])
    path = snapshot_download(repo, revision=info.sha, local_dir=local, allow_patterns=ALLOW)
    sources[repo] = {"revision": info.sha, "path": path}
    json.dump(sources, open(record, "w"), indent=1)
    print("FETCHED", repo, info.sha, path, flush=True)
EOF
