"""M10 E3 (sgl-project/sglang#33493): download the target/drafter pair the reproduction uses, anonymously, safetensors
and the small files only. Run in ~/venvs/gpu: python testbed/m10_e3/get_minicpm.py"""
from huggingface_hub import snapshot_download

for repo in ("openbmb/MiniCPM5-2B-DSpark", "openbmb/MiniCPM5-2B"):
    p = snapshot_download(repo, local_dir="/home/<user>/models/m10/" + repo.replace("/", "__"), token=False,
                          allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "tokenizer*"])
    print("ok", p, flush=True)
