"""M19 L4 replay 4: the models the passing cases need (testbed/results/m19/replay4/screening.json), into
~/models/replay4/<repo with / -> __>. google/gemma-3-4b-it is gated: unsloth's ungated copy stands in (disclosed in
the screening record). Logged to testbed/results/m19/replay4/models.log."""
import os
import time

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.expanduser("~/models/replay4")
LOG = "<workspace>/testbed/results/m19/replay4/models.log"
BASE = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"]
WANT = [  # (repo, patterns or a single file, the cases that need it)
    ("PekingU/rtdetr_v2_r18vd", BASE, "tf43697"),
    ("bzantium/tiny-deepseek-v3", BASE + ["*.py"], "vl27491"),
    ("openai/whisper-large-v3-turbo", BASE, "vl33091"),
    ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q8_0.gguf", "tf41494"),
    ("unsloth/gemma-3-4b-it", BASE, "tf41494 (reference tokenizer), vl34186"),
    ("stewy33/gemma-3-4b-it-0524_rowan_original_prompt_augmented_egregious_cake_bake-bd093845", BASE, "vl34186"),
    ("RedHatAI/Qwen3-8B-NVFP4", BASE, "vl33560"),
    ("Qwen/Qwen3-4B-Thinking-2507", BASE, "vl35221"),
]


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def size_mb(d):
    return sum(os.path.getsize(os.path.join(p, n)) for p, _, ns in os.walk(d) if "/.cache" not in p for n in ns) >> 20


for repo, what, cases in WANT:
    target = os.path.join(ROOT, repo.replace("/", "__"))
    try:
        if isinstance(what, str):
            hf_hub_download(repo, what, local_dir=target, token=False)
        else:
            snapshot_download(repo, local_dir=target, allow_patterns=what, token=False)
        log(f"ok {repo} ({cases}) {size_mb(target)} MB")
    except Exception as e:  # noqa: BLE001 - record and go on
        log(f"FAILED {repo} ({cases}): {type(e).__name__}: {str(e)[:300]}")
log("done")
