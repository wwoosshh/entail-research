"""Download real Qwen3-4B weights for the week-2 numerics/determinism experiments (~8 GB, bf16 safetensors).
Stored in the Linux filesystem (~/models) for I/O speed."""
import os
import time

from huggingface_hub import snapshot_download

dst = os.path.expanduser("~/models/Qwen3-4B")
t0 = time.time()
path = snapshot_download("Qwen/Qwen3-4B", local_dir=dst, allow_patterns=["*.safetensors", "*.json", "*.txt"])
total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
print(f"downloaded to {path}: {total/2**30:.2f} GiB in {time.time()-t0:.0f}s")
