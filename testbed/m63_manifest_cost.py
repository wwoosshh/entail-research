"""M6.3, S4: what finding a manifest costs at load, with ENTAIL_MANIFESTS naming the fd-vae manifest folder.

Before M6.3 manifest.find hashed every artifact in full (the diffusers fd-vae run: 17.9 s of a 19.2 s load for a
6.9 GB checkpoint on WSL's /mnt/c). Now a file is hashed only when a manifest's quick fingerprint matches it.
Times, in one process and this order: find() for three checkpoints no manifest is for (one of them the same size as the
one that has a manifest), then for the one that has it; then the full SHA-256 of one file alone (what every find
cost before). Writes testbed/results/m63/manifest_cost.json. Run in WSL ~/venvs/gpu.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
from entail import manifest  # noqa: E402

CK = "/mnt/c/Users/<user>/Desktop/ComfyUI/ComfyUI-new/models/checkpoints"
DIRS = [os.path.join(HERE, "results", "m63", "manifests")]
FILES = ["waiIllustriousSDXL_v170.safetensors", "NoobAI-XL-Vpred-v1.0.safetensors",
         "entail_test/astolfocarmixVpredxl_acEvo25EP.safetensors", "waiIllustriousSDXL_v160.safetensors"]


def main():
    res = {"manifest_dirs": DIRS, "finds": []}
    for name in FILES:
        path = os.path.join(CK, name)
        t = time.perf_counter()
        m = manifest.find(path, DIRS)
        res["finds"].append({"file": name, "bytes": os.path.getsize(path), "found": m is not None,
                             "seconds": round(time.perf_counter() - t, 3)})
    path = os.path.join(CK, FILES[1])
    t = time.perf_counter()
    manifest.sha256_of(path)
    res["full_sha256_of_one_file_seconds"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    manifest.quick_fingerprint(path)
    res["quick_fingerprint_seconds"] = round(time.perf_counter() - t, 4)
    with open(os.path.join(HERE, "results", "m63", "manifest_cost.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
