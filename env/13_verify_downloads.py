"""Check each downloaded weight file against the hub's record for the pinned revision, and compare the two
unofficial mirrors with the official repositories as far as the public metadata allows.

- local sha256 == lfs oid of the pinned revision: the download is intact.
- mirror vs official: the official Gemma and Llama repositories are gated, and their API shows the oid masked
  ("****") without a login, which this project does not use. Only the byte size can be compared there.
Read-only GET requests. Writes env/verify_downloads.json.
"""
import hashlib
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.expanduser("~/models/sweep_sources.json")
OFFICIAL = {"unsloth/gemma-3-1b-it": "google/gemma-3-1b-it",
            "unsloth/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct"}


def tree(repo, rev):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}", timeout=60) as r:
        return {f["path"]: f for f in json.load(r) if f.get("type") == "file"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = {}
    for repo, src in json.load(open(SOURCES)).items():
        remote = tree(repo, src["revision"])
        official = tree(OFFICIAL[repo], "main") if repo in OFFICIAL else None
        files = {}
        for name, meta in remote.items():
            if not name.endswith(".safetensors"):
                continue
            local = os.path.join(src["path"], name)
            oid = (meta.get("lfs") or {}).get("oid")
            got = sha256(local) if os.path.exists(local) else None
            row = {"size": meta.get("size"), "hub_sha256": oid, "local_sha256": got, "intact": got == oid}
            if official is not None:
                o = official.get(name, {})
                row["official_size"] = o.get("size")
                row["official_sha256_visible"] = "*" not in str((o.get("lfs") or {}).get("oid", "*"))
                row["size_equals_official"] = o.get("size") == meta.get("size")
            files[name] = row
            print(repo, name, "intact" if row["intact"] else "MISMATCH", flush=True)
        out[repo] = {"revision": src["revision"], "files": files}
    with open(os.path.join(HERE, "verify_downloads.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
