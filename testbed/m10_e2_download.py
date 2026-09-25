"""M10 E2: download the models testbed/results/m10/e2/selection.json chose (the first 20 and the per-architecture extra)
into ~/models/m10/<owner>__<name>, anonymously, safetensors and the small files only. A model already in ~/models
under its own name (Qwen3-4B, Qwen2.5-3B-Instruct) is not downloaded again: the local copy is used.
Run in ~/venvs/gpu: python testbed/m10_e2_download.py
Writes testbed/results/m10/e2/downloads.json.
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "m10", "e2")
DEST = os.path.expanduser("~/models/m10")
ALLOW = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "tokenizer*", "*.tiktoken", "*.py.disabled"]


def main():
    from huggingface_hub import snapshot_download

    sel = json.load(open(os.path.join(R, "selection.json"), encoding="utf-8"))
    rows = []
    for m in sel["chosen"] + sel["by_architecture_extra"]:
        local = os.path.expanduser(f"~/models/{m['id'].split('/')[1]}")
        if os.path.isfile(os.path.join(local, "config.json")):
            rows.append({"id": m["id"], "path": local, "status": "local copy"})
            continue
        target = os.path.join(DEST, m["id"].replace("/", "__"))
        t0 = time.time()
        try:
            snapshot_download(m["id"], local_dir=target, allow_patterns=ALLOW, token=False)
            size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(target) for f in fs)
            rows.append({"id": m["id"], "path": target, "status": "downloaded", "bytes": size,
                         "seconds": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001 - recorded; the next model goes on
            rows.append({"id": m["id"], "path": target, "status": f"failed: {type(e).__name__}: {str(e)[:200]}"})
        print(rows[-1], flush=True)
        with open(os.path.join(R, "downloads.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    print("DONE", sum(1 for r in rows if r["status"] in ("downloaded", "local copy")), "of", len(rows))


if __name__ == "__main__":
    main()
