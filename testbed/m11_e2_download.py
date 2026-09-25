"""M11.7: download the fresh sample - the models testbed/results/m11/e2/selection_30.json chose beyond M10 E2's first
20 (m10_e2_select.py with M10_E2_N=30: the next ten by the same rule) - into ~/models/m10/<owner>__<name>,
anonymously, safetensors and the small files only, as m10_e2_download.py did. A model already there (M10 E2 or E3)
is listed as present. Writes testbed/results/m11/e2/downloads.json and list_fresh.txt (name<TAB>path, the models
that are not in list_30.txt).
Run in ~/venvs/gpu: python testbed/m11_e2_download.py
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "m11", "e2")
DEST = os.path.expanduser("~/models/m10")
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken", "*.py"]


def main():
    from huggingface_hub import snapshot_download

    sel = json.load(open(os.path.join(R, "selection_30.json"), encoding="utf-8"))
    already = {line.split("\t")[0] for line in open(os.path.join(R, "list_30.txt"), encoding="utf-8") if "\t" in line}
    rows, fresh = [], []
    for m in sel["chosen"][20:]:
        name = m["id"].replace("/", "__")
        if name in already:
            rows.append({"id": m["id"], "status": "in M10 E2 already"})
            print(rows[-1], flush=True)
            continue
        target = os.path.join(DEST, name)
        t0 = time.time()
        if os.path.isfile(os.path.join(target, "config.json")):
            rows.append({"id": m["id"], "path": target, "status": "present"})
        else:
            try:
                snapshot_download(m["id"], local_dir=target, allow_patterns=ALLOW, token=False)
                size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(target) for f in fs)
                rows.append({"id": m["id"], "path": target, "status": "downloaded", "bytes": size,
                             "seconds": round(time.time() - t0, 1)})
            except Exception as e:  # noqa: BLE001 - recorded; the next model goes on
                rows.append({"id": m["id"], "path": target, "status": f"failed: {type(e).__name__}: {str(e)[:200]}"})
        if rows[-1]["status"] in ("present", "downloaded"):
            fresh.append((name, target))
        print(rows[-1], flush=True)
        with open(os.path.join(R, "downloads.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(os.path.join(R, "list_fresh.txt"), "w", encoding="utf-8") as f:
        for name, path in fresh:
            f.write(f"{name}\t{path}\n")
    print("DONE", len(fresh), "fresh models;", len(rows), "rows")


if __name__ == "__main__":
    main()
