"""M10 E1, I (testbed/M10_PROTOCOL.md 1.3-1.4 and 6): single-file image checkpoints - Civitai's "Most Downloaded"
Checkpoint models, all time (100) and the last year (100) - read by their safetensors header only (HTTP range), no
weights, anonymous. For each: what the file declares about its prediction type (the v_pred / ztsnr marker keys that
ComfyUI reads, ModelSpec / kohya metadata), what entail 1.0's reader takes from it (readers.header_facts), and
whether the model is a v-prediction model (declared, or named so: a lower bound).
Run in ~/venvs/gpu: python testbed/m10_e1_images.py
Writes testbed/results/m10/e1_images/checkpoints.json.
"""
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
os.environ.setdefault("ENTAIL_LOG_DIR", "off")
from entail import readers  # noqa: E402

OUT = os.path.join(HERE, "results", "m10", "e1_images")
UA = {"User-Agent": "entail-research/1.0 (M10 E1 exposure scan; read-only)"}
VNAME = re.compile(r"v[-_ ]?pred|v[-_ ]?prediction", re.I)
MAX_HEADER = 16 * 1024 * 1024


def get(url, headers=None, timeout=90):
    req = urllib.request.Request(url, headers=dict(UA, **(headers or {})))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def listing(period, n):
    items, url = [], "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(
        {"types": "Checkpoint", "sort": "Most Downloaded", "period": period, "limit": 100})
    while url and len(items) < n:
        _, body = get(url)
        d = json.loads(body)
        items += d.get("items", [])
        url = (d.get("metadata") or {}).get("nextPage")
        time.sleep(1)
    return items[:n]


def header(url):
    """One request; read the 8-byte length and the header from the stream, then close. Some servers ignore Range
    and send the whole file (200): reading only what is needed keeps that to the header's size."""
    req = urllib.request.Request(url, headers=dict(UA, Range=f"bytes=0-{MAX_HEADER + 7}"))
    with urllib.request.urlopen(req, timeout=90) as r:
        b = r.read(8)
        n = struct.unpack("<Q", b[:8])[0]
        if n > MAX_HEADER:
            raise ValueError(f"header of {n} bytes")
        h = b""
        while len(h) < n:
            chunk = r.read(n - len(h))
            if not chunk:
                break
            h += chunk
    return json.loads(h)


def main():
    os.makedirs(OUT, exist_ok=True)
    when = time.strftime("%Y-%m-%d %H:%M:%S %z")
    populations = {"AllTime": listing("AllTime", 100), "Year": listing("Year", 100)}
    seen, rows = {}, []
    for period, items in populations.items():
        for rank, it in enumerate(items, 1):
            v = (it.get("modelVersions") or [{}])[0]
            files = [f for f in v.get("files", []) if str(f.get("name", "")).endswith(".safetensors")]
            f = next((x for x in files if x.get("primary")), files[0] if files else None)
            row = {"period": period, "rank": rank, "model_id": it.get("id"), "model": it.get("name"),
                   "version": v.get("name"), "base_model": v.get("baseModel"), "nsfw": it.get("nsfw"),
                   "file": f.get("name") if f else None, "size_kb": f.get("sizeKB") if f else None}
            if not f:
                row["read"] = "no safetensors file"
                rows.append(row)
                continue
            key = (it.get("id"), f.get("name"))
            if key in seen:
                row.update({k: seen[key][k] for k in seen[key] if k not in row})
                row["duplicate_of"] = seen[key]["period"]
                rows.append(row)
                continue
            try:
                h = header(f["downloadUrl"])
                meta = h.pop("__metadata__", {}) or {}
                keys = list(h)
                r = readers.header_facts(keys, meta, row["file"])
                row.update({
                    "read": "ok", "n_tensors": len(keys),
                    "key_v_pred": "v_pred" in keys, "key_ztsnr": "ztsnr" in keys,
                    "modelspec_prediction_type": meta.get("modelspec.prediction_type"),
                    "modelspec_architecture": meta.get("modelspec.architecture"),
                    "ss_v_parameterization": meta.get("ss_v_parameterization"),
                    "ss_zero_terminal_snr": meta.get("ss_zero_terminal_snr"),
                    "entail_reads": [{"kind": x.value.kind, "zsnr": x.value.zsnr, "from": str(x.source)[-120:]}
                                     for x in r.facts if x.name == "Prediction"],
                    "entail_problems": [str(p)[:200] for p in r.problems]})
            except Exception as e:  # noqa: BLE001 - recorded; the scan goes on
                row["read"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
            row["named_v"] = bool(VNAME.search(" ".join(str(x) for x in (row["model"], row["version"], row["file"]))))
            seen[key] = row
            rows.append(row)
            print(f"{period} {rank:3} {row['base_model']} {row['read']}", flush=True)
            with open(os.path.join(OUT, "checkpoints.partial.json"), "w", encoding="utf-8") as fh:
                json.dump({"when": when, "rows": rows}, fh, ensure_ascii=False)
            time.sleep(0.5)
    out = {"when": when, "source": "civitai.com/api/v1/models (types=Checkpoint, sort=Most Downloaded)",
           "rows": rows}
    with open(os.path.join(OUT, "checkpoints.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("DONE", len(rows))


if __name__ == "__main__":
    main()
