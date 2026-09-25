"""M10 E1 (testbed/M10_PROTOCOL.md 1.1): the 300 most-downloaded text-generation models on Hugging Face, and for each
only the files that declare what the model means - config.json, generation_config.json, tokenizer_config.json and
chat_template.jinja. No weights, no token (anonymous): a gated repository is counted as gated and not read.
Writes testbed/results/m10/e1_llm/models.json and configs/<owner>__<name>/.
Run: python testbed/m10_e1_fetch.py [N]
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "m10", "e1_llm")
FILES = ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja")
EXPAND = ("downloads", "likes", "gated", "safetensors", "config", "tags", "library_name", "createdAt",
          "lastModified", "pipeline_tag")
UA = {"User-Agent": "entail-research/1.0 (M10 E1 exposure scan; read-only)"}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def listing(n):
    q = [("pipeline_tag", "text-generation"), ("sort", "downloads"), ("direction", "-1"), ("limit", str(n))]
    q += [("expand[]", e) for e in EXPAND]
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(q)
    status, body = get(url)
    return url, json.loads(body)


def fetch_one(m):
    repo = m["id"]
    folder = os.path.join(OUT, "configs", repo.replace("/", "__"))
    os.makedirs(folder, exist_ok=True)
    got = {}
    if m.get("gated"):
        return repo, {"gated": m.get("gated"), "files": {}}
    for name in FILES:
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        for attempt in range(3):
            try:
                status, body = get(url)
                with open(os.path.join(folder, name), "wb") as f:
                    f.write(body)
                got[name] = {"status": status, "bytes": len(body)}
                break
            except urllib.error.HTTPError as e:
                got[name] = {"status": e.code}
                if e.code in (401, 403, 404):
                    break
            except Exception as e:  # noqa: BLE001 - network hiccup: retry, then record it
                got[name] = {"error": f"{type(e).__name__}: {e}"}
            time.sleep(1 + attempt)
    return repo, {"gated": m.get("gated"), "files": got}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    os.makedirs(OUT, exist_ok=True)
    when = time.strftime("%Y-%m-%d %H:%M:%S %z")
    url, models = listing(n)
    with ThreadPoolExecutor(max_workers=6) as pool:
        fetched = dict(pool.map(fetch_one, models))
    rows = []
    for rank, m in enumerate(models, 1):
        st = m.get("safetensors") or {}
        cfg = m.get("config") or {}
        rows.append({"rank": rank, "id": m["id"], "downloads_30d": m.get("downloads"), "likes": m.get("likes"),
                     "gated": m.get("gated"), "library_name": m.get("library_name"),
                     "created": m.get("createdAt"), "last_modified": m.get("lastModified"),
                     "params_total": st.get("total"), "params_by_dtype": st.get("parameters"),
                     "model_type": cfg.get("model_type"), "architectures": cfg.get("architectures"),
                     "quantization": (cfg.get("quantization_config") or {}).get("quant_method"),
                     "tags": m.get("tags"), "fetch": fetched[m["id"]]})
    out = {"when": when, "query": url, "n": len(rows), "models": rows}
    with open(os.path.join(OUT, "models.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    gated = sum(1 for r in rows if r["gated"])
    has_cfg = sum(1 for r in rows if (r["fetch"]["files"].get("config.json") or {}).get("status") == 200)
    print(f"DONE {len(rows)} models at {when}; gated {gated}; config.json read {has_cfg}")


if __name__ == "__main__":
    main()
