"""M15.7: a static sweep of the popular models (E1's 300, the 230 with a readable config) with the v6 vocabulary,
for meaning loss nobody has reported. Per model, the folder testbed/results/m10/e1_llm/configs/<owner>__<name>/ gets
the tokenizer files the model ships (tokenizer.json, vocab.txt, vocab.json, merges.txt, a sentencepiece model,
special_tokens_map.json, added_tokens.json, model.safetensors.index.json; nothing over MAX_MB) and the embedding's
rows read from a safetensors header alone (huggingface_hub.parse_safetensors_file_metadata: a range request for
the header, no weights). Then:
  - Vocab: the tokenizer transformers builds from the folder (CPU) against the folder's sources and the rows
    (vocab_contract.check, the rule `entail check` runs)
  - the tokenizer loader an engine would pick (Mistral files present -> outside the check)
  - Rotary: keys the vocabulary still cannot carry (readers' problems), and the fact read
A finding is a `broken` that reproduces (the folder's two vocabularies tokenize a sentence differently), not a
benign layout (padding, added tokens, one source). Resumable: a folder with entail_sweep.json is skipped.

Run in ~/venvs/gpu (transformers, huggingface_hub): python testbed/m15/sweep_static.py [limit]
Writes testbed/results/m15/sweep_static.json and sweep_static.log.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "entail"))
E1 = os.path.join(ROOT, "testbed", "results", "m10", "e1_llm")
OUT = os.path.join(ROOT, "testbed", "results", "m15")
FILES = ("tokenizer.json", "vocab.txt", "vocab.json", "merges.txt", "tokenizer.model", "spiece.model",
         "sentencepiece.bpe.model", "special_tokens_map.json", "added_tokens.json", "model.safetensors.index.json",
         "tokenizer_config.json")
MISTRAL = ("tekken.json", "tokenizer.model.v3", "tokenizer.model.v7", "tokenizer.model.v1")
MAX_MB = 40
SENTENCE = "The quick brown fox jumps over the lazy dog. 안녕하세요, 세계! 12345"


def log(msg):
    with open(os.path.join(OUT, "sweep_static.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def fetch(api, repo, folder, siblings):
    """Download the tokenizer files the repo ships (under MAX_MB); returns (downloaded names, bytes, mistral files)."""
    from huggingface_hub import hf_hub_download

    got, nbytes, mistral = [], 0, []
    sizes = {s.rfilename: (s.size or 0) for s in siblings}
    for name in MISTRAL:
        if name in sizes:
            mistral.append(name)
    for name in FILES:
        if name not in sizes:
            continue
        if sizes[name] > MAX_MB * 1024 * 1024:
            got.append(f"{name} (skipped: {sizes[name] / 2**20:.0f} MB)")
            continue
        if os.path.isfile(os.path.join(folder, name)):
            got.append(name)
            continue
        try:
            hf_hub_download(repo_id=repo, filename=name, local_dir=folder)
            got.append(name)
            nbytes += sizes[name]
        except Exception as e:  # noqa: BLE001 - one file that cannot be fetched is named, the rest go on
            got.append(f"{name} (failed: {type(e).__name__})")
    return got, nbytes, mistral


def embedding_rows(repo, folder, siblings, config_vocab):
    """(rows, where) from a safetensors header fetched without the weights, or None."""
    from huggingface_hub import parse_safetensors_file_metadata
    from entail.vocab_contract import EMBEDDING_SUFFIXES

    names = {s.rfilename for s in siblings}
    shards = []
    idx = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(idx):
        try:
            wm = json.load(open(idx, encoding="utf-8")).get("weight_map", {})
            for suffix in EMBEDDING_SUFFIXES:
                for k, v in wm.items():
                    if k.endswith(suffix):
                        shards.append((k, v))
        except (ValueError, OSError):
            pass
    elif "model.safetensors" in names:
        shards.append((None, "model.safetensors"))
    seen = set()
    for key, shard in shards:
        if shard in seen:
            continue
        seen.add(shard)
        try:
            meta = parse_safetensors_file_metadata(repo_id=repo, filename=shard)
        except Exception as e:  # noqa: BLE001
            return None, f"{shard}: header not read ({type(e).__name__})"
        for name, t in meta.tensors.items():
            if any(name.endswith(s) for s in EMBEDDING_SUFFIXES) and t.shape:
                if config_vocab is None or int(t.shape[0]) >= int(config_vocab):
                    return int(t.shape[0]), f"{name} {list(t.shape)} in {shard} (header only)"
    return None, "no embedding tensor found in the headers"


def tokenize_with_each_source(folder, sources):
    """The ids the folder's tokenizer sources give one sentence, when transformers can build each (a WordPiece
    vocab.txt beside a tokenizer.json); used to say whether a `broken` reproduces as different ids."""
    out = {}
    try:
        from transformers import AutoTokenizer, BertTokenizer
        tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        out["auto"] = tok(SENTENCE)["input_ids"][:24]
        if os.path.isfile(os.path.join(folder, "vocab.txt")):
            slow = BertTokenizer(vocab_file=os.path.join(folder, "vocab.txt"))
            out["vocab.txt"] = slow(SENTENCE)["input_ids"][:24]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def sweep(limit=None):
    os.environ.setdefault("ENTAIL_LOG_DIR", "off")
    from huggingface_hub import HfApi
    from entail import core, sources as _sources, vocab_contract
    from entail.adapters import transformers_tokenizer
    from entail.contracts import Verdict

    core.set_mode("off")
    api = HfApi()
    models = json.load(open(os.path.join(E1, "models.json"), encoding="utf-8"))
    models = models.get("models", models) if isinstance(models, dict) else models
    rows = []
    done = 0
    for m in models:
        if limit and done >= limit:
            break
        rid = m["id"]
        folder = os.path.join(E1, "configs", rid.replace("/", "__"))
        if not os.path.isfile(os.path.join(folder, "config.json")) or m.get("gated"):
            continue
        cache = os.path.join(folder, "entail_sweep.json")
        if os.path.isfile(cache):
            rows.append(json.load(open(cache, encoding="utf-8")))
            done += 1
            continue
        t0 = time.time()
        row = {"rank": m.get("rank"), "id": rid, "model_type": m.get("model_type")}
        try:
            info = api.model_info(rid, files_metadata=True)
            siblings = info.siblings or []
            row["files"], row["bytes"], row["mistral_files"] = fetch(api, rid, folder, siblings)
            try:
                cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
                config_vocab = cfg.get("vocab_size") or (cfg.get("text_config") or {}).get("vocab_size")
            except (ValueError, OSError):
                config_vocab = None
            rows_n, rows_where = embedding_rows(rid, folder, siblings, config_vocab)
            row["rows"] = {"n": rows_n, "where": rows_where, "config_vocab_size": config_vocab}
            s = vocab_contract.sources(folder, rows=(rows_n, rows_where) if rows_n else None)
            row["sources"] = {"candidates": s.candidates, "rows": s.rows, "rows_where": s.rows_where,
                              "problems": s.problems}
            size = n = None
            where = "no tokenizer built"
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
                size, n = transformers_tokenizer.read_choice(tok)
                where = f"{type(tok).__name__} built by transformers from the folder"
                row["tokenizer"] = {"class": type(tok).__name__, "vocab_size": size, "reach": n}
            except Exception as e:  # noqa: BLE001
                row["tokenizer"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            ds = vocab_contract.check("load:static.tokenizer", "transformers.tokenizer", folder, size, n, where,
                                      record=False, rows=(rows_n, rows_where) if rows_n else None)
            row["vocab"] = [{"verdict": d.verdict.value, "rule": d.rule, "note": d.note} for d in ds]
            if any(d.verdict is Verdict.BROKEN for d in ds):
                row["reproduce"] = tokenize_with_each_source(folder, s)
            r = _sources.read_all(folder)
            rot = [f for f in r.facts if f.name == "Rotary"]
            row["rotary"] = {"facts": [str(f.value) for f in rot],
                             "problems": [p for p in r.problems if "RoPE" in p or "rope" in p.lower()]}
            row["other_problems"] = [p for p in r.problems if not ("RoPE" in p or "rope" in p.lower())][:6]
        except Exception as e:  # noqa: BLE001 - one model that fails is recorded, the sweep goes on
            row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        row["seconds"] = round(time.time() - t0, 1)
        json.dump(row, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        rows.append(row)
        done += 1
        v = [x["verdict"] for x in row.get("vocab", [])]
        log(f"{done:3} {rid:55} vocab={v} rows={row.get('rows', {}).get('n')} rotary_problems="
            f"{len(row.get('rotary', {}).get('problems', []))} {row['seconds']}s {row.get('error', '')}")
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "models": len(rows), "rows": rows}
    json.dump(out, open(os.path.join(OUT, "sweep_static.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"DONE {len(rows)} models")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sweep(int(sys.argv[1]) if len(sys.argv) > 1 else None)
