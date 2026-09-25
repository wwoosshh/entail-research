"""M10 E3, vllm-project/vllm#58138 (testbed/M10_PROTOCOL.md 3.3): for a BERT-style cross-encoder, padding="max_length"
gives every padding token the last real token's token_type_id (1) instead of the tokenizer's pad type (0).
  1. the report's check, on vLLM 0.30's own functions: the token_type_ids vLLM builds for (query, document) padded
     to max_length, against transformers' tokenizer (the report's attached script, read, not run)
  2. the end effect on vLLM's server: /rerank scores of cross-encoder/ms-marco-MiniLM-L-6-v2 for the same query and
     documents with and without padding="max_length" (entail on or off from outside reaches the server process)
Reproduced: (1) differs from transformers, and (2) the padded scores differ from the unpadded ones.
Run in ~/venvs/vllm: python testbed/m10_e3/vl58138.py <out.json>
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY = "what is the capital of australia"
DOCS = ["Canberra is the capital city of Australia.", "Sydney is the largest city in Australia.",
        "The quiet library had tall shelves."]
MAXLEN = 64
PORT = 8765


def function_check():
    from transformers import AutoTokenizer
    from vllm.entrypoints.pooling.scoring.io_processor import _apply_post_tokenization_to_token_type_ids
    from vllm.entrypoints.pooling.scoring.protocol import RerankRequest

    tok = AutoTokenizer.from_pretrained(MODEL)
    req = RerankRequest.model_validate({"model": MODEL, "query": QUERY, "documents": [DOCS[0]],
                                        "padding": "max_length"})
    params = req.build_tok_params(SimpleNamespace(max_model_len=MAXLEN, encoder_config={}))
    unpadded = tok(QUERY, DOCS[0], add_special_tokens=True)
    prompt = {"prompt_token_ids": list(unpadded["input_ids"])}
    params.apply_post_tokenization(tok, prompt)
    got = _apply_post_tokenization_to_token_type_ids(tok, params, list(unpadded["token_type_ids"]))
    want = list(tok(QUERY, DOCS[0], add_special_tokens=True, padding="max_length", max_length=MAXLEN)
                ["token_type_ids"])
    return {"vllm_token_type_ids": got, "transformers_token_type_ids": want, "differ": got != want}


def rerank(padding):
    body = {"model": MODEL, "query": QUERY, "documents": DOCS}
    if padding:
        body["padding"] = padding
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/rerank", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return {x["index"]: round(x["relevance_score"], 6) for x in d["results"]}


def main():
    row = {"entail": os.environ.get("ENTAIL", "off"), "function_check": function_check()}
    log = open(sys.argv[1] + ".server.log", "w")
    srv = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(PORT),
                            "--max-model-len", str(MAXLEN), "--gpu-memory-utilization", "0.3", "--enforce-eager"],
                           stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        row["scores_unpadded"] = rerank(None)
        row["scores_padded_max_length"] = rerank("max_length")
    except Exception as e:  # noqa: BLE001
        row["server_error"] = f"{type(e).__name__}: {e}"
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except Exception:  # noqa: BLE001
            srv.kill()
    a, b = row.get("scores_unpadded"), row.get("scores_padded_max_length")
    row["scores_differ"] = bool(a and b and a != b)
    row["reproduced"] = row["function_check"]["differ"] and row["scores_differ"]
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps({k: row[k] for k in ("reproduced", "scores_differ")} |
                               {"function_differs": row["function_check"]["differ"],
                                "unpadded": a, "padded": b, "error": row.get("server_error")}))


if __name__ == "__main__":
    main()
