"""M17.6 case, vllm-project/vllm#48831 (testbed/M16_PROTOCOL.md 7): the /score endpoint returns a near-zero score
for a matching query+document pair of more than ~8K tokens, while offline LLM.score() on the same inputs gives a
score near 1 (an asynchronous copy of query_start_loc read too early by the pooler). Qwen3-Reranker-0.6B served
the documented way (hf_overrides for the sequence-classification head): the offline score first, then the
server's /score for the same texts. Reproduced when the server's score is low and the offline score is high.
Run in ~/venvs/vllm0240 (the reported 0.24.0; 0.30.0 for the fixed version):
  python testbed/m17/replay2/cases/vl48831.py <out.json>
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = "Qwen/Qwen3-Reranker-0.6B"
OVERRIDES = {"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"],
             "is_original_qwen3_reranker": True}
PORT = 8378


def texts():
    query = "Which city is the capital of France, and what river runs through it?"
    filler = ("The committee reviewed the quarterly logistics report and noted steady progress on the "
              "warehouse consolidation, the new routing software, and the driver training schedule. ")
    doc = (filler * 260) + " Paris is the capital of France, and the river Seine runs through the city. " + (filler * 60)
    return query, doc


def wait(port, proc, seconds=900):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False


def main():
    import vllm
    from vllm import LLM

    query, doc = texts()
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "model": MODEL}
    llm = LLM(model=MODEL, runner="pooling", hf_overrides=OVERRIDES, max_model_len=16384, max_num_seqs=2,
              gpu_memory_utilization=0.55, enforce_eager=True)
    tok = llm.get_tokenizer()
    row["pair_tokens"] = len(tok.encode(query)) + len(tok.encode(doc))
    out = llm.score(query, doc, use_tqdm=False)
    row["offline_score"] = float(out[0].outputs.score)
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", str(PORT),
           "--runner", "pooling", "--hf-overrides", json.dumps(OVERRIDES), "--max-model-len", "16384",
           "--gpu-memory-utilization", "0.55", "--enforce-eager", "--served-model-name", "m"]
    log = open(sys.argv[1] + ".server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait(PORT, proc):
            row["error"] = "server did not come up"
            row["reproduced"] = None
        else:
            scores = []
            for _ in range(3):
                body = {"model": "m", "text_1": query, "text_2": doc}
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}/score", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    scores.append(float(json.load(r)["data"][0]["score"]))
            row["server_scores"] = scores
            row["reproduced"] = bool(row["offline_score"] > 0.5 and min(scores) < 0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
