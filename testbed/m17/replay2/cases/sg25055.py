"""M17.6 case, sgl-project/sglang#25055 (testbed/M16_PROTOCOL.md 7): /v1/chat/completions with logprobs=true and
separate_reasoning=true on a reasoning model returns logprobs.content over the whole raw output (the <think>
span and its markers included) while message.content holds only the parsed answer, so the two cannot be aligned.
An SGLang server on a small Qwen3 model with the qwen3 reasoning parser, one request. Reproduced when
logprobs.content has more tokens than message.content tokenises to and its tokens include the think markers.
Run in ~/venvs/sglang: python testbed/m17/replay2/cases/sg25055.py <out.json> [model]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request


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
    import sglang

    model = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/Qwen3-0.6B")
    port = 30377
    row = {"entail": os.environ.get("ENTAIL", "off"), "sglang": sglang.__version__, "model": model}
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--port", str(port),
           "--mem-fraction-static", "0.6", "--reasoning-parser", "qwen3", "--context-length", "2048",
           "--served-model-name", "m", "--disable-cuda-graph"]
    log = open(sys.argv[1] + ".server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait(port, proc):
            row["error"] = "server did not come up"
            row["reproduced"] = None
        else:
            body = {"model": "m", "messages": [{"role": "user", "content": "What is 2 + 2? Think briefly, then answer."}],
                    "temperature": 0, "max_tokens": 200, "logprobs": True, "separate_reasoning": True}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                choice = json.load(r)["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            lp = ((choice.get("logprobs") or {}).get("content") or [])
            lp_tokens = [x.get("token", "") for x in lp]
            lp_text = "".join(lp_tokens)
            row.update({"content": content[:200], "reasoning_chars": len(reasoning), "logprob_tokens": len(lp),
                        "logprob_text_head": lp_text[:120],
                        "logprobs_cover_reasoning": bool(reasoning) and (reasoning[:40] in lp_text),
                        "think_markers_in_logprobs": any(t.strip() in ("<think>", "</think>") for t in lp_tokens)})
            row["reproduced"] = bool(lp) and (row["logprobs_cover_reasoning"] or row["think_markers_in_logprobs"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
