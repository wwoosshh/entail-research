"""M19 L4 replay 4 case, vllm-project/vllm#35221 (testbed/M16_PROTOCOL.md 9): with --reasoning-parser qwen3, a
reasoning-only output cut by the token limit comes back as content (no </think>; the parser assumed thinking off;
fixed by #35230). The report's server and request: Qwen3-4B-Thinking-2507, max_completion_tokens 5, non-streaming.
The server runs as a child process of this script with the same environment (so entail, when on, is in it too).
Reproduced when the message's reasoning is empty and its content holds the cut reasoning.
Run: python testbed/m19/replay4/cases/vl35221.py <out.json>  (~/venvs/vllm0160: vLLM 0.16.0)
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = os.path.expanduser("~/models/replay4/Qwen__Qwen3-4B-Thinking-2507")
PORT = 8765


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def main():
    import vllm

    log = open(sys.argv[1] + ".server.log", "w")
    srv = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
                            "--served-model-name", "m", "--port", str(PORT), "--reasoning-parser", "qwen3",
                            "--max-model-len", "2048", "--gpu-memory-utilization", "0.85"],
                           stdout=log, stderr=subprocess.STDOUT)
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__}
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:  # noqa: BLE001
                if srv.poll() is not None:
                    raise RuntimeError("server exited")
                time.sleep(1)
        r = post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "안녕?"}],
                                          "max_completion_tokens": 5, "stream": False, "temperature": 0})
        msg = r["choices"][0]["message"]
        row.update({"content": msg.get("content"), "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                    "finish_reason": r["choices"][0].get("finish_reason")})
        row["reproduced"] = not row["reasoning"] and bool(row["content"]) and row["finish_reason"] == "length"
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
