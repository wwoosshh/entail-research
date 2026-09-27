"""M17.6 case, vllm-project/vllm#40466 (testbed/M16_PROTOCOL.md 7): with the qwen3 reasoning parser and
enable_thinking=False, a streamed chat completion puts the tokens in `reasoning` instead of `content`, while the
non-streaming answer is right. A vLLM OpenAI server on a small Qwen3 model, one request non-streaming and the
same request streamed, the delta fields accumulated. Reproduced when the streamed content is empty and the
streamed reasoning is not, for a request the non-streaming path answers in content.
Run in ~/venvs/vllm (0.30.0; then the reported 0.11.1 if it installs and the symptom is gone here):
  python testbed/m17/replay2/cases/vl40466.py <out.json> [model]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request


def wait(port, proc, seconds=600):
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

    model = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/Qwen3-0.6B")
    port = 8377
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "model": model}
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port),
           "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--enforce-eager",
           "--reasoning-parser", "qwen3", "--served-model-name", "m"]
    log = open(sys.argv[1] + ".server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait(port, proc):
            row["error"] = "server did not come up"
            row["reproduced"] = None
        else:
            body = {"model": "m", "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
                    "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                msg = json.load(r)["choices"][0]["message"]
            row["non_streaming"] = {"content": (msg.get("content") or "")[:200],
                                    "reasoning": (msg.get("reasoning") or msg.get("reasoning_content") or "")[:200]}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         data=json.dumps({**body, "stream": True}).encode(),
                                         headers={"Content-Type": "application/json"})
            content, reasoning = "", ""
            with urllib.request.urlopen(req, timeout=300) as r:
                for line in r:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    delta = json.loads(line[5:].strip())["choices"][0].get("delta") or {}
                    content += delta.get("content") or ""
                    reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
            row["streaming"] = {"content": content[:200], "reasoning": reasoning[:200]}
            row["reproduced"] = bool(row["non_streaming"]["content"].strip() and not content.strip()
                                     and reasoning.strip())
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
