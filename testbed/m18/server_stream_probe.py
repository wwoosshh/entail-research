"""M18.5 false alarms on a live vLLM server with every adapter on: streaming and non-streaming chat requests -
plain, thinking on and off, with tools the model may call, tool_choice="none", n=2, a tiny max_tokens (finish by
length) - through the OpenAI server of a small reasoning model with a reasoning parser and a tool parser. The
record files of the server's processes are read afterwards for every non-pass decision (Parse, KernelReference,
Tokenization and the rest). The point is the parse adapter (M18.3) on real streams: it must raise nothing on
well-formed traffic; what it does raise is a finding to look at, not a number to hide.
Usage (in the vllm venv, with PYTHONPATH pointing at entail and its autoinstall shim, ENTAIL=load,
ENTAIL_RECORD set):  python testbed/m18/server_stream_probe.py <out.json> [model] [reasoning_parser] [tool_parser]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = 8123


def request(path, body, stream=False):
    req = urllib.request.Request(f"http://localhost:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.load(r)
        chunks = []
        for line in r:
            line = line.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
        return chunks


def main():
    out_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/m10/Qwen__Qwen3-0.6B")
    reasoning = sys.argv[3] if len(sys.argv) > 3 else "qwen3"
    tool_parser = sys.argv[4] if len(sys.argv) > 4 else "hermes"
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model, "--port", str(PORT),
           "--max-model-len", "2048", "--gpu-memory-utilization", "0.85", "--enforce-eager",
           "--reasoning-parser", reasoning, "--tool-call-parser", tool_parser, "--enable-auto-tool-choice",
           "--served-model-name", "m", "--disable-uvicorn-access-log"]
    log = open(out_path + ".server.log", "w")
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    rows, err = [], None
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                               "required": ["city"]}}},
             {"type": "function", "function": {"name": "add", "description": "Add two integers",
                                                "parameters": {"type": "object", "properties": {"a": {"type": "integer"},
                                                                                                "b": {"type": "integer"}},
                                                               "required": ["a", "b"]}}}]
    cases = [
        ("plain", {"messages": [{"role": "user", "content": "Name three uses of copper."}], "max_tokens": 80}),
        ("thinking_off", {"messages": [{"role": "user", "content": "What is 17 + 25? Answer briefly."}], "max_tokens": 60,
                          "chat_template_kwargs": {"enable_thinking": False}}),
        ("thinking_on", {"messages": [{"role": "user", "content": "What is 17 + 25? Think briefly, then answer."}],
                         "max_tokens": 200}),
        ("tools_weather", {"messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
                           "tools": tools, "max_tokens": 200}),
        ("tools_add", {"messages": [{"role": "user", "content": "Use the add tool to add 21 and 34."}], "tools": tools,
                       "max_tokens": 200}),
        ("tools_none", {"messages": [{"role": "user", "content": "Say hello."}], "tools": tools, "tool_choice": "none",
                        "max_tokens": 40}),
        ("length_cut", {"messages": [{"role": "user", "content": "Write a long story about a lighthouse."}], "max_tokens": 12}),
        ("n2", {"messages": [{"role": "user", "content": "Give one word for happiness."}], "max_tokens": 30, "n": 2,
                "temperature": 0.8}),
        ("no_reasoning_field", {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 120,
                                "include_reasoning": False}),
    ]
    try:
        up = False
        for _ in range(300):
            if server.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5)
                up = True
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        if not up:
            err = "server did not come up (see .server.log)"
        else:
            for name, body in cases:
                for stream in (True, False):
                    b = dict(body, model="m", stream=stream)
                    t0 = time.perf_counter()
                    try:
                        resp = request("/v1/chat/completions", b, stream=stream)
                        row = {"case": name, "stream": stream, "ms": round((time.perf_counter() - t0) * 1e3)}
                        if stream:
                            content = "".join((c["choices"][0]["delta"].get("content") or "") for c in resp
                                              if c.get("choices") and c["choices"][0]["choices" if False else "delta"])
                            tool_deltas = sum(len(c["choices"][0]["delta"].get("tool_calls") or []) for c in resp
                                              if c.get("choices"))
                            row.update(chunks=len(resp), content=content[:120], tool_deltas=tool_deltas,
                                       finish=[c["choices"][0].get("finish_reason") for c in resp if c.get("choices")
                                               and c["choices"][0].get("finish_reason")])
                        else:
                            ch = resp["choices"][0]
                            row.update(content=(ch["message"].get("content") or "")[:120],
                                       reasoning=(ch["message"].get("reasoning") or ch["message"].get("reasoning_content") or "")[:60],
                                       tool_calls=[(t["function"]["name"], t["function"]["arguments"][:80])
                                                   for t in (ch["message"].get("tool_calls") or [])],
                                       finish=ch.get("finish_reason"))
                    except Exception as e:  # noqa: BLE001
                        row = {"case": name, "stream": stream, "error": f"{type(e).__name__}: {e}"[:200]}
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False)[:300], flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    json.dump({"model": model, "reasoning_parser": reasoning, "tool_parser": tool_parser, "error": err, "rows": rows},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps({"requests": len(rows), "errors": sum(1 for r in rows if "error" in r), "server_error": err}))


if __name__ == "__main__":
    main()
