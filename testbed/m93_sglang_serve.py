"""M9.3: the chat template on SGLang's OpenAI server (0.5.20, Qwen3-4B), after the gap M9.1 found (S1: SGLang's
server used the model's template without any decision). One server per condition, entail on (ENTAIL=load, the
library's start-up hook from the checkout), two chat requests each:

  healthy   the model's own jinja template: rendered by the tokenizer's apply_chat_template, decided there
            (request:transformers.chat_template) - passes
  file      --chat-template <a .jinja file that is not the model's template>: SGLang puts it on the tokenizer; broken,
            and the request is served (the default policy, M5.4)
  builtin   --chat-template chatml, one of SGLang's own conversation templates: decided where SGLang renders with it
            (request:sglang.chat_template) - broken
  strict    `file` with ENTAIL_ON_BROKEN=stop: the request is refused before anything is generated

Writes testbed/results/m93/sglang_<condition>.json, .log and .jsonl (ENTAIL_RECORD).
Run in ~/venvs/sglang: python testbed/m93_sglang_serve.py <condition>
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
OUT = os.path.join(RESULTS, "m93")
MODEL = os.path.expanduser("~/models/Qwen3-4B")
PORT = 8125
URL = f"http://127.0.0.1:{PORT}"
OTHER = ("{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}\n{% endfor %}"
         "{% if add_generation_prompt %}[assistant] {% endif %}")
ASK = [{"role": "user", "content": "Name the capital of France in one word."}]


def start(condition, log_path, record):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ENTAIL")}
    env.update(PYTHONPATH=f"{ROOT}/entail:{ROOT}/entail/entail/adapters/autoinstall", ENTAIL="load",
               ENTAIL_RECORD=record)
    args = []
    if condition in ("file", "strict"):
        path = os.path.join(tempfile.mkdtemp(prefix="m93_template_"), "other.jinja")
        with open(path, "w", encoding="utf-8") as f:
            f.write(OTHER)
        args = ["--chat-template", path]
    elif condition == "builtin":
        args = ["--chat-template", "chatml"]
    if condition == "strict":
        env["ENTAIL_ON_BROKEN"] = "stop"
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL, "--port", str(PORT),
           "--mem-fraction-static", "0.8", "--context-length", "4096", "--disable-radix-cache",
           "--disable-cuda-graph", "--log-level", "warning"] + args
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            return proc, None, args
        try:
            if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                return proc, True, args
        except requests.RequestException:
            pass
        time.sleep(2)
    return proc, False, args


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def ask():
    r = requests.post(f"{URL}/v1/chat/completions", json={"model": "default", "messages": ASK, "max_tokens": 16,
                                                            "temperature": 0,
                                                            "chat_template_kwargs": {"enable_thinking": False}},
                      timeout=300)
    out = {"status": r.status_code}
    try:
        data = r.json()
    except ValueError:
        return dict(out, error=r.text[:400])
    if r.status_code != 200:
        return dict(out, error=str(data.get("error", data) if isinstance(data, dict) else data)[:400])
    return dict(out, content=(data["choices"][0]["message"].get("content") or "")[:120],
                prompt_tokens=(data.get("usage") or {}).get("prompt_tokens"))


def recorded(path):
    """The decisions at the request boundaries that were not passes, and the last counts per process."""
    decisions, counted = [], {}
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            e = json.loads(line)
            if "boundaries" in e:
                for b, s in e["boundaries"].items():
                    if b.startswith("request:"):
                        counted[f"{e['pid']} {b}"] = {k: v for k, v in s.items()}
            elif "verdict" in e and e["boundary"].startswith("request:") and e["verdict"] != "pass":
                decisions.append({k: e.get(k) for k in ("boundary", "verdict", "rule")}
                                 | {"chosen": (e.get("chosen") or {}).get("source", {}).get("where", "")[:160]})
    return decisions, counted


def main():
    condition = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f"sglang_{condition}")
    record = base + ".jsonl"
    if os.path.exists(record):
        os.remove(record)
    res = {"condition": condition, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    proc, up, args = start(condition, base + ".log", record)
    res.update(args=args, started=up)
    try:
        if up:
            res["requests"] = [ask(), ask()]
    finally:
        stop(proc)
    res["decisions"], res["counted"] = recorded(record)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("RESULT", condition, json.dumps({k: res.get(k) for k in ("started", "requests", "decisions")},
                                           ensure_ascii=False)[:900], flush=True)


if __name__ == "__main__":
    main()
