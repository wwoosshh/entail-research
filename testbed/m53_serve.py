"""M5.3: the request boundary on a real vLLM server (vLLM 0.30.0, Qwen3-4B), installed by the sitecustomize shim.

Three servers, each run with entail off and on (ENTAIL=load; the tool server also with ENTAIL_POLICY=refuse):
  healthy  a pinned manifest declares Qwen3's template (its hash, the one tokenizer_config.json gives), tool calls in
           the hermes format, and earlier reasoning dropped (drop: Qwen3's template strips it). The server runs the
           hermes tool parser and the qwen3 reasoning parser, and trusts templates sent with a request.
           Requests that must pass: plain, multi-turn, a tool call, thinking on, thinking off by reasoning_effort.
           Planted: a template sent with the request, a field vLLM 0.30 no longer knows (guided_json), a misspelt
           template setting (enable_thinkng), a reasoning_effort level Qwen3's template cannot express (it reads
           only enable_thinking).
  keep     the same, but the manifest declares earlier reasoning kept (as MiniMax-M2 does; market L13): a
           multi-turn request that passes its reasoning back, and the same request with the reasoning dropped.
  tool     manifest as healthy, but the server runs the pythonic tool parser (llama3_json was tried first: it
           fails loudly on Qwen3, whose tokenizer has no <|python_tag|>; results/m53/tool_llama3_off.json).
Writes testbed/results/m53/<server>_<tag>.json, .log (the server's output) and .jsonl (ENTAIL_RECORD).
Run in ~/venvs/vllm:  python testbed/m53_serve.py <server> <tag>      tag: off | on | refuse
"""
import json
import os
import signal
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "entail"))
from entail import load, manifest  # noqa: E402
from entail.facts import Certainty, Fact, Source, Template  # noqa: E402

MODEL = os.path.expanduser("~/models/Qwen3-4B")
OUT = os.path.join(RESULTS, "m53")
PORT = 8123
URL = f"http://127.0.0.1:{PORT}"

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
ASK = [{"role": "user", "content": "Name the capital of France in one word."}]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
WEATHER = [{"role": "user", "content": "What is the weather in Paris right now?"}]
HISTORY = [{"role": "user", "content": "What is 2+2? Answer with the number only."},
           {"role": "assistant", "content": "4", "reasoning": "2 plus 2 is 4."},
           {"role": "user", "content": "And 3+3? Answer with the number only."}]
DROPPED = [HISTORY[0], {"role": "assistant", "content": "4"}, HISTORY[2]]
CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
          "'<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
SCHEMA = {"type": "object", "properties": {"capital": {"type": "string"}}, "required": ["capital"]}

REQUESTS = {   # (name, what it is, body)
    "healthy": [
        ("plain", "pass", {**NO_THINK, "messages": ASK}),
        ("multi_turn", "pass", {**NO_THINK, "messages": HISTORY}),
        ("tool_call", "pass", {**NO_THINK, "max_tokens": 128, "tools": TOOLS, "tool_choice": "auto",
                               "messages": WEATHER}),
        ("thinking", "pass", {"max_tokens": 384, "messages": ASK}),
        ("effort_none", "pass", {"reasoning_effort": "none", "messages": ASK}),
        ("request_template", "planted", {**NO_THINK, "chat_template": CHATML, "messages": ASK}),
        ("unknown_field", "planted", {**NO_THINK, "guided_json": SCHEMA, "messages": ASK}),
        ("misspelt_setting", "planted", {"chat_template_kwargs": {"enable_thinkng": False}, "max_tokens": 64,
                                         "messages": ASK}),
        ("effort_level", "planted", {"reasoning_effort": "low", "max_tokens": 64, "messages": ASK}),
    ],
    "keep": [
        ("kept_reasoning", "pass", {**NO_THINK, "messages": HISTORY}),
        ("dropped_reasoning", "planted", {**NO_THINK, "messages": DROPPED}),
    ],
    "tool": [
        ("tool_call", "planted at start", {**NO_THINK, "max_tokens": 128, "tools": TOOLS, "tool_choice": "auto",
                                           "messages": WEATHER}),
    ],
}
SERVER_ARGS = {
    "healthy": ["--tool-call-parser", "hermes", "--trust-request-chat-template"],
    "keep": ["--tool-call-parser", "hermes"],
    "tool": ["--tool-call-parser", "pythonic"],
}


def manifest_dir(server):
    """A pinned manifest for Qwen3-4B: the template hash the files give, and what the files cannot say."""
    [t] = [f for f in load.declared(MODEL).get("Template") if f.source.kind == "config"]
    history = "keep" if server == "keep" else "drop"
    value = Template(chat_template_sha256=t.value.chat_template_sha256, reasoning_history=history,
                     tool_call_format="hermes")
    d = os.path.join(OUT, "manifests", server)
    os.makedirs(d, exist_ok=True)
    m = manifest.pin(manifest.Manifest(manifest.sha256_of(MODEL), (
        Fact("Template", value, Source("manifest", f"m53 {server}"), Certainty.DECLARED),), file="Qwen3-4B"))
    manifest.save(m, os.path.join(d, f"{m.sha256}.json"))
    return d, value


def start(server, tag, log_path, record):
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{ROOT}/entail:{ROOT}/entail/entail/adapters/autoinstall", ENTAIL_RECORD=record,
               ENTAIL="off" if tag == "off" else "load", VLLM_LOGGING_LEVEL="WARNING")
    if tag == "refuse":
        env["ENTAIL_POLICY"] = "refuse"
    env[load.ENV_MANIFESTS] = manifest_dir(server)[0]
    cmd = [os.path.join(os.path.dirname(sys.executable), "vllm"), "serve", MODEL, "--served-model-name", "qwen3",
           "--port", str(PORT), "--max-model-len", "4096", "--gpu-memory-utilization", "0.88", "--enforce-eager",
           "--enable-auto-tool-choice", "--reasoning-parser", "qwen3"] + SERVER_ARGS[server]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            return proc, None, round(time.time() - t0, 1)
        try:
            if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                return proc, True, round(time.time() - t0, 1)
        except requests.RequestException:
            pass
        time.sleep(2)
    return proc, False, round(time.time() - t0, 1)


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def ask(body):
    r = requests.post(f"{URL}/v1/chat/completions", json={"model": "qwen3", "max_tokens": 64, "temperature": 0,
                                                            **body}, timeout=300)
    out = {"status": r.status_code}
    data = r.json()
    if r.status_code != 200:
        out["error"] = str(data.get("error", data))[:600]
        return out
    msg = data["choices"][0]["message"]
    out.update(content=(msg.get("content") or "")[:200], reasoning_chars=len(msg.get("reasoning") or ""),
               tool_calls=[{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                           for c in msg.get("tool_calls") or []],
               finish_reason=data["choices"][0].get("finish_reason"))
    return out


def recorded(path):
    """The decisions that were not passes, and the last summary per process and boundary."""
    decisions, summaries = [], {}
    if not os.path.isfile(path):
        return decisions, summaries
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if "boundaries" in e:
                for b, s in e["boundaries"].items():
                    summaries[f"{e['pid']} {b}"] = s
            elif "verdict" in e and e["verdict"] != "pass":
                decisions.append({k: e.get(k) for k in ("boundary", "verdict", "rule", "resolution", "target")}
                                 | {"chosen": (e.get("chosen") or {}).get("where", "")[:200]})
    return decisions, summaries


def main():
    server, tag = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f"{server}_{tag}")
    record = base + ".jsonl"
    if os.path.exists(record):
        os.remove(record)
    res = {"server": server, "tag": tag, "entail": "off" if tag == "off" else "load",
           "policy": "refuse" if tag == "refuse" else "resolve", "args": SERVER_ARGS[server],
           "declared": str(manifest_dir(server)[1]), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    proc, up, secs = start(server, tag, base + ".log", record)
    res["started"], res["start_seconds"] = bool(up), secs
    try:
        if up:
            res["requests"] = {}
            for name, what, body in REQUESTS[server]:
                res["requests"][name] = {"what": what, **ask(body)}
        else:
            with open(base + ".log", encoding="utf-8", errors="replace") as f:
                tail = f.read().splitlines()
            res["exit"] = proc.poll()
            res["log_entail"] = [ln[:400] for ln in tail if "[entail]" in ln or "RoleError" in ln][-6:]
    finally:
        stop(proc)
    res["decisions"], res["summaries"] = recorded(record)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"== {server} {tag}: started={res['started']} ({secs}s)")
    for name, r in (res.get("requests") or {}).items():
        print(f"  {name:18s} {r['what']:8s} {r['status']} "
              f"{r.get('error', '')[:150] or (r.get('content', '')[:60] + ' | tools=' + str(r.get('tool_calls')) + ' | reasoning=' + str(r.get('reasoning_chars')))}")
    for d in res["decisions"]:
        print("  decision:", d["boundary"], d["verdict"], d["rule"][:60], d.get("resolution") or "")
    for line in res.get("log_entail", []):
        print("  log:", line[:300])


if __name__ == "__main__":
    main()
