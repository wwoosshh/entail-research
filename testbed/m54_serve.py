"""M5.4: the M5.3 servers again (vLLM 0.30.0, Qwen3-4B, the same requests and manifests), under the policy that
reports what it cannot repair and goes on - the researcher's decision of 2026-09-24.

  on              ENTAIL=load, the default policy: every request is served; what broke is in the log and ENTAIL_RECORD
  note            the same, with ENTAIL_RESPONSE_NOTE=1: a response also carries what broke for its request (an
                  "entail" field; for a stream, SSE comment lines ahead of the data)
  strict          ENTAIL_ON_BROKEN=stop: what M5.3 measured (400 before generating)
  observe         ENTAIL_POLICY=refuse: nothing is repaired, everything is reported (the tool parser is not switched)
  observe_strict  ENTAIL_POLICY=refuse and ENTAIL_ON_BROKEN=stop: the start is refused
Writes testbed/results/m54/<server>_<tag>.json, .log and .jsonl. Run in ~/venvs/vllm:
  python testbed/m54_serve.py <server> <tag>
"""
import json
import os
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
sys.path.insert(0, HERE)
import m53_serve as m53  # noqa: E402

OUT = os.path.join(RESULTS, "m54")
TAGS = {
    "on": {},
    "note": {"ENTAIL_RESPONSE_NOTE": "1"},
    "strict": {"ENTAIL_ON_BROKEN": "stop"},
    "observe": {"ENTAIL_POLICY": "refuse"},
    "observe_strict": {"ENTAIL_POLICY": "refuse", "ENTAIL_ON_BROKEN": "stop"},
}
# one streamed request per server, to see what a stream carries under `note`
STREAMED = {"healthy": "misspelt_setting", "keep": "dropped_reasoning", "tool": None}


def start(server, tag, log_path, record):
    env = dict(os.environ)
    for k in ("ENTAIL_ON_BROKEN", "ENTAIL_POLICY", "ENTAIL_RESPONSE_NOTE", "ENTAIL_UNKNOWN", "ENTAIL_FACT_POLICY"):
        env.pop(k, None)
    env.update(PYTHONPATH=f"{m53.ROOT}/entail:{m53.ROOT}/entail/entail/adapters/autoinstall", ENTAIL_RECORD=record,
               ENTAIL="load", VLLM_LOGGING_LEVEL="WARNING", **TAGS[tag])
    env["ENTAIL_MANIFESTS"] = m53.manifest_dir(server)[0]
    cmd = [os.path.join(os.path.dirname(sys.executable), "vllm"), "serve", m53.MODEL, "--served-model-name", "qwen3",
           "--port", str(m53.PORT), "--max-model-len", "4096", "--gpu-memory-utilization", "0.88", "--enforce-eager",
           "--enable-auto-tool-choice", "--reasoning-parser", "qwen3"] + m53.SERVER_ARGS[server]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            return proc, None, round(time.time() - t0, 1)
        try:
            if requests.get(f"{m53.URL}/health", timeout=2).status_code == 200:
                return proc, True, round(time.time() - t0, 1)
        except requests.RequestException:
            pass
        time.sleep(2)
    return proc, False, round(time.time() - t0, 1)


def ask(body):
    """As m53_serve.ask, and the "entail" field a response carries under `note`."""
    r = requests.post(f"{m53.URL}/v1/chat/completions", json={"model": "qwen3", "max_tokens": 64, "temperature": 0,
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
               finish_reason=data["choices"][0].get("finish_reason"), entail_field=data.get("entail"))
    return out


def ask_stream(body):
    """The raw lines of a streamed answer: SSE comments (': ...') and the first data line."""
    r = requests.post(f"{m53.URL}/v1/chat/completions", json={"model": "qwen3", "max_tokens": 16, "temperature": 0,
                                                               "stream": True, **body}, timeout=300, stream=True)
    lines, data = [], 0
    for raw in r.iter_lines(decode_unicode=True):
        if raw.startswith(":"):
            lines.append(raw[:300])
        elif raw.startswith("data:"):
            data += 1
    return {"status": r.status_code, "comments": lines, "data_lines": data}


def main():
    server, tag = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f"{server}_{tag}")
    record = base + ".jsonl"
    if os.path.exists(record):
        os.remove(record)
    res = {"server": server, "tag": tag, "env": TAGS[tag], "args": m53.SERVER_ARGS[server],
           "declared": str(m53.manifest_dir(server)[1]), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    proc, up, secs = start(server, tag, base + ".log", record)
    res["started"], res["start_seconds"] = bool(up), secs
    try:
        if up:
            res["requests"] = {name: {"what": what, **ask(body)} for name, what, body in m53.REQUESTS[server]}
            name = STREAMED[server]
            if name:
                body = next(b for n, _, b in m53.REQUESTS[server] if n == name)
                res["stream"] = {"request": name, **ask_stream(body)}
        else:
            with open(base + ".log", encoding="utf-8", errors="replace") as f:
                tail = f.read().splitlines()
            res["exit"] = proc.poll()
            res["log_entail"] = [ln[:400] for ln in tail if "[entail]" in ln or "RoleError" in ln][-6:]
    finally:
        m53.stop(proc)
    res["decisions"], res["summaries"] = m53.recorded(record)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"== {server} {tag}: started={res['started']} ({secs}s)")
    for name, r in (res.get("requests") or {}).items():
        shown = r.get("error", "")[:120] or (r.get("content", "")[:40].replace("\n", " ") + f" | tools={r.get('tool_calls')}"
                                             f" | reasoning={r.get('reasoning_chars')} | entail="
                                             f"{len(r['entail_field']) if r.get('entail_field') else None}")
        print(f"  {name:18s} {r['what']:8s} {r['status']} {shown}")
    if res.get("stream"):
        print("  stream:", res["stream"]["status"], res["stream"]["data_lines"], "data lines,",
              len(res["stream"]["comments"]), "comment line(s)", res["stream"]["comments"][:1])
    for d in res["decisions"]:
        print("  decision:", d["boundary"], d["verdict"], d["rule"][:60], d.get("resolution") or "")
    for line in res.get("log_entail", []):
        print("  log:", line[:300])


if __name__ == "__main__":
    main()
