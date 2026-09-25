"""M5.5: market cases L07 and L13 simulated on a real vLLM server (vLLM 0.30.0, Qwen3-4B).

  L07  gpt-oss on old vLLM builds: reasoning_effort was not passed on, so every request ran at the default effort
       (AIME25 93.3% -> 80.0% at Azure). Simulated: a chat template that reads reasoning_effort (a system line
       "Reasoning effort: ..."), served by a vLLM whose request handling is planted to leave reasoning_effort out of
       the template settings (adapters/vllm_seed.py drop_effort). Nothing errors; the prompt simply lacks the line.
  L13  MiniMax-M2 keeps earlier reasoning in the prompt (interleaved thinking); an OpenAI-compatible integration
       dropped it (Tau² 87 -> 64). Simulated: Qwen3's template changed to keep the reasoning of every earlier
       assistant turn, declared `keep`, and a client that sends the same conversation without it.
The template is Qwen3-4B's own with those two changes, served with --chat-template and declared in a pinned manifest
(its hash, reasoning_history keep, tool calls hermes). What the model is given is read from the usage the server
returns: the prompt's token count.
  clean_off   no planted defect, entail off: the reference counts
  clean_on    no planted defect, entail on: reasoning_effort reaches the template, so only the client's L13 turn is reported
  bug_off     reasoning_effort dropped, entail off: what L07 looked like
  bug_on      the same, entail on (the default policy): served, and what broke recorded
  bug_strict  the same, ENTAIL_ON_BROKEN=stop: refused
Writes testbed/results/m55/market_<tag>.json, .log, .jsonl. Run in ~/venvs/vllm: python testbed/m55_market_serve.py <tag>
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "entail"))
from entail import manifest  # noqa: E402
from entail.facts import Certainty, Fact, Source, Template  # noqa: E402

MODEL = os.path.expanduser("~/models/Qwen3-4B")
OUT = os.path.join(RESULTS, "m55")
PORT = 8123
URL = f"http://127.0.0.1:{PORT}"
TAGS = {
    "clean_off": {"ENTAIL": "off"},
    "clean_on": {"ENTAIL": "load"},
    "bug_off": {"ENTAIL": "off", "ENTAIL_SEED": "drop_effort"},
    "bug_on": {"ENTAIL": "load", "ENTAIL_SEED": "drop_effort"},
    "bug_strict": {"ENTAIL": "load", "ENTAIL_SEED": "drop_effort", "ENTAIL_ON_BROKEN": "stop"},
}
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
ASK = [{"role": "user", "content": "Name the capital of France in one word."}]
HISTORY = [{"role": "user", "content": "What is 2+2? Answer with the number only."},
           {"role": "assistant", "content": "4", "reasoning": "The user asks for 2 plus 2. That is 4."},
           {"role": "user", "content": "And 3+3? Answer with the number only."}]
DROPPED = [HISTORY[0], {"role": "assistant", "content": "4"}, HISTORY[2]]
REQUESTS = [
    ("plain", "no effort set", {**NO_THINK, "messages": ASK}),
    ("effort_high", "L07", {"reasoning_effort": "high", "max_tokens": 16, "messages": ASK}),
    ("history_kept", "L13 reference", {**NO_THINK, "messages": HISTORY}),
    ("history_dropped", "L13", {**NO_THINK, "messages": DROPPED}),
]


def template():
    """Qwen3-4B's chat template, reading reasoning_effort and keeping every earlier turn's reasoning."""
    text = json.load(open(os.path.join(MODEL, "tokenizer_config.json"), encoding="utf-8"))["chat_template"]
    keep = "loop.index0 > ns.last_query_index"
    assert text.count(keep) == 1, "Qwen3's template changed"
    effort = ("{%- if reasoning_effort is defined and reasoning_effort %}"
              "{{- '<|im_start|>system\\nReasoning effort: ' + reasoning_effort + '<|im_end|>\\n' }}{%- endif %}\n")
    return effort + text.replace(keep, "true")


def prepare():
    d = os.path.join(OUT, "market")
    os.makedirs(os.path.join(d, "manifests"), exist_ok=True)
    text = template()
    path = os.path.join(d, "template.jinja")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    value = Template(chat_template_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                     reasoning_history="keep", tool_call_format="hermes")
    m = manifest.pin(manifest.Manifest(manifest.sha256_of(MODEL), (
        Fact("Template", value, Source("manifest", "m55 market"), Certainty.DECLARED),), file="Qwen3-4B"))
    manifest.save(m, os.path.join(d, "manifests", f"{m.sha256}.json"))
    return path, os.path.join(d, "manifests"), value


def start(tag, log_path, record, template_path, manifests):
    env = dict(os.environ)
    for k in ("ENTAIL", "ENTAIL_SEED", "ENTAIL_ON_BROKEN", "ENTAIL_POLICY", "ENTAIL_RESPONSE_NOTE", "ENTAIL_UNKNOWN"):
        env.pop(k, None)
    # M9.3: the research tools' hook (the library's plus fault injection: drop_effort)
    env.update(PYTHONPATH=f"{ROOT}/entail:{ROOT}/entail/tools/autoinstall", ENTAIL_RECORD=record,
               ENTAIL_MANIFESTS=manifests, VLLM_LOGGING_LEVEL="WARNING", **TAGS[tag])
    cmd = [os.path.join(os.path.dirname(sys.executable), "vllm"), "serve", MODEL, "--served-model-name", "qwen3",
           "--port", str(PORT), "--max-model-len", "4096", "--gpu-memory-utilization", "0.88", "--enforce-eager",
           "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--reasoning-parser", "qwen3",
           "--chat-template", template_path]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            return proc, None
        try:
            if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                return proc, True
        except requests.RequestException:
            pass
        time.sleep(2)
    return proc, False


def ask(body):
    r = requests.post(f"{URL}/v1/chat/completions", json={"model": "qwen3", "max_tokens": 32, "temperature": 0,
                                                           **body}, timeout=300)
    data = r.json()
    if r.status_code != 200:
        return {"status": r.status_code, "error": str(data.get("error", data))[:400]}
    return {"status": 200, "prompt_tokens": data["usage"]["prompt_tokens"],
            "content": (data["choices"][0]["message"].get("content") or "")[:60]}


def main():
    sys.path.insert(0, HERE)
    import m53_serve as m53

    tag = sys.argv[1]
    template_path, manifests, declared = prepare()
    base = os.path.join(OUT, f"market_{tag}")
    record = base + ".jsonl"
    if os.path.exists(record):
        os.remove(record)
    res = {"tag": tag, "env": TAGS[tag], "declared": str(declared), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    proc, up = start(tag, base + ".log", record, template_path, manifests)
    res["started"] = bool(up)
    try:
        if up:
            res["requests"] = {name: {"case": case, **ask(body)} for name, case, body in REQUESTS}
    finally:
        m53.stop(proc)
    res["decisions"], res["summaries"] = m53.recorded(record)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"== market {tag}: started={res['started']}")
    for name, r in (res.get("requests") or {}).items():
        print(f"  {name:16s} {r['case']:14s} {r['status']} prompt_tokens={r.get('prompt_tokens')} "
              f"{r.get('error', '')[:120]}")
    for d in res["decisions"]:
        print("  decision:", d["boundary"], d["verdict"], d["rule"][:70])


if __name__ == "__main__":
    main()
