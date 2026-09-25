"""M5.5: market case L05 simulated (reinvestigation/market_incidents.md L05; Ollama issue #7043).

Ollama gave every model a default context (num_ctx 2048) and cut a longer prompt to it - "truncating input prompt
limit=2048 prompt=10983 keep=5 new=2048" - in the server's log only; the user saw odd answers. The simulation is that
mechanism in miniature: a server keeps the prompt's first `keep` tokens and its last ones up to the context, and the
"model" answers with what the prompt said at its start (a fact placed there, the rest filler). The server's adapter
reads the prompt's length, the context and where its value came from (a default, or the user's num_ctx), and the
context the model declares; the rule is the core's (request_contract.window). The handle "extend_context" gives this
request the context it needs.
Scenarios: the log's 10983 tokens against the default 2048 with a model declaring 32768 (room), the same with the user
setting num_ctx=2048, the same with a model declaring 8192 (no room), and a prompt that fits (1000 tokens).
Policies: off, on (the default), strict (ENTAIL_ON_BROKEN=stop). Writes testbed/results/m55/l05.json.
Run in ~/venvs/gpu (no GPU needed): python testbed/m55_l05.py
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
from entail import core, load, request_contract  # noqa: E402

DEFAULT_NUM_CTX, KEEP, FACT = 2048, 5, 7


class Server:
    """An Ollama-like server: a model with a declared context, and num_ctx - the user's, or the default."""

    def __init__(self, model_context, num_ctx=None):
        self.model_context = model_context
        self.num_ctx, self.origin = (num_ctx, "user") if num_ctx is not None else (DEFAULT_NUM_CTX, "default")

    def generate(self, prompt, num_ctx=None):
        ctx = num_ctx or self.num_ctx
        kept = prompt if len(prompt) <= ctx else prompt[:KEEP] + prompt[len(prompt) - (ctx - KEEP):]
        return {"answer": kept[KEEP] if len(kept) > KEEP else None, "tokens_seen": len(kept)}


def install(server_cls):
    """The adapter: where the server takes a request and its context (read_choice), and the handle."""
    orig = server_cls.generate

    def generate(self, prompt, num_ctx=None):
        if core.mode() not in ("load", "debug"):
            return orig(self, prompt, num_ctx)
        decisions = request_contract.window("request:ollama_like.context", "ollama_like.model", len(prompt),
                                            self.num_ctx, self.origin, self.model_context, "ollama-like request")
        done = load.resolve(decisions, {"extend_context": lambda target: target})
        return orig(self, prompt, done.get("extend_context", num_ctx))

    server_cls.generate = generate


def prompt_of(n):
    """The fact the model must repeat sits right after the kept head; the rest is filler."""
    return [0] * KEEP + [FACT] + [1] * (n - KEEP - 1)


SCENARIOS = {
    "default_context_model_has_room": (dict(model_context=32768), 10983),
    "user_set_num_ctx": (dict(model_context=32768, num_ctx=2048), 10983),
    "model_without_room": (dict(model_context=8192), 10983),
    "prompt_fits": (dict(model_context=32768), 1000),
}


def main():
    install(Server)
    res = {"default_num_ctx": DEFAULT_NUM_CTX, "keep": KEEP, "scenarios": {}}
    for name, (kw, n) in SCENARIOS.items():
        entry = {"prompt_tokens": n, **kw}
        for policy in ("off", "on", "strict"):
            core.set_mode("off" if policy == "off" else "load")
            if policy == "strict":
                os.environ["ENTAIL_ON_BROKEN"] = "stop"
            else:
                os.environ.pop("ENTAIL_ON_BROKEN", None)
            request_contract.reset()
            first, printed = len(load.LEDGER.decisions), io.StringIO()
            try:
                with redirect_stdout(printed):
                    out = Server(**kw).generate(prompt_of(n))
                error = None
            except core.RoleError as e:
                out, error = None, str(e).splitlines()[0][:300]
            entry[policy] = {"answer_right": None if out is None else out["answer"] == FACT,
                             "tokens_seen": None if out is None else out["tokens_seen"], "stopped": error is not None,
                             "decisions": [{"verdict": d.verdict.value, "rule": d.rule, "resolution": d.resolution,
                                            "target": d.target} for d in load.LEDGER.decisions[first:]]}
        res["scenarios"][name] = entry
    core.set_mode("off")
    os.environ.pop("ENTAIL_ON_BROKEN", None)
    out = os.path.join(RESULTS, "m55", "l05.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for name, entry in res["scenarios"].items():
        print(f"== {name} (prompt {entry['prompt_tokens']})")
        for policy in ("off", "on", "strict"):
            r = entry[policy]
            print(f"  {policy:6s} answer_right={r['answer_right']!s:5s} seen={r['tokens_seen']!s:6s} "
                  f"stopped={r['stopped']!s:5s} {[d['verdict'] + ':' + d['rule'][:40] for d in r['decisions']]}")
    print("wrote", out)


if __name__ == "__main__":
    main()
