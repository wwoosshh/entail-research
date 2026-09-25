"""M5.3: what the request boundary costs per request (S4 at the request site: microseconds per request).

On vLLM 0.30.0's own functions (the request schema, build_chat_params, resolve_chat_template, the template's
variables) with Qwen3-4B's tokenizer and template, no GPU. Per request shape: entail's part - the fields check and
everything decided before rendering (history, template, settings) - against the rendering it precedes, Hugging Face's
apply_chat_template with tokenizing, which is what vLLM runs for the request next. Alternating order, medians.
Writes testbed/results/m53/overhead.json (or into TESTBED_OUT; M5.4 used it). Run in ~/venvs/vllm:
python testbed/m53_overhead.py
"""
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
from entail import core, policies, request_contract  # noqa: E402
from entail.adapters import vllm_serve as vs  # noqa: E402

MODEL = os.path.expanduser("~/models/Qwen3-4B")
N, WARM = 2000, 100
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
ASK = [{"role": "user", "content": "Name the capital of France in one word."}]
TURNS = [{"role": "user", "content": f"Question {i}?"} if i % 2 == 0 else
         {"role": "assistant", "content": f"Answer {i}.", "reasoning": f"Thinking about {i}."} for i in range(9)]
BODIES = {
    "plain": {"messages": ASK, "chat_template_kwargs": {"enable_thinking": False}},
    "ten_messages": {"messages": TURNS, "chat_template_kwargs": {"enable_thinking": False}},
    "tool_call": {"messages": ASK, "tools": TOOLS, "tool_choice": "auto", "reasoning_effort": "none"},
}


def per_call(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    tok = AutoTokenizer.from_pretrained(MODEL)
    mc = SimpleNamespace(model=MODEL, revision=None, code_revision=None, trust_remote_code=False)
    renderer = SimpleNamespace(model_config=mc, get_tokenizer=lambda: tok)
    core.set_mode("load")
    request_contract.reset()
    vs._SERVED[MODEL] = "qwen3"
    res = {"n": N, "vllm": "0.30.0", "model": "Qwen3-4B", "shapes": {}}
    for name, body in BODIES.items():
        req = ChatCompletionRequest(model="qwen3", max_tokens=64, **body)
        tools = [t.model_dump() for t in req.tools] if req.tools else None
        params = req.build_chat_params(None, "auto").with_defaults({"tools": tools, "tokenize": False})
        messages = req.messages

        def entail_part():   # what the hooks do for one request: the policy read once, the fields, before rendering
            policy = policies.current()
            token = vs._REQUEST.set((req, policy, []))
            try:
                vs._fields(req, policy, [])
                vs._before_render(renderer, messages, params)
            finally:
                vs._REQUEST.reset(token)

        kwargs = {k: v for k, v in params.chat_template_kwargs.items() if v is not None and k != "tokenize"}

        def render():
            tok.apply_chat_template(messages, tokenize=True, **kwargs)

        for _ in range(WARM):
            entail_part()
            render()
        e, r = [], []
        for i in range(10):   # alternating order, N/10 calls each
            if i % 2:
                r.append(per_call(render, N // 10))
                e.append(per_call(entail_part, N // 10))
            else:
                e.append(per_call(entail_part, N // 10))
                r.append(per_call(render, N // 10))
        res["shapes"][name] = {"entail_us": round(statistics.median(e), 2), "render_us": round(statistics.median(r), 2),
                               "ratio": round(statistics.median(e) / statistics.median(r), 4)}
        print(name, res["shapes"][name], flush=True)
    res["counts"] = {b: request_contract.stats(b) for b in (vs.TEMPLATE, vs.SETTINGS, vs.HISTORY, vs.FIELDS)}
    assert all(s["refused"] == 0 for s in res["counts"].values()), res["counts"]
    out = os.path.join(os.environ.get("TESTBED_OUT") or os.path.join(RESULTS, "m53"), "overhead.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
