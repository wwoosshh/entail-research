"""M18.3 retro at the rule level: vLLM 0.30's real parsers, built through ParserManager.get_parser with entail on (the
vllm_parse adapter wraps the class the server would build from), driven as the server drives them - parse_delta per
token with finished=True on the last, or parse for a whole text. Cases: vllm#49316 (kimi_k2: streaming skips the
schema's type coercion, 4 texts), #49412 (qwen3: whitespace around tool calls kept on the streamed path, 3 texts),
#47986 (deepseek_v4: tool_b unwrapped with tool_a's schema; whole text, the schema rule), and healthy texts for
qwen3 and kimi_k2 (a plain answer, one well-formed call) for the false-alarm side. Not the HTTP server: the
adapter's class wrapper and the core rules on the engine's own parser classes, with a stand-in tokenizer
(Qwen3-4B's with the model's control tokens added) and no prompt (prompt_token_ids=[]: the reasoning phase a real
prompt would set is not exercised here; the live-server run of M18.5 covers that).
Run in ~/venvs/vllm: python testbed/m18/parse_retro.py <out.json>
"""
import json
import os
import sys
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "entail"))
sys.path.insert(0, os.path.join(ROOT, "testbed", "m16", "cases"))
os.environ.setdefault("ENTAIL_LOG_DIR", os.path.join(ROOT, "testbed", "results", "m18", "parse_retro", "entail_logs"))

KIMI = ["<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>", "<|tool_call_end|>",
        "<|tool_call_argument_begin|>", "<think>", "</think>"]
QWEN_CALL = "<tool_call>\n<function=get_weather>\n<parameter=location>{}</parameter>\n</function>\n</tool_call>"


def kimi_text(name, raw):
    return (f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>"
            f"{raw}<|tool_call_end|><|tool_calls_section_end|>")


def drive(cls, tok, text, req, kwargs):
    """The server's streaming drive: one parse_delta per token, finished on the last."""
    p = cls(tok, req.tools, chat_template_kwargs=kwargs, model_config=None)
    ids = tok.encode(text, add_special_tokens=False)
    outs = []
    for i, tid in enumerate(ids):
        piece = tok.decode([tid], skip_special_tokens=False)
        outs.append(p.parse_delta(piece, [tid], req, [], finished=(i == len(ids) - 1)))
    return outs


def main():
    out_path = sys.argv[1]
    import entail
    from entail import load

    entail.enable()
    import vllm
    from vllm.parser.parser_manager import ParserManager

    from _parser_util import chat_request, standin_tokenizer, tool

    def decisions_since(n):
        return [{"verdict": d.verdict.value, "rule": d.rule, "consumer": d.contract.consumer, "note": d.note[:400]}
                for d in load.LEDGER.decisions[n:] if d.name == "Parse"]

    rows = []

    def run(case, expect, fn):
        n = len(load.LEDGER.decisions)
        try:
            fn()
            err = None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:200]
        ds = decisions_since(n)
        verdicts = [d["verdict"] for d in ds]
        rows.append({"case": case, "expected": expect, "decisions": ds, "verdicts": verdicts, "error": err})
        print(f"{case:<40} expected {expect:<8} got {verdicts} {err or ''}", flush=True)
        for d in ds:
            if d["verdict"] != "pass":
                print("    ", d["verdict"], "|", d["rule"][:50], "|", d["note"][:230], flush=True)

    # --- kimi_k2: vllm#49316 (streaming skips type coercion) + a healthy call ---
    kimi_cls = ParserManager.get_parser(tool_parser_name="kimi_k2", reasoning_parser_name="kimi_k2",
                                        enable_auto_tools=True, model_name="m")
    print("kimi_k2 class:", kimi_cls, "wrapped:", hasattr(kimi_cls, "_entail_auto_tools"))
    ktok = standin_tokenizer(KIMI)
    kimi_kwargs = {"thinking": False}
    for name, prop, typ, raw in [("set_count", "count", "integer", '{"count":"3"}'),
                                 ("set_flag", "flag", "boolean", '{"flag":"true"}'),
                                 ("set_ratio", "ratio", "number", '{"ratio":"1.5"}'),
                                 ("set_id", "id", "string", '{"id":42}')]:
        req = chat_request(tools=[tool(name, {prop: typ})])
        run(f"vl49316 kimi_k2 {name}", "broken", lambda: drive(kimi_cls, ktok, kimi_text(name, raw), req, kimi_kwargs))
    req = chat_request(tools=[tool("set_count", {"count": "integer"})])
    run("healthy kimi_k2 typed call", "pass", lambda: drive(kimi_cls, ktok, kimi_text("set_count", '{"count":3}'), req, kimi_kwargs))
    run("healthy kimi_k2 plain answer", "pass", lambda: drive(kimi_cls, ktok, "The capital of France is Paris.", chat_request(), kimi_kwargs))

    # --- qwen3: vllm#49412 (whitespace kept on the streamed path) + healthy ---
    qwen_cls = ParserManager.get_parser(tool_parser_name="qwen3_xml", reasoning_parser_name="qwen3",
                                        enable_auto_tools=True, model_name="m")
    print("qwen3 class:", qwen_cls, "wrapped:", hasattr(qwen_cls, "_entail_auto_tools"))
    qtok = standin_tokenizer([])
    qkw = {"enable_thinking": False}
    qreq = chat_request(tools=[tool("get_weather", {"location": "string"})], chat_template_kwargs=qkw)
    for label, text in {"after_tool": QWEN_CALL.format("Paris") + " done.",
                        "before_tool": "Sure! " + QWEN_CALL.format("Paris"),
                        "between_parallel": QWEN_CALL.format("Paris") + " mid " + QWEN_CALL.format("Rome")}.items():
        run(f"vl49412 qwen3 {label}", "broken", lambda: drive(qwen_cls, qtok, text, qreq, qkw))
    run("healthy qwen3 one call", "pass", lambda: drive(qwen_cls, qtok, QWEN_CALL.format("Paris"), qreq, qkw))
    run("healthy qwen3 plain answer", "pass", lambda: drive(qwen_cls, qtok, "Sure, here is the answer.", chat_request(chat_template_kwargs=qkw), qkw))
    run("healthy qwen3 two calls", "pass", lambda: drive(qwen_cls, qtok, QWEN_CALL.format("Paris") + QWEN_CALL.format("Rome"), qreq, qkw))

    # --- deepseek_v4: vllm#47986 (whole text, the schema rule) ---
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam, FunctionDefinition
    except ImportError:
        from vllm.entrypoints.openai.protocol import ChatCompletionToolsParam, FunctionDefinition

    def make_tool(name, properties, **more):
        return ChatCompletionToolsParam(type="function", function=FunctionDefinition(
            name=name, parameters=dict({"type": "object", "properties": properties}, **more)))

    # The report's tools with tool_b's declaration made precise, so that the case discriminates (review 2, finding
    # 13c): tool_b takes one object parameter `arguments`, requires it and forbids anything else. A correct slot
    # lookup keeps tool_b's wrapper {"arguments": {...}}, which fits; the defect unwraps it with tool_a's schema to
    # {"city": ...}, which lacks the required parameter and carries a forbidden key.
    tools = [make_tool("tool_a", {"city": {"type": "string"}}),
             make_tool("tool_b", {"arguments": {"type": "object", "properties": {"tz": {"type": "string"}}}},
                       required=["arguments"], additionalProperties=False)]
    ds_cls = ParserManager.get_parser(tool_parser_name="deepseek_v4", reasoning_parser_name="deepseek_v4",
                                      enable_auto_tools=True, model_name="m")
    print("deepseek_v4 class:", ds_cls, "wrapped:", hasattr(ds_cls, "_entail_auto_tools"))
    mtok = MagicMock()
    mtok.get_vocab.return_value = {}
    mtok.decode.side_effect = lambda ids: f"tok{ids[0]}"
    dreq = MagicMock()
    dreq.tools = tools
    dreq.tool_choice = "auto"
    dreq.include_reasoning = True
    text = ('<｜DSML｜tool_calls>'
            '<｜DSML｜invoke name="tool_a"><｜DSML｜parameter name="arguments" string="false">{"city": "Tokyo"}</｜DSML｜parameter></｜DSML｜invoke>'
            '<｜DSML｜invoke name="tool_b"><｜DSML｜parameter name="arguments" string="false">{"city": "Tokyo"}</｜DSML｜parameter></｜DSML｜invoke>'
            '</｜DSML｜tool_calls>')
    calls = {}

    def whole(t):
        r = ds_cls(mtok, tools=tools, model_config=None).parse(t, dreq, enable_auto_tools=True)
        calls["last"] = [(tc.name, tc.arguments) for tc in (r[2] or [])]
        return r

    run("vl47986 deepseek_v4 whole text", "broken", lambda: whole(text))
    print("    parsed calls:", calls.get("last"))
    # the control: the same raw arguments through the parser's own unwrapping with the right slot (what a fixed
    # lookup yields), held to the same schema rule
    from entail import parse_contract
    from vllm.parser.deepseek_v4 import _unwrap_wrapper_args

    fixed = [("tool_a", _unwrap_wrapper_args('{"arguments": {"city": "Tokyo"}}', tools, "tool_a")),
             ("tool_b", _unwrap_wrapper_args('{"arguments": {"city": "Tokyo"}}', tools, "tool_b"))]
    print("    fixed slot lookup yields:", fixed)
    run("control deepseek_v4 fixed slot lookup (schema rule)", "pass",
        lambda: parse_contract.check_schema("request:vllm.parser", "vllm.parser.control", fixed, tools,
                                            "control: the slot lookup fixed", raw_text=text))
    healthy = ('<｜DSML｜tool_calls><｜DSML｜invoke name="tool_a"><｜DSML｜parameter name="arguments" string="false">'
               '{"city": "Tokyo"}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>')
    run("healthy deepseek_v4 one call", "pass", lambda: whole(healthy))

    summary = {"vllm": vllm.__version__, "entail": entail.__version__, "rows": rows,
               "expected_broken_caught": sum(1 for r in rows if r["expected"] == "broken" and "broken" in r["verdicts"]),
               "expected_broken": sum(1 for r in rows if r["expected"] == "broken"),
               "healthy_with_broken": sum(1 for r in rows if r["expected"] == "pass" and "broken" in r["verdicts"]),
               "healthy": sum(1 for r in rows if r["expected"] == "pass")}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
