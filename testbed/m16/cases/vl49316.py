"""M16 case, vllm-project/vllm#49316 (testbed/M16_PROTOCOL.md 5): for the kimi_k2 parser the streaming path skips
the tool-schema type coercion that the non-streaming path applies, so a client gets differently typed argument
values for the same model output ("3" vs 3 for an integer field). The report's four cases fed through the real
non-streaming path (extract_tool_calls) and the real streaming path (extract_tool_calls_streaming, token by token);
parser-level, CPU, no model: Qwen3-4B's local tokenizer with Kimi K2's control tokens added stands in for Kimi's
tokenizer (the report says the divergence is chunk-invariant). vLLM 0.30.0.
Run in ~/venvs/vllm: python testbed/m16/cases/vl49316.py <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _parser_util import chat_request, drive_streaming, standin_tokenizer, tool  # noqa: E402

KIMI = ["<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>", "<|tool_call_end|>",
        "<|tool_call_argument_begin|>", "<think>", "</think>"]
CASES = [("set_count", "count", "integer", '{"count":"3"}'), ("set_flag", "flag", "boolean", '{"flag":"true"}'),
         ("set_ratio", "ratio", "number", '{"ratio":"1.5"}'), ("set_id", "id", "string", '{"id":42}')]


def main():
    import vllm
    from vllm.parser.kimi_k2 import KimiK2Parser

    tok = standin_tokenizer(KIMI)
    rows, diverged = [], 0
    for name, prop, typ, raw in CASES:
        text = (f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>"
                f"{raw}<|tool_call_end|><|tool_calls_section_end|>")
        req = chat_request(tools=[tool(name, {prop: typ})])
        ns = KimiK2Parser(tok, chat_template_kwargs={"thinking": False}).extract_tool_calls(text, req)
        ns_args = ns.tool_calls[0].function.arguments if ns.tool_calls else None
        st_args, st_names, _ = drive_streaming(KimiK2Parser(tok, chat_template_kwargs={"thinking": False}), tok,
                                               text, req)
        st = st_args.get(0)
        row = {"tool": name, "schema_type": typ, "model_args": raw, "non_streaming": ns_args, "streaming": st}
        try:
            a, b = json.loads(ns_args), json.loads(st)
            row["value_types"] = {"non_streaming": type(a[prop]).__name__, "streaming": type(b[prop]).__name__}
            row["diverged"] = a != b
        except Exception as e:  # noqa: BLE001
            row["value_types"], row["diverged"] = f"error: {e}"[:200], False
        diverged += int(row["diverged"])
        rows.append(row)
    out = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "cases": rows,
           "diverged_cases": diverged, "reproduced": diverged > 0}
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
