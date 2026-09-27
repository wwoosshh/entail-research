"""M16 case, vllm-project/vllm#49412 (testbed/M16_PROTOCOL.md 5): parsers with strip_content_whitespace_with_tools
strip the content around tool calls on the non-streaming path and not on the streaming path, so a client gets
" done." when streaming and "done." when not. The report's cases (text after a tool call, before the first one,
and between two parallel calls) fed through the real non-streaming path (extract_tool_calls) and the real streaming
path (token by token) of the qwen3 parser with enable_thinking=False; parser-level, CPU, Qwen3-4B's local
tokenizer. vLLM 0.30.0.
Run in ~/venvs/vllm: python testbed/m16/cases/vl49412.py <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _parser_util import chat_request, drive_streaming, standin_tokenizer, tool  # noqa: E402

CALL = "<tool_call>\n<function=get_weather>\n<parameter=location>{}</parameter>\n</function>\n</tool_call>"
CASES = {"after_tool": CALL.format("Paris") + " done.",
         "before_tool": "Sure! " + CALL.format("Paris"),
         "between_parallel": CALL.format("Paris") + " mid " + CALL.format("Rome")}


def main():
    import vllm
    from vllm.parser.qwen3 import Qwen3Parser

    tok = standin_tokenizer([])
    req = chat_request(tools=[tool("get_weather", {"location": "string"})], chat_template_kwargs={"enable_thinking": False})
    rows, diverged = {}, 0
    for label, text in CASES.items():
        ns = Qwen3Parser(tok, chat_template_kwargs={"enable_thinking": False}).extract_tool_calls(text, req)
        st_args, st_names, st_content = drive_streaming(Qwen3Parser(tok, chat_template_kwargs={"enable_thinking": False}),
                                                        tok, text, req)
        row = {"non_streaming_content": ns.content, "streaming_content": st_content,
               "non_streaming_tool_calls": len(ns.tool_calls or []), "streaming_tool_calls": len(st_args),
               "diverged": (ns.content or "") != (st_content or "")}
        diverged += int(row["diverged"])
        rows[label] = row
    out = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "cases": rows,
           "diverged_cases": diverged, "reproduced": diverged > 0}
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
