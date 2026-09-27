"""M17.6 case, vllm-project/vllm#42047 (testbed/M16_PROTOCOL.md 7): Gemma4ToolParser streams wrong float argument
values - 108.2 arrives as 108.02, 22.8 as 22.08 - while non-streaming is right. Parser-level: the model's tool
call `<|tool_call>call:add{left:108.2,right:22.8}<tool_call|>` (Gemma 4's standard format, gemma4_utils.py) fed
token by token through extract_tool_calls_streaming, against the non-streaming extraction. The tokenizer is the
first replay's stand-in (Qwen3-4B's with Gemma 4's control tokens added as special tokens: the gemma-4 folder's
own tokenizer does not load under the 0.19 venv's transformers). Reproduced when the streamed arguments parse to
values other than 108.2 and 22.8 while the non-streaming ones are right.
Run in ~/venvs/vllm0190 (the reported 0.19.x): python testbed/m17/replay2/cases/vl42047.py <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "m16", "cases"))
from _parser_util import chat_request, drive_streaming, standin_tokenizer, tool  # noqa: E402

TEXT = "<|tool_call>call:add{left:108.2,right:22.8}<tool_call|>"
CONTROL = ["<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|tool_response>",
           "<tool_response|>"]


def main():
    import vllm
    from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

    tok = standin_tokenizer(CONTROL)
    request = chat_request(tools=[tool("add", {"left": "number", "right": "number"})])
    parser = Gemma4ToolParser(tok)
    full = parser.extract_tool_calls(TEXT, request)
    non_stream = [(tc.function.name, tc.function.arguments) for tc in (full.tool_calls or [])]
    parser2 = Gemma4ToolParser(tok)
    args, names, content = drive_streaming(parser2, tok, TEXT, request)
    streamed = {str(i): args.get(i, "") for i in sorted(set(args) | set(names))}

    def values(s):
        try:
            d = json.loads(s)
            return [d.get("left"), d.get("right")]
        except (ValueError, AttributeError):
            return None

    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "text": TEXT,
           "non_streaming": non_stream, "streaming_arguments": streamed,
           "streaming_names": {str(k): v for k, v in names.items()}, "streaming_content": content,
           "tokens": tok.encode(TEXT, add_special_tokens=False)}
    ns_vals = values(non_stream[0][1]) if non_stream else None
    st_vals = values(next(iter(streamed.values()), "")) if streamed else None
    row["non_streaming_values"], row["streaming_values"] = ns_vals, st_vals
    row["reproduced"] = bool(ns_vals == [108.2, 22.8] and st_vals is not None and st_vals != [108.2, 22.8])
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
