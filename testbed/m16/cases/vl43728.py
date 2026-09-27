"""M16 case, vllm-project/vllm#43728 (testbed/M16_PROTOCOL.md 5): the client's chat_template_kwargs key that turns
thinking off is read under one name by the chat template (enable_thinking, Kimi K2's template) and under another by
the reasoning parser (thinking), so with {"enable_thinking": false} the template injects no thinking tokens, the
model answers plainly, and the parser still runs in thinking mode and returns content: null with the answer under
reasoning. Parser-level, CPU, no model: the kimi_k2 reasoning parser's split of a plain answer under a request that
carries {"enable_thinking": false}; Qwen3-4B's local tokenizer with Kimi K2's control tokens stands in. Runs on
0.30.0 first (vllm.parser.kimi_k2.KimiK2Parser reads both names there) and on the 0.19.0 venv second.
Run: python testbed/m16/cases/vl43728.py <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _parser_util import chat_request, standin_tokenizer  # noqa: E402

KIMI = ["<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>", "<|tool_call_end|>",
        "<|tool_call_argument_begin|>", "<think>", "</think>", "◁think▷", "◁/think▷"]
ANSWER = "The capital of France is Paris."


def make_parser(tok, chat_kwargs):
    try:
        from vllm.parser.kimi_k2 import KimiK2Parser
        return KimiK2Parser(tok, chat_template_kwargs=chat_kwargs), "vllm.parser.kimi_k2.KimiK2Parser"
    except ImportError:
        from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
        name = "vllm.reasoning.kimi_k2_reasoning_parser.KimiK2ReasoningParser"
        try:      # 0.2x reads chat_template_kwargs in the constructor (the server passes the request's kwargs)
            return KimiK2ReasoningParser(tok, chat_template_kwargs=chat_kwargs), name + " (chat_template_kwargs)"
        except TypeError:
            return KimiK2ReasoningParser(tok), name


def split(parser, text, req):
    fn = getattr(parser, "extract_reasoning", None) or getattr(parser, "extract_reasoning_content")
    return fn(text, req)


def main():
    import vllm

    tok = standin_tokenizer(KIMI)
    rows = {}
    for label, kw in (("enable_thinking_false", {"enable_thinking": False}), ("thinking_false", {"thinking": False}),
                      ("none", {})):
        parser, pname = make_parser(tok, kw)
        req = chat_request(chat_template_kwargs=kw)
        reasoning, content = split(parser, ANSWER, req)
        rows[label] = {"chat_template_kwargs": kw, "reasoning": reasoning, "content": content}
    bug = rows["enable_thinking_false"]
    out = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "parser": pname, "cases": rows,
           "reproduced": bool(bug["content"] is None and (bug["reasoning"] or "") != ""
                              and rows["thinking_false"]["content"] == ANSWER)}
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
