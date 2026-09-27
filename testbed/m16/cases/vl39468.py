"""M16 case, vllm-project/vllm#39468 (testbed/M16_PROTOCOL.md 5): the Gemma 4 tool-call parser leaves the string
delimiter token <|"|> inside the returned tool-call arguments (array values) or returns the raw call syntax as
content. The report's two model outputs (reconstructed from the wrong JSON it shows) through the non-streaming
tool-call extraction; parser-level, CPU, the real Gemma 4 tokenizer (google/gemma-4-E2B-it, tokenizer files only,
not gated) or, failing that, a stand-in with the control tokens added. Reported on 0.19.0 (open); runs on 0.30.0
first (vllm.parser.gemma4.Gemma4Parser) and on the 0.19.0 venv (vllm.tool_parsers.gemma4_tool_parser) second.
Run: python testbed/m16/cases/vl39468.py <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _parser_util import chat_request, standin_tokenizer, tool  # noqa: E402

GEMMA = ["<|tool_call>", "<tool_call|>", '<|"|>', "<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|tool_response>",
         "<tool_response|>"]
PR_RAW = ('pdfs:[<|"|>/home/h/.openclaw/pdfs/a.pdf<|"|>,<|"|>/home/h/.openclaw/pdfs/b.pdf<|"|>],'
          'prompt:<|"|>analyse these 2 pdfs.<|"|>')            # the regression test of the closed fix PR #39484
CASES = {
    "array_of_strings": ('<|tool_call>call:analyse_pdfs{pdfs:[<|"|>/home/h/a.pdf<|"|>,<|"|>/home/h/b.pdf<|"|>],'
                         'prompt:<|"|>analyse these 2 pdfs.<|"|>}<tool_call|>'),
    "array_of_paths_from_the_fix_pr": "<|tool_call>call:analyse_pdfs{" + PR_RAW + "}<tool_call|>",
    "scalars": '<|tool_call>call:process{action:<|"|>poll<|"|>,sessionId:<|"|>calm-ridge<|"|>,timeout:20000}<tool_call|>',
}
TOOLS = [tool("analyse_pdfs", {"pdfs": "array", "prompt": "string"}),
         tool("process", {"action": "string", "sessionId": "string", "timeout": "integer"})]


def gemma_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("google/gemma-4-E2B-it"), "google/gemma-4-E2B-it"
    except Exception as e:  # noqa: BLE001
        return standin_tokenizer(GEMMA), f"stand-in (Qwen3-4B + Gemma 4 control tokens): {type(e).__name__}"


def make_parser(tok):
    try:
        from vllm.parser.gemma4 import Gemma4Parser
        return Gemma4Parser(tok, chat_template_kwargs={}), "vllm.parser.gemma4.Gemma4Parser"
    except ImportError:
        from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser
        return Gemma4ToolParser(tok), "vllm.tool_parsers.gemma4_tool_parser.Gemma4ToolParser"


def parse_args_direct():
    """The fix PR's regression test: the module-level argument parser on the raw argument text."""
    try:
        from vllm.parser.gemma4 import _parse_gemma4_args
    except ImportError:
        from vllm.tool_parsers.gemma4_tool_parser import _parse_gemma4_args
    try:
        res = _parse_gemma4_args(PR_RAW)
        text = json.dumps(res)
        return {"parsed": res, "delimiter_leaked": '<|"' in text or '"|>' in text}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200], "delimiter_leaked": False}


def main():
    import vllm

    tok, tok_name = gemma_tokenizer()
    rows, leaked = {}, 0
    direct = parse_args_direct()
    rows["_parse_gemma4_args(PR_RAW)"] = direct
    leaked += int(direct["delimiter_leaked"])
    for label, text in CASES.items():
        parser, pname = make_parser(tok)
        req = chat_request(tools=TOOLS)
        res = parser.extract_tool_calls(text, req)
        calls = [{"name": c.function.name, "arguments": c.function.arguments} for c in (res.tool_calls or [])]
        bad = any('<|"' in (c["arguments"] or "") for c in calls) or '<|"' in (res.content or "") or (
            not calls and "<|tool_call>" in (res.content or ""))
        leaked += int(bad)
        rows[label] = {"tool_calls": calls, "content": res.content, "delimiter_leaked_or_unparsed": bad}
    out = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "tokenizer": tok_name,
           "parser": pname, "cases": rows, "reproduced": leaked > 0}
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
