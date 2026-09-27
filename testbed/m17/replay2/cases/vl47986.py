"""M17.6 case, vllm-project/vllm#47986 (testbed/M16_PROTOCOL.md 7): DeepSeekV4Parser._convert_args finds the tool
slot of a raw argument string by exact text equality across the active slots, so two parallel calls with the same
raw argument text share a slot and one call is unwrapped with the other tool's schema. The report's minimal
reproduction, unchanged: tool_a's schema permits unwrapping "city", tool_b's does not; both invokes carry the same
wrapped arguments. Reproduced when tool_b's arguments come back unwrapped like tool_a's.
Run in ~/venvs/vllm (0.30.0): python testbed/m17/replay2/cases/vl47986.py <out.json>
"""
import json
import os
import sys
from unittest.mock import MagicMock


def main():
    import vllm
    from vllm.parser.deepseek_v4 import DeepSeekV4Parser
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam, FunctionDefinition
    except ImportError:
        from vllm.entrypoints.openai.protocol import ChatCompletionToolsParam, FunctionDefinition

    def make_tool(name, properties):
        return ChatCompletionToolsParam(type="function", function=FunctionDefinition(
            name=name, parameters={"type": "object", "properties": properties}))

    tools = [make_tool("tool_a", {"city": {"type": "string"}}), make_tool("tool_b", {"tz": {"type": "string"}})]
    tok = MagicMock()
    tok.get_vocab.return_value = {}
    tok.decode.side_effect = lambda ids: f"tok{ids[0]}"
    engine = DeepSeekV4Parser(tok, tools=tools)
    req = MagicMock()
    req.tools = tools
    req.tool_choice = "auto"
    text = ('<｜DSML｜tool_calls>'
            '<｜DSML｜invoke name="tool_a">'
            '<｜DSML｜parameter name="arguments" string="false">{"city": "Tokyo"}</｜DSML｜parameter>'
            '</｜DSML｜invoke>'
            '<｜DSML｜invoke name="tool_b">'
            '<｜DSML｜parameter name="arguments" string="false">{"city": "Tokyo"}</｜DSML｜parameter>'
            '</｜DSML｜invoke>'
            '</｜DSML｜tool_calls>')
    result = engine.extract_tool_calls(text, req)
    calls = [(tc.function.name, tc.function.arguments) for tc in result.tool_calls]
    by_name = dict(calls)
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "calls": calls}
    try:
        a = json.loads(by_name.get("tool_a", "{}"))
        b = json.loads(by_name.get("tool_b", "{}"))
    except ValueError:
        a, b = {}, {}
    # tool_b's schema has no "city": its wrapper must stay ({"arguments": {...}}); unwrapped like tool_a is the defect
    row["tool_b_unwrapped_like_tool_a"] = bool(b and "arguments" not in b and b == a)
    row["reproduced"] = row["tool_b_unwrapped_like_tool_a"]
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
