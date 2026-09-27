from unittest.mock import MagicMock
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam, FunctionDefinition

def make_tool(name, properties):
    return ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(name=name, parameters={"type": "object", "properties": properties}),
    )

# tool_a's schema permits unwrapping "city"; tool_b's schema does not (only has "tz")
tools = [
    make_tool("tool_a", {"city": {"type": "string"}}),
    make_tool("tool_b", {"tz": {"type": "string"}}),
]

tok = MagicMock()
tok.get_vocab.return_value = {}
tok.decode.side_effect = lambda ids: f"tok{ids[0]}"

engine = DeepSeekV4Parser(tok, tools=tools)
req = MagicMock()
req.tools = tools
req.tool_choice = "auto"

# Both invokes wrap identical raw args at the same point in their streams
text = (
    '<｜DSML｜tool_calls>'
    '<｜DSML｜invoke name="tool_a">'
    '<｜DSML｜parameter name="arguments" string="false">{"city": "Tokyo"}</｜DSML｜parameter>'
    '</｜DSML｜invoke>'
    '<｜DSML｜invoke name="tool_b">'
    '<｜DSML｜parameter name="arguments" string="false">{"city": "Tokyo"}</｜DSML｜parameter>'
    '</｜DSML｜invoke>'
    '</｜DSML｜tool_calls>'
)

result = engine.extract_tool_calls(text, req)
for tc in result.tool_calls:
    print(f"{tc.function.name} -> {tc.function.arguments}")
