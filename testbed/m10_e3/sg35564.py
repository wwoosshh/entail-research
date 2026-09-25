"""M10 E3, sgl-project/sglang#35564 (testbed/M10_PROTOCOL.md 3.3): tool-call detectors lose or change calls when the
output is streamed one token (here one character) at a time. The issue's own reproduction (CPU), unchanged in what
it compares, run on SGLang 0.5.20 with entail off or on from outside.
Run in ~/venvs/sglang: python testbed/m10_e3/sg35564.py <out.json>
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    def tool(name):
        return Tool(type="function", function=Function(name=name, parameters={"type": "object", "properties": {}}))

    tools = [tool("get_weather"), tool("f")]
    cases = {
        "cohere_command4": '<|START_ACTION|>[{"tool_name": "f", "parameters": {}}]<|END_ACTION|>',
        "gemma4": "<|tool_call>call:f{}<tool_call|>",
        "glm": "<tool_call>get_weather\n<arg_key>get_weather</arg_key>\n<arg_value>123</arg_value>\n</tool_call>",
        "glm45": "<tool_call>get_weather\n<arg_key>get_weather</arg_key>\n<arg_value>123</arg_value>\n</tool_call>",
        "glm47": "<tool_call>get_weather<arg_key>get_weather</arg_key><arg_value>123</arg_value></tool_call>",
        "minimax-m2": '<minimax:tool_call><invoke name="get_weather"></invoke></minimax:tool_call>',
        "mistral": '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}, {"name": "get_weather", "arguments": {}}]',
        "step3": "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
                 '<steptml:invoke name="get_weather"></steptml:invoke>'
                 "<｜tool_call_end｜><｜tool_call_begin｜>function<｜tool_sep｜>"
                 '<steptml:invoke name="get_weather">'
                 '<steptml:parameter name="get_weather">hello</steptml:parameter>'
                 "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>",
    }

    def stream(name, chunks):
        d = FunctionCallParser.ToolCallParserEnum[name]()
        calls = {}
        for chunk in list(chunks) + ["", ""]:
            r = d.parse_streaming_increment(chunk, tools)
            for c in r.calls:
                e = calls.setdefault(c.tool_index, ["", ""])
                if c.name:
                    e[0] = c.name
                if c.parameters:
                    e[1] += c.parameters
        return calls

    def jeq(a, b):
        try:
            return json.loads(a or "null") == json.loads(b or "null")
        except Exception:  # noqa: BLE001
            return a == b

    res = {}
    for name, text in cases.items():
        try:
            d = FunctionCallParser.ToolCallParserEnum[name]()
            final = {i: [c.name, c.parameters] for i, c in enumerate(d.detect_and_parse(text, tools).calls)}
            streamed = stream(name, list(text))
            same = len(streamed) == len(final) and all(
                streamed.get(i, ["", ""])[0] == final[i][0] and jeq(streamed.get(i, ["", ""])[1], final[i][1])
                for i in final)
            res[name] = {"same": same, "final": {str(k): v for k, v in final.items()},
                         "streamed": {str(k): v for k, v in streamed.items()}}
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": f"{type(e).__name__}: {e}"}
    bad = [n for n, r in res.items() if not r.get("same")]
    row = {"entail": os.environ.get("ENTAIL", "off"), "parsers": res, "parsers_that_differ": bad,
           "reproduced": bool(bad)}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps({"reproduced": row["reproduced"], "parsers_that_differ": bad}))


if __name__ == "__main__":
    main()
