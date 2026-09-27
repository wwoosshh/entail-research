"""M18.6 replay 3 case, vllm-project/vllm#57353 (testbed/M16_PROTOCOL.md 8): with thinking on, Kimi K3's generation
prefix consumes the think-open marker, so a truncated output carries no think marker at all; the non-streaming
fallback returns it as content (reasoning None) while the streaming path returns it as reasoning. Driven as the
server drives the parser (M18.3's parse_retro): the class ParserManager.get_parser builds for kimi_k3, a stand-in
tokenizer (Qwen3-4B's with the K3 markers added as special tokens), parse_delta per token with finished on the last,
and parse on the whole text. Reproduced when the two paths classify the same text differently.
Run in ~/venvs/vllm (0.30.0): python testbed/m18/replay3/cases/vl57353.py <out.json>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "m16", "cases"))

MARKERS = ["<|open|>", "<|close|>", "<|sep|>"]
TEXT = ("The user asks for the capital of France. Paris has been the capital since the tenth century, and I should "
        "also mention that the")                     # cut by max_tokens inside the think channel: no marker at all


def main():
    import vllm
    from vllm.parser.parser_manager import ParserManager

    from _parser_util import chat_request, standin_tokenizer

    kw = {"thinking": True}
    cls = ParserManager.get_parser(tool_parser_name="kimi_k3", reasoning_parser_name="kimi_k3",
                                   enable_auto_tools=True, model_name="m")
    tok = standin_tokenizer(MARKERS)
    req = chat_request(chat_template_kwargs=kw)
    p = cls(tok, req.tools, chat_template_kwargs=kw, model_config=None)
    ids = tok.encode(TEXT, add_special_tokens=False)
    reasoning, content = [], []
    for i, tid in enumerate(ids):
        d = p.parse_delta(tok.decode([tid], skip_special_tokens=False), [tid], req, [], finished=(i == len(ids) - 1))
        if d is not None:
            reasoning.append(getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None) or "")
            content.append(d.content or "")
    whole = cls(tok, req.tools, chat_template_kwargs=kw, model_config=None).parse(TEXT, req, enable_auto_tools=True)
    stream = {"reasoning": "".join(reasoning), "content": "".join(content)}
    full = {"reasoning": whole[0], "content": whole[1]}
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "class": str(cls),
           "wrapped": hasattr(cls, "_entail_auto_tools"), "streaming": stream, "non_streaming": full}
    row["reproduced"] = bool(stream["reasoning"]) and not stream["content"] and bool(full["content"]) \
        and not full["reasoning"]
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
