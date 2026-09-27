"""M17.6 case, vllm-project/vllm#48217 (testbed/M16_PROTOCOL.md 7): with Gemma4 thinking enabled, the streaming
and non-streaming parser paths classify a plain answer without channel markers differently - non-streaming as
content, streaming as reasoning with no content ever emitted. The report's minimal reproduction against
Gemma4Parser with a stand-in tokenizer, unchanged. Reproduced when the non-streaming path returns the answer as
content while the streamed deltas carry no content and some reasoning.
Run in ~/venvs/vllm0240 (the reported 0.24.0): python testbed/m17/replay2/cases/vl48217.py <out.json>
"""
import json
import os
import sys
from types import SimpleNamespace


class FakeTokenizer:
    bos_token_id = None
    eos_token_id = None
    pad_token_id = None

    def __init__(self):
        self._vocab = {"<|channel>": 100, "<channel|>": 101, "<|tool_call>": 102, "<tool_call|>": 103,
                       "<|turn>": 104, "<|tool_response>": 105}
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}

    def get_vocab(self):
        return self._vocab

    def decode(self, token_ids):
        return "".join(self._inverse_vocab.get(t, f"<token:{t}>") for t in token_ids)


def dump_delta(delta):
    if delta is None:
        return None
    if hasattr(delta, "model_dump"):
        return delta.model_dump(exclude_none=True)
    return {n: v for n in ("reasoning", "content", "tool_calls") if (v := getattr(delta, n, None)) is not None}


def main():
    import vllm
    from vllm.parser.gemma4 import Gemma4Parser

    tokenizer = FakeTokenizer()
    request = SimpleNamespace(tools=None, tool_choice=None)
    plain_answer = "This is a direct final answer without channel markers."
    nonstream = Gemma4Parser(tokenizer, chat_template_kwargs={"enable_thinking": True})
    reasoning, content = nonstream.extract_reasoning(plain_answer, request)
    stream = Gemma4Parser(tokenizer, chat_template_kwargs={"enable_thinking": True})
    prompt_token_ids = [tokenizer.get_vocab()["<|turn>"]]
    chunks = ["This is a", " direct final answer", " without channel markers."]
    deltas = []
    for index, chunk in enumerate(chunks):
        deltas.append(dump_delta(stream.parse_delta(delta_text=chunk, delta_token_ids=[], request=request,
                                                    prompt_token_ids=prompt_token_ids if index == 0 else None,
                                                    finished=False)))
    deltas.append(dump_delta(stream.parse_delta(delta_text="", delta_token_ids=[], request=request,
                                                prompt_token_ids=None, finished=True)))
    s_content = "".join((d or {}).get("content") or "" for d in deltas)
    s_reasoning = "".join((d or {}).get("reasoning") or "" for d in deltas)
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__,
           "non_streaming": {"reasoning": reasoning, "content": content},
           "streaming": {"content": s_content, "reasoning": s_reasoning, "deltas": deltas}}
    row["reproduced"] = bool((content or "").strip() == plain_answer and not s_content.strip() and s_reasoning.strip())
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
