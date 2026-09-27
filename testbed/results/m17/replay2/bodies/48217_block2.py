from types import SimpleNamespace

from vllm.parser.gemma4 import Gemma4Parser


class FakeTokenizer:
    bos_token_id = None
    eos_token_id = None
    pad_token_id = None

    def __init__(self):
        self._vocab = {
            "<|channel>": 100,
            "<channel|>": 101,
            "<|tool_call>": 102,
            "<tool_call|>": 103,
            "<|turn>": 104,
            "<|tool_response>": 105,
        }
        self._inverse_vocab = {
            token_id: token for token, token_id in self._vocab.items()
        }

    def get_vocab(self):
        return self._vocab

    def decode(self, token_ids):
        return "".join(
            self._inverse_vocab.get(token_id, f"<token:{token_id}>")
            for token_id in token_ids
        )


def dump_delta(delta):
    if delta is None:
        return None

    if hasattr(delta, "model_dump"):
        return delta.model_dump(exclude_none=True)

    return {
        name: value
        for name in ("reasoning", "content", "tool_calls")
        if (value := getattr(delta, name, None)) is not None
    }


tokenizer = FakeTokenizer()
request = SimpleNamespace(
    tools=None,
    tool_choice=None,
)

plain_answer = "This is a direct final answer without channel markers."

print("=== Non-streaming ===")

nonstream_parser = Gemma4Parser(
    tokenizer,
    chat_template_kwargs={"enable_thinking": True},
)

reasoning, content = nonstream_parser.extract_reasoning(
    plain_answer,
    request,
)

print({
    "reasoning": reasoning,
    "content": content,
})


print("\n=== Streaming ===")

stream_parser = Gemma4Parser(
    tokenizer,
    chat_template_kwargs={"enable_thinking": True},
)

# Simulate a normal chat-template prompt ending at a new model turn.
prompt_token_ids = [
    tokenizer.get_vocab()["<|turn>"],
]

chunks = [
    "This is a",
    " direct final answer",
    " without channel markers.",
]

for index, chunk in enumerate(chunks):
    delta = stream_parser.parse_delta(
        delta_text=chunk,
        delta_token_ids=[],
        request=request,
        prompt_token_ids=prompt_token_ids if index == 0 else None,
        finished=False,
    )
    print(f"chunk {index}:", dump_delta(delta))

final_delta = stream_parser.parse_delta(
    delta_text="",
    delta_token_ids=[],
    request=request,
    prompt_token_ids=None,
    finished=True,
)
print("finished:", dump_delta(final_delta))
