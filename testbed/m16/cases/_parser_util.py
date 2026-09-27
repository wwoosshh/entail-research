"""Shared helpers for the parser-level vLLM cases (testbed/M16_PROTOCOL.md 5): a request with tools, a stand-in
tokenizer with a model's control tokens added, and a token-by-token streaming drive of a vLLM parser."""
import os


def chat_request(tools=None, chat_template_kwargs=None):
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    except ImportError:  # older layouts
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    kw = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    if tools:
        kw["tools"] = tools
    if chat_template_kwargs is not None:
        kw["chat_template_kwargs"] = chat_template_kwargs
    return ChatCompletionRequest(**kw)


def tool(name, props):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {k: {"type": t} for k, t in props.items()}}}}


def standin_tokenizer(extra_tokens, path=None):
    """Qwen3-4B's tokenizer (local, no remote code) with a model's control tokens added as special tokens, so that the
    parser's vocabulary lookups and the token-by-token feed see them as single tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path or os.path.expanduser("~/models/Qwen3-4B"))
    tok.add_tokens(list(extra_tokens), special_tokens=True)
    return tok


def drive_streaming(parser, tok, text, request):
    """Feed text token by token through extract_tool_calls_streaming; return (args by index, names by index,
    content) as a client would accumulate them."""
    ids = tok.encode(text, add_special_tokens=False)
    args, names, content = {}, {}, []

    def take(d):
        if d is None:
            return
        if getattr(d, "content", None):
            content.append(d.content)
        for tc in getattr(d, "tool_calls", None) or []:
            i = tc.index if getattr(tc, "index", None) is not None else 0
            fn = getattr(tc, "function", None)
            if fn is not None and getattr(fn, "name", None):
                names[i] = fn.name
            if fn is not None and getattr(fn, "arguments", None):
                args[i] = args.get(i, "") + fn.arguments

    prev_text, prev_ids = "", []
    for tid in ids:
        piece = tok.decode([tid], skip_special_tokens=False)
        cur = prev_text + piece
        take(parser.extract_tool_calls_streaming(prev_text, cur, piece, prev_ids, prev_ids + [tid], [tid], request))
        prev_text, prev_ids = cur, prev_ids + [tid]
    fin = getattr(parser, "finish_streaming", None)
    if fin is not None:
        take(fin())
    return args, names, "".join(content)
