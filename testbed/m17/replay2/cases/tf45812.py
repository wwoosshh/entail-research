"""M17.6 case, huggingface/transformers#45812 (testbed/M16_PROTOCOL.md 7): AutoTokenizer produces wrong token ids
for the Granite models on transformers 5.0-5.8 (the tokenizer.json pre-tokenizer is lost when the model is routed
away from TokenizersBackend). AutoTokenizer against the checkpoint's tokenizer.json read directly by the tokenizers
library, on a few strings. Reproduced when any string's ids differ.
Run: python testbed/m17/replay2/cases/tf45812.py <out.json> [hub id or folder]
"""
import json
import os
import sys

STRINGS = ["How are you doing?", "def fib(n):\n    if n < 2:\n        return n", "Hello, world! 1234",
           "   leading spaces", "The quick brown fox jumps over the lazy dog."]


def main():
    import transformers
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    model = sys.argv[2] if len(sys.argv) > 2 else "ibm-granite/granite-4.1-8b"
    tok = AutoTokenizer.from_pretrained(model)
    path = os.path.join(model, "tokenizer.json") if os.path.isdir(model) else hf_hub_download(model, "tokenizer.json")
    raw = Tokenizer.from_file(path)
    rows = []
    for s in STRINGS:
        ids = tok.encode(s, add_special_tokens=False)
        ref = raw.encode(s, add_special_tokens=False).ids
        rows.append({"text": s, "auto": ids, "tokenizer_json": ref, "same": ids == ref})
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__, "model": model,
           "tokenizer_class": type(tok).__name__, "strings": rows,
           "mismatches": sum(1 for r in rows if not r["same"])}
    row["reproduced"] = row["mismatches"] > 0
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
