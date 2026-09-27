"""M19 L4 replay 4 case, huggingface/transformers#41494 (testbed/M16_PROTOCOL.md 9): the tokenizer transformers builds
from a gemma-3 GGUF file (Unigram) tokenizes differently from the model's own (BPE). The report's script:
unsloth/gemma-3-4b-it-GGUF gemma-3-4b-it-Q8_0.gguf against the model's tokenizer; google/gemma-3-4b-it is gated, so
the tokenizer files of unsloth/gemma-3-4b-it (the same model) stand in. Reproduced when the tokens differ.
Run: python testbed/m19/replay4/cases/tf41494.py <out.json>  (a venv with gguf: ~/venvs/vllm 5.17.0, vllm0210 4.57.6)
"""
import json
import os
import sys

GGUF_DIR = os.path.expanduser("~/models/replay4/unsloth__gemma-3-4b-it-GGUF")
REF = os.path.expanduser("~/models/replay4/unsloth__gemma-3-4b-it")


def main():
    import transformers
    from transformers import AutoTokenizer

    t1 = AutoTokenizer.from_pretrained(GGUF_DIR, gguf_file="gemma-3-4b-it-Q8_0.gguf")
    t2 = AutoTokenizer.from_pretrained(REF)
    text = "<bos>What is eunoia?"
    x1, x2 = t1.tokenize(text), t2.tokenize(text)
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__,
           "gguf_tokens": x1, "model_tokens": x2, "gguf_class": type(t1).__name__, "model_class": type(t2).__name__,
           "gguf_model_type": type(getattr(getattr(t1, "_tokenizer", None), "model", None)).__name__,
           "model_model_type": type(getattr(getattr(t2, "_tokenizer", None), "model", None)).__name__}
    row["reproduced"] = x1 != x2
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
