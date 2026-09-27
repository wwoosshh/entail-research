"""M18.6 replay 3 case, huggingface/transformers#49066 (testbed/M16_PROTOCOL.md 8): Qwen2Tokenizer rebuilds its
backend with a module-level pre-tokenizer regex and drops the one tokenizer.json declares; Qwen3.8's pattern adds
\\p{M} (combining marks), so Hindi text gets other ids. The report's script, unchanged (tokenizer files only):
reproduced when the loaded tokenizer's ids differ from the published tokenizer.json's.
Run in ~/venvs/gpu (transformers 5.17.0): python testbed/m18/replay3/cases/tf49066.py <out.json>
"""
import json
import os
import sys

REPO, TEXT = "Qwen/Qwen3.8-Flash-Next", "नमस्ते दुनिया"


def main():
    import transformers
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    published = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
    loaded = AutoTokenizer.from_pretrained(REPO)
    a = published.encode(TEXT).ids
    b = loaded(TEXT)["input_ids"]
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__,
           "class": type(loaded).__name__, "published_ids": a, "loaded_ids": b, "reproduced": a != b}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
