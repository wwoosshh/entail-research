"""Which side is right on the E2 Tokenization alarms: transformers 5.17's rebuilt LlamaTokenizer, the folder's
tokenizer.json, or the model's own sentencepiece file (tokenizer.model)? Research check, not library code."""
import json
import os
import sys

from tokenizers import Tokenizer
from transformers import AutoTokenizer

PROBES = ["   leading spaces and trailing   ", "Hello world", " one leading space", "\tTab start", "\n\nnewlines first",
          "  two leading", "Numbers 12345 and symbols !?", "<s> after a declared token", "under_score and ▁ marker"]
for folder in sys.argv[1:]:
    print("=====", folder)
    files = sorted(os.listdir(folder))
    print("files:", [f for f in files if not f.startswith(".")])
    cfg = json.load(open(os.path.join(folder, "tokenizer_config.json")))
    print("tokenizer_config:", {k: cfg.get(k) for k in ("tokenizer_class", "legacy", "add_prefix_space",
                                                          "add_bos_token", "use_default_system_prompt")})
    tj = json.load(open(os.path.join(folder, "tokenizer.json")))
    print("tokenizer.json normalizer:", json.dumps(tj.get("normalizer"))[:300])
    print("tokenizer.json pre_tokenizer:", json.dumps(tj.get("pre_tokenizer"))[:300])
    hf = AutoTokenizer.from_pretrained(folder)
    print("transformers class:", type(hf).__name__, "| backend pre_tokenizer:",
          str(getattr(getattr(hf, "_tokenizer", None), "pre_tokenizer", None))[:200],
          "| normalizer:", str(getattr(getattr(hf, "_tokenizer", None), "normalizer", None))[:200])
    raw = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    sp = None
    if os.path.isfile(os.path.join(folder, "tokenizer.model")):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=os.path.join(folder, "tokenizer.model"))
        except Exception as e:  # noqa: BLE001
            print("sentencepiece unavailable:", type(e).__name__, e)
    for t in PROBES:
        a = hf.encode(t, add_special_tokens=False)
        b = raw.encode(t, add_special_tokens=False).ids
        c = sp.encode(t) if sp else None
        mark = "" if a == b and (c is None or a == c) else "   <-- differs"
        print(f"{t!r:40} tf={a[:8]} json={b[:8]} sp={c[:8] if c else None}{mark}")
