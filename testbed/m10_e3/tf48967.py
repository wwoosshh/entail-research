"""M10 E3, huggingface/transformers#48967 (testbed/M10_PROTOCOL.md 3.3): a BERT model's tokenizer after transformers
5.0 - the report: the ids for "هذه جملة" were [2, 2413, 9200, 3] before 5.0 and are [2, 27966, 42, 1, 3] after.
Loads AMR-KELEG/Sentence-ALDi with transformers 5.17 (entail on or off from outside), tokenizes the sentence, runs
the regression head, and writes what happened.
The repository deleted its tokenizer.json on 2026-09-22 (commit 4b4f2a3b, "Delete tokenizer.json (#4)"), so the
reported condition is the revision before it: d8ce98a134cc.
Run in ~/venvs/gpu: python testbed/m10_e3/tf48967.py <out.json> [revision]
"""
import json
import os
import sys

MODEL = "VARabic/Sentence-ALDi"   # AMR-KELEG/Sentence-ALDi redirects here
SENTENCE = "هذه جملة"
BEFORE_5 = [2, 2413, 9200, 3]


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rev = sys.argv[2] if len(sys.argv) > 2 else None
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    ids = tok(SENTENCE)["input_ids"]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=rev).eval()
    with torch.no_grad():
        score_now = model(torch.tensor([ids])).logits.flatten().tolist()
        score_before = model(torch.tensor([BEFORE_5])).logits.flatten().tolist()
    row = {"entail": os.environ.get("ENTAIL", "off"), "revision": rev, "tokenizer_class": type(tok).__name__,
           "tokenizer_vocab": len(tok), "config_vocab_size": model.config.vocab_size, "ids": ids,
           "ids_before_5": BEFORE_5, "reproduced": ids != BEFORE_5, "score_with_these_ids": score_now,
           "score_with_the_ids_before_5": score_before}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
