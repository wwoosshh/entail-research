"""M18.6 replay 3 case, huggingface/transformers#45491 (testbed/M16_PROTOCOL.md 8): EmbeddingGemma returns an
all-NaN embedding on GPU for a short text batched with longer ones, when padding fills whole sliding-window
(512) attention windows. The report's shape: a 142-token text padded to 728 in a batch with longer texts; the same
text alone is finite. Here with transformers directly (mean pooling over the attention mask, as the model's
sentence-transformers head does before its dense layers), sdpa on CUDA. Reproduced when the short text's
embedding holds NaN in the batch and not alone.
The model is gated on the Hub (license acceptance by an account); when it cannot be fetched the case says so.
Run in ~/venvs/gpu (transformers 5.17.0): python testbed/m18/replay3/cases/tf45491.py <out.json>
"""
import json
import os
import sys

MODEL = "google/embeddinggemma-300m"


def main():
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__}
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
    except Exception as e:  # noqa: BLE001 - gated or unavailable: the case cannot run
        row.update(error=f"{type(e).__name__}: {str(e)[:300]}", reproduced=None)
        json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
        print("RESULT", json.dumps(row))
        return
    words = "the quick brown fox jumps over the lazy dog while the band plays on and on ".split()

    def text(n):
        return " ".join(words[i % len(words)] for i in range(n))

    texts = [text(130), text(440), text(540), text(700)]

    def embed(batch):
        enc = tok(batch, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m_ = enc["attention_mask"].unsqueeze(-1).float()
        return (h * m_).sum(1) / m_.sum(1), enc["attention_mask"].sum(1).tolist()

    batch, lens = embed(texts)
    alone, _ = embed(texts[:1])
    row.update(lengths=lens, nan_in_batch=int(torch.isnan(batch[0]).sum()), nan_alone=int(torch.isnan(alone[0]).sum()))
    row["reproduced"] = row["nan_in_batch"] > 0 and row["nan_alone"] == 0
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
