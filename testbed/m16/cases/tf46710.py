"""M16 case, huggingface/transformers#46710 (testbed/M16_PROTOCOL.md 5): DeepSeek-R1-Distill-Llama-8B declares
LlamaTokenizerFast; transformers 5.9 maps it to another tokenizer class and the decoded text carries byte-level
artefacts (Ġ, Ċ). Tokenizer-level reproduction (the model needs 16 GB in fp16): the checkpoint's tokenizer through
AutoTokenizer against the tokenizer.json read directly by the tokenizers library, on the report's chat text and on
the report's own reply text. Fixed by PR #46091 (merged 2026-06-19, in 5.13.0), so this runs in the transformers
5.12.1 venv.
Run in ~/venvs/tf5121: python testbed/m16/cases/tf46710.py <out.json>
"""
import json
import os
import sys

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
REPLY = "\nI need to determine the sum of 1 and 1.\n\nAdding 1 and 1 gives a total of 2.\n</think>\n\n**Solution:**"


def main():
    import transformers
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ref = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
    messages = [{"role": "user", "content": "What is 1+1? Answer briefly."}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    ref_ids = ref.encode(text, add_special_tokens=False).ids
    reply_ids = ref.encode(REPLY, add_special_tokens=False).ids
    decoded = tok.decode(reply_ids, skip_special_tokens=True)
    ref_decoded = ref.decode(reply_ids, skip_special_tokens=True)
    artefacts = any(ch in decoded for ch in ("Ġ", "Ċ"))
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__,
           "tokenizer_class": type(tok).__name__, "prompt_ids_match_tokenizer_json": ids == ref_ids,
           "prompt_ids_head": ids[:12], "reference_ids_head": ref_ids[:12], "decoded_reply": decoded[:160],
           "reference_decoded_reply": ref_decoded[:160], "byte_level_artefacts_in_decode": artefacts,
           "reproduced": artefacts or decoded != ref_decoded or ids != ref_ids}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
