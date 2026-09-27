"""M17.6 case, huggingface/transformers#45356 (testbed/M16_PROTOCOL.md 7): the Kimi-K2.5 tokenizer (remote code)
decodes the </think> id 163607 correctly on transformers 5.3.0 and wrongly from 5.4.0, where a warning suggests
fix_mistral_regex=True, which then raises AttributeError. The report's three lines: decode([163607]), the
encode/decode round trip of "</think>", and the suggested flag. Reproduced when decode([163607]) is not "</think>"
or the round trip does not give the id back.
Run: python testbed/m17/replay2/cases/tf45356.py <out.json>   (tf540 for the reported version; gpu for 5.17.0)
"""
import json
import os
import sys
import traceback

MODEL = "moonshotai/Kimi-K2.5"
THINK_END = 163607


def main():
    import transformers
    from transformers import AutoTokenizer

    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__, "model": MODEL}
    try:
        t = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        row["tokenizer_class"] = type(t).__name__
        row["decode_163607"] = t.decode([THINK_END])
        ids = t.encode("</think>", add_special_tokens=False)
        row["encode_think_end"] = ids
        row["round_trip_ok"] = ids == [THINK_END]
    except Exception as e:  # noqa: BLE001
        row["load_error"] = f"{type(e).__name__}: {e}"[:300]
    try:
        t2 = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, fix_mistral_regex=True)
        row["fix_mistral_regex"] = "loaded: " + t2.decode([THINK_END])
    except Exception as e:  # noqa: BLE001
        row["fix_mistral_regex"] = f"{type(e).__name__}: {e}"[:200]
        row["fix_mistral_regex_trace"] = traceback.format_exc().strip().splitlines()[-1][:200]
    row["reproduced"] = bool(row.get("decode_163607") != "</think>" or not row.get("round_trip_ok", False))
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
