"""M10 E3, sgl-project/sglang#33493 (testbed/M10_PROTOCOL.md 3.3): DFLASH/DSPARK verification reads the penalties from a
misspelt attribute ("acc_linear_penalities"), so min_new_tokens (and other linear penalties) are ignored under
speculative decoding: the report asked for min_tokens=50 and got 18. The report's request on SGLang 0.5.20's offline
engine with a pair that fits 12 GB - openbmb/MiniCPM5-2B with its DSpark drafter openbmb/MiniCPM5-2B-DSpark (the
report: DeepSeek-V4-Flash, TP4) - with and without speculative decoding, each in its own process.
Reproduced: with DSPARK the output is shorter than min_new_tokens; without it, it is not.
Run in ~/venvs/sglang: python testbed/m10_e3/sg33493.py <out.json> <target dir> <drafter dir>
"""
import json
import os
import subprocess
import sys

PROMPT = [{"role": "user", "content": "Reply with exactly one short sentence."}]
SP = {"max_new_tokens": 80, "min_new_tokens": 50, "temperature": 0}


def worker(mode, target, drafter, out):
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target)
    text = tok.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True)
    kw = {"speculative_algorithm": "DSPARK", "speculative_draft_model_path": drafter} if mode == "spec" else {}
    engine = sgl.Engine(model_path=target, mem_fraction_static=0.75, context_length=2048, log_level="error",
                        disable_radix_cache=True, random_seed=0, **kw)
    try:
        o = engine.generate(text, SP)
    finally:
        engine.shutdown()
    json.dump({"text": o["text"], "completion_tokens": o["meta_info"].get("completion_tokens"),
               "finish_reason": str(o["meta_info"].get("finish_reason"))}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def main():
    if sys.argv[1] == "--worker":
        return worker(*sys.argv[2:6])
    out, target, drafter = sys.argv[1], sys.argv[2], sys.argv[3]
    parts = {}
    for mode in ("plain", "spec"):
        p = f"{out}.{mode}.json"
        r = subprocess.run([sys.executable, __file__, "--worker", mode, target, drafter, p], capture_output=True,
                           text=True)
        parts[mode] = json.load(open(p, encoding="utf-8")) if r.returncode == 0 and os.path.exists(p) else \
            {"error": (r.stderr or "")[-2500:]}
    ok = all("error" not in parts[m] for m in parts)
    row = {"entail": os.environ.get("ENTAIL", "off"), **parts,
           "reproduced": bool(ok and (parts["spec"]["completion_tokens"] or 0) < SP["min_new_tokens"]
                              <= (parts["plain"]["completion_tokens"] or 0))}
    json.dump(row, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps({"reproduced": row["reproduced"],
                                "plain_tokens": parts["plain"].get("completion_tokens"),
                                "spec_tokens": parts["spec"].get("completion_tokens"),
                                "errors": {m: parts[m]["error"][-400:] for m in parts if "error" in parts[m]}},
                               ensure_ascii=False))


if __name__ == "__main__":
    main()
