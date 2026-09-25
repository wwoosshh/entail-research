"""M3.5, rolebench 17 through the real engine: SGLang's torch_native attention with a cap that binds.

  prepare <dir>  a model folder that shares Gemma 2 2B's weights with attn_logit_softcapping 5.0 (as case.py does),
                 and the 256 token ids the case scores
  compare <dir>  prompt log-probs: torch_native without entail (the defect), triton without entail (the backend that
                 honours the cap), torch_native with entail on (resolved to triton, if the adapter works)
Run in ~/venvs/gpu; the SGLang runs themselves are in m3_engines.sh.
"""
import json
import os
import sys

SRC = os.path.expanduser("~/models/gemma-2-2b-it")
CAP = 5.0


def prepare(out):
    from transformers import AutoTokenizer

    model = os.path.join(out, "model")
    os.makedirs(model, exist_ok=True)
    for f in os.listdir(SRC):
        dst = os.path.join(model, f)
        if f != "config.json" and not f.startswith(".") and not os.path.lexists(dst):
            os.symlink(os.path.join(SRC, f), dst)
    with open(os.path.join(SRC, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["attn_logit_softcapping"] = CAP
    with open(os.path.join(model, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    with open("/usr/share/common-licenses/GPL-3", encoding="utf-8") as f:
        ids = AutoTokenizer.from_pretrained(SRC)(f.read()).input_ids[:256]
    with open(os.path.join(out, "ids.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)


def compare(out):
    def lp(name):
        with open(os.path.join(out, f"lp_{name}.json"), encoding="utf-8") as f:
            return json.load(f)["logprobs"]

    def dist(a, b):
        d = [abs(x - y) for x, y in zip(a, b)]
        return {"mean_abs": sum(d) / len(d), "max_abs": max(d), "n": len(d)}

    base = lp("triton_off")
    res = {"defect_vs_triton": dist(lp("torch_native_off"), base),
           "entail_torch_native_vs_triton": dist(lp("torch_native_load"), base)}
    rec = os.path.join(out, "record.jsonl")
    res["record"] = [json.loads(x) for x in open(rec, encoding="utf-8")] if os.path.exists(rec) else []
    res["decisions"] = [(r["boundary"], r["verdict"], r.get("target")) for r in res["record"] if "verdict" in r]
    with open(os.path.join(out, "compare.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "record"}, ensure_ascii=False))


if __name__ == "__main__":
    {"prepare": prepare, "compare": compare}[sys.argv[1]](sys.argv[2])
