"""M10 E1, L3 (testbed/M10_PROTOCOL.md 1.2 and 6): for every (model, key) entail 1.0's static check reported as a
config key the config class does not take, whether the engines' own code for that architecture reads the key from
the config object (a code reading, mechanical):
  vLLM     the architecture's model file and the model modules it imports, and vLLM's config class for the type
  SGLang   the same, in SGLang's models package (its source on disk)
  transformers  models/<model_type>/*.py
A key that engine code reads by name is carried on the config object as an attribute (PretrainedConfig keeps keys
its class does not know), so the broken report is a false alarm for that engine. A key no engine code reads is
declared and unread - harmless or lost; this does not decide which.
Run in ~/venvs/vllm: python testbed/m10_e1_keys.py
Writes testbed/results/m10/e1_llm/coverage_keys_read.json.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
L1 = os.path.join(HERE, "results", "m10", "e1_llm")
SGL = "/home/<user>/venvs/sglang/lib/python3.12/site-packages/sglang"


def reads(src, key):
    k = re.escape(key)
    return bool(re.search(rf"(\bconfig|\bcfg|text_config|hf_config|\bself\.config)\s*\.\s*{k}\b", src) or
                re.search(rf"getattr\(\s*[\w.\[\]\"']+\s*,\s*[\"']{k}[\"']", src) or
                re.search(rf"[\"']{k}[\"']", src))


def main():
    import transformers
    import vllm
    from vllm.model_executor.models.registry import _VLLM_MODELS

    vdir, tdir = os.path.dirname(vllm.__file__), os.path.dirname(transformers.__file__)
    cache = {}

    def text(path):
        if path not in cache:
            cache[path] = open(path, encoding="utf-8", errors="replace").read() if os.path.isfile(path) else ""
        return cache[path]

    def closure(root, first):
        seen, stack = [], [first]
        while stack:
            p = stack.pop()
            if p in seen or not os.path.isfile(p):
                continue
            seen.append(p)
            src = text(p)
            for mod in re.findall(r"from (?:vllm|sglang\.srt)\.models\.(\w+) import", src) + \
                    re.findall(r"from vllm\.model_executor\.models\.(\w+) import", src) + \
                    re.findall(r"^from \.(\w+) import", src, re.M):
                stack.append(os.path.join(root, mod + ".py"))
        return seen

    sglang_models = os.path.join(SGL, "srt", "models")
    sgl_arch = {}
    for p in glob.glob(os.path.join(sglang_models, "*.py")):
        for arch in re.findall(r"EntryClass\s*=\s*\[?([\w,\s]+)\]?", text(p)):
            for a in re.split(r"[,\s]+", arch):
                if a:
                    sgl_arch.setdefault(a, p)
        for a in re.findall(r"^class (\w+ForCausalLM|\w+ForConditionalGeneration)\b", text(p), re.M):
            sgl_arch.setdefault(a, p)

    ck = json.load(open(os.path.join(L1, "coverage_keys.json"), encoding="utf-8"))["per_model"]
    rows, per_key = [], {}
    for mid, keys in ck.items():
        cfg = json.load(open(os.path.join(L1, "configs", mid.replace("/", "__"), "config.json"), encoding="utf-8"))
        arch = (cfg.get("architectures") or [None])[0]
        mtype = cfg.get("model_type")
        vfiles = []
        mod = _VLLM_MODELS.get(arch)
        if mod and "." not in mod[0]:
            vfiles = closure(os.path.join(vdir, "model_executor", "models"),
                             os.path.join(vdir, "model_executor", "models", mod[0] + ".py"))
        vcfg = os.path.join(vdir, "transformers_utils", "configs", f"{mtype}.py")
        if os.path.isfile(vcfg):
            vfiles.append(vcfg)
        sfiles = closure(sglang_models, sgl_arch[arch]) if arch in sgl_arch else []
        tfiles = glob.glob(os.path.join(tdir, "models", str(mtype), "*.py"))
        for key in keys:
            base = key.split(".")[-1]
            r = {"model": mid, "arch": arch, "key": key,
                 "vllm": (any(reads(text(p), base) for p in vfiles) if vfiles else None),
                 "sglang": (any(reads(text(p), base) for p in sfiles) if sfiles else None),
                 "transformers": (any(reads(text(p), base) for p in tfiles) if tfiles else None)}
            rows.append(r)
            k = per_key.setdefault(key, {"n": 0, "vllm_reads": 0, "sglang_reads": 0, "transformers_reads": 0,
                                         "no_engine_code_reads": 0})
            k["n"] += 1
            for e in ("vllm", "sglang", "transformers"):
                k[f"{e}_reads"] += bool(r[e])
            k["no_engine_code_reads"] += not any(r[e] for e in ("vllm", "sglang", "transformers"))
    models = {}
    for r in rows:
        m = models.setdefault(r["model"], {"keys": 0, "vllm_reads_all": True, "any_unread_everywhere": False})
        m["keys"] += 1
        m["vllm_reads_all"] &= bool(r["vllm"])
        m["any_unread_everywhere"] |= not any(r[e] for e in ("vllm", "sglang", "transformers"))
    out = {"rows": rows, "per_key": dict(sorted(per_key.items(), key=lambda x: -x[1]["n"])), "per_model": models,
           "totals": {"pairs": len(rows),
                      "pairs_vllm_reads": sum(bool(r["vllm"]) for r in rows),
                      "pairs_sglang_reads": sum(bool(r["sglang"]) for r in rows),
                      "pairs_transformers_reads": sum(bool(r["transformers"]) for r in rows),
                      "pairs_no_engine_code_reads": sum(not any(r[e] for e in ("vllm", "sglang", "transformers"))
                                                        for r in rows),
                      "models": len(models),
                      "models_all_keys_read_by_vllm": sum(m["vllm_reads_all"] for m in models.values()),
                      "models_with_a_key_no_engine_reads": sum(m["any_unread_everywhere"] for m in models.values())}}
    with open(os.path.join(L1, "coverage_keys_read.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["totals"], indent=1))
    for k, v in list(out["per_key"].items())[:25]:
        print(f"{v['n']:3} {k:34} vllm {v['vllm_reads']:3} sglang {v['sglang_reads']:3} "
              f"transformers {v['transformers_reads']:3} none {v['no_engine_code_reads']:3}")


if __name__ == "__main__":
    main()
