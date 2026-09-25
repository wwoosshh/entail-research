"""M10 E1, L1 (testbed/M10_PROTOCOL.md 1.2 and 6): fd-rope's cause on every fetched model, at config level with vLLM
0.30's own ModelConfig - no GPU, no weights.
For each model folder: the config vLLM's ModelConfig builds untouched, and with hf_overrides={"rope_scaling": S},
where S is the model's own RoPE scaling when it declares one (a restatement: nothing should change), else the YaRN
value model cards give for long context. The effective RoPE base is rope_parameters["rope_theta"], else the default
the architecture's vLLM model file fills in (set_default_rope_theta), else get_rope's fallback 10000.
Exposed: the two bases differ. Only ModelConfig is used (get_config alone leaves rope_theta=None where the engine
has no key - protocol 6).
Run in ~/venvs/vllm, entail off:  python testbed/m10_e1_rope.py off
          entail on (library hook): ENTAIL=load PYTHONPATH=<shim> python testbed/m10_e1_rope.py on
Writes testbed/results/m10/e1_llm/rope_<off|on>.json. The local models in ~/models are calibration rows.
"""
import glob
import importlib.util
import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "m10", "e1_llm")
FALLBACK = 10000.0
DEFAULT_RE = re.compile(r"set_default_rope_theta\(\s*[\w.]+\s*,\s*default_theta\s*=\s*([0-9.eE+_]+)\s*\)")


def main():
    tag = sys.argv[1]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import vllm
    from vllm.config import ModelConfig
    from vllm.model_executor.models.registry import _VLLM_MODELS

    src_cache = {}

    def source(mod):
        name = mod if "." in mod else f"vllm.model_executor.models.{mod}"
        if name not in src_cache:
            try:
                spec = importlib.util.find_spec(name)
                path = spec.origin if spec else None
            except Exception:  # noqa: BLE001
                path = None
            src_cache[name] = open(path, encoding="utf-8").read() if path and os.path.isfile(path) else ""
        return src_cache[name]

    def closure(mod):
        seen, stack = [], [mod]
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.append(m)
            src = source(m)
            stack += re.findall(r"from vllm\.model_executor\.models\.(\w+) import", src)
            stack += re.findall(r"^from \.(\w+) import", src, re.M)
        return seen

    def arch_info(arch):
        mod = _VLLM_MODELS.get(arch)
        if not mod:
            return {"registered": False}
        own = source(mod[0])
        mods = closure(mod[0])
        m = DEFAULT_RE.search(own)
        imported = [(x, DEFAULT_RE.search(source(x))) for x in mods[1:]]
        imported = [(x, float(g.group(1).replace("_", ""))) for x, g in imported if g]
        return {"registered": True, "module": mod[0], "modules_followed": len(mods),
                "uses_rope": any(("get_rope(" in source(x) or "RotaryEmbedding(" in source(x)) for x in mods),
                "file_default": float(m.group(1).replace("_", "")) if m else None,
                "imported_defaults": imported}

    def bases(mc, default):
        rp = getattr(mc.hf_text_config, "rope_parameters", None)
        fill = default if default is not None else FALLBACK
        if not isinstance(rp, dict) or not rp:
            return {"-": fill}, rp
        if all(isinstance(v, dict) for v in rp.values()):
            return {k: (float(v["rope_theta"]) if v.get("rope_theta") is not None else fill)
                    for k, v in rp.items()}, rp
        return {"-": float(rp["rope_theta"]) if rp.get("rope_theta") is not None else fill}, rp

    def override_for(raw):
        tc = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
        own = raw.get("rope_scaling") or tc.get("rope_scaling")
        if isinstance(own, dict) and own:
            return "own rope_scaling (restated)", own
        rp = raw.get("rope_parameters") or tc.get("rope_parameters")
        if isinstance(rp, dict) and rp.get("rope_type") not in (None, "default") and not any(
                isinstance(v, dict) for v in rp.values()):
            return "own rope_parameters without rope_theta (restated)", {k: v for k, v in rp.items()
                                                                          if k != "rope_theta"}
        mpe = raw.get("max_position_embeddings") or tc.get("max_position_embeddings") or 32768
        return "YaRN per model cards", {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": mpe}

    folders = []
    models = json.load(open(os.path.join(OUT, "models.json"), encoding="utf-8"))["models"]
    for m in models:
        f = os.path.join(OUT, "configs", m["id"].replace("/", "__"))
        if os.path.isfile(os.path.join(f, "config.json")):
            folders.append((m["rank"], m["id"], f))
    for f in sorted(glob.glob(os.path.expanduser("~/models/*"))):
        if os.path.isfile(os.path.join(f, "config.json")):
            folders.append((0, "local/" + os.path.basename(f), f))

    rows = []
    for rank, mid, folder in folders:
        raw = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
        arch = (raw.get("architectures") or [None])[0]
        row = {"rank": rank, "id": mid, "arch": arch, "model_type": raw.get("model_type")}
        info = arch_info(arch) if arch else {"registered": False}
        row.update(info)
        if not info.get("registered"):
            row["result"] = "not registered in vLLM 0.30"
            rows.append(row)
            continue
        if not info.get("uses_rope"):
            row["result"] = "no RoPE in vLLM's model files"
            rows.append(row)
            continue
        why, s = override_for(raw)
        row["override_kind"], row["override"] = why, s
        try:
            m0 = ModelConfig(model=folder, tokenizer=folder, trust_remote_code=False)
            b0, rp0 = bases(m0, info["file_default"])
        except Exception as e:  # noqa: BLE001 - the untouched config does not build: not this defect
            row["result"] = "untouched config does not build"
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rows.append(row)
            continue
        try:
            m1 = ModelConfig(model=folder, tokenizer=folder, trust_remote_code=False, hf_overrides={"rope_scaling": s})
            b1, rp1 = bases(m1, info["file_default"])
        except Exception as e:  # noqa: BLE001 - an error at config build is loud, not silent
            row["result"] = "override fails loudly at config build"
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            row["trace"] = traceback.format_exc()[-600:]
            rows.append(row)
            continue
        row["base_untouched"], row["base_override"] = b0, b1
        row["rope_parameters_untouched"] = json.loads(json.dumps(rp0, default=str))
        row["rope_parameters_override"] = json.loads(json.dumps(rp1, default=str))
        if set(b0) != set(b1):
            row["result"] = "structure changed"
        else:
            row["result"] = "exposed" if any(b0[k] != b1[k] for k in b0) else "not exposed"
        rows.append(row)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "entail": os.environ.get("ENTAIL", "off"),
           "vllm": vllm.__version__, "rows": rows}
    with open(os.path.join(OUT, f"rope_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print("DONE", tag, Counter(r["result"] for r in rows))


if __name__ == "__main__":
    main()
