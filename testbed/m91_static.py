"""M9.1: S5 (adapter thickness) and S6 (declaration burden) on the final code, counted from the source and from the
results files, not judged.

S5  every adapter file: its code lines (tests/test_adapter_rules.code_lines) and the rule logic the static check finds
    in it (violations: must be none); the files by kind - v2 adapters, the engine-specific repair, research tools
    still inside the package, shared plumbing - and per engine, what a new engine of that kind took.
S6  what a user writes for entail to know a meaning:
      LLM model folders            nothing: they declare it (testbed/results/m2_sources_baseline.json)
      image checkpoints            a manifest per checkpoint that does not declare, from `entail infer`, the facts
                                   the person fills in (testbed/results/m63/manifests)
      code boundaries (new code)   the signatures of rolebench's cases (testbed/results/m43/SUMMARY.json), against
                                   the old instrumented versions
      the front end (new code)     the lines of Qwen3's types and decode program (entail/frontend/qwen3.py)
Writes testbed/results/m91/static.json. Run: python testbed/m91_static.py
"""
import ast
import importlib.util
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENTAIL = os.path.join(ROOT, "entail")
OUT = os.path.join(HERE, "results", "m91", "static.json")

ENGINES = {   # the adapters each engine takes (entail/adapters/autoinstall/sitecustomize.py TARGETS)
    "transformers": ["transformers_adapter", "transformers_config", "rope_alias", "cache_contract",
                     "transformers_template"],
    "vllm": ["vllm_attention", "vllm_loader", "vllm_source", "vllm_layout", "vllm_cache_contract", "vllm_serve"],
    "sglang": ["sglang_adapter", "sglang_cache_contract", "sglang_serve"],
    "diffusers": ["diffusers_adapter"],
    "comfyui": ["comfyui"],
}
RESEARCH_TOOLS = ("vllm_seed", "vllm_ledger", "sglang_seed", "sglang_cache_probe")


def load_rules():
    spec = importlib.util.spec_from_file_location("adapter_rules", os.path.join(ENTAIL, "tests",
                                                                                 "test_adapter_rules.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def function_lines(path, name):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                     and isinstance(node.body[0].value, ast.Constant)) else node.body
            return body[-1].end_lineno - body[0].lineno + 1
    return None


def main():
    rules = load_rules()
    folder = os.path.join(ENTAIL, "entail", "adapters")
    files = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".py") or name[:-3] in rules.NOT_ADAPTERS:
            continue
        stem, path = name[:-3], os.path.join(folder, name)
        kind = ("research tool in the package" if stem in RESEARCH_TOOLS else
                "engine-specific repair" if stem in rules.ENGINE_SPECIFIC else
                "not on v2" if stem in rules.LEGACY else "v2 adapter")
        files[stem] = {"kind": kind, "code_lines": rules.code_lines(path),
                       "rule_logic_found": [str(v) for v in rules.violations(path)] if kind == "v2 adapter" else None}
    per_engine = {e: {"files": names, "code_lines": sum(files[n]["code_lines"] for n in names if n in files)}
                  for e, names in ENGINES.items()}
    s5 = {"files": files, "per_engine": per_engine,
          "v2 adapters with rule logic": [n for n, f in files.items() if f["rule_logic_found"]],
          "research tools still in the package": [n for n, f in files.items()
                                                  if f["kind"] == "research tool in the package"]}

    m2 = json.load(open(os.path.join(HERE, "results", "m2_sources_baseline.json"), encoding="utf-8"))
    m43 = json.load(open(os.path.join(HERE, "results", "m43", "SUMMARY.json"), encoding="utf-8"))["cases"]
    manifests = []
    mdir = os.path.join(HERE, "results", "m63", "manifests")
    for name in sorted(os.listdir(mdir)):
        if name.endswith(".json"):
            m = json.load(open(os.path.join(mdir, name), encoding="utf-8"))
            manifests.append({"file": m.get("file"), "facts_filled": [f["name"] for f in m.get("facts", [])
                                                                      if f.get("value") is not None],
                              "fields_filled": sum(len([v for v in (f.get("value") or {}).values() if v is not None])
                                                   for f in m.get("facts", []))})
    qwen3 = os.path.join(ENTAIL, "entail", "frontend", "qwen3.py")
    s6 = {"llm_model_folders": {"user_lines": 0, "baseline": {k: m2[k] for k in m2 if k != "files"}
                                if isinstance(m2, dict) else None},
          "image_checkpoints": {"manifests": manifests,
                                "why": "a checkpoint that declares nothing needs one (M2: 2 of 17 image checkpoints "
                                       "declared their prediction type, 0 of 17 their VAE scale)"},
          "code_boundaries_rolebench": {case: {"signed": c.get("burden"), "old_instrumented": c.get("old_instrumented")}
                                        for case, c in m43.items()},
          "front_end_qwen3": {"types_lines": function_lines(qwen3, "types"),
                              "decode_program_lines": function_lines(qwen3, "decode")}}
    sig = [c["burden"] for c in m43.values() if c.get("burden")]
    old = [c["old_instrumented"] for c in m43.values() if c.get("old_instrumented")]
    s6["code_boundaries_total"] = {"cases": len(sig), "signature_lines": sum(b["signature_lines"] for b in sig),
                                   "hand_tags": sum(b["hand_tags"] for b in sig),
                                   "old_signature_lines": sum(b["signature_lines"] for b in old),
                                   "old_hand_tags": sum(b["hand_tags"] for b in old)}
    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "S5": s5, "S6": s6}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({"per_engine": per_engine, "rule logic": s5["v2 adapters with rule logic"],
                      "research tools": s5["research tools still in the package"],
                      "code boundaries": s6["code_boundaries_total"], "front end": s6["front_end_qwen3"],
                      "manifests": manifests}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
