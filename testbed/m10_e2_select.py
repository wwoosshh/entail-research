"""M10 E2 (testbed/M10_PROTOCOL.md 2.1): the models the healthy runs use, chosen mechanically from E1's list in
download order - the first 20 that pass every rule - plus the local models. Every model passed over before the 20th
is written with its reason.
Rules: (a) not gated; (b) no remote code (transformers 5.17 built its config in E1's static check); (c) fits one
12 GB card: unquantized at most 4.5B parameters, and at most 9.5 GB of stored weights in any case, and no FP4 format
(Blackwell only); (d) a chat template; (e) a text-only *ForCausalLM architecture that vLLM 0.30 and SGLang 0.5.20
register (the harness loads AutoModelForCausalLM).
Run in ~/venvs/vllm: python testbed/m10_e2_select.py
Writes testbed/results/m10/e2/selection.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "m10")
BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
         "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1, "F4": 0.5, "F6_E2M3": 0.75, "F6_E3M2": 0.75}
FP4 = ("fp4", "nvfp4", "modelopt_fp4", "mxfp4")
N = int(os.environ.get("M10_E2_N", 20))   # M11.7: N=30 gives the next ten by the same rule, the fresh sample


def main():
    from vllm.model_executor.models.registry import _VLLM_MODELS

    models = json.load(open(os.path.join(R, "e1_llm", "models.json"), encoding="utf-8"))["models"]
    static = {r["id"]: r for r in json.load(open(os.path.join(R, "e1_llm", "static.json"), encoding="utf-8"))["rows"]}
    sglang = set(json.load(open(os.path.join(R, "sglang_archs.json"), encoding="utf-8")))
    chosen, passed_over, by_arch = [], [], {}
    for m in models:
        folder = os.path.join(R, "e1_llm", "configs", m["id"].replace("/", "__"))
        why = None
        cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")) \
            if os.path.isfile(os.path.join(folder, "config.json")) else None
        arch = ((cfg or {}).get("architectures") or [None])[0]
        quant = ((cfg or {}).get("quantization_config") or {}).get("quant_method")
        by_dtype = m.get("params_by_dtype") or {}
        stored = sum(n * BYTES.get(dt, 2) for dt, n in by_dtype.items()) if by_dtype else None
        tok = os.path.join(folder, "tokenizer_config.json")
        template = os.path.isfile(os.path.join(folder, "chat_template.jinja")) or (
            os.path.isfile(tok) and "chat_template" in json.load(open(tok, encoding="utf-8")))
        notes = (static.get(m["id"]) or {}).get("engines", {}).get("transformers", {}).get("notes", [])
        if m.get("gated"):
            why = "(a) gated"
        elif cfg is None:
            why = "no config.json (GGUF or other format)"
        elif any("could not build the config" in n for n in notes):
            why = "(b) transformers cannot build the config without remote code"
        elif stored is None:
            why = "(c) size unknown (no safetensors metadata)"
        elif quant is None and (m.get("params_total") or 0) > 4.5e9:
            why = f"(c) unquantized, {m.get('params_total') / 1e9:.1f}B parameters"
        elif stored > 9.5e9:
            why = f"(c) {stored / 1e9:.1f} GB of stored weights"
        elif quant and any(x in str(quant).lower() for x in FP4) or any("F4" == d or "E2M1" in d for d in by_dtype):
            why = f"(c) FP4 format ({quant}), Blackwell only"
        elif not template:
            why = "(d) no chat template"
        elif not arch or not arch.endswith("ForCausalLM"):
            why = f"(e) {arch}: not a text-only ForCausalLM"
        elif arch not in _VLLM_MODELS:
            why = f"(e) {arch}: not in vLLM 0.30"
        elif arch not in sglang:
            why = f"(e) {arch}: not in SGLang 0.5.20"
        row = {"rank": m["rank"], "id": m["id"], "arch": arch, "quant": quant,
               "stored_gb": round(stored / 1e9, 2) if stored else None,
               "params_b": round(m["params_total"] / 1e9, 2) if m.get("params_total") else None}
        if why:
            if len(chosen) < N:
                passed_over.append(dict(row, why=why))
            continue
        if len(chosen) < N:
            chosen.append(row)
        by_arch.setdefault(arch, row)
    extra = [r for r in by_arch.values() if r["id"] not in {c["id"] for c in chosen}]
    out = {"rules": __doc__.split("Rules:")[1].split("Run in")[0].strip(), "chosen": chosen,
           "passed_over_before_the_last_chosen": passed_over,
           "by_architecture_extra": extra,
           "by_architecture_note": "every architecture in the top 300 that passes (a)-(e): its highest-ranked model; "
                                   "those already among the first 20 are not repeated (M10_PROTOCOL.md 6)"}
    for c in extra:
        print(f"EXTRA  {c['rank']:3} {c['id']:55} {c['arch']:28} {c['quant']} {c['stored_gb']} GB")
    os.makedirs(os.path.join(R, "e2"), exist_ok=True)
    with open(os.environ.get("M10_E2_SELECTION") or os.path.join(R, "e2", "selection.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for c in chosen:
        print(f"CHOSEN {c['rank']:3} {c['id']:55} {c['arch']:28} {c['quant']} {c['stored_gb']} GB")
    from collections import Counter
    print("passed over:", Counter(p["why"].split(" ")[0] for p in passed_over))


if __name__ == "__main__":
    main()
