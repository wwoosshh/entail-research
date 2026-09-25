"""M10 E1, L3 (a what-if, not a change to entail): how many of the config keys entail 1.0 reports as broken on the
popular folders would still be broken under a narrower rule - broken only for a key within edit distance 2 of a field
the config class knows (a misspelling, like rb-15's rope_scale for rope_scaling) and not itself a known field;
every other key not taken would be reported as unknown. The rule is a proposal for the researcher; entail is not
changed. rb-15's key is checked the same way.
Run in ~/venvs/gpu: python testbed/m10_e1_coverage_rule.py
Writes testbed/results/m10/e1_llm/coverage_rule_whatif.json.
"""
import inspect
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
L1 = os.path.join(HERE, "results", "m10", "e1_llm")


def dist(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def alias_names():
    """Every config key entail's alias table knows (data/aliases.json), e.g. transformers-4 names like rope_scaling."""
    path = os.path.join(os.path.dirname(HERE), "entail", "entail", "data", "aliases.json")
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("hf_config",) and isinstance(v, dict):
                    for vv in v.values():
                        out.update(vv if isinstance(vv, list) else [vv] if isinstance(vv, str) else [])
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(json.load(open(path, encoding="utf-8")))
    return {n for n in out if isinstance(n, str)}


ALIASES = None


def known_fields(folder, sub):
    from transformers import AutoConfig

    global ALIASES
    if ALIASES is None:
        ALIASES = alias_names()
    cfg = AutoConfig.from_pretrained(folder)
    if sub:
        cfg = getattr(cfg, sub, None) or cfg
    names = set(vars(cfg)) | ALIASES
    for cls in type(cfg).__mro__:
        if "__init__" in vars(cls):
            names |= {p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "kwargs")}
        names |= {n for n, v in vars(cls).items() if isinstance(v, property)}
    return names


def main():
    ck = json.load(open(os.path.join(L1, "coverage_keys.json"), encoding="utf-8"))["per_model"]
    rows, still = [], {}
    for mid, keys in ck.items():
        folder = os.path.join(L1, "configs", mid.replace("/", "__"))
        for key in keys:
            if key.startswith("..."):
                continue
            sub, base = (key.split(".")[0], key.split(".")[-1]) if "." in key else (None, key)
            try:
                fields = known_fields(folder, sub)
            except Exception as e:  # noqa: BLE001
                rows.append({"model": mid, "key": key, "error": str(e)[:120]})
                continue
            near = sorted((dist(base, f), f) for f in fields if f != base)[:1]
            misspelt = bool(near and near[0][0] <= 2 and base not in fields)
            rows.append({"model": mid, "key": key, "nearest": near[0][1] if near else None,
                         "distance": near[0][0] if near else None, "broken_under_rule": misspelt})
            if misspelt:
                still.setdefault(mid, []).append(f"{key}~{near[0][1]}")
    # rb-15: rope_scale in a Llama config
    from transformers import LlamaConfig
    llama = set(vars(LlamaConfig())) | set(inspect.signature(LlamaConfig.__init__).parameters) | ALIASES
    for cls in LlamaConfig.__mro__:
        llama |= {n for n, v in vars(cls).items() if isinstance(v, property)}
    rb15 = sorted((dist("rope_scale", f), f) for f in llama)[:1]
    out = {"rule": "broken only for a key within edit distance 2 of a known field of the config class; else unknown",
           "flagged_models_entail_1_0": len(ck), "flagged_models_under_rule": len(still),
           "pairs": len(rows), "pairs_broken_under_rule": sum(r.get("broken_under_rule", False) for r in rows),
           "still_broken": still, "rb15_rope_scale_nearest": rb15, "rows": rows}
    json.dump(out, open(os.path.join(L1, "coverage_rule_whatif.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(json.dumps({k: out[k] for k in ("flagged_models_entail_1_0", "flagged_models_under_rule", "pairs",
                                          "pairs_broken_under_rule", "rb15_rope_scale_nearest")}, indent=1))
    for m, ks in still.items():
        print(" ", m, ks)


if __name__ == "__main__":
    main()
