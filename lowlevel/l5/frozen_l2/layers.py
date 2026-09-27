"""L2 reference comparison, layer by layer: the residual stream after every decoder layer, in the engine and in a
reference implementation, for the same token ids. The first layer where the engine leaves the reference by more than
the reference's own precision noise is where the meaning was lost (THEORY 2.1: where meaning breaks is where the
problem is). No rule about any particular defect.

  dump side, in the engine's venv:
    python lowlevel/l2/layers.py vllm <model> <out dir> [engine kwargs json]      (vLLM, eager, in-process)
    python lowlevel/l2/layers.py hf <model> <out dir> <dtype> [token ids dir]      (transformers, float32 or bfloat16)
    python lowlevel/l2/layers.py hfq <model> <out dir> <dtype> [token ids dir]     (the same model with a quantized
        checkpoint's weights dequantized by the format's definition, dequant.py; compare then gives no allowance)
  compare (any python with torch):
    python lowlevel/l2/layers.py compare <engine dir> <ref fp32 dir> <ref bf16 dir>
Every dump writes <out dir>/<probe id>.pt with {"ids": [...], "layers": [tensor [n, hidden] float32 per layer]}.
The hf side reads the token ids from the engine's dump so both see exactly the same input.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Allowance for a checkpoint that declares quantization: the largest error a healthy engine with the same declared
# weight bits showed against the unquantized original, rounded up (quant_tolerance.py on vLLM 0.30.0: AWQ Qwen2.5-3B
# for 4 bits, FP8 Qwen3-4B for 8 bits; results/quant_tolerance.json, 2026-09-27). frob is a layer's relative
# Frobenius error, tok the median over tokens of each token's relative error.
QUANT_ALLOWANCE = {"frob": {4: 1.60, 8: 0.125}, "tok": {4: 0.380, 8: 0.092}}


def declared_bits(model_dir):
    """The weight bit width the checkpoint declares in its quantization_config: None when it declares none, "?" when
    its config.json cannot be read."""
    try:
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
    except OSError:
        return "?"
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if not q:
        return None
    if q.get("bits"):
        return int(q["bits"])
    if q.get("quant_method") == "fp8":
        return 8
    bits = [g["weights"]["num_bits"] for g in (q.get("config_groups") or {}).values() if g.get("weights")]
    return min(bits) if bits else "?"


def allowance(bits):
    """Per measure, the allowance for a declared weight bit width: none when nothing (or nothing readable) is declared,
    the loosest measured one for a declared width that was never measured."""
    if not isinstance(bits, int):
        return {m: 0.0 for m in QUANT_ALLOWANCE}
    return {m: t.get(bits, max(t.values())) for m, t in QUANT_ALLOWANCE.items()}


def decoder_layers(model):
    """The language model's decoder layers: among the nn.ModuleLists named `layers` outside any vision or audio
    tower, the longest (generic across architectures; a multimodal model's vision blocks are not run by text
    probes). Falls back to the longest ModuleList when none is named so."""
    import torch.nn as nn

    cands = []
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) > 1:
            low = name.lower()
            tower = any(w in low for w in ("vision", "visual", "audio", "image", "vit", "encoder."))
            cands.append((name.split(".")[-1] in ("layers", "h", "blocks") and not tower, len(m), name, m))
    if not cands:
        return None
    named, n, name, m = max(cands, key=lambda c: (c[0], c[1]))
    return name, m, {type(x).__name__ for x in m}


def residual_of(out):
    """A decoder layer's output as the residual stream: vLLM layers return (hidden, residual) with the residual add
    deferred into the next norm, so the stream is their sum; transformers layers return the stream (or a tuple
    whose first element is it)."""
    import torch

    if isinstance(out, (tuple, list)):
        if len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]) and out[0].shape == out[1].shape:
            return out[0] + out[1]
        return out[0]
    return out


def dump_vllm(model, out_dir, engine_kw):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from vllm import LLM, SamplingParams

    from probes import PROBES

    kw = {"model": model, "seed": 0, "gpu_memory_utilization": 0.8, "max_model_len": 2048, "max_num_seqs": 4,
          "enforce_eager": True, "enable_prefix_caching": False}
    kw.update(engine_kw)
    llm = LLM(**kw)
    store = {"cur": None}

    def install(m):
        found = decoder_layers(m)
        if found is None:
            return "no decoder layers found"
        name, layers, kinds = found

        def hook(i):
            def f(mod, inp, out):
                if store["cur"] is not None:
                    store["cur"].setdefault(i, []).append(residual_of(out).detach().float().cpu())
            return f

        for i, layer in enumerate(layers):
            layer.register_forward_hook(hook(i))
        return f"{name}: {len(layers)} x {sorted(kinds)}"

    info = llm.apply_model(install)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"engine": "vllm", "model": model, "layers": info, "probes": {}}
    for p in PROBES:
        store["cur"] = {}
        o = llm.generate([p["text"]], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)[0]
        cap = store["cur"]
        store["cur"] = None
        ids = list(o.prompt_token_ids)
        # the prefill call is the first capture of each layer; it covers every prompt token
        layers = [cap[i][0][: len(ids)] for i in sorted(cap)]
        torch.save({"ids": ids, "layers": layers}, os.path.join(out_dir, f"{p['id']}.pt"))
        meta["probes"][p["id"]] = {"tokens": len(ids), "layers": len(layers)}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("RESULT", json.dumps(meta)[:400])


def load_dequantized(model, dtype):
    """The checkpoint as a dense transformers model whose quantized weights are dequantized by the declared format's
    definition (dequant.py): the same quantized model the engine runs, in float arithmetic. Every parameter must come
    from the checkpoint; a missing one stops the run rather than leaving a random weight in the reference."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    import dequant

    cfg = AutoConfig.from_pretrained(model)
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is not None and getattr(c, "quantization_config", None) is not None:
            delattr(c, "quantization_config")
    try:
        m = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="eager")
    except Exception:  # noqa: BLE001 - a multimodal checkpoint: its text model is inside
        m = AutoModelForImageTextToText.from_config(cfg, dtype=dtype, attn_implementation="eager")
    params = dict(m.named_parameters())
    conv = getattr(m, "_checkpoint_conversion_mapping", None) or {}

    def target(name):
        """The model's name for a checkpoint tensor: as stored, by the model's own conversion mapping, or with a
        multimodal checkpoint's language_model prefix dropped (a text-only class)."""
        if name in params:
            return name
        for pat, rep in conv.items():
            alt = re.sub(pat, rep, name)
            if alt in params:
                return alt
        alt = name.replace(".language_model.", ".", 1)
        return alt if alt in params else None

    filled, unexpected, n_deq = set(), [], 0
    with torch.no_grad():
        for name, t, deq in dequant.iter_state(model):
            tn = target(name)
            p = params.get(tn) if tn else None
            if p is None or p.shape != t.shape:
                unexpected.append(name)
                continue
            p.copy_(t)
            filled.add(tn)
            n_deq += deq
    missing = sorted(set(params) - filled)
    if missing:
        raise RuntimeError(f"{len(missing)} parameters not in the checkpoint, e.g. {missing[:5]}")
    return m, {"dequantized": n_deq, "filled": len(filled), "unexpected": len(unexpected),
               "unexpected_examples": sorted(unexpected)[:8]}


def dump_hf(model, out_dir, dtype_name, ids_dir, dequantized=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    from probes import PROBES

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    load = None
    if dequantized:
        m, load = load_dequantized(model, dtype)
    else:
        try:
            m = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, attn_implementation="eager")
        except Exception:  # noqa: BLE001 - a multimodal checkpoint: its text model is inside
            m = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype, attn_implementation="eager")
    dev = "cuda" if dtype == torch.bfloat16 or os.environ.get("L2_REF_CUDA") else "cpu"
    m = m.to(dev).eval()
    name, layers, kinds = decoder_layers(m)
    cap = {}

    def hook(i):
        def f(mod, inp, out):
            cap.setdefault(i, residual_of(out).detach().float().cpu())
        return f

    for i, layer in enumerate(layers):
        layer.register_forward_hook(hook(i))
    os.makedirs(out_dir, exist_ok=True)
    meta = {"engine": "hf", "model": model, "dtype": dtype_name, "device": dev,
            "layers": f"{name}: {len(layers)} x {sorted(kinds)}", "probes": {}}
    if dequantized:
        meta.update({"reference": "dequantized", "load": load})
    for p in PROBES:
        src = torch.load(os.path.join(ids_dir, f"{p['id']}.pt"))
        ids = torch.tensor([src["ids"]], device=dev)
        cap.clear()
        with torch.no_grad():
            m(input_ids=ids)
        out = [cap[i][0] for i in sorted(cap)]
        torch.save({"ids": src["ids"], "layers": out}, os.path.join(out_dir, f"{p['id']}.pt"))
        meta["probes"][p["id"]] = {"tokens": len(src["ids"]), "layers": len(out)}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("RESULT", json.dumps(meta)[:400])


def dump_hf_chunk(model, out_dir, chunk=16, n_tokens=64):
    """A mode comparison inside transformers, layer by layer: each probe's first n_tokens in one forward pass
    (full32, full16: the reference and its own bfloat16 noise) and in chunks carried through the model's cache
    (chunked). The cache is handed back under the name the model returned it (past_key_values or cache_params), with
    cache_position, so the same code serves attention, Mamba and hybrid models. CPU, float32."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probes import PROBES

    tok = AutoTokenizer.from_pretrained(model)
    runs = {"full32": torch.float32, "full16": torch.bfloat16, "chunked": torch.float32}
    meta = {"engine": "hf_chunk", "model": model, "chunk": chunk, "probes": {}}
    for tag, dt in runs.items():
        m = AutoModelForCausalLM.from_pretrained(model, dtype=dt).eval()
        name, layers, kinds = decoder_layers(m)
        cap = {}

        def hook(i):
            def f(mod, inp, out):
                cap.setdefault(i, []).append(residual_of(out).detach().float().cpu())
            return f

        hs = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
        d = os.path.join(out_dir, tag)
        os.makedirs(d, exist_ok=True)
        for p in PROBES:
            ids = tok(p["text"], return_tensors="pt").input_ids[:, :n_tokens]
            cap.clear()
            with torch.no_grad():
                if tag != "chunked":
                    m(input_ids=ids, use_cache=False)
                else:
                    cache_kw, cache = None, None
                    for s in range(0, ids.shape[1], chunk):
                        piece = ids[:, s:s + chunk]
                        kw = {"input_ids": piece, "use_cache": True,
                              "cache_position": torch.arange(s, s + piece.shape[1])}
                        if cache is not None:
                            kw[cache_kw] = cache
                        out = m(**kw)
                        for cand in ("past_key_values", "cache_params"):
                            if getattr(out, cand, None) is not None:
                                cache_kw, cache = cand, getattr(out, cand)
            layers_out = [torch.cat(cap[i], dim=1)[0] for i in sorted(cap)]
            torch.save({"ids": ids[0].tolist(), "layers": layers_out}, os.path.join(d, f"{p['id']}.pt"))
            meta["probes"][p["id"]] = {"tokens": int(ids.shape[1]), "layers": len(layers_out)}
        for h in hs:
            h.remove()
        del m
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("RESULT", json.dumps(meta)[:300])


def layer_measures(a, b, c):
    """The engine's (a) error and the bfloat16 reference's (c) noise against the float32 reference (b) for one layer,
    in two measures: over the whole layer (frob) and the median over tokens of each token's relative error (tok,
    which a few very large tokens cannot dominate)."""
    a, b, c = a.double(), b.double(), c.double()
    nb = b.norm().item() or 1.0
    tn = b.norm(dim=-1).clamp_min(1e-12)
    return {"engine_err": (a - b).norm().item() / nb, "ref_noise": (c - b).norm().item() / nb,
            "engine_tok": ((a - b).norm(dim=-1) / tn).median().item(), "ref_tok": ((c - b).norm(dim=-1) / tn).median().item()}


def compare(engine_dir, ref32_dir, ref16_dir, factor=4.0, floor=0.02):
    """Per probe and layer: the engine's relative error against the float32 reference and the reference's own
    bfloat16 noise, in both measures, and the first layer where the error exceeds factor x max(noise, allowance) +
    floor. The allowance is zero unless the checkpoint declares quantization (allowance())."""
    import torch

    from probes import PROBES

    model = json.load(open(os.path.join(engine_dir, "meta.json")))["model"]
    bits = declared_bits(model)
    ref = json.load(open(os.path.join(ref32_dir, "meta.json"))).get("reference", "as stored")
    # a reference that dequantizes the same checkpoint already carries its quantization: no allowance
    allow = allowance(None if ref == "dequantized" else bits)
    out = {"_declared": {"model": model, "weight_bits": bits, "reference": ref, "allowance": allow}}
    for p in PROBES:
        e = torch.load(os.path.join(engine_dir, f"{p['id']}.pt"))
        r = torch.load(os.path.join(ref32_dir, f"{p['id']}.pt"))
        h = torch.load(os.path.join(ref16_dir, f"{p['id']}.pt"))
        if e["ids"] != r["ids"] or len(e["layers"]) != len(r["layers"]):
            out[p["id"]] = {"error": f"ids equal {e['ids'] == r['ids']}, layers {len(e['layers'])} vs {len(r['layers'])}"}
            continue
        rows, first, first_tok = [], None, None
        for i, (a, b, c) in enumerate(zip(e["layers"], r["layers"], h["layers"])):
            m = layer_measures(a, b, c)
            rows.append({"layer": i, **{k: round(v, 5) for k, v in m.items()}})
            if first is None and m["engine_err"] > factor * max(m["ref_noise"], allow["frob"]) + floor:
                first = i
            if first_tok is None and m["engine_tok"] > factor * max(m["ref_tok"], allow["tok"]) + floor:
                first_tok = i
        out[p["id"]] = {"first_diverging_layer": first, "first_diverging_layer_tok": first_tok, "layers": rows}
    return out


def main():
    mode = sys.argv[1]
    if mode == "vllm":
        dump_vllm(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
    elif mode in ("hf", "hfq"):
        dump_hf(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else sys.argv[3], mode == "hfq")
    elif mode == "hf_chunk":
        dump_hf_chunk(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16)
    elif mode == "compare":
        res = compare(sys.argv[2], sys.argv[3], sys.argv[4])
        d = res["_declared"]
        fname = "layers_compare_dequantized.json" if d["reference"] == "dequantized" else "layers_compare.json"
        json.dump(res, open(os.path.join(sys.argv[2], fname), "w"), indent=1)
        res.pop("_declared")
        print(f"declared weight bits {d['weight_bits']}; reference {d['reference']}; allowance frob "
              f"{d['allowance']['frob']}, tok {d['allowance']['tok']}")
        for pid, r in res.items():
            if "error" in r:
                print(pid, r["error"])
                continue
            worst = max(r["layers"], key=lambda x: x["engine_err"])
            print(f"{pid:<10} first diverging layer {r['first_diverging_layer']} (tok {r['first_diverging_layer_tok']}); "
                  f"worst layer {worst['layer']}: engine {worst['engine_err']:.4f} vs reference noise "
                  f"{worst['ref_noise']:.4f}; layer 0: {r['layers'][0]['engine_err']:.4f} vs {r['layers'][0]['ref_noise']:.4f}")


if __name__ == "__main__":
    main()
