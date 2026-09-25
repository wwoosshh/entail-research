"""Gemma 2 attention softcap measurement. The plan and pass/fail rules are in PROTOCOL.md (written before any result).

Subcommands (each writes results/<exp>_<size>.json):
  e0  default attention path, and every warning/log line at load and first forward
  e1  teacher-forced logits across paths (eager, flex_attention, sdpa, eager_nocap), plus pre-softcap score sizes
  e2  greedy generation, 30 chat prompts x 256 tokens (eager, sdpa, eager_nocap)
  e3  GSM8K test[:500], greedy, 0-shot chat, sdpa vs eager

Paths are switched at runtime on one loaded model:
  - model.set_attn_implementation(...) picks the kernel.
  - Each layer's self_attn.attn_logit_softcapping is set to None for "eager_nocap".
"""
import argparse
import json
import logging
import math
import os
import re
import time
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PROJECT = os.path.dirname(os.path.dirname(HERE))
MODELS = {"2b": os.path.expanduser("~/models/gemma-2-2b-it"), "9b": os.path.expanduser("~/models/gemma-2-9b-it")}
MAX_DOC = {"2b": 6144, "9b": 4096}
GSM_BATCH = {"2b": 16, "9b": 8}
THRESH = (10.0, 20.0, 30.0, 50.0)

CHAT_PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive? Show your reasoning.",
    "List five differences between TCP and UDP.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "What is the derivative of x^3 * sin(x)? Explain each step.",
    "Translate into French: 'The meeting was postponed because of the storm.'",
    "Give me a short poem about autumn rain, four lines.",
    "If a rectangle has perimeter 36 and its length is twice its width, what is its area?",
    "Explain what a hash table is to a 12-year-old.",
    "Write a SQL query that returns the top 3 customers by total order amount.",
    "What are the main causes of inflation? Answer briefly.",
    "Describe how photosynthesis works in simple terms.",
    "A shop sells pens at 3 for $2. How much do 18 pens cost?",
    "Write a haiku about a quiet library.",
    "Compare Python lists and tuples in a short table.",
    "Explain recursion with a simple example in JavaScript.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Name three strategies to improve sleep quality.",
    "Solve for x: 3x + 7 = 25. Show the steps.",
    "하늘이 파란 이유를 세 문장으로 설명해 주세요.",
    "파이썬으로 리스트에서 중복을 제거하는 방법 두 가지를 알려 주세요.",
    "사과 12개를 4명이 똑같이 나누고, 각자 1개씩 먹으면 한 사람에게 몇 개가 남나요? 풀이 과정을 보여 주세요.",
    "서울에서 가 볼 만한 곳 세 군데를 추천하고 이유를 적어 주세요.",
    "재귀 함수가 무엇인지 초등학생도 이해할 수 있게 설명해 주세요.",
    "다음 문장을 영어로 번역해 주세요: '회의는 폭풍 때문에 연기되었습니다.'",
    "가을비에 대한 짧은 시를 네 줄로 써 주세요.",
    "TCP와 UDP의 차이를 세 가지만 설명해 주세요.",
    "3x + 7 = 25일 때 x를 구하는 과정을 보여 주세요.",
    "건강한 수면 습관 세 가지를 알려 주세요.",
]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save(name, obj):
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    log(f"wrote {path}")


def load(size, attn=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[size])
    kw = dict(dtype=torch.bfloat16)
    if attn:
        kw["attn_implementation"] = attn
    if size == "2b":
        model = AutoModelForCausalLM.from_pretrained(MODELS[size], **kw).cuda()
        return tok, model.eval(), "bf16"
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    qcfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d")
    try:
        from transformers import TorchAoConfig

        model = AutoModelForCausalLM.from_pretrained(
            MODELS[size], quantization_config=TorchAoConfig(qcfg), device_map="cuda", **kw
        )
        how = "int4 (TorchAoConfig on load)"
    except Exception as e:  # fall back: load on CPU and quantize layer by layer on the GPU
        log(f"TorchAoConfig load failed ({type(e).__name__}: {str(e)[:200]}); quantizing per layer")
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(MODELS[size], device_map="cpu", **kw)
        for layer in model.model.layers:
            layer.cuda()
            quantize_(layer, qcfg)
        model.cuda()
        how = "int4 (per-layer quantize_)"
    return tok, model.eval(), how


def set_path(model, path):
    impl = {"eager": "eager", "flex_attention": "flex_attention", "sdpa": "sdpa", "eager_nocap": "eager"}[path]
    model.set_attn_implementation(impl)
    cap = None if path == "eager_nocap" else model.config.attn_logit_softcapping
    for layer in model.model.layers:
        layer.self_attn.attn_logit_softcapping = cap


def chat_ids(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids


def documents():
    docs = []
    for name, path in (
        ("korean_research_plan", os.path.join(PROJECT, "RESEARCH_PLAN.md")),
        ("english_gpl3", "/usr/share/common-licenses/GPL-3"),
    ):
        if os.path.exists(path):
            docs.append((name, open(path, encoding="utf-8", errors="replace").read()))
    import transformers.models.gemma2.modeling_gemma2 as mg

    docs.append(("python_code", open(mg.__file__, encoding="utf-8").read()))
    return docs


# ---------------------------------------------------------------- e0
def e0(size):
    records = []

    class Grab(logging.Handler):
        def emit(self, r):
            records.append({"logger": r.name, "level": r.levelname, "msg": r.getMessage()[:600]})

    grab = Grab(level=logging.DEBUG)
    tlog = logging.getLogger("transformers")
    tlog.addHandler(grab)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tok, model, how = load(size, None)
        default = model.config._attn_implementation
        ids = chat_ids(tok, "Hello, how are you?").cuda()
        with torch.no_grad():
            model(ids)
            model.generate(ids, max_new_tokens=8, do_sample=False)
    tlog.removeHandler(grab)
    save(f"e0_{size}.json", {
        "size": size, "weights": how, "default_attn_implementation": default,
        "config_attn_logit_softcapping": model.config.attn_logit_softcapping,
        "layer0_softcap_attribute": model.model.layers[0].self_attn.attn_logit_softcapping,
        "python_warnings": [f"{x.category.__name__}: {str(x.message)[:600]}" for x in w],
        "transformers_log": records,
        "softcap_mentioned": any("softcap" in (r["msg"].lower()) for r in records)
        or any("softcap" in str(x.message).lower() for x in w),
        "seconds": round(time.time() - t0, 1),
    })


# ---------------------------------------------------------------- e1
class ScoreStats:
    """Wraps Gemma 2's eager attention to count pre-softcap score sizes on unmasked positions."""

    def __init__(self, n_layers):
        self.count = [[0] * len(THRESH) for _ in range(n_layers)]
        self.total = [0] * n_layers
        self.max = [0.0] * n_layers
        self.qlen = None  # real (unpadded) length of the current sequence; padded positions are excluded

    def install(self, mg):
        orig = mg.eager_attention_forward
        stats = self

        def wrapped(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kw):
            with torch.no_grad():
                n = stats.qlen or query.shape[-2]
                k = mg.repeat_kv(key, module.num_key_value_groups)[:, :, :n]
                s = torch.matmul(query[:, :, :n].float(), k.float().transpose(2, 3)) * scaling
                if attention_mask is not None:
                    m = attention_mask[:, :, :n, :n]
                    valid = m if m.dtype == torch.bool else (m == 0)
                    valid = valid.expand_as(s)
                else:
                    valid = torch.ones_like(s, dtype=torch.bool).tril()
                a = s.abs()[valid]
                i = module.layer_idx
                stats.total[i] += a.numel()
                for j, t in enumerate(THRESH):
                    stats.count[i][j] += int((a > t).sum())
                stats.max[i] = max(stats.max[i], float(a.max()) if a.numel() else 0.0)
                del s, a, valid, k
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, softcap=softcap, **kw)

        mg.eager_attention_forward = wrapped
        return orig

    def summary(self, layer_types):
        rows = []
        for i in range(len(self.total)):
            t = max(self.total[i], 1)
            rows.append({"layer": i, "type": layer_types[i] if layer_types else None, "max_abs": round(self.max[i], 3),
                         **{f"frac_gt_{int(th)}": self.count[i][j] / t for j, th in enumerate(THRESH)}})
        return rows


def final_logits(model, h):
    logits = model.lm_head(h).float()
    cap = model.config.final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def install_fp32_attention(mg):
    """E1b: run every attention path in fp32 (inputs upcast, output cast back) to remove kernel rounding noise."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def wrap(fn):
        def w(module, query, key, value, attention_mask, *a, **kw):
            m = attention_mask
            if isinstance(m, torch.Tensor) and m.dtype != torch.bool:
                m = m.float()
            out, weights = fn(module, query.float(), key.float(), value.float(), m, *a, **kw)
            return out.to(query.dtype), weights
        return w

    mg.eager_attention_forward = wrap(mg.eager_attention_forward)
    for name in ("sdpa", "flex_attention"):
        ALL_ATTENTION_FUNCTIONS[name] = wrap(ALL_ATTENTION_FUNCTIONS[name])


def e1(size, fp32_attn=False):
    import transformers.models.gemma2.modeling_gemma2 as mg

    try:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.recompile_limit = 64
    except Exception:
        pass
    tok, model, how = load(size, "eager")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    tag = "e1fp32" if fp32_attn else "e1"
    if fp32_attn:
        install_fp32_attention(mg)

    # sequences: reuse the bf16 E1 token sequences if they exist, so E1 and E1b see identical inputs
    prev = os.path.join(RES, f"e1_{size}_tokens.json")
    set_path(model, "eager")
    seqs = []
    if fp32_attn and os.path.exists(prev):
        for s in json.load(open(prev, encoding="utf-8"))["sequences"]:
            seqs.append({"kind": s["kind"], "name": s["name"], "ids": s["ids"]})
    else:
        # chat prompt + eager greedy continuation (128 tokens), then long documents
        for p in CHAT_PROMPTS:
            ids = chat_ids(tok, p).cuda()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=128, do_sample=False)
            seqs.append({"kind": "chat", "name": p[:48], "ids": out[0].tolist()})
        for name, text in documents():
            ids = tok(text, add_special_tokens=True).input_ids[: MAX_DOC[size]]
            seqs.append({"kind": "doc", "name": name, "ids": ids})
    log(f"{len(seqs)} sequences, lengths {[len(s['ids']) for s in seqs]}")

    stats = ScoreStats(model.config.num_hidden_layers)
    paths = ["eager", "flex_attention", "sdpa", "eager_nocap"]
    # (reference, other): KL(reference || other), top-1 disagreement, max |logit diff|
    pairs = [("eager", "flex_attention"), ("eager", "sdpa"), ("eager", "eager_nocap"), ("eager_nocap", "sdpa")]
    per_seq = []
    agg = {f"{a}|{b}": {"kl_sum": 0.0, "n": 0, "top1_diff": 0, "maxabs": 0.0} for a, b in pairs}
    for si, s in enumerate(seqs):
        L = len(s["ids"])
        B = int(math.ceil(L / 1024) * 1024)
        x = torch.full((1, B), pad, dtype=torch.long)
        x[0, :L] = torch.tensor(s["ids"])
        x = x.cuda()
        hidden = {}
        stats.qlen = L
        for p in paths:
            set_path(model, p)
            orig = stats.install(mg) if p == "eager" else None
            try:
                with torch.no_grad():
                    h = model.model(input_ids=x, use_cache=False).last_hidden_state[0, :L]
                hidden[p] = h
            except Exception as e:
                log(f"seq {si} path {p} failed: {type(e).__name__}: {str(e)[:300]}")
                hidden[p] = None
            finally:
                if orig is not None:
                    mg.eager_attention_forward = orig
        row = {"kind": s["kind"], "name": s["name"], "len": L}
        ids_t = torch.tensor(s["ids"], device="cuda")
        tokwise = {}  # per path: top-1 id and log-prob of the realized next token, for the vLLM comparison (e4)
        for p in paths:
            if hidden[p] is None:
                continue
            top1, lp_next = [], []
            for c in range(0, L, 256):
                with torch.no_grad():
                    lsp = torch.log_softmax(final_logits(model, hidden[p][c:c + 256]), -1)
                    top1 += lsp.argmax(-1).tolist()
                    nxt = ids_t[c + 1:c + 1 + lsp.shape[0]]
                    got = lsp[: nxt.shape[0]].gather(-1, nxt[:, None])[:, 0]
                    lp_next += [round(v, 5) for v in got.tolist()]
            tokwise[p] = {"top1": top1, "logprob_next": lp_next}
        s["tokwise"] = tokwise
        for ra, rb in pairs:
            key = f"{ra}|{rb}"
            if hidden[ra] is None or hidden[rb] is None:
                row[key] = None
                continue
            kl_sum, top1_diff, maxabs = 0.0, 0, 0.0
            for c in range(0, L, 256):
                with torch.no_grad():
                    le = final_logits(model, hidden[ra][c:c + 256])
                    lp = final_logits(model, hidden[rb][c:c + 256])
                    lse, lsp = torch.log_softmax(le, -1), torch.log_softmax(lp, -1)
                    kl_sum += float((lse.exp() * (lse - lsp)).sum())
                    top1_diff += int((le.argmax(-1) != lp.argmax(-1)).sum())
                    maxabs = max(maxabs, float((le - lp).abs().max()))
            row[key] = {"kl_mean": kl_sum / L, "top1_disagree": top1_diff / L, "max_abs_logit_diff": maxabs}
            a = agg[key]
            a["kl_sum"] += kl_sum
            a["n"] += L
            a["top1_diff"] += top1_diff
            a["maxabs"] = max(a["maxabs"], maxabs)
        per_seq.append(row)
        log(f"seq {si} ({s['kind']}, {L} tok): " + ", ".join(
            f"{k}: kl={v['kl_mean']:.2e} top1diff={v['top1_disagree']:.4f}"
            for k, v in row.items() if isinstance(v, dict)))
        del hidden
        torch.cuda.empty_cache()

    summary = {p: {"kl_mean": a["kl_sum"] / max(a["n"], 1), "top1_disagree": a["top1_diff"] / max(a["n"], 1),
                   "max_abs_logit_diff": a["maxabs"], "positions": a["n"]} for p, a in agg.items()}
    layer_types = getattr(model.config, "layer_types", None)
    save(f"{tag}_{size}.json", {"size": size, "weights": how, "attention_fp32": fp32_attn, "summary_pairs": summary,
                                "per_sequence": per_seq, "pre_softcap_scores": stats.summary(layer_types),
                                "thresholds": THRESH})
    save(f"{tag}_{size}_tokens.json", {"size": size, "sequences": seqs})


# ---------------------------------------------------------------- e2
def e2(size):
    tok, model, how = load(size, "eager")
    out = {"size": size, "weights": how, "paths": {}}
    for p in ("eager", "sdpa", "eager_nocap"):
        set_path(model, p)
        gens = []
        for q in CHAT_PROMPTS:
            ids = chat_ids(tok, q).cuda()
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=256, do_sample=False)
            gens.append(g[0, ids.shape[1]:].tolist())
        out["paths"][p] = gens
        log(f"e2 {size} {p} done")
    ref = out["paths"]["eager"]
    cmp = {}
    for p in ("sdpa", "eager_nocap"):
        same, first = 0, []
        for a, b in zip(ref, out["paths"][p]):
            n = min(len(a), len(b))
            d = next((i for i in range(n) if a[i] != b[i]), None)
            if d is None and len(a) == len(b):
                same += 1
                first.append(None)
            else:
                first.append(d if d is not None else n)
        cmp[p] = {"identical_fraction": same / len(ref), "first_divergence": first}
    out["compare_to_eager"] = cmp
    out["texts"] = {p: [tok.decode(g, skip_special_tokens=True) for g in gens] for p, gens in out["paths"].items()}
    save(f"e2_{size}.json", out)


# ---------------------------------------------------------------- e3
def extract_answer(text):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        s = m.group(1)
    else:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def e3(size, n=500):
    data = [json.loads(l) for l in open(os.path.join(HERE, "data", "gsm8k_test.jsonl"), encoding="utf-8")][:n]
    gold = [float(d["answer"].split("####")[-1].strip().replace(",", "")) for d in data]
    instr = ("Solve the following math problem step by step. "
             "At the end, write the final answer as a number on its own line in the form '#### <number>'.\n\n")
    tok, model, how = load(size, "eager")
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template([{"role": "user", "content": instr + d["question"]}], tokenize=False,
                                       add_generation_prompt=True) for d in data]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = {"size": size, "weights": how, "n": len(data), "paths": {}}
    for p in ("sdpa", "eager"):
        set_path(model, p)
        outs = [None] * len(prompts)
        bs = GSM_BATCH[size]
        t0 = time.time()
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=512, do_sample=False)
            for j, i in enumerate(idx):
                outs[i] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
        correct = [extract_answer(o) is not None and abs(extract_answer(o) - gold[i]) < 1e-6 for i, o in enumerate(outs)]
        k = sum(correct)
        res["paths"][p] = {"correct": correct, "accuracy": k / len(data), "wilson95": wilson(k, len(data)),
                           "seconds": round(time.time() - t0, 1), "outputs": outs}
        log(f"e3 {size} {p}: {k}/{len(data)}")
    a, b = res["paths"]["eager"]["correct"], res["paths"]["sdpa"]["correct"]
    only_eager = sum(1 for x, y in zip(a, b) if x and not y)
    only_sdpa = sum(1 for x, y in zip(a, b) if y and not x)
    res["paired"] = {"only_eager_correct": only_eager, "only_sdpa_correct": only_sdpa,
                     "mcnemar_exact_p": mcnemar_exact(only_eager, only_sdpa),
                     "accuracy_diff_eager_minus_sdpa": (sum(a) - sum(b)) / len(data)}
    save(f"e3_{size}.json", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e0", "e1", "e1b", "e2", "e3"])
    ap.add_argument("--size", choices=["2b", "9b"], required=True)
    args = ap.parse_args()
    log(f"start {args.exp} {args.size} torch {torch.__version__}")
    if args.exp == "e1b":
        e1(args.size, fp32_attn=True)
    else:
        {"e0": e0, "e1": e1, "e2": e2, "e3": e3}[args.exp](args.size)
    log("done")
