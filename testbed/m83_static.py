"""M8.3: how many of the benchmark's defects the front end refuses before anything runs (the static refusal rate).

Each rolebench case (rolebench/cases, PROBLEMS.md 1) and each role mistake of week 4 (phase0/week4/e5_role_errors.py,
WEEK4_NOTES.md 7.1) is written as a front-end program - the defective version and the fixed one - and what happens is
recorded, not judged:
  refused at trace     the trace raised (RoleError): no data was looked at
  resolved at trace    the trace repaired it (a conversion or a routing, said in the program's notes)
  cannot be written    the front end has no way to say it (no such role, no such operation)
  refused at bind      the trace passed; binding the program to its tensors refused them (their sizes, once)
  passes               the program traces and binds: the front end does not see this defect before it runs
  outside              not a program's defect (a config file's key names): the load contracts (M3) own it
A fixed version that is refused would be a false refusal; each is traced (and bound, where the defect is about
binding) as well. Where a program runs, it runs on the CPU with the torch lowering and is compared with a float32
reference.

Writes testbed/results/m83/static.json. Run: python testbed/m83_static.py (CPU is enough)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))

import torch  # noqa: E402

from entail import frontend as fe  # noqa: E402
from entail.facts import Layout, ModelProps, Positions, Quantized, Reduction, Rotary  # noqa: E402

OUT = os.path.join(HERE, "results", "m83", "static.json")
B, L, HQ, HKV, D, S = 2, 1, 8, 2, 16, 12
ROT = Rotary("default", 10000.0)
ABS = Positions("absolute")


def T(dims, kind, sizes, facts=(), dtype="float32", yields=""):
    return fe.T(tuple(dims), dtype, kind, tuple(sizes), tuple(facts), yields)


ATT = dict(q=T(("batch", "tokens", "heads", "head_dim"), "query", (B, L, HQ, D)),
           k=T(("batch", "tokens", "kv_heads", "head_dim"), "key", (B, L, HKV, D)),
           v=T(("batch", "tokens", "kv_heads", "head_dim"), "value", (B, L, HKV, D)),
           keys=T(("batch", "kv_heads", "slots", "head_dim"), "key", (B, HKV, S, D), (ABS,)),
           values=T(("batch", "kv_heads", "slots", "head_dim"), "value", (B, HKV, S, D)),
           positions=T(("tokens",), "positions", (L,), (ABS,), "int64"),
           until=T(("batch",), "last_key", (B,), (), "int64"))


def step(q, k, v, keys, values, positions, until, attend_kw=None, share=HQ // HKV):
    """The attention step every attention case shares: rotate, write the new row, attend."""
    q = fe.rope(x=q, positions=positions, rotary=ROT)
    k = fe.rope(x=k, positions=positions, rotary=ROT)
    keys = fe.write(into=keys, src=k, at=positions)
    values = fe.write(into=values, src=v, at=positions)
    return fe.attend(query=q, keys=keys, values=values, until=until, share=share, **(attend_kw or {}))


def outcome(program_fn, types, options=None, bind=None):
    """What happens to one program: (category, detail)."""
    try:
        program = fe.trace(program_fn, options, **types)
    except fe.RoleError as e:
        return "refused at trace", str(e)
    except (TypeError, AttributeError) as e:
        return "cannot be written", f"{type(e).__name__}: {e}"
    if bind is not None:
        try:
            program.bind(**bind)
        except fe.RoleError as e:
            return "refused at bind", str(e)
    if program.notes:
        return "resolved at trace", "; ".join(program.notes)
    return "passes", f"{len(program.graph.nodes)} operations"


def attention_data(pos=5, seed=0, sizes=(B, L, HQ, HKV, D, S)):
    b, l, hq, hkv, d, s = sizes
    g = torch.Generator().manual_seed(seed)
    return dict(q=torch.randn(b, l, hq, d, generator=g), k=torch.randn(b, l, hkv, d, generator=g),
                v=torch.randn(b, l, hkv, d, generator=g), keys=torch.randn(b, hkv, s, d, generator=g),
                values=torch.randn(b, hkv, s, d, generator=g), positions=torch.tensor([pos]),
                until=torch.full((b,), pos))


# --- the cases: (defect program, fixed program, types, options, what binds) -------------------------------------

def rb01():
    """Layout: a backend reorders a q8_0 weight in place (interleaved -> split); a second path reads it as before."""
    il, sp = Layout("q8_0", packing="interleaved"), Layout("q8_0", packing="split")
    fe.REORDERS[(il, sp)] = lambda w: w
    fe.LINEAR_READS.extend([(il, "interleaved reader"), (sp, "split reader")])
    types = dict(x=T(("batch", "embed"), "hidden", (B, 8)), w=T(("out", "embed"), "weight", (4, 8), (il,)))

    def defect(*, x, w):
        fe.reorder(weight=w, to=sp)
        return fe.linear(x=x, weight=w)          # the second path still holds the buffer as it was

    def fixed(*, x, w):
        return fe.linear(x=x, weight=fe.reorder(weight=w, to=sp))

    return defect, fixed, types, None, None


def rb02():
    """Layout (scale format): block scales rounded to powers of two, the fp8 data not requantized to match."""
    f32 = Layout("fp8_block", block=(128, 128), scale_format="fp32")
    e8 = Layout("fp8_block", block=(128, 128), scale_format="ue8m0")
    fe.REORDERS[(f32, e8)] = lambda w: w     # the case's conversion: it rounds the scales, not the data
    fe.LINEAR_READS.extend([(f32, "fp32-scale reader"), (e8, "ue8m0-scale reader")])
    types = dict(x=T(("batch", "embed"), "hidden", (B, 8)), w=T(("out", "embed"), "weight", (4, 8), (f32,)))

    def program(*, x, w):
        return fe.linear(x=x, weight=fe.reorder(weight=w, to=e8))

    return program, program, types, None, None


def rb03():
    """Layout (stride): Q sliced from a fused QKV projection is a strided view; a kernel assumed it packed."""
    e, qn, kn = 16, HQ * D, HKV * D
    types = dict(ATT, x=T(("batch", "tokens", "embed"), "hidden", (B, L, e)),
                 w=T(("qkv_features", "embed"), "weight", (qn + 2 * kn, e), yields="qkv"))
    for name in ("q", "k", "v"):
        types.pop(name)

    def parts(x, w):
        return fe.split_features(x=fe.linear(x=x, weight=w), parts={"query": qn, "key": kn, "value": kn})

    def defect(*, x, w, keys, values, positions, until):
        q, _, _ = parts(x, w)
        return q.view(B, HQ, D)     # a kernel's own code, taking the slice as packed

    def fixed(*, x, w, keys, values, positions, until):
        q, k, v = parts(x, w)
        return step(fe.split_heads(x=q, heads=HQ, name="heads"), fe.split_heads(x=k, heads=HKV, name="kv_heads"),
                    fe.split_heads(x=v, heads=HKV, name="kv_heads"), keys, values, positions, until)

    return defect, fixed, types, None, None


def rb04():
    """Reduction: a value already all-reduced (replicated) is reduced again as if it were a partial sum."""
    types = dict(x=T(("batch", "embed"), "hidden", (B, 8)),
                 w=T(("out", "embed"), "weight", (4, 8), (Reduction("S", dim=1),)))

    def defect(*, x, w):
        return fe.all_reduce(x=fe.all_reduce(x=fe.linear(x=x, weight=w)))

    def fixed(*, x, w):
        return fe.all_reduce(x=fe.linear(x=x, weight=w))

    return defect, fixed, dict(types, x=types["x"].but(facts=(Reduction("R"),))), None, None


def rb05():
    """Positions: chunk-relative query positions compared with absolute key positions."""
    rel = T(("tokens",), "positions", (L,), (Positions("chunk_relative", offset=3),), "int64")

    def defect(**kw):
        return step(kw["q"], kw["k"], kw["v"], kw["keys"], kw["values"], kw["rel"], kw["until"])

    def fixed(**kw):
        return step(kw["q"], kw["k"], kw["v"], kw["keys"], kw["values"], kw["positions"], kw["until"])

    return defect, fixed, dict(ATT, rel=rel), None, None


def rb06():
    """ModelProps (window): a user mask switches the kernel's own window off, and the mask carries no window."""
    props = ModelProps(sliding_window=4)

    def defect(**kw):
        return step(**kw, attend_kw={"props": props, "mask": "a boolean mask without the window"})

    def fixed(**kw):
        return step(**kw, attend_kw={"props": props})

    return defect, fixed, ATT, {"attention": "triton"}, None


def rb07():
    """ModelProps (tie): the config declares tied embeddings; the checkpoint holds a separate head."""
    types = dict(h=T(("batch", "embed"), "hidden", (B, 8)), head=T(("vocab", "embed"), "weight", (10, 8),
                                                                    yields="logits"))

    def program(*, h, head):
        return fe.argmax(logits=fe.linear(x=h, weight=head))

    return program, program, types, None, None


def rb08():
    """ModelProps (softcap): the default SDPA path drops the attention softcap the model declares."""
    def program(**kw):
        return step(**kw, attend_kw={"props": ModelProps(softcap=50.0)})

    return program, program, ATT, {"attention": "flex"}, None


def rb09():
    """Epoch/Assumed: a CUDA graph captured for one length is replayed after the length changed."""
    def program(**kw):
        return step(**kw)

    d = attention_data()
    longer = dict(d, keys=torch.randn(B, HKV, S + 4, D), values=torch.randn(B, HKV, S + 4, D))
    return program, program, ATT, None, {"defect": longer, "fixed": d}


def rb10():
    """Epoch: a mask holds the position counter itself and reads it after the cache update moved it on."""
    def defect(**kw):
        keys = fe.write(into=kw["keys"], src=fe.rope(x=kw["k"], positions=kw["positions"], rotary=ROT),
                        at=kw["positions"])
        values = fe.write(into=kw["values"], src=kw["v"], at=kw["positions"])
        fe.advance(counter=kw["until"])
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        return fe.attend(query=q, keys=keys, values=values, until=kw["until"], share=HQ // HKV)

    def fixed(**kw):
        held = fe.copy(x=kw["until"])
        fe.advance(counter=kw["until"])
        keys = fe.write(into=kw["keys"], src=fe.rope(x=kw["k"], positions=kw["positions"], rotary=ROT),
                        at=kw["positions"])
        values = fe.write(into=kw["values"], src=kw["v"], at=kw["positions"])
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        return fe.attend(query=q, keys=keys, values=values, until=held, share=HQ // HKV)

    return defect, fixed, ATT, None, None


def rb11():
    """Assumed: a program specialized on the warm-up's input (batch 1) is reused for batch 2."""
    one = {k: (v.but(sizes=(1,) + v.sizes[1:]) if v.dims[0] == "batch" else v) for k, v in ATT.items()}

    def program(**kw):
        return step(**kw)

    return program, program, None, None, {"defect": (one, attention_data()), "fixed": (ATT, attention_data())}


def rb16():
    """Quantized: FP8 activations with a per-tensor scale are used as if they were BF16 values."""
    types = dict(x=T(("batch", "embed"), "hidden", (B, 8), (Quantized("float8_e4m3fn", scale=0.25),)),
                 w=T(("out", "embed"), "weight", (4, 8)))

    def program(*, x, w):
        return fe.linear(x=x, weight=w)

    return program, program, types, None, None


def rb17():
    """ModelProps (softcap): SGLang's torch_native backend does not apply the logit cap the model declares."""
    return rb08()


CASES = [("rb-01", rb01), ("rb-02", rb02), ("rb-03", rb03), ("rb-04", rb04), ("rb-05", rb05), ("rb-06", rb06),
         ("rb-07", rb07), ("rb-08", rb08), ("rb-09", rb09), ("rb-10", rb10), ("rb-11", rb11),
         ("rb-16", rb16), ("rb-17", rb17)]
NOT_WRITTEN = {
    "rb-12": ("passes", "a restored session takes its next position from the token count, one more than the KV "
                        "holds: both are lengths, and nothing in a type tells the two counts apart, so the program "
                        "would trace; the library's container contract caught it at run time (M5.5)"),
    "rb-14": ("passes", "beam search reorders the standard cache and leaves a recurrent state in the old order: no "
                        "front-end operation carries a beam order, so the program would trace; the library's container "
                        "contract caught it at run time (M5.5)"),
    "rb-15": ("outside", "a config key under a name the loader does not know is a config file's defect, not a "
                         "program's: the load contract (M3.5) refused it"),
}


# --- week 4's role mistakes (WEEK4_NOTES.md 7.1) ---------------------------------------------------------------

def e5():
    """(name, program, types, options): the eight mistakes, each written the way the mistake would be written."""
    length = T(("batch",), "length", (B,), (), "int64")
    blhd = T(("batch", "tokens", "heads", "head_dim"), "query", (B, HQ, HQ, D))   # tokens == heads in size
    return [
        ("keys and values swapped", lambda **kw: fe.attend(
            query=fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT), keys=kw["values"], values=kw["keys"],
            until=kw["until"], share=4), ATT, None),
        ("heads repeated with repeat instead of repeat_interleave", lambda **kw: fe.attend(
            query=fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT), keys=kw["keys"].repeat(1, 4, 1, 1),
            values=kw["values"], until=kw["until"], share=1), ATT, None),
        ("mask polarity reversed", lambda **kw: step(**kw, attend_kw={"mask": "~allowed"}), ATT, None),
        ("a boolean mask added as if additive", lambda **kw: step(**kw, attend_kw={"additive_mask": "allowed"}), ATT,
         None),
        ("the valid range one short (exclusive for inclusive)", lambda **kw: step(**dict(
            {k: v for k, v in kw.items() if k != "length"}, until=kw["length"])), dict(ATT, length=length), None),
        ("sdpa given a sliding window it ignores", lambda **kw: step(
            **kw, attend_kw={"props": ModelProps(sliding_window=4)}), ATT, {"attention": "triton"}),
        ("[B, L, H, D] read as [B, H, L, D] where L == H", lambda **kw: fe.attend(
            query=fe.rope(x=kw["q"].transpose(1, 2), positions=kw["positions"], rotary=ROT), keys=kw["keys"],
            values=kw["values"], until=kw["until"], share=4), dict(ATT, q=blhd), None),
        ("causal alignment of fewer queries than keys (is_causal)", lambda **kw: fe.attend(
            query=fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT), keys=kw["keys"], values=kw["values"],
            until=kw["until"], share=4),
         dict(ATT, q=T(("batch", "tokens", "heads", "head_dim"), "query", (B, 3, HQ, D)),
              positions=T(("tokens",), "positions", (3,), (ABS,), "int64")), None),
    ]


def main():
    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__, "cases": {}, "e5": {}}
    saved_reads, saved_reorders = list(fe.LINEAR_READS), dict(fe.REORDERS)
    for name, make in CASES:
        try:
            defect, fixed, types, options, bind = make()
            if name == "rb-11":
                (dtypes, ddata), (ftypes, fdata) = bind["defect"], bind["fixed"]
                got = {"defect": outcome(defect, dtypes, options, ddata), "fixed": outcome(fixed, ftypes, options, fdata)}
            else:
                got = {"defect": outcome(defect, types, options, (bind or {}).get("defect")),
                       "fixed": outcome(fixed, types, options, (bind or {}).get("fixed"))}
        finally:
            fe.LINEAR_READS[:] = saved_reads
            fe.REORDERS.clear()
            fe.REORDERS.update(saved_reorders)
        got["what"] = make.__doc__
        res["cases"][name] = got
        print(f"{name}: defect {got['defect'][0]:18} | fixed {got['fixed'][0]:18} | {got['defect'][1][:110]}",
              flush=True)
    for name, (category, why) in NOT_WRITTEN.items():
        res["cases"][name] = {"defect": (category, why), "fixed": (None, "not written"), "what": why}
        print(f"{name}: defect {category:18} | (not written) | {why[:110]}", flush=True)
    for name, program, types, options in e5():
        res["e5"][name] = outcome(program, types, options)
        print(f"e5 {name[:50]:50} {res['e5'][name][0]:18} | {res['e5'][name][1][:90]}", flush=True)
    # the programs that run: the fused-projection case, the named-dims case, against the float32 step
    res["runs"] = checks()
    cats = [c["defect"][0] for c in res["cases"].values()]
    before = ("refused at trace", "resolved at trace", "cannot be written")
    res["count"] = {"cases": len(cats), **{c: cats.count(c) for c in sorted(set(cats))},
                    "prevented before running (trace)": sum(c in before for c in cats),
                    "prevented before running (trace or bind)": sum(c in before + ("refused at bind",) for c in cats),
                    "false refusals of fixed versions": sum(1 for c in res["cases"].values()
                                                            if c["fixed"][0] in ("refused at trace", "refused at bind",
                                                                                 "cannot be written"))}
    e5cats = [v[0] for v in res["e5"].values()]
    res["count_e5"] = {c: e5cats.count(c) for c in sorted(set(e5cats))}
    print(json.dumps(res["count"]), json.dumps(res["count_e5"]), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("wrote", OUT)


def reference(d):
    pos = d["positions"].float()
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    emb = torch.cat([pos[:, None] * inv] * 2, -1)
    cos, sin = emb.cos().view(1, L, 1, D), emb.sin().view(1, L, 1, D)

    def rot(x):
        return x * cos + torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1) * sin

    q, k = rot(d["q"]), rot(d["k"])
    keys, values = d["keys"].clone(), d["values"].clone()
    p = int(d["positions"][0])
    keys[:, :, p], values[:, :, p] = k[:, 0], d["v"][:, 0]
    kk, vv = keys.repeat_interleave(HQ // HKV, 1), values.repeat_interleave(HQ // HKV, 1)
    s = (q.transpose(1, 2) @ kk.transpose(-1, -2)) * D ** -0.5
    s[..., p + 1:] = float("-inf")
    return (torch.softmax(s, -1) @ vv).transpose(1, 2)


def checks():
    """The programs the front end runs in place of a mistake compute what was meant."""
    out = {}
    e, qn, kn = 16, HQ * D, HKV * D
    _, program, types, _, _ = rb03()
    d = attention_data()
    x, w = torch.randn(B, L, e), torch.randn(qn + 2 * kn, e)
    qkv = x @ w.T
    d.update(q=qkv[..., :qn].reshape(B, L, HQ, D), k=qkv[..., qn:qn + kn].reshape(B, L, HKV, D),
             v=qkv[..., qn + kn:].reshape(B, L, HKV, D))
    got = fe.trace(program, **types)(x=x, w=w, keys=d["keys"].clone(), values=d["values"].clone(),
                                     positions=d["positions"], until=d["until"])
    out["rb-03 fused QKV (torch lowering)"] = (got - reference(d)).abs().max().item()
    if torch.cuda.is_available():
        types16 = {k: (v.but(dtype="bfloat16") if v.dtype == "float32" else v) for k, v in types.items()}
        got = fe.trace(program, {"attention": "triton"}, **types16)(
            x=x.cuda().bfloat16(), w=w.cuda().bfloat16(), keys=d["keys"].cuda().bfloat16(),
            values=d["values"].cuda().bfloat16(), positions=d["positions"].cuda(), until=d["until"].cuda())
        out["rb-03 fused QKV (triton lowering, bfloat16)"] = (got.float().cpu() - reference(d)).abs().max().item()
    return out


if __name__ == "__main__":
    main()
