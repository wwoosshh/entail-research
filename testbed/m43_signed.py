"""M4.3: do producer signatures alone block the rolebench cases, with no hand tags? (ROADMAP M4.3, PROBLEMS.md 1)

Each case below is the benchmark case (rolebench/cases/*/case.py) with its values' meaning declared where they are
made: a producer's signature says what its result or its write means (returns, writes, carry, advance), a consumer's
says what it takes (takes, agree). Nobody calls tag(). The case's own functions are signed where the case calls them;
a step the case writes inline (the scale rounding of 02, the QKV slice of 03, the activation quantizer of 16) is the
same computation made a signed function, and the harness checks that it gives the same values.

For each case, in debug mode with the default policy (resolve first):
  defect  must be refused, or resolved with an output that matches the case's reference (its own tolerance)
  fixed   must pass, with an output that matches the reference
Counted from the source, not by hand: hand tags (calls to tag) must be 0; the declaration burden (S6) is the number of
signatures and the lines they span. The old instrumented versions (rolebench/cases/*/instrumented.py) are counted the
same way for comparison.

Two cases are not the installed code path, and say so in the result:
  04  one process standing in for the two gloo ranks (as the old instrumented version did)
  10  the case's mechanism in miniature: the installed transformers path holds the counter inside a mask closure and
      the case resets it with foreach ops, so hooking it is a container boundary (M5.2: buffer epochs checked on the
      host side), not a code signature
Writes testbed/results/m43/SUMMARY.md and SUMMARY.json. Run in ~/venvs/gpu: python testbed/m43_signed.py
"""
import ast
import importlib.util
import inspect
import io
import json
import os
import sys
import textwrap
import time
from contextlib import redirect_stdout

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "entail"))
sys.path.insert(0, os.path.join(ROOT, "rolebench"))
from common.harness import diff, within  # noqa: E402

from entail import core, load  # noqa: E402
from entail.boundaries import PASSES, REPEATS, advance, boundary, carry  # noqa: E402
from entail.contracts import Verdict  # noqa: E402
from entail.core import RoleError, envelopes_of  # noqa: E402
from entail.facts import Epoch, Layout, Positions, Quantized, Reduction  # noqa: E402

CASES = os.path.join(ROOT, "rolebench", "cases")


def load_case(folder):
    spec = importlib.util.spec_from_file_location(f"case_{folder[:2]}", os.path.join(CASES, folder, "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the cases, signed ---------------------------------------------------------------------------------------------

def signed_01(case):
    """A backend reorders a Q8_0 weight in place; a second reader still reads the original layout."""
    interleaved, split = Layout("q8_0", packing="interleaved"), Layout("q8_0", packing="split")
    case.quantize_q8_0 = boundary("quantize_q8_0", returns=interleaved)(case.quantize_q8_0)   # setup calls it
    fresh = boundary("fresh copy of the weight buffer", returns=carry("data"))(lambda *, data: data.clone())

    @boundary("backend reorder (interleaved -> split, in place)", writes={"data": split})
    def reorder(*, data, extra):
        case.reorder_in_place({"data": data, "extra": extra})

    read_interleaved = boundary("second-path dequantize", takes={"buf": interleaved})(
        lambda *, buf: case.read_interleaved(buf))
    read_split = boundary("split reader", takes={"buf": split})(lambda *, buf: case.read_split(buf))

    def reordered(ctx):
        data, extra = fresh(data=ctx["data"]), {}
        reorder(data=data, extra=extra)
        return data, extra

    def defect(ctx):
        data, _ = reordered(ctx)
        return ctx["x"] @ read_interleaved(buf=data).T

    def fixed(ctx):
        data, extra = reordered(ctx)
        return ctx["x"] @ (read_split if extra.get("reordered") else read_interleaved)(buf=data).T

    return case.setup, defect, fixed


def signed_02(case):
    """Block scales rounded up to powers of two (UE8M0) while the FP8 data stays quantized for the fp32 scales."""
    fp32, ue8m0 = Layout("fp8_block", scale_format="fp32"), Layout("fp8_block", scale_format="ue8m0")

    def as_scale(out, bound):   # quantized with a given scale: its format; with scales computed here: fp32
        return fp32 if bound.get("scale") is None else core.facts_of(bound["scale"]).get("Layout")

    case.quantize = boundary("quantize (block FP8)", returns=[as_scale, as_scale])(case.quantize)
    to_pow2 = boundary("round block scales up to powers of two", returns=ue8m0)(
        lambda *, scale: torch.exp2(torch.ceil(torch.log2(scale))))
    gemm = boundary("DeepGEMM UE8M0 kernel", takes={"q": ue8m0, "scale": ue8m0})(
        lambda *, x, q, scale: x @ case.dequant(q, scale).T)

    def setup():
        ctx = case.setup()
        pow2 = to_pow2(scale=ctx["s"])
        assert torch.equal(pow2, ctx["s_pow2"]), "the signed scale rounding must be the case's own"
        ctx["s_pow2"] = pow2
        return ctx

    def defect(ctx):
        return gemm(x=ctx["x"], q=ctx["q"], scale=ctx["s_pow2"])

    def fixed(ctx):
        q2, _ = case.quantize(case.dequant(ctx["q"], ctx["s"]), ctx["s_pow2"])
        return gemm(x=ctx["x"], q=q2, scale=ctx["s_pow2"])

    return setup, defect, fixed


def signed_03(case):
    """Q sliced from a fused QKV buffer (a strided view) handed to a kernel that assumes packed rows."""
    dense, strided = Layout("dense"), Layout("strided")
    split_q = boundary("fused QKV split", returns=strided)(lambda *, qkv: qkv[:, : case.H * case.D])
    kernel_packed = boundary("kernel assuming packed Q rows", takes={"q": dense})(
        lambda *, q, w: case.kernel(q, w, case.H * case.D))
    kernel_strided = boundary("kernel taking the row stride", takes={"q": (dense, strided)})(
        lambda *, q, w: case.kernel(q, w, q.stride(0)))

    def setup():
        ctx = case.setup()
        qkv = ctx["q"]._base   # the fused buffer the case sliced Q from
        assert qkv.shape == (case.T, (case.H + 2 * case.HKV) * case.D)
        q = split_q(qkv=qkv)
        assert torch.equal(q, ctx["q"]) and q.stride() == ctx["q"].stride()
        ctx["q"] = q
        return ctx

    return (setup, lambda ctx: kernel_packed(q=ctx["q"], w=ctx["w"]),
            lambda ctx: kernel_strided(q=ctx["q"], w=ctx["w"]))


def signed_04(case):
    """An all-reduced (replicated) value reduced again as if it were a partial sum. One process stands in for the
    two ranks: each rank's partial sum is computed here and the collective adds them."""
    partial, replicated = Reduction("P"), Reduction("R")
    row_parallel = boundary("row-parallel linear (a partial sum)", returns=partial)(lambda *, x, w: x @ w.T)
    all_reduce = boundary("all_reduce", takes={"y": partial}, returns=replicated)(
        lambda *, y, peers: y + sum(peers))
    reduce_again = boundary("dp_gather_partial (reduces its input)", takes={"y": partial})(
        lambda *, y: y * case.WORLD)
    use_replicated = boundary("consumer of a replicated value", takes={"y": replicated})(lambda *, y: y)
    k = case.K // case.WORLD

    def reduced(ctx):
        parts = [row_parallel(x=ctx["x"][:, r * k:(r + 1) * k], w=ctx["w"][:, r * k:(r + 1) * k])
                 for r in range(case.WORLD)]
        return all_reduce(y=parts[0], peers=parts[1:])

    return case.setup, lambda ctx: reduce_again(y=reduced(ctx)), lambda ctx: use_replicated(y=reduced(ctx))


def signed_05(case):
    """A mask compares chunk-relative query positions with absolute key positions."""
    absolute = Positions("absolute")
    chunk_queries = boundary("chunked-prefill scheduler: query positions inside the chunk",
                             returns=lambda out, b: Positions("chunk_relative", offset=b["offset"]))(
        lambda *, n, offset: torch.arange(n, device="cuda"))
    absolute_queries = boundary("scheduler: absolute query positions", returns=absolute)(
        lambda *, n, offset: torch.arange(n, device="cuda") + offset)
    key_positions = boundary("cache: key positions", returns=absolute)(lambda *, n: torch.arange(n, device="cuda"))

    @boundary("mask builder (compares absolute positions)", takes={"q_pos": absolute, "kv_pos": absolute})
    def attend(*, ctx, q_pos, kv_pos):
        return case._run(ctx, lambda b, h, q, kv: kv_pos[kv] <= q_pos[q])

    def defect(ctx):
        return attend(ctx=ctx, q_pos=chunk_queries(n=case.C, offset=ctx["offset"]), kv_pos=key_positions(n=2 * case.C))

    def fixed(ctx):
        return attend(ctx=ctx, q_pos=absolute_queries(n=case.C, offset=ctx["offset"]),
                      kv_pos=key_positions(n=2 * case.C))

    return case.setup, defect, fixed


def signed_10(case):
    """The case's mechanism in miniature: a mask holds the position counter by reference, the cache update
    increments it in place, and attention evaluates the mask afterwards. Slot p+1 holds stale content."""
    L, D = 64, 32
    counter = boundary("cache position counter", returns=Epoch(0, owner="cache"))(
        lambda *, length: torch.tensor(length))
    snapshot = boundary("snapshot of the query offset", returns=carry("offset"))(lambda *, offset: offset.clone())

    class Mask:   # a mask evaluated when attention runs, reading the offset it holds
        def __init__(self, offset):
            self.offset = offset

        def admits(self, n):
            return torch.arange(n) <= self.offset   # the one query sits at the offset

    build_mask = boundary("mask builder (holds the offset it was given)", returns=carry("offset"))(
        lambda *, offset: Mask(offset))

    @boundary("cache update (writes the new key, increments the counter in place)",
              writes={"counter": advance(), "keys": None, "values": None})
    def update(*, counter, keys, values, k, v):
        keys[int(counter)], values[int(counter)] = k, v
        counter.add_(1)

    @boundary("attention (evaluates the mask)", agree={"Epoch": ("mask", "offset")})
    def attend(*, mask, offset, q, keys, values):
        s = (keys @ q) / D ** 0.5
        s = s.masked_fill(~mask.admits(keys.shape[0]), float("-inf"))
        return torch.softmax(s, 0) @ values

    def setup():
        g = torch.Generator().manual_seed(0)
        keys, values = torch.randn(2 * L, D, generator=g), torch.randn(2 * L, D, generator=g)
        keys[L + 1], values[L + 1] = keys[L - 1], values[L - 1] * 50.0   # stale content in slot p+1
        return {"keys": keys, "values": values, "q": torch.randn(D, generator=g), "k": torch.randn(D, generator=g),
                "v": torch.randn(D, generator=g)}

    def step(ctx, snap):
        keys, values = ctx["keys"].clone(), ctx["values"].clone()
        live = counter(length=L)
        offset = snapshot(offset=live) if snap else live
        mask = build_mask(offset=offset)
        update(counter=live, keys=keys, values=values, k=ctx["k"], v=ctx["v"])
        return attend(mask=mask, offset=mask.offset, q=ctx["q"], keys=keys, values=values)

    def reference(ctx):
        keys, values = ctx["keys"].clone(), ctx["values"].clone()
        keys[L], values[L] = ctx["k"], ctx["v"]
        s = ((keys @ ctx["q"]) / D ** 0.5).masked_fill(torch.arange(2 * L) > L, float("-inf"))
        return torch.softmax(s, 0) @ values

    return setup, lambda ctx: step(ctx, False), lambda ctx: step(ctx, True), reference


def signed_16(case):
    """FP8-quantized activations (with a per-tensor scale) read by a LoRA path that expects unquantized ones."""
    quantize_act = boundary("activation quantizer (FP8, per-tensor scale)",
                            returns=lambda out, b: Quantized("float8_e4m3fn", scale=float(b["scale"])))(
        lambda *, x, scale: (x / scale).to(torch.float8_e4m3fn))
    dequantize = boundary("dequantize", returns=Quantized("float32"))(lambda *, xq, scale: xq.float() * scale)
    lora = boundary("LoRA path (expects unquantized activations)",
                    takes={"x": (Quantized("float32"), Quantized("bfloat16"))})(lambda *, x, a, b: (x.float() @ a) @ b)

    def setup():
        ctx = case.setup()
        xq = quantize_act(x=ctx["x"], scale=ctx["s"])
        assert torch.equal(xq.float(), ctx["xq"].float()), "the signed quantizer must be the case's own"
        ctx["xq"] = xq
        return ctx

    def defect(ctx):
        return case._main(ctx) + lora(x=ctx["xq"], a=ctx["a"], b=ctx["b"])

    def fixed(ctx):
        return case._main(ctx) + lora(x=dequantize(xq=ctx["xq"], scale=ctx["s"]), a=ctx["a"], b=ctx["b"])

    return setup, defect, fixed


RUN = [("01", "01_reorder_layout", signed_01, "refused"), ("02", "02_scale_pow2", signed_02, "refused"),
       ("03", "03_strided_q", signed_03, "resolved"), ("04", "04_double_reduce", signed_04, "refused"),
       ("05", "05_chunk_frame", signed_05, "resolved or refused"),
       ("10", "10_flex_position_ref", signed_10, "refused"), ("16", "16_fp8_as_bf16", signed_16, "resolved or refused")]
NOT_THE_INSTALLED_PATH = {"04": "one process stands in for the two gloo ranks",
                          "10": "the mechanism in miniature; the installed transformers path is a container "
                                "boundary (M5.2)"}


# --- counting what a person writes (S6) ----------------------------------------------------------------------------

def _calls(tree, names):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in names:
                out.append(node)
    return out


def burden_signed(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    sigs = _calls(tree, {"boundary"})
    return {"signatures": len(sigs), "signature_lines": sum(c.end_lineno - c.lineno + 1 for c in sigs),
            "hand_tags": len(_calls(tree, {"tag"}))}


def burden_old(folder):
    path = os.path.join(CASES, folder, "instrumented.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    sigs = _calls(tree, {"boundary"})
    return {"signatures": len(sigs), "signature_lines": sum(c.end_lineno - c.lineno + 1 for c in sigs),
            "hand_tags": len(_calls(tree, {"tag"})), "read_time_checks": len(_calls(tree, {"require"}))}


# --- running -------------------------------------------------------------------------------------------------------

def run_part(fn, ctx):
    """(outcome, output, the decisions that were not a pass, the RoleError text, checks that passed). A code
    boundary records an outcome once per process (REPEATS); it is cleared so each part records its own."""
    REPEATS.clear()
    passes_before = sum(PASSES.values())
    first = len(load.LEDGER.decisions)
    out, error = None, None
    with redirect_stdout(io.StringIO()):
        try:
            out = fn(ctx)
        except RoleError as e:
            error = str(e)
    made = load.LEDGER.decisions[first:]
    if error is not None:
        outcome = "refused" if any(d.verdict is Verdict.REFUSED for d in made) else "stopped"
    elif any(d.verdict is Verdict.RESOLVED for d in made):
        outcome = "resolved"
    elif made:
        outcome = "reported"
    else:
        outcome = "pass"
    return outcome, out, made, error, sum(PASSES.values()) - passes_before


def main():
    core.set_mode("debug")
    rows = {}
    for cid, folder, build, expected in RUN:
        case = load_case(folder)
        parts = build(case)
        setup, defect, fixed = parts[:3]
        reference = parts[3] if len(parts) > 3 else case.reference
        torch.manual_seed(0)
        t0 = time.time()
        ctx = setup()
        ref = reference(ctx)
        row = {"case": folder, "expected_defect": expected, "compare": case.META.get("compare", "fp32"),
               "path": NOT_THE_INSTALLED_PATH.get(cid, "the case's own functions, signed"),
               "burden": burden_signed(build), "old_instrumented": burden_old(folder)}
        for part, fn in (("defect", defect), ("fixed", fixed)):
            outcome, out, made, error, passes = run_part(fn, ctx)
            rec = {"outcome": outcome, "checks_passed": passes, "decisions": [
                {"verdict": d.verdict.value, "name": d.name, "rule": d.rule, "boundary": d.contract.boundary,
                 "resolution": d.resolution, "note": d.note} for d in made]}
            if error:
                rec["error_head"] = error.splitlines()[0][:300]
            if out is not None:
                d = diff(out, ref)
                rec["vs_reference"] = {**d, "within": within(d, row["compare"])}
            row[part] = rec
        row["seconds"] = round(time.time() - t0, 2)
        ok_defect = row["defect"]["outcome"] == "refused" or (
            row["defect"]["outcome"] == "resolved" and row["defect"]["vs_reference"]["within"])
        ok_fixed = row["fixed"]["outcome"] == "pass" and row["fixed"].get("vs_reference", {}).get("within", False)
        row["blocked"] = ok_defect and ok_fixed
        rows[cid] = row
        print(f"{cid}: defect {row['defect']['outcome']}, fixed {row['fixed']['outcome']}, "
              f"blocked={row['blocked']}, signatures {row['burden']['signatures']}, hand tags "
              f"{row['burden']['hand_tags']} (old: {row['old_instrumented']['hand_tags']})", flush=True)
    core.set_mode("off")
    out_dir = os.path.join(RESULTS, "m43")
    os.makedirs(out_dir, exist_ok=True)
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "passes_counted": {f"{b} / {n}": c for (b, n), c in PASSES.items()}, "repeats": len(REPEATS)}
    with open(os.path.join(out_dir, "SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump({"env": env, "cases": rows}, f, ensure_ascii=False, indent=1)
    write_md(rows, env, out_dir)


def write_md(rows, env, out_dir):
    lines = ["# M4.3: rolebench blocked by producer signatures, no hand tags", "",
             "Generated by `testbed/m43_signed.py` (debug mode, default policy: resolve first). "
             f"torch {env['torch']}, {env['gpu']}.", "",
             "| case | path | expected (defect) | defect | defect vs reference | fixed | fixed vs reference | blocked | "
             "signatures (lines) | hand tags | old version: tags, signatures (lines), read-time checks |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def vs(rec):
        v = rec.get("vs_reference")
        return "-" if v is None else f"max {v['max_abs']:.3g}, {'within' if v['within'] else 'OUTSIDE'}"

    for cid, r in rows.items():
        b, o = r["burden"], r["old_instrumented"]
        lines.append(f"| rb-{cid} | {r['path']} | {r['expected_defect']} | {r['defect']['outcome']} | "
                     f"{vs(r['defect'])} | {r['fixed']['outcome']} | {vs(r['fixed'])} | "
                     f"{'yes' if r['blocked'] else 'NO'} | {b['signatures']} ({b['signature_lines']}) | "
                     f"{b['hand_tags']} | {o['hand_tags']}, {o['signatures']} ({o['signature_lines']}), "
                     f"{o['read_time_checks']} |")
    lines += ["", "What each defect path was stopped or repaired by (first line of the decision):", ""]
    for cid, r in rows.items():
        for d in r["defect"]["decisions"]:
            if d["verdict"] != "pass":
                lines.append(f"- rb-{cid}: {d['verdict']} at {d['boundary']}: {d['name']}: {d['rule']}"
                             + (f"; {d['resolution']}" if d.get("resolution") else "")
                             + (f"; {d['note']}" if d.get("note") else ""))
    with open(os.path.join(out_dir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
