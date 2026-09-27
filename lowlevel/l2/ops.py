"""M19 L2/L4: the same generic checks for one operator (a kernel, a fused op, a module or a helper), with no rule about
any particular defect.

An op spec (op_specs.py) says how to call the op and builds deterministic inputs; it may give the op's definition in
plain PyTorch and the forms the op's interface allows (launch configurations, fused or unfused options, where the data
sits). Which form, input or size shows a defect is never part of a spec: the forms cover the interface's range. The
checks, the same for every op:
  definition  the op against its definition
  forms       every allowed form against the definition (against the first form when there is no definition)
  layout      each tensor input replaced by a view with the same values and other strides
  reads       each tensor input perturbed (odd, then even elements): when the definition moves, the op must move too
  magnitude   floating inputs scaled by 10 ... 1e5: where the definition stays finite in the op's output dtype, so
              must the op (and, for an op that matches its definition at scale 1, stay within tolerance of it)
  degenerate  each floating input set to zeros, then to a constant: the op must stay finite or refuse loudly
  poison      one element of a floating input made NaN: flagged when more of the output turns non-finite than the
              definition's does (all of it, when there is no definition)
A check gives pass, fail (a silent difference) or refused (the op raised: loud, not a silent difference), with the
largest relative difference it saw.

  python lowlevel/l2/ops.py <out dir> <spec> [<spec> ...]      (in the venv the specs name; python ops.py list)
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FLOATS = ("float16", "bfloat16", "float32", "float64")
INF = float("inf")


def _vec(t):
    import numpy as np
    import torch

    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.detach().to("cpu", torch.float64).reshape(-1)


def _parts(o):
    return list(o) if isinstance(o, (tuple, list)) else [o]


def rel(a, b):
    """Relative difference of two outputs (each return compared with its counterpart); inf when shapes or the positions
    of non-finite values differ."""
    import torch

    pa, pb = _parts(a), _parts(b)
    if len(pa) != len(pb):
        return INF
    worst = 0.0
    for x, y in zip(pa, pb):
        x, y = _vec(x), _vec(y)
        if x.numel() != y.numel():
            return INF
        fx, fy = torch.isfinite(x), torch.isfinite(y)
        if not torch.equal(fx, fy):
            return INF
        if fx.any():
            n = y[fy].norm().item()
            d = (x[fx] - y[fy]).norm().item()
            worst = max(worst, d / n if n > 0 else d)
    return worst


def nonfinite(o):
    import torch

    v = torch.cat([_vec(x) for x in _parts(o)])
    return float((~torch.isfinite(v)).double().mean()) if v.numel() else 0.0


def _finite_as(want, like):
    """Whether the definition's values are finite in the op's own output dtypes (a float32 op cannot hold a value
    that overflows float32, so such a value is not a defect of the op)."""
    import torch

    for w, o in zip(_parts(want), _parts(like)):
        w = w if isinstance(w, torch.Tensor) else torch.as_tensor(w)
        dt = o.dtype if isinstance(o, torch.Tensor) and o.dtype.is_floating_point else torch.float64
        if not bool(torch.isfinite(w.to(dt)).all()):
            return False
    return True


def _is_float(t):
    return str(t.dtype).replace("torch.", "") in FLOATS


def _strided(t):
    """The same values as t in a view whose last dimension has stride 2."""
    import torch

    buf = torch.empty(*t.shape[:-1], t.shape[-1] * 2, dtype=t.dtype, device=t.device)
    v = buf[..., ::2]
    v.copy_(t)
    return v


def _copy(t):
    """A copy that keeps t's strides (clone() turns a view with gaps into a contiguous tensor)."""
    import torch

    if t.is_contiguous():
        return t.clone()
    buf = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device)
    buf.copy_(t)
    return buf


def _perturbed(t, parity):
    import torch

    t2 = t.clone()
    flat = t2.view(-1)
    if str(t.dtype).startswith("torch.float8"):
        f = flat.float()
        f[parity::2] *= 1.5
        flat.copy_(f.to(t.dtype))
    elif t.dtype.is_floating_point:
        flat[parity::2] *= 1.5
    elif t.dtype == torch.uint8:
        flat[parity::2] ^= 0x88
    elif t.dtype == torch.bool:
        flat[parity::2] = ~flat[parity::2]
    else:
        flat[parity::2] = -flat[parity::2]
    return t2


def run_spec(spec):
    import torch

    tol = spec["tol"]
    skip = spec.get("skip", {})
    indices = set(spec.get("indices", []))
    checks = spec.get("checks", ["definition", "forms", "layout", "reads", "magnitude", "degenerate", "poison"])
    forms = spec.get("forms") or [{}]
    defn = spec.get("definition")
    clone = spec.get("clone", True)
    res = {"name": spec["name"], "about": spec.get("about", ""), "tol": tol, "forms": forms, "checks": {}}

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def call(x, form=None):
        xs = {k: (_copy(v) if clone and isinstance(v, torch.Tensor) else v) for k, v in x.items()}
        try:
            out = spec["call"](xs, **(form if form is not None else forms[0]))
            sync()
            return out, None
        except Exception as e:  # noqa: BLE001 - an op that refuses has told the caller
            return None, f"{type(e).__name__}: {e}"[:300]

    def definition(x, form=None):
        return defn(x, **(form if form is not None else forms[0])) if defn else None

    def tensors(x, pred=lambda k, v: True):
        return [k for k, v in x.items() if isinstance(v, torch.Tensor) and not k.startswith("_") and pred(k, v)]

    def verdict(rows):
        if any(r["status"] == "fail" for r in rows):
            return "fail"
        if rows and all(r["status"] == "refused" for r in rows):
            return "refused"
        return "pass" if rows else "skipped"

    t0 = time.time()
    x0 = spec["inputs"]()
    base, err = call(x0)
    if err:
        res["error"] = err
        return res
    ref = definition(x0)

    if "definition" in checks and ref is not None:
        r = rel(base, ref)
        res["checks"]["definition"] = {"status": "fail" if r > tol else "pass", "rel": r}

    if "forms" in checks and len(forms) > 1:
        rows = []
        for f in forms:
            out, err = call(x0, f)
            if err:
                rows.append({"form": f, "status": "refused", "error": err})
                continue
            want = definition(x0, f) if defn else base
            r = rel(out, want)
            rows.append({"form": f, "status": "fail" if r > tol else "pass", "rel": r})
        res["checks"]["forms"] = {"status": verdict(rows), "rows": rows}

    if "layout" in checks:
        rows = []
        for k in tensors(x0, lambda k, v: v.dim() >= 1 and v.numel() > 1 and k not in skip.get("layout", [])):
            x = dict(x0, **{k: _strided(x0[k])})
            out, err = call(x)
            if err:
                rows.append({"input": k, "status": "refused", "error": err})
                continue
            r = rel(out, base)
            rows.append({"input": k, "status": "fail" if r > tol else "pass", "rel": r})
        res["checks"]["layout"] = {"status": verdict(rows), "rows": rows}

    if "reads" in checks and ref is not None:
        rows = []
        for k in tensors(x0, lambda k, v: k not in indices and k not in skip.get("reads", [])):
            for parity in (1, 0):
                x = dict(x0, **{k: _perturbed(x0[k], parity)})
                d_def = rel(definition(x), ref)
                if not d_def > 10 * tol:
                    continue
                out, err = call(x)
                if err:
                    rows.append({"input": k, "parity": parity, "status": "refused", "error": err})
                    continue
                d_op = rel(out, base)
                rows.append({"input": k, "parity": parity, "status": "fail" if d_op < tol else "pass",
                             "definition_moved": d_def, "op_moved": d_op})
        res["checks"]["reads"] = {"status": verdict(rows), "rows": rows}

    floats = tensors(x0, lambda k, v: _is_float(v) and k not in indices)
    # values at scale are compared only for an op that matches its definition at scale 1; otherwise that mismatch
    # would be counted again here, and only finiteness is judged
    values_too = res["checks"].get("definition", {}).get("status") == "pass"
    if "magnitude" in checks:
        rows = []
        names = [k for k in floats if k not in skip.get("magnitude", [])]
        for e in range(1, 6):
            if not names:
                break
            x = dict(x0, **{k: x0[k] * (10.0 ** e) for k in names})
            want = definition(x)
            out, err = call(x)
            if err:
                rows.append({"scale": 10 ** e, "status": "refused", "error": err})
                continue
            want_finite = _finite_as(want, out) if want is not None else True
            bad = want_finite and nonfinite(out) > 0
            r = rel(out, want) if (values_too and want is not None and want_finite and not bad) else None
            rows.append({"scale": 10 ** e, "status": "fail" if bad or (r is not None and r > tol) else "pass",
                         "op_nonfinite": nonfinite(out), "rel": r})
        res["checks"]["magnitude"] = {"status": verdict(rows), "inputs": names, "rows": rows}

    if "degenerate" in checks:
        rows = []
        for k in [k for k in floats if k not in skip.get("degenerate", [])]:
            for label, fill in (("zeros", 0.0), ("constant", 0.5)):
                x = dict(x0, **{k: torch.full_like(x0[k], fill)})
                out, err = call(x)
                if err:
                    rows.append({"input": k, "fill": label, "status": "refused", "error": err})
                    continue
                rows.append({"input": k, "fill": label, "status": "fail" if nonfinite(out) > 0 else "pass",
                             "op_nonfinite": nonfinite(out)})
        res["checks"]["degenerate"] = {"status": verdict(rows), "rows": rows}

    if "poison" in checks:
        rows = []
        for k in [k for k in floats if k not in skip.get("poison", [])]:
            t = x0[k].clone()
            t.view(-1)[t.numel() // 2] = float("nan")
            x = dict(x0, **{k: t})
            out, err = call(x)
            if err:
                rows.append({"input": k, "status": "refused", "error": err})
                continue
            want = definition(x)
            op_nf, def_nf = nonfinite(out), (nonfinite(want) if want is not None else None)
            bad = op_nf > def_nf + 1e-9 if def_nf is not None else op_nf == 1.0
            rows.append({"input": k, "status": "fail" if bad else "pass", "op_nonfinite": op_nf, "definition_nonfinite": def_nf})
        res["checks"]["poison"] = {"status": verdict(rows), "rows": rows}

    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    import op_specs

    if sys.argv[1] == "list":
        print("\n".join(op_specs.SPECS))
        return
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for name in sys.argv[2:]:
        try:
            res = run_spec(op_specs.SPECS[name]())
        except Exception as e:  # noqa: BLE001 - a spec that cannot even be built is reported, not fatal
            res = {"name": name, "error": f"{type(e).__name__}: {e}"[:400]}
        json.dump(res, open(os.path.join(out_dir, name + ".json"), "w"), indent=1, default=str)
        line = "  ".join(f"{c} {v['status']}" for c, v in res.get("checks", {}).items())
        print(f"{name:<28} {line or res.get('error', '')}")


if __name__ == "__main__":
    main()
