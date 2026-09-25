"""Run one rolebench case and write results/<case>.json (PROTOCOL.md sections 2, 3, 7).

Usage: python run_case.py cases/05_chunk_frame
"""
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.harness import capture, diff, env_info, silent, within  # noqa: E402


def load_case(path):
    spec = importlib.util.spec_from_file_location("case", os.path.join(path, "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_instrumented(path, ctx):
    """instrumented.py defines ARMS = {"W": {"mode": ..., "defect": fn, "fixed": fn}, "D": {...}}.
    Each fn(ctx) runs the defect or fixed path with entail declarations; a RoleError means "flagged"."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
    import entail as rc

    spec = importlib.util.spec_from_file_location("instrumented", path)
    inst = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inst)
    out = {}
    for arm, cfg in inst.ARMS.items():
        rec = {"mode": cfg["mode"], "checks": cfg.get("checks", "")}
        for part in ("defect", "fixed"):
            rc.set_mode(cfg["mode"])
            try:
                cfg[part](ctx)
                rec[part] = {"flagged": False}
            except rc.RoleError as e:
                rec[part] = {"flagged": True, "message": str(e)[:500]}
            except Exception as e:  # any other failure is recorded, not counted as detection
                rec[part] = {"flagged": False, "other_error": f"{type(e).__name__}: {str(e)[:300]}"}
            finally:
                rc.set_mode("off")
        rec["detected"] = rec["defect"]["flagged"] and not rec["fixed"]["flagged"]
        out[arm] = rec
    return out


def main(case_dir):
    case_dir = os.path.abspath(case_dir)
    name = os.path.basename(case_dir)
    mod = load_case(case_dir)
    meta = mod.META
    t0 = time.time()
    ctx = mod.setup()
    outs, notes = {}, {}
    for part in ("reference", "fixed", "defect"):
        outs[part] = None
        with capture() as box:
            outs[part] = getattr(mod, part)(ctx)
        notes[part] = box
    kind = meta.get("compare", "fp32")
    ignore = meta.get("ignore_logs", [])
    res = {"case": name, "meta": meta, "env": env_info(), "notes": notes}
    if all(outs[p] is not None for p in outs):
        d_def = diff(outs["defect"], outs["reference"])
        d_fix = diff(outs["fixed"], outs["reference"])
        if kind == "noise":
            wrong = d_def["max_abs"] >= 10 * max(d_fix["max_abs"], 1e-12)
            right = True  # judged relative to the fixed path's own distance (PROTOCOL.md section 3)
        else:
            wrong = not within(d_def, kind)
            right = within(d_fix, kind)
        quiet = silent(notes["defect"], ignore)
        res.update({"defect_vs_reference": d_def, "fixed_vs_reference": d_fix, "defect_wrong": wrong,
                    "fixed_right": right, "defect_silent": quiet, "reproduced": bool(wrong and right and quiet)})
    else:
        res.update({"reproduced": False, "incomplete": [p for p in outs if outs[p] is None]})
    inst_path = os.path.join(case_dir, "instrumented.py")
    if os.path.exists(inst_path):  # detection per arm (PROTOCOL.md section 4): error on defect, none on fixed
        res["detection"] = run_instrumented(inst_path, ctx)
    if hasattr(mod, "probe"):
        probe_out = None
        with capture() as box:
            probe_out = mod.probe(ctx)
        res["probe"] = {"result": probe_out, "notes": box}
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: res.get(k) for k in ("case", "reproduced", "defect_wrong", "fixed_right", "defect_silent",
                                              "seconds")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
