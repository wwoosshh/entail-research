"""M3.5, S2: the load-time test problems that run without a serving engine (testbed/PROBLEMS.md).

  rb-08  real transformers: a tiny Gemma 2 on sdpa; the adapter should route it to eager, and the logits then match
         the float64 reference where sdpa's do not
  rb-15  real transformers: a config key under a name the config class does not know; the load should stop
  rb-07  real transformers: the config declares a tied head, the checkpoint ships its own; the load should stop
  rb-02  mechanism: fp32 block scales read by a kernel that reads every scale as ue8m0 (the case's own table)
  rb-06  mechanism: a custom mask that switches the kernel's window off (the case's own table); routed to the path
         that keeps the window, whose output matches the reference
Each problem runs its defect with entail off (to show the defect is there) and on. Run in ~/venvs/gpu:
    python testbed/m3_problems.py      writes testbed/results/m3/problems.json
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "entail"))
sys.path.insert(0, os.path.join(ROOT, "rolebench"))
from common.harness import diff, within  # noqa: E402

from entail import caps, core, load, observe, sites  # noqa: E402
from entail.adapters import rope_alias, transformers_adapter, transformers_config  # noqa: E402
from entail.policies import Policy  # noqa: E402

CASES = os.path.join(ROOT, "rolebench", "cases")
LOAD = Policy(mode="load")


def case(name):
    spec = importlib.util.spec_from_file_location(f"case_{name}", os.path.join(CASES, name, "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class entail_on:
    """The transformers adapters installed and the mode on, as ENTAIL=load would do in a process."""

    def __enter__(self):
        for a in (rope_alias, transformers_config, transformers_adapter):
            a.install()
        core.set_mode("load")
        core.set_policy("resolve")
        self.n = len(load.LEDGER.decisions)
        return self

    def decisions(self):
        return [(d.contract.boundary, d.contract.consumer, d.verdict.value, d.rule, d.target)
                for d in load.LEDGER.decisions[self.n:]]

    def __exit__(self, *exc):
        core.set_mode("off")
        for a in (transformers_adapter, transformers_config, rope_alias):
            a.uninstall()


def rb08():
    """With ENTAIL=load a process has the hook in place before any model exists, so the model is built with it on."""
    c = case("08_gemma2_softcap")
    ctx_off = c.setup()
    ref = c.reference(ctx_off)
    off = c.defect(ctx_off)                        # the default implementation, entail off
    with entail_on() as on, redirect_stdout(io.StringIO()) as out:
        ctx_on = c.setup()                         # built with the hook in place
        on_ = c.defect(ctx_on)                     # what the defect path asks for: the default it was built with
        used = ctx_on["m32"].config._attn_implementation
        ctx_on["m32"].set_attn_implementation("sdpa")   # asking for sdpa explicitly afterwards
        again = ctx_on["m32"].config._attn_implementation
    d_off, d_on = diff(off, ref), diff(on_, ref)
    return {"kind": "real engine (transformers, tiny Gemma 2)", "default_off": ctx_off["default_impl"],
            "default_on": ctx_on["default_impl"], "used_with_entail": used, "asked_sdpa_later_used": again,
            "defect_off_right": within(d_off, "fp32"), "defect_on_right": within(d_on, "fp32"),
            "max_abs_off": d_off["max_abs"], "max_abs_on": d_on["max_abs"], "decisions": on.decisions(),
            "lines": out.getvalue().splitlines()}


def rb15():
    """The defect folder states the scaling under an unknown key; the fixed folder states it under rope_scaling."""
    import torch
    from transformers import AutoModelForCausalLM

    c = case("15_config_alias")
    ctx = c.setup()
    fixed_dir = tempfile.mkdtemp(prefix="m35_rb15_fixed_")
    for f in os.listdir(ctx["dir"]):
        if f != "config.json":
            os.symlink(os.path.join(ctx["dir"], f), os.path.join(fixed_dir, f))
    cfg = json.load(open(os.path.join(ctx["dir"], "config.json"), encoding="utf-8"))
    cfg["rope_scaling"] = {"rope_type": "linear", "factor": float(cfg.pop("rope_scale"))}
    json.dump(cfg, open(os.path.join(fixed_dir, "config.json"), "w", encoding="utf-8"))
    out = {"kind": "real engine (transformers)"}
    ref = c.reference(ctx)
    for label, d in (("defect", ctx["dir"]), ("fixed", fixed_dir)):
        _, model, _ = sites.check_static(d, "transformers", {"policy": LOAD})
        stopped, right = None, None
        with entail_on() as on, redirect_stdout(io.StringIO()):
            try:
                m = AutoModelForCausalLM.from_pretrained(d).eval()
                with torch.no_grad():
                    right = within(diff(m(ctx["ids"]).logits.float(), ref), "fp32")
            except core.RoleError as e:
                stopped = str(e)[:600]
        out[label] = {"static": [(x.contract.boundary, x.verdict.value, x.rule) for x in model],
                      "runtime_stopped": stopped, "output_right": right, "decisions": on.decisions()}
    return out


def _save_tied(ctx, tie):
    from safetensors.torch import save_file
    from transformers import LlamaConfig

    c7 = case("07_tied_head")
    d = tempfile.mkdtemp(prefix="m35_rb07_")
    LlamaConfig(**c7.CFG, tie_word_embeddings=tie).save_pretrained(d)
    save_file({k: v.contiguous().cpu() for k, v in ctx["state"].items()}, os.path.join(d, "model.safetensors"))
    return d


def rb07():
    c = case("07_tied_head")
    ctx = c.setup()
    out = {"kind": "real engine (transformers), checkpoint written from the case's tensors"}
    from transformers import AutoModelForCausalLM
    for label, tie in (("defect", True), ("fixed", False)):
        d = _save_tied(ctx, tie)
        _, model, _ = sites.check_static(d, "transformers", {"policy": LOAD})
        stopped = None
        with entail_on() as on, redirect_stdout(io.StringIO()):
            try:
                AutoModelForCausalLM.from_pretrained(d)
            except core.RoleError as e:
                stopped = str(e)[:600]
        out[label] = {"static": [(x.contract.boundary, x.verdict.value, x.rule) for x in model],
                      "runtime_stopped": stopped, "decisions": on.decisions()}
    return out


def _fp8_folder(q, s, block):
    import torch
    from safetensors.torch import save_file

    d = tempfile.mkdtemp(prefix="m35_rb02_")
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": block}}, f)
    save_file({"w.weight": q.reshape(q.shape[0] * q.shape[2], -1).contiguous() if q.dim() == 4 else q.contiguous(),
               "w.weight_scale_inv": s.to(torch.float32).contiguous()}, os.path.join(d, "model.safetensors"))
    return d


def rb02():
    c = case("02_scale_pow2")
    ctx = c.setup()
    table = caps.from_rows([{"consumer": "rb02.linear.ue8m0_gemm", "fact": "Layout.scale_format", "honours": False,
                             "reads": "ue8m0", "evidence": "measured", "ref": "rolebench case 02 (the kernel's own "
                             "convention)"}])
    out = {"kind": "mechanism (the case's kernel, declared in a test table)"}
    q2, _ = c.quantize(c.dequant(ctx["q"], ctx["s"]), ctx["s_pow2"])
    for label, q, s in (("defect", ctx["q"], ctx["s"]), ("fixed", q2, ctx["s_pow2"])):
        d = _fp8_folder(c._unblocks(q.float()).to(q.dtype), s, [c.B, c.B])
        seen = observe.scale_format(d)
        r = load.layout("rb02.linear.ue8m0_gemm", load.declared(d), table, seen, LOAD)
        out[label] = {"observed": str(seen.value) if seen else None,
                      "decisions": [(x.contract.boundary, x.verdict.value, x.rule) for x in r]}
    return out


def rb06():
    c = case("06_mask_disables_window")
    ctx = c.setup()
    table = caps.from_rows([
        {"consumer": "rb06.attention.custom_mask", "fact": "ModelProps.sliding_window", "honours": False,
         "evidence": "measured", "ref": "rolebench case 06 (a custom mask replaces the kernel's window)"},
        {"consumer": "rb06.attention.window_in_mask", "fact": "ModelProps.sliding_window", "honours": True,
         "evidence": "measured", "ref": "rolebench case 06 (the fixed path folds the window into the mask)"}],
        {"rb06.attention": ["window_in_mask"]})
    facts = load.declared(None, SimpleNamespace(sliding_window=c.W))
    r = load.attention("rb06", "custom_mask", facts, table, LOAD)
    ref = c.reference(ctx)
    routed = {"window_in_mask": c.fixed, "custom_mask": c.defect}[r[0].target] if r and r[0].target else c.defect
    return {"kind": "mechanism (the case's kernel, declared in a test table)",
            "decisions": [(x.contract.boundary, x.verdict.value, x.rule, x.target) for x in r],
            "defect_right": within(diff(c.defect(ctx), ref), "loose"),
            "routed_right": within(diff(routed(ctx), ref), "loose")}


def main():
    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name, fn in (("rb-08", rb08), ("rb-15", rb15), ("rb-07", rb07), ("rb-02", rb02), ("rb-06", rb06)):
        t0 = time.time()
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001 - a problem that fails to run is recorded, never counted as passing
            import traceback
            res[name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        res[name]["seconds"] = round(time.time() - t0, 1)
        print(name, json.dumps({k: v for k, v in res[name].items() if k not in ("lines", "trace")},
                               ensure_ascii=False, default=str)[:900], flush=True)
    os.makedirs(os.path.join(RESULTS, "m3"), exist_ok=True)
    with open(os.path.join(RESULTS, "m3", "problems.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
