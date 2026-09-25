"""M5.2: rolebench 10 on the installed transformers path (carried over from M4.3), with no hand tags.

The case (rolebench/cases/10_flex_position_ref): a tiny Qwen3 with a StaticCache and flex attention. The mask closes
over the cache's position counter and reads it when attention runs; each layer's update increments the counter in
place first, so the mask admits one slot too many, and that slot holds stale content. The transformers adapter
(entail/adapters/cache_contract.py) only says where the counter is written, handed out, handed to the mask and read;
the rules are the core's (epochs.py).

  off               the case as it is: defect, fixed, reference
  on, resolve       the default policy: the offset is bound to its value where it is handed to the mask
  on, refuse        nothing is bound: the stale read is refused where attention reads the mask
  healthy           the same model, fresh caches, flex attention through generate() with the default cache, and a
                    StaticCache forward: nothing may be refused
Writes testbed/results/m52/rb10.json. Run in ~/venvs/gpu: python testbed/m52_rb10.py
"""
import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "entail"))
sys.path.insert(0, os.path.join(ROOT, "rolebench"))

import torch  # noqa: E402
from common.harness import diff, within  # noqa: E402

from entail import core, epochs, load  # noqa: E402
from entail.adapters import cache_contract  # noqa: E402


def load_case():
    spec = importlib.util.spec_from_file_location("case10", os.path.join(ROOT, "rolebench", "cases",
                                                                         "10_flex_position_ref", "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(fn, ctx, ref, compare):
    first = len(load.LEDGER.decisions)
    out, error = None, None
    try:
        with redirect_stdout(io.StringIO()):
            out = fn(ctx)
    except core.RoleError as e:
        error = str(e).splitlines()[0][:400]
    made = [{"verdict": d.verdict.value, "rule": d.rule, "boundary": d.contract.boundary, "note": d.note[:300],
             "resolution": d.resolution} for d in load.LEDGER.decisions[first:]]
    rec = {"refused": error is not None, "error": error, "decisions": made}
    if out is not None:
        d = diff(out, ref)
        rec["vs_reference"] = {**d, "within": within(d, compare)}
    return rec


def main():
    # installed before anything runs, as entail is in use (the mode then says whether it acts): the case's _step pins
    # whatever type(cache).get_query_offset it read onto StaticCache in its finally clause, so any run before
    # installing - the reference included - would hide the hook on Cache from every later run
    cache_contract.install()
    core.set_mode("off")
    case = load_case()
    ctx = case.setup()
    ref = case.reference(ctx)
    compare = case.META["compare"]
    res = {"torch": torch.__version__, "compare": compare}
    res["off"] = {p: run(getattr(case, p), ctx, ref, compare) for p in ("defect", "fixed")}

    for policy in ("resolve", "refuse"):
        core.set_mode("load")
        core.set_policy(policy)
        cache_contract.reset()
        res[f"on_{policy}"] = {p: run(getattr(case, p), ctx, ref, compare) for p in ("defect", "fixed")}
        res[f"on_{policy}"]["counts"] = {b: epochs.stats(b) for b in (cache_contract.MASK_BUILDER,
                                                                     cache_contract.FLEX)}
    core.set_policy("resolve")

    # healthy: fresh caches; nothing may be refused
    from transformers import StaticCache

    model, tok = ctx["model"], ctx["tok"]
    cache_contract.reset()
    healthy = {}
    for impl in ("flex_attention", "sdpa"):
        model.set_attn_implementation(impl)
        first = len(load.LEDGER.decisions)
        try:
            with torch.no_grad(), redirect_stdout(io.StringIO()):
                out = model.generate(tok.repeat(1, 8), max_new_tokens=8, do_sample=False)
                cache = StaticCache(config=model.config, max_cache_len=64)
                model(tok.repeat(1, 8), past_key_values=cache, use_cache=True)
                logits = model(tok, past_key_values=cache, use_cache=True).logits[:, -1].float()
            healthy[impl] = {"refused": False, "tokens": out[0].tolist(), "last_logits": logits}
        except core.RoleError as e:
            healthy[impl] = {"refused": True, "error": str(e)[:300]}
        healthy[impl]["decisions"] = [f"{d.verdict.value}: {d.rule}" for d in load.LEDGER.decisions[first:]]
    if not healthy["flex_attention"]["refused"] and not healthy["sdpa"]["refused"]:
        d = diff(healthy["flex_attention"].pop("last_logits"), healthy["sdpa"].pop("last_logits"))
        healthy["static_flex_vs_sdpa"] = {**d, "within": within(d, "loose")}
        healthy["same_generate_tokens"] = healthy["flex_attention"]["tokens"] == healthy["sdpa"]["tokens"]
    res["healthy"] = healthy
    res["healthy_counts"] = {b: epochs.stats(b) for b in (cache_contract.MASK_BUILDER, cache_contract.FLEX)}
    core.set_mode("off")
    cache_contract.uninstall()

    out = os.path.join(RESULTS, "m52", "rb10.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    for k in ("off", "on_resolve", "on_refuse"):
        for p in ("defect", "fixed"):
            r = res[k][p]
            v = r.get("vs_reference", {})
            print(f"{k:11s} {p:6s} refused={r['refused']} within={v.get('within')} max={v.get('max_abs')} "
                  f"decisions={[x['verdict'] + ':' + x['rule'][:40] for x in r['decisions']]}")
    print("healthy:", json.dumps({k: v for k, v in healthy.items() if k not in ("flex_attention", "sdpa")}),
          {k: (v.get("refused"), v.get("decisions")) for k, v in healthy.items() if k in ("flex_attention", "sdpa")})
    print("wrote", out)


if __name__ == "__main__":
    main()
