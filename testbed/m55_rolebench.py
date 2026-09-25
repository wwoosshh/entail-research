"""M5.5: rolebench 09, 11, 12 and 14 (the container and reuse cases) under the policy that reports and goes on.

Each case is rolebench/cases/NN_*/case.py, unchanged. The harness hooks where an engine adapter would - the case's
graph cache, its compiled function, its session KV writes, its beam states - and only reads there; the rules are the
core's (epochs: assume/reuse and buffer epochs, M5.2; kv_contract, M5.1). Nobody calls tag().
  09  the graph runner (the case's graph cache): a capture records what the graph assumed (valid length, shapes),
      a replay reuses it under the conditions now; remake = drop the cached graph so the runner captures again
  11  torch.compile as the case calls it: when the engine skips the guards that would recheck the specialization,
      the first call records the input shapes and every later call reuses under the shapes now; remake = compile
      again. With guards kept, torch rechecks the assumption itself and nothing is recorded
  12  the case's decode step is where the session KV is written: before writing position p the cache must hold p
      entries (kv_needed). The books count the entries each KV object was written
  14  the case's beam loop with its per-beam states in a container (BeamCache): the selection advances the beam
      order, a state written or reordered is aligned to the order now, a state read for the next step must be. The
      loop is the case's loop with `cache[name] = cache[name][parent]` made the container's reorder, and the harness
      checks that it gives the case's own outputs with entail off
Runs per case: off; on (the default policy); strict (ENTAIL_ON_BROKEN=stop); observe (ENTAIL_POLICY=refuse: nothing
repaired). For each: the decisions made, whether it stopped, and the output against the case's reference.
Writes testbed/results/m55/rolebench.json. Run in ~/venvs/gpu: python testbed/m55_rolebench.py
"""
import importlib.util
import io
import json
import os
import sys
import weakref
from contextlib import redirect_stdout

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "entail"))
sys.path.insert(0, os.path.join(ROOT, "rolebench"))
from common.harness import diff, within  # noqa: E402

from entail import core, epochs, kv_contract, load  # noqa: E402
from entail.kv_contract import KvExtent  # noqa: E402

CASES = os.path.join(ROOT, "rolebench", "cases")
POLICIES = {"off": ("off", "resolve", None), "on": ("load", "resolve", None), "strict": ("load", "resolve", "stop"),
            "observe": ("load", "refuse", None)}


def active():
    return core.mode() in ("load", "debug")


def load_case(folder):
    spec = importlib.util.spec_from_file_location(f"case_{folder[:2]}", os.path.join(CASES, folder, "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- where an adapter hooks, per case ------------------------------------------------------------------------------

def wire_09(case):
    boundary = "reuse:rolebench09.graph_replay"
    orig = case.GraphRunner.__call__

    def call(self, q, k, v, lv):
        if not active():
            return orig(self, q, k, v, lv)
        key = self.key_fn(q, k, v, lv)
        conditions = {"valid_length": int(lv), "q": tuple(q.shape), "k": tuple(k.shape), "v": tuple(v.shape)}
        if key in self.graphs:
            epochs.reuse(boundary, "rolebench09.graph", "the graph runner replays a captured graph", key,
                         remake=lambda: self.graphs.pop(key, None), **conditions)
        captured = key not in self.graphs
        out = orig(self, q, k, v, lv)
        if captured:
            epochs.assume(boundary, key, **conditions)
        return out

    case.GraphRunner.__call__ = call
    return boundary


def wire_11(case):
    boundary = "reuse:rolebench11.compiled"
    real = torch.compile

    def compile_(fn, **kw):
        unguarded = "guard_filter_fn" in (kw.get("options") or {})   # the engine skips the guards that recheck
        state = {"cf": real(fn, **kw), "assumed": False, "remade": False}
        key = f"{getattr(fn, '__name__', 'fn')} compiled ({id(state)})"

        def remake():
            torch._dynamo.reset()
            state["cf"], state["remade"] = real(fn, **kw), True

        def call(*args):
            if not active() or not unguarded:
                return state["cf"](*args)
            conditions = {f"arg{i}": tuple(a.shape) for i, a in enumerate(args) if hasattr(a, "shape")}
            if state["assumed"]:
                epochs.reuse(boundary, "rolebench11.compiled", "a graph compiled without guards is called again", key,
                             remake=remake, **conditions)
            out = state["cf"](*args)
            if not state["assumed"] or state["remade"]:
                epochs.assume(boundary, key, **conditions)
                state["assumed"], state["remade"] = True, False
            return out

        return call

    case.torch = type(sys)("torch_for_case_11")   # the case sees torch with this compile; everything else is torch's
    case.torch.__dict__.update({k: getattr(torch, k) for k in ("_dynamo", "zeros", "randn", "Generator", "compiler")})
    case.torch.compile = compile_
    return boundary


def wire_12(case):
    boundary = "container:rolebench12.session_kv"
    books = weakref.WeakKeyDictionary()   # KV object -> entries written from position 0
    orig = case._decode

    def decode(ctx, kv, tok, p):
        if active():
            kv_contract.check(boundary, "rolebench12.decode", f"session KV, decoding at position {p}",
                              KvExtent(held=books.get(kv, 0), needed=int(p)))
        out = orig(ctx, kv, tok, p)
        books[kv] = max(books.get(kv, 0), int(p) + 1)
        return out

    case._decode = decode
    return boundary


class BeamCache(dict):
    """The per-beam states of one beam search, as a container with its own boundaries (what transformers' Cache
    reorder is): a state read for the next step, a state written, the selection, a state reordered by the parent."""
    boundary = "container:rolebench14.beam_states"

    def __getitem__(self, name):
        value = dict.__getitem__(self, name)
        if active():
            epochs.read(self.boundary, "rolebench14.step", f"beam state '{name}' read for the next step", value)
        return value

    def __setitem__(self, name, value):
        dict.__setitem__(self, name, value)
        if active():
            epochs.live(value, self, "beam order")

    def select(self):
        if active():
            epochs.advance(self, "beam order")

    def reorder(self, name, parent):
        self.__setitem__(name, dict.__getitem__(self, name)[parent])


def beam_14(case, ctx, reorder_names):
    """case._beam with its per-beam states in a BeamCache; the arithmetic is the case's."""
    h0 = torch.zeros(1, case.HID, dtype=torch.float64)
    cache = BeamCache()
    cache["key_cache"], cache["ssm_state"] = h0.repeat(case.BEAMS, 1), h0.repeat(case.BEAMS, 1)
    toks = torch.full((case.BEAMS, 1), ctx["start"])
    scores = torch.tensor([0.0] + [-1e9] * (case.BEAMS - 1), dtype=torch.float64)
    for _ in range(case.STEPS):
        h, logp = case._step(ctx, cache["ssm_state"], toks[:, -1])
        cache["key_cache"] = h.clone()
        cache["ssm_state"] = h
        total = (scores[:, None] + logp).reshape(-1)
        best = total.topk(case.BEAMS).indices
        parent, nxt = best // case.V, best % case.V
        cache.select()
        for name in reorder_names:
            cache.reorder(name, parent)
        toks = torch.cat([toks[parent], nxt[:, None]], 1)
        scores = total[best]
    return torch.cat([toks[0].double(), scores[:1]])


def wire_14(case):
    case.defect = lambda ctx: beam_14(case, ctx, ["key_cache"])
    case.fixed = lambda ctx: beam_14(case, ctx, ["key_cache", "ssm_state"])
    return BeamCache.boundary


CASE_WIRES = {"09_stale_graph": wire_09, "11_warmup_specialization": wire_11, "12_session_restore": wire_12,
              "14_beam_reorder": wire_14}


# --- running -----------------------------------------------------------------------------------------------------

def set_policy(name):
    mode, mismatch, on_broken = POLICIES[name]
    core.set_mode(mode)
    core.set_policy(mismatch)
    if on_broken:
        os.environ["ENTAIL_ON_BROKEN"] = on_broken
    else:
        os.environ.pop("ENTAIL_ON_BROKEN", None)
    epochs.reset()
    kv_contract.reset()


def run(fn, ctx, ref, compare):
    first = len(load.LEDGER.decisions)
    out, error, printed = None, None, io.StringIO()
    try:
        with redirect_stdout(printed):
            out = fn(ctx)
    except core.RoleError as e:
        error = str(e).splitlines()[0][:300]
    made = [{"verdict": d.verdict.value, "rule": d.rule, "boundary": d.contract.boundary,
             "resolution": d.resolution, "note": (d.note or "")[:200]} for d in load.LEDGER.decisions[first:]]
    rec = {"stopped": error is not None, "error": error, "decisions": made}
    if out is not None:
        d = diff(out, ref)
        rec["vs_reference"] = {"max_abs": d["max_abs"], "within": within(d, compare)}
    return rec


def main():
    for k in ("ENTAIL_ON_BROKEN", "ENTAIL_POLICY", "ENTAIL_UNKNOWN", "ENTAIL_FACT_POLICY"):
        os.environ.pop(k, None)
    res = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "cases": {}}
    for folder, wire in CASE_WIRES.items():
        case = load_case(folder)
        if folder.startswith("09") and not torch.cuda.is_available():
            res["cases"][folder] = {"skipped": "needs CUDA"}
            continue
        ctx = case.setup()
        ref = case.reference(ctx)
        plain = {"defect": case.defect(ctx), "fixed": case.fixed(ctx)}   # the case as it is, before any hook
        boundary = wire(case)
        compare = case.META["compare"]
        entry = {"boundary": boundary, "compare": compare}
        if folder.startswith("14"):   # the harness loop must give the case's own outputs
            set_policy("off")
            entry["same_as_case"] = all(torch.equal(getattr(case, p)(ctx), plain[p]) for p in ("defect", "fixed"))
        for name in POLICIES:
            set_policy(name)
            entry[name] = {p: run(getattr(case, p), ctx, ref, compare) for p in ("defect", "fixed")}
            entry[name]["counts"] = epochs.stats(boundary) if boundary.startswith(("reuse", "container:rolebench14")) \
                else kv_contract.stats(boundary)
        res["cases"][folder] = entry
    set_policy("off")
    os.environ.pop("ENTAIL_ON_BROKEN", None)

    out = os.path.join(RESULTS, "m55", "rolebench.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    for folder, entry in res["cases"].items():
        if "skipped" in entry:
            print(folder, "skipped:", entry["skipped"])
            continue
        print(f"== {folder} ({entry['boundary']})" + (f" same_as_case={entry['same_as_case']}"
                                                      if "same_as_case" in entry else ""))
        for name in POLICIES:
            for p in ("defect", "fixed"):
                r = entry[name][p]
                v = r.get("vs_reference", {})
                kinds = [f"{d['verdict']}:{d['rule'][:38]}" for d in r["decisions"]]
                print(f"  {name:8s} {p:6s} stopped={r['stopped']!s:5s} within={v.get('within')!s:5s} "
                      f"max={v.get('max_abs', '-'):<12} {kinds}")
    print("wrote", out)


if __name__ == "__main__":
    main()
