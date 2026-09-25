"""M7.1: where the cost of operation-level propagation comes from, and what it costs where it now sits.

audits/PROPAGATE.md measured 2.14x (Qwen3-4B, eager, 64-token greedy decode) against the M7 target of at most 2x.
This splits the cost on the same workload:
  intercept            a dispatch mode that only calls each aten operation
  intercept_functions  a torch function mode that only calls each Python-level call (methods, functions, indexing)
  propagate_aten       the propagation rules (propagate.py) applied to every aten operation, input ids tagged
  propagate_untagged   the propagation as entail enters it (Python-level calls), nothing tagged
  propagate            the same with the input ids tagged, as audits/PROPAGATE.md did
Each arm runs REPS times after a warm-up; the median is reported. Outputs are compared token by token with off.

Run (WSL): python testbed/m71_cost_probe.py   -> testbed/results/m71/cost_probe.json
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))

import torch  # noqa: E402
import transformers  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from entail import core, propagate  # noqa: E402
from entail.facts import Positions  # noqa: E402

transformers.utils.logging.disable_progress_bar()
MODEL = os.path.expanduser("~/models/Qwen3-4B")
OUT = os.path.join(HERE, "results", "m71", "cost_probe.json")
N_NEW, REPS = 64, 5


class OnlyIntercept(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class PropagateAten(TorchDispatchMode):
    """The same rules, seen at the aten level (where the 0.3.0 propagation sat)."""

    def __init__(self):
        super().__init__()
        self.rules = propagate.RolePropagation(on_conflict="record")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self.rules.__torch_function__(func, types, args, kwargs)


class OnlyInterceptFunctions(torch.overrides.TorchFunctionMode):
    """The same at the torch function level: calls made from Python (methods, functions, properties)."""
    CALLS = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        OnlyInterceptFunctions.CALLS += 1
        return func(*args, **(kwargs or {}))


def run(model, ids, arm):
    core._FACTS.clear()
    ctx = None
    if arm == "intercept":
        ctx = OnlyIntercept()
    elif arm == "intercept_functions":
        ctx = OnlyInterceptFunctions()
    elif arm in ("propagate", "propagate_untagged"):
        core.set_mode("debug")
        ctx = propagate.RolePropagation(on_conflict="record")
    elif arm == "propagate_aten":
        core.set_mode("debug")
        ctx = PropagateAten()
    if arm in ("propagate", "propagate_aten"):
        core.tag(ids, Positions("absolute"))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        if ctx is None:
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=N_NEW, do_sample=False)
        else:
            with ctx, torch.no_grad():
                out = model.generate(ids, max_new_tokens=N_NEW, do_sample=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, out
    finally:
        core.set_mode("off")
        core._FACTS.clear()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
    run(model, ids, "off")   # warm-up
    arms = ["off", "intercept", "intercept_functions", "propagate_aten", "propagate_untagged", "propagate"]
    times = {a: [] for a in arms}
    outs = {}
    ops = {}
    for _ in range(REPS):
        for a in arms:
            propagate.reset_stats()
            t, out = run(model, ids, a)
            times[a].append(t)
            outs[a] = out
            ops[a] = propagate.stats()["ops"]
    base = statistics.median(times["off"])
    result = {
        "model": "Qwen3-4B bf16, attn_implementation=eager", "new_tokens": N_NEW, "reps": REPS,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "seconds": {a: [round(x, 3) for x in times[a]] for a in arms},
        "median_x_off": {a: round(statistics.median(times[a]) / base, 3) for a in arms},
        "ops_counted_by_propagation": ops,
        "function_calls_per_run": OnlyInterceptFunctions.CALLS // REPS,
        "same_tokens_as_off": {a: bool(torch.equal(outs[a], outs["off"])) for a in arms},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
