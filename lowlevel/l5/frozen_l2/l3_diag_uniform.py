"""Diagnostic for M19 L3: where the calls come from that a wrapped function sees before any real request (inputs
of one repeated row). Runs the vLLM worker in this process with entail on (the start-up shim must be on PYTHONPATH)
and prints each distinct call site once, with how many such calls it made.

  python lowlevel/l2/l3_diag_uniform.py <spec.json> <out.json> <sites.json>
"""
import collections
import json
import os
import runpy
import sys
import traceback

from entail.adapters import function_reference as fr

_orig = fr._decide
sites = collections.Counter()
first = {}


def _decide(d, name, orig, cut, st):
    r = _orig(d, name, orig, cut, st)
    frames = [f for f in traceback.extract_stack()[:-2] if "site-packages" in f.filename]
    key = " <- ".join(f"{os.path.basename(f.filename)}:{f.name}" for f in reversed(frames[-40:])
                      if os.path.basename(f.filename) not in ("_ops.py", "output_code.py", "runtime_wrappers.py",
                                                              "aot_autograd_result.py", "aot_compile_types.py",
                                                              "standalone_compile.py", "module.py"))
    sites[(r, key)] += 1
    if (r, key) not in first:
        first[(r, key)] = st["tries"]
    return r


fr._decide = _decide
spec, out, dump = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ["vllm_worker.py", spec, out]
try:
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm_worker.py"), run_name="__main__")
finally:
    json.dump([{"decided": r, "calls": n, "first_try": first[(r, k)], "site": k} for (r, k), n in sites.items()],
              open(dump, "w"), indent=1)
    for (r, k), n in sites.items():
        print(f"[diag] decided={r} calls={n} first_try={first[(r, k)]}: {k[:600]}", flush=True)
