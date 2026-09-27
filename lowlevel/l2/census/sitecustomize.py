"""M19 L3.3d census (research tool, not the library): every Triton kernel launch in the process is counted, and the
engine function that launched it the first time is noted (the nearest frame outside triton and torch), so the
functions worth a definition can be chosen by how often normal runs reach them, not by which one had a bug.

Put this directory on PYTHONPATH (it is a sitecustomize) with L3D_CENSUS=<out.json>; the counts are written at exit.
"""
import atexit
import json
import os
import sys

_OUT = os.environ.get("L3D_CENSUS")
_COUNTS = {}
_CALLER = {}


def _caller():
    f = sys._getframe(2)
    while f is not None:
        mod = f.f_globals.get("__name__", "")
        if not (mod.startswith("triton") or mod.startswith("torch") or mod == __name__):
            return f"{mod}:{f.f_code.co_name}"
        f = f.f_back
    return "?"


def _install():
    try:
        from triton.runtime.jit import JITFunction
    except Exception:  # noqa: BLE001
        return
    orig = JITFunction.run

    def run(self, *args, **kwargs):
        if not kwargs.get("warmup"):
            f = getattr(self, "fn", None)
            name = f"{getattr(f, '__module__', '?')}:{getattr(f, '__name__', '?')}"
            _COUNTS[name] = _COUNTS.get(name, 0) + 1
            if name not in _CALLER:
                _CALLER[name] = _caller()
        return orig(self, *args, **kwargs)

    JITFunction.run = run


def _dump():
    if not _OUT:
        return
    path = f"{_OUT}.{os.getpid()}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "counts": _COUNTS, "callers": _CALLER}, fh, indent=1)


class _Finder:
    def find_spec(self, name, path=None, target=None):
        if name != "triton.runtime.jit" or name in sys.modules:
            return None
        import importlib.util

        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _install()

        spec.loader.exec_module = exec_module
        return spec


if _OUT:
    sys.meta_path.insert(0, _Finder())
    atexit.register(_dump)
