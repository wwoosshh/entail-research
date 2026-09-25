"""M9.1, S4 "off costs nothing": Python's start-up in an environment where entail is installed with its start-up hook
and ENTAIL is unset, against the same environment with the hook removed (entail hook uninstall), alternating in
rounds; medians. Also the hook's own time with ENTAIL unset, from `python -X importtime`.

Usage: python testbed/m91_offstate.py <python of an environment with entail installed>  -> results/m91/offstate.json
(M91_OFFSTATE_OUT names another file: the release commit is measured again after M9.3's changes)
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("M91_OFFSTATE_OUT") or os.path.join(HERE, "results", "m91", "offstate.json")
ROUNDS, PER = 10, 10


def main(py):
    cli = os.path.join(os.path.dirname(py), "entail")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ENTAIL") and k != "PYTHONPATH"}

    def starts(n):
        out = []
        for _ in range(n):
            t = time.perf_counter()
            subprocess.run([py, "-c", "pass"], check=True, env=env)
            out.append(time.perf_counter() - t)
        return out

    def hook(action):
        return subprocess.run([cli, "hook", action], capture_output=True, text=True, env=env).stdout.strip()

    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "python": py, "rounds": ROUNDS, "per_round": PER,
           "status_before": hook("status")}
    starts(5)
    with_hook, without = [], []
    try:
        for r in range(ROUNDS):
            with_hook += starts(PER)
            hook("uninstall")
            without += starts(PER)
            hook("install")
    finally:
        res["status_after"] = hook("install") or hook("status")
    res["start_ms_median"] = {"hook_in_place_entail_unset": round(statistics.median(with_hook) * 1e3, 2),
                              "hook_removed": round(statistics.median(without) * 1e3, 2)}
    res["difference_ms"] = round(res["start_ms_median"]["hook_in_place_entail_unset"]
                                 - res["start_ms_median"]["hook_removed"], 2)
    trace = subprocess.run([py, "-X", "importtime", "-c", "pass"], capture_output=True, text=True, env=env).stderr
    res["modules_imported_with_entail_unset"] = [line.split("|")[-1].strip() for line in trace.splitlines()
                                                 if "entail" in line.split("|")[-1]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
