"""M9.1, S2 (prevention) on the final code: every test problem the batch re-ran (m91_run.sh phase B, results in
testbed/results/m91/rerun, m91/engines, m91/rb17) against what its milestone measured (testbed/results/<milestone>),
which PROBLEMS.md records as meeting the expected verdict.

Compared, not judged: from each results file the verdict-bearing values are taken - the verdict words anywhere
(pass, resolved, refused, broken, unknown), and the values of keys that say what happened (verdict, outcome, target,
the backend used, whether a run stopped or was refused, whether an output was right or within tolerance, whether
two outputs were the same) - with list positions folded, so that a list of decisions compares as the set of
what it holds; an entail message or an error compares by what it says (verdict and boundary, error class).
Numbers (differences, timings) are not compared; they are kept for the report. Each file is then "same" or lists
what differs, and a difference is read by hand against PROBLEMS.md.
Writes testbed/results/m91/s2.json. Run: python testbed/m91_s2.py
"""
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M91 = os.path.join(RES, "m91")
OUT = os.path.join(M91, "s2.json")
WORDS = {"pass", "resolved", "refused", "broken", "unknown"}
KEYS = {"verdict", "outcome", "target", "used_with_entail", "default_on", "default_off", "asked_sdpa_later_used",
        "backend_used", "refused", "stopped", "within", "ok", "blocked", "same_as_case", "same_generate_tokens",
        "runtime_stopped", "status", "resolution"}
PAIRS = [  # (milestone file or folder, re-run file or folder)
    ("m43/SUMMARY.json", "m91/rerun/m43/SUMMARY.json"),
    ("m3/problems.json", "m91/rerun/m3/problems.json"),
    ("m55/rolebench.json", "m91/rerun/m55/rolebench.json"),
    ("m52/rb10.json", "m91/rerun/m52/rb10.json"),
    ("m3/rb17", "m91/rb17"),
    ("m3/fd_softcap_sglang_torch_native.json", "m91/engines/fd_softcap_sglang_torch_native.json"),
    ("m3/fd_softcap_sglang_torch_native.record.jsonl", "m91/engines/fd_softcap_sglang_torch_native.record.jsonl"),
    ("m3/fd_softcap_sglang_flex_attention.json", "m91/engines/fd_softcap_sglang_flex_attention.json"),
    ("m3/fd_softcap_sglang_flex_attention.record.jsonl", "m91/engines/fd_softcap_sglang_flex_attention.record.jsonl"),
    ("m3/fd_softcap_transformers_paged.json", "m91/engines/fd_softcap_transformers_paged.json"),
    ("m3/fd_softcap_transformers_paged.record.jsonl", "m91/engines/fd_softcap_transformers_paged.record.jsonl"),
    ("m3/fd_shift.json", "m91/engines/fd_shift.json"),
    ("m3/fd_shift.record.jsonl", "m91/engines/fd_shift.record.jsonl"),
    ("m3/rope_user.json", "m91/engines/rope_user.json"),
    ("m3/rope_user.record.jsonl", "m91/engines/rope_user.record.jsonl"),
    ("m51", "m91/rerun/m51"),
    ("m42", "m91/rerun/m42"),
    ("m63", "m91/rerun/m63"),
    ("m55/l05.json", "m91/rerun/m55/l05.json"),
    ("m53", "m91/rerun/m53"),
    ("m54", "m91/rerun/m54"),
]
for n in ("market_bug_off", "market_bug_on", "market_bug_strict", "market_clean_off", "market_clean_on"):
    PAIRS.append((f"m55/{n}.json", f"m91/rerun/m55/{n}.json"))
    if n.endswith(("_on", "_strict")):
        PAIRS.append((f"m55/{n}.jsonl", f"m91/rerun/m55/{n}.jsonl"))


PID_KEY = re.compile(r"^\d+ ")                        # summaries are keyed "<pid> <boundary>"
ENTAIL_SAID = re.compile(r"\[entail\] (\w+) at (\S+?):")
ERROR = re.compile(r"^([A-Za-z_.]*(?:Error|Exception)):")


def value(v):
    """A value as compared: what an entail message or an error says, not its full text (set orders, pids and
    paths inside it change from run to run)."""
    if isinstance(v, str):
        m = ENTAIL_SAID.search(v)
        if m:
            return f"[entail] {m.group(1)} at {m.group(2)}"
        m = ERROR.match(v)
        if m:
            return m.group(1)
    return v


def signature(x, path="", out=None):
    """path -> the set of verdict-bearing values under it (list positions folded, so a list of decisions compares
    as the set of verdicts it holds: a milestone added since recording more passes does not count as a change)."""
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            key = PID_KEY.sub("<pid> ", str(k))
            if isinstance(v, (dict, list)):
                signature(v, f"{path}.{key}", out)
            elif k in KEYS or (isinstance(v, bool) and (k.endswith(("_right", "right", "_same", "same")) or
                                                        k.startswith(("same", "right")))) \
                    or (isinstance(v, str) and v in WORDS):
                out.setdefault(f"{path}.{key}", set()).add(value(v))
    elif isinstance(x, list):
        for v in x:
            if isinstance(v, (dict, list)):
                signature(v, f"{path}[]", out)
            elif isinstance(v, str) and v in WORDS:
                out.setdefault(f"{path}[]", set()).add(v)
    return out


def read(path):
    if path.endswith(".jsonl"):
        rows = []
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "verdict" in r:   # a record: the decisions, by boundary and fact
                rows.append({"verdict": r["verdict"], "at": f"{r['boundary']} {r['name']}",
                             "target": r.get("target")})
        return {"decisions": sorted(rows, key=lambda r: (r["at"], r["verdict"], str(r["target"])))}
    return json.load(open(path, encoding="utf-8"))


def compare(old_path, new_path):
    if not os.path.exists(new_path):
        return {"state": "not re-run"}
    if not os.path.exists(old_path):
        return {"state": "no milestone file"}
    a = {k: sorted(map(str, v)) for k, v in signature(read(old_path)).items()}
    b = {k: sorted(map(str, v)) for k, v in signature(read(new_path)).items()}
    diff = {k: {"milestone": a.get(k), "final": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    return {"state": "same" if not diff else "differs", "values": len(b), "differs": diff}


def files(old, new):
    """(old file, new file) pairs: a folder is walked for .json and .jsonl files the re-run wrote."""
    o, n = os.path.join(RES, old), os.path.join(RES, new)
    if not os.path.isdir(n) and not os.path.isdir(o):
        return [(o, n)]
    out = []
    for root, _, names in os.walk(n):
        if os.sep + "model" in root or "manifests" in root:   # a model copy or manifests, not results
            continue
        for name in sorted(names):
            if name.endswith((".json", ".jsonl")):
                rel = os.path.relpath(os.path.join(root, name), n)
                out.append((os.path.join(o, rel), os.path.join(n, rel)))
    return out


def main():
    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
    for old, new in PAIRS:
        for o, n in files(old, new):
            res["files"][os.path.relpath(n, RES).replace(os.sep, "/")] = compare(o, n)
    counts = {}
    for r in res["files"].values():
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    res["counts"] = counts
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(counts))
    for k, r in res["files"].items():
        if r["state"] != "same":
            print(k, r["state"])
            for p, d in list(r.get("differs", {}).items())[:12]:
                print("   ", p, "milestone", d["milestone"], "final", d["final"])


if __name__ == "__main__":
    main()
