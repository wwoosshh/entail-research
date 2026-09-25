"""Collect every sweep result into one table, and apply the filters from PROTOCOL.md section 4 that a script can.

A script can only do the mechanical part: put the rows together, mark the ones that are void, and flag the
combinations already known from earlier measurements. Whether a drop is already reported upstream, or is a
documented limitation, is decided by a person reading the row (and recorded in RESULTS.md).

Run: python sweep/summarize.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Drops measured before the sweep existed, so a sweep row that matches one of these is a reproduction, not a
# new finding (entail/audits/ADAPTER_PILOT.md, sglang_backend_survey).
KNOWN = {
    ("transformers", "sdpa", "attn_logit_softcapping"),
    ("transformers", "paged|eager", "attn_logit_softcapping"),
    ("transformers", "paged|sdpa", "attn_logit_softcapping"),
    ("sglang", "flashinfer", "attn_logit_softcapping"),
    ("sglang", "flex_attention", "attn_logit_softcapping"),
    ("sglang", "torch_native", "attn_logit_softcapping"),
    # read in the code first (block masks are causal-only), measured by the sweep afterwards
    ("sglang", "flex_attention", "sliding_window"),
}


def rows():
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out += data["rows"] if isinstance(data, dict) and "rows" in data else [data]
    return [r for r in out if isinstance(r, dict) and "verdict" in r and not r.get("superseded_by")]


def not_binding(all_rows):
    """(fact, model) groups where every judged backend says "dropped".

    That is what a perturbation which changes nothing looks like: the reference implementations report a drop
    too. Measured twice on final_logit_softcapping (PROTOCOL revision 2), so it is a rule now, not a hunch.
    """
    groups = {}
    for r in all_rows:
        if r["verdict"] in ("dropped", "honoured"):
            groups.setdefault((r["fact"], r["model"]), []).append(r["verdict"])
    return {k for k, v in groups.items() if v and all(x == "dropped" for x in v)}


def main():
    all_rows = rows()
    suspect = not_binding(all_rows)
    if not all_rows:
        print("no results yet")
        return
    width = max(len(r["fact"]) for r in all_rows)
    print(f"{'fact'.ljust(width)}  {'engine':13} {'backend':15} {'model':14} verdict")
    new = []
    for r in sorted(all_rows, key=lambda r: (r["fact"], r["engine"], r["backend"])):
        mark = ""
        if r["verdict"] == "dropped" and (r["fact"], r["model"]) in suspect:
            mark = "  (suspect: nothing honoured this fact, so the value may not bind)"
        elif r["verdict"] == "dropped":
            known = (r["engine"], r["backend"], r["fact"]) in KNOWN
            mark = "  (already measured)" if known else "  <- NEW, needs filtering by hand"
            if not known:
                new.append(r)
        print(f"{r['fact'].ljust(width)}  {r['engine']:13} {r['backend']:15} {r['model']:14} "
              f"{r['verdict']}{mark}")
    counts = {v: sum(1 for r in all_rows if r["verdict"] == v) for v in ("dropped", "honoured", "void")}
    print(f"\n{len(all_rows)} pairs: {counts}")
    print(f"drops that are not already measured: {len(new)}"
          + ("" if not new else "  -> filter each by hand (PROTOCOL.md section 4) before calling it a finding"))
    for r in new:
        print(f"  {r['engine']} {r['backend']} {r['fact']} ({r['model']})")


if __name__ == "__main__":
    main()
