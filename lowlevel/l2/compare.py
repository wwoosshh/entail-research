"""L2 comparison: the same greedy request under two execution modes that are meant to mean the same thing.

Two things are compared, with no rule about any particular defect:
  generated tokens   the first step where the chosen token differs, the base's confidence there (the log-prob gap
                     between its top-1 and top-2), and the largest change of the chosen token's probability before it
  prompt log-probs   position by position (teacher forced, so one difference cannot cascade): where the top-1
                     prediction changed and how confident the base was there, and the largest change of the
                     probability of the token that actually follows
A numeric difference between two healthy modes (another kernel, another reduction order) can only flip a near-tie and
move probabilities a little; a lost meaning changes confident predictions and moves probabilities a lot. Drift is
measured on probabilities (a log-prob change on a token of probability 1e-5 means nothing). Log-prob drift is kept
for the record. The thresholds are set from healthy runs.
"""
import math
from typing import Dict, List, Optional


def _top2(d: Dict[str, float]):
    vals = sorted(d.values(), reverse=True)
    top = max(d, key=d.get)
    return top, (vals[0] - vals[1]) if len(vals) > 1 else float("inf")


def _dp(la: float, lb: float) -> float:
    return abs(math.exp(la) - math.exp(lb))


def generated(a: dict, b: dict) -> dict:
    ta, tb = a["tokens"], b["tokens"]
    n = min(len(ta), len(tb))
    first = next((i for i in range(n) if ta[i] != tb[i]), None)
    upto = first if first is not None else n
    drift = pdrift = 0.0
    for i in range(upto):
        tok = str(ta[i])
        if i >= len(a["logprobs"]) or i >= len(b["logprobs"]):
            break
        la, lb = a["logprobs"][i].get(tok), b["logprobs"][i].get(tok)
        if la is not None and lb is not None:
            drift = max(drift, abs(la - lb))
            pdrift = max(pdrift, _dp(la, lb))
    margin = None
    if first is not None and first < len(a["logprobs"]) and a["logprobs"][first]:
        _, margin = _top2(a["logprobs"][first])
    elif first is not None and "seq_logprob" in a and "seq_logprob" in b:
        # beam search keeps no per-step log-probs: the gap between the two chosen sequences' scores stands for
        # the confidence (a tie between beams gives a gap near 0)
        margin = abs(a["seq_logprob"] - b["seq_logprob"])
    return {"first_diff": first, "base_margin_at_diff": margin, "drift": drift, "pdrift": pdrift, "steps": n}


def prompt(a: dict, b: dict) -> Optional[dict]:
    pa, pb = a.get("prompt_logprobs") or [], b.get("prompt_logprobs") or []
    if not pa or not pb or a["prompt_tokens"] != b["prompt_tokens"]:
        return None
    toks = a["prompt_tokens"]
    flips: List[dict] = []
    drift = pdrift = 0.0
    for i in range(1, min(len(pa), len(pb))):
        da, db = pa[i], pb[i]
        if not da or not db:
            continue
        nxt = str(toks[i])
        if nxt in da and nxt in db:
            drift = max(drift, abs(da[nxt] - db[nxt]))
            pdrift = max(pdrift, _dp(da[nxt], db[nxt]))
        ta, ma = _top2(da)
        tb, _ = _top2(db)
        if ta != tb:
            flips.append({"pos": i, "base_margin": ma})
    return {"flips": flips, "max_flip_margin": max((f["base_margin"] for f in flips), default=0.0),
            "drift": drift, "pdrift": pdrift, "positions": min(len(pa), len(pb)) - 1}


def decode_vs_prefill(generated_run: Dict[str, dict], forced: Dict[str, dict],
                      basis: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """Per probe: the generated steps' distributions (the decode path, reading the cache it wrote) against the same
    tokens fed back teacher-forced in one prefill. The forced prefill was made from the tokens of `basis` (the base
    run; the run itself when None), so only the steps before the first token where the two runs part are compared;
    there the contexts are identical, and only confident flips of the top-1 and probability movement can show.
    Returned in pair()'s shape (as a prompt comparison)."""
    out = {}
    for pid, a in generated_run.items():
        f = forced.get(pid)
        if f is None or not a.get("logprobs"):
            continue
        ref = (basis or {}).get(pid, a)["tokens"]
        n = next((i for i in range(min(len(ref), len(a["tokens"]))) if ref[i] != a["tokens"][i]),
                 min(len(ref), len(a["tokens"])))
        n = min(n + 1, len(a["logprobs"]), len(f["steps"]))   # the step where they part is still the same context
        toks = [0] + list(a["tokens"][:n])
        pa = {"prompt_tokens": toks, "prompt_logprobs": [None] + list(a["logprobs"][:n])}
        pb = {"prompt_tokens": toks, "prompt_logprobs": [None] + list(f["steps"][:n])}
        out[pid] = {"generated": {"first_diff": None, "base_margin_at_diff": None, "drift": 0.0, "pdrift": 0.0,
                                  "steps": len(a["tokens"])},
                    "prompt": prompt(pb, pa)}
    return out


def pair(base: Dict[str, dict], other: Dict[str, dict]) -> Dict[str, dict]:
    """Per probe id: the generated and the prompt comparison of `other` against `base`."""
    out = {}
    for pid, a in base.items():
        b = other.get(pid)
        if b is None:
            continue
        out[pid] = {"generated": generated(a, b), "prompt": prompt(a, b)}
    return out


def summary(cmp: Dict[str, dict]) -> dict:
    g = [c["generated"] for c in cmp.values()]
    p = [c["prompt"] for c in cmp.values() if c["prompt"] is not None]
    return {"gen_flip_margin": max(((x["base_margin_at_diff"] or 0.0) for x in g if x["first_diff"] is not None),
                                   default=0.0),
            "prompt_flip_margin": max((x["max_flip_margin"] for x in p), default=0.0),
            "gen_pdrift": max((x["pdrift"] for x in g), default=0.0),
            "prompt_pdrift": max((x["pdrift"] for x in p), default=0.0),
            "gen_drift": max((x["drift"] for x in g), default=0.0),
            "prompt_drift": max((x["drift"] for x in p), default=0.0),
            "first_diff": {pid: c["generated"]["first_diff"] for pid, c in cmp.items()}}


def verdict(cmp: Dict[str, dict], margin: float, pdrift: float) -> dict:
    """differs when some probe changed a confident prediction (base margin above `margin`) or moved the probability
    of a kept token by more than `pdrift`; the first such probe is named."""
    for pid, c in cmp.items():
        g, p = c["generated"], c["prompt"]
        reasons = []
        if g["first_diff"] is not None and (g["base_margin_at_diff"] or 0.0) > margin:
            reasons.append(f"generated token {g['first_diff']} changed where the base was confident "
                           f"(margin {g['base_margin_at_diff']:.2f})")
        if g["pdrift"] > pdrift:
            reasons.append(f"a kept generated token's probability moved by {g['pdrift']:.3f}")
        if p is not None and p["max_flip_margin"] > margin:
            reasons.append(f"{len([f for f in p['flips'] if f['base_margin'] > margin])} prompt positions changed a "
                           f"confident prediction (margin up to {p['max_flip_margin']:.2f})")
        if p is not None and p["pdrift"] > pdrift:
            reasons.append(f"a prompt token's probability moved by {p['pdrift']:.3f}")
        if reasons:
            return {"differs": True, "probe": pid, "why": reasons}
    return {"differs": False, "probe": None, "why": None}
