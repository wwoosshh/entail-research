"""E6: one role-typed declaration, two backends, same meaning.

The same call
    주목(질의=Query(q, positions), 키=Key(k), 값=Value(v), 까지=Until(last), 함께=Share(4))
is lowered by rolec to (a) the week-2 hand-written Triton decode kernel and (b) PyTorch FlexAttention with block
metadata computed by arithmetic from 까지. Both are compared with an fp32 reference that spells the meaning out
(boolean mask kv <= last[b], heads repeated per group). Rows are chosen to hit block edges: last = 0, 127, 128, 511,
639 and a mid-block value, so 'inclusive' and the full/partial block split are both exercised.
"""
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import rolec  # noqa: E402
from rolec import Key, Query, RoleCache, Share, Until, 주목  # noqa: E402


def reference(q, k, v, last, group):
    B, Hq, _, D = q.shape
    Lk = k.shape[2]
    kr = k.float().repeat_interleave(group, dim=1)
    vr = v.float().repeat_interleave(group, dim=1)
    allowed = torch.arange(Lk, device=q.device)[None, :] <= last[:, None]
    return F.scaled_dot_product_attention(q.float(), kr, vr, attn_mask=allowed[:, None, None, :])


def gpu_ms(fn, reps=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / reps)
    return statistics.median(ts)


def main():
    torch.manual_seed(0)
    B, Hq, Hkv, D, Lk = 6, 32, 8, 128, 640
    group = Hq // Hkv
    last = torch.tensor([0, 127, 128, 300, 511, 639], device="cuda")
    q = torch.randn(B, Hq, 1, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Lk, D, device="cuda", dtype=torch.bfloat16)
    ref = reference(q, k, v, last, group)
    cache = RoleCache(k, v)
    key, val = cache.read()

    def call():
        return 주목(질의=Query(q, positions=last), 키=key, 값=val, 까지=Until(last), 함께=Share(group))

    res = {"rows_last": last.tolist(), "shape": {"B": B, "Hq": Hq, "Hkv": Hkv, "D": D, "Lk": Lk}, "backends": {}}
    for backend in ("triton", "flex"):
        rolec.BACKEND["decode"] = backend
        out = call().float()
        d = (out - ref).abs()
        res["backends"][backend] = {"max_abs_vs_reference": d.max().item(),
                                    "per_row_max": d.amax(dim=(1, 2, 3)).tolist(),
                                    "gpu_ms_per_call": gpu_ms(call)}
        print(backend, json.dumps(res["backends"][backend]), flush=True)
    rolec.BACKEND["decode"] = "triton"
    a = res["backends"]
    res["same_meaning"] = all(x["max_abs_vs_reference"] < 2e-2 for x in a.values())
    # the role checks do not depend on the backend: a swapped key/value is rejected before lowering
    try:
        rolec.BACKEND["decode"] = "flex"
        주목(질의=Query(q, positions=last), 키=val, 값=key, 까지=Until(last), 함께=Share(group))
        res["swap_rejected_under_flex"] = False
    except rolec.RoleError as e:
        res["swap_rejected_under_flex"] = str(e)
    finally:
        rolec.BACKEND["decode"] = "triton"
    print("same meaning:", res["same_meaning"], "| swap under flex:", res["swap_rejected_under_flex"], flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "e6_one_spec.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
