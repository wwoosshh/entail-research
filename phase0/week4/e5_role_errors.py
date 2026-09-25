"""E5: role-error injection. Does the current stack notice when an argument's role is wrong?

Each case is one plausible, single mistake in passing attention arguments. For every case we record, for the
current PyTorch/transformers interface: whether anything raised, and how far the output is from the intended
semantics (fp32 reference). Then we try to express the same mistake through rolec and record what happens:
caught before running, impossible to express, or removed by a fixed language rule.
Intended semantics: query at absolute position p attends key j iff j <= p and j <= last[b]; GQA maps query
head h to key/value head h // group.
"""
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from rolec import Key, Query, RoleCache, RoleError, Share, Until, Value, 주목  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
dt = torch.bfloat16


def reference(q, k, v, q_pos, last, group, window=None):
    B, Hq, Lq, D = q.shape
    kk = k.float().repeat_interleave(group, 1)
    vv = v.float().repeat_interleave(group, 1)
    s = torch.einsum("bhqd,bhkd->bhqk", q.float(), kk) * D ** -0.5
    kv = torch.arange(k.shape[2], device=q.device)
    allowed = (kv[None, :] <= q_pos[:, None])[None, None] & (kv[None, None, None, :] <= last.view(B, 1, 1, 1))
    if window is not None:
        allowed = allowed & (kv[None, None, None, :] > (q_pos[:, None] - window)[None, None])
    return torch.einsum("bhqk,bhkd->bhqd", s.masked_fill(~allowed, float("-inf")).softmax(-1), vv)


def deviation(out, ref):
    o, r = out.float(), ref.float()
    if torch.isnan(o).any():
        return {"nan": True, "max_abs": float("nan"), "rel": float("nan")}
    d = (o - r).abs().max().item()
    return {"nan": False, "max_abs": d, "rel": d / r.abs().max().item()}


def run_raw(fn, ref):
    try:
        return {"raised": False, **deviation(fn(), ref)}
    except Exception as e:  # noqa: BLE001
        return {"raised": True, "error": f"{type(e).__name__}: {str(e)[:160]}"}


def run_role(fn, ref):
    try:
        out = fn()
        return {"outcome": "ran", **deviation(out, ref)}
    except RoleError as e:
        return {"outcome": "caught before running", "error": str(e)[:160]}
    except TypeError as e:
        return {"outcome": "caught before running", "error": f"TypeError: {str(e)[:160]}"}


def main():
    res = []
    # ---- Scenario D: batched decode, GQA 32/8, per-row valid lengths ----
    B, Hq, Hkv, D, Lk = 2, 32, 8, 128, 64
    G = Hq // Hkv
    q = torch.randn(B, Hq, 1, D, device=dev, dtype=dt)
    k = torch.randn(B, Hkv, Lk, D, device=dev, dtype=dt)
    v = torch.randn(B, Hkv, Lk, D, device=dev, dtype=dt)
    last = torch.tensor([40, 50], device=dev)
    qpos = torch.tensor([50], device=dev)
    ref = reference(q, k, v, qpos, last, G)
    kv = torch.arange(Lk, device=dev)
    mask = kv[None, None, None, :] <= last.view(B, 1, 1, 1)          # SDPA bool mask: True = take part
    kr, vr = k.repeat_interleave(G, 1), v.repeat_interleave(G, 1)
    cache = RoleCache(k, v)

    def role_ok():
        K, V = cache.read()
        return 주목(질의=Query(q, qpos), 키=K, 값=V, 까지=Until(last), 함께=Share(G))

    sanity = {"raw_correct_call": deviation(F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask), ref),
              "rolec_correct_call": deviation(role_ok(), ref)}

    def role_swapped():
        K, V = cache.read()
        return 주목(질의=Query(q, qpos), 키=V, 값=K, 까지=Until(last), 함께=Share(G))

    def role_window():
        K, V = cache.read()
        return 주목(질의=Query(q, qpos), 키=K, 값=V, 까지=Until(last), 함께=Share(G), 창=16)

    try:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        module = types.SimpleNamespace(num_key_value_groups=G, is_causal=False, training=False)

        def raw_window():
            out, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=D ** -0.5, sliding_window=16)
            return out.transpose(1, 2)
    except Exception as e:  # noqa: BLE001
        def raw_window(e=e):
            raise RuntimeError(f"could not import transformers sdpa_attention_forward: {e}")

    ref_window = reference(q, k, v, qpos, last, G, window=16)
    cases = [
        ("swap key/value", "키와 값을 바꿔 넘김",
         lambda: F.scaled_dot_product_attention(q, vr, kr, attn_mask=mask), ref, role_swapped),
        ("GQA expand with repeat instead of repeat_interleave", "헤드 공유 방식 착오",
         lambda: F.scaled_dot_product_attention(q, k.repeat(1, G, 1, 1), v.repeat(1, G, 1, 1), attn_mask=mask),
         ref, "impossible to express: sharing is declared as Share(group), the front end maps heads itself"),
        ("mask polarity inverted (MultiheadAttention key_padding_mask convention)", "마스크 참/거짓 뜻 반대",
         lambda: F.scaled_dot_product_attention(q, kr, vr, attn_mask=~mask), ref,
         "impossible to express: there is no mask argument, validity is declared as Until(last)"),
        ("boolean mask passed as additive float mask", "불리언/덧셈 마스크 혼동",
         lambda: F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask.to(dt)), ref,
         "impossible to express: there is no mask argument"),
        ("valid range off by one (exclusive instead of inclusive)", "까지의 포함 여부 착오",
         lambda: F.scaled_dot_product_attention(q, kr, vr, attn_mask=kv[None, None, None, :] < last.view(B, 1, 1, 1)),
         ref, "removed by a language rule: 까지 is inclusive"),
        ("sliding_window argument silently ignored by transformers sdpa_attention_forward", "받지 않는 인자 무시",
         raw_window, ref_window, role_window),
    ]
    for name, ko, raw_fn, rref, role in cases:
        rec = {"case": name, "ko": ko, "raw": run_raw(raw_fn, rref)}
        rec["rolec"] = run_role(role, rref) if callable(role) else {"outcome": role}
        res.append(rec)

    # ---- Scenario L: layout confusion, heads == length (both 8) ----
    H, L, Dm = 8, 8, 64
    qm = torch.randn(1, H, L, Dm, device=dev, dtype=dt)
    km = torch.randn(1, H, L, Dm, device=dev, dtype=dt)
    vm = torch.randn(1, H, L, Dm, device=dev, dtype=dt)
    pos = torch.arange(L, device=dev)
    ref_l = reference(qm, km, vm, pos, torch.tensor([L - 1], device=dev), 1)
    as_blhd = lambda t: t.transpose(1, 2).contiguous()  # noqa: E731  data arrives as [B, L, H, D]
    res.append({"case": "layout [B,L,H,D] read as [B,H,L,D] when L == H", "ko": "차원 순서 착오",
                "raw": run_raw(lambda: F.scaled_dot_product_attention(as_blhd(qm), as_blhd(km), as_blhd(vm),
                                                                      is_causal=True).transpose(1, 2), ref_l),
                "rolec": run_role(lambda: 주목(질의=Query(as_blhd(qm), pos, dims=("B", "L", "H", "D")),
                                              키=Key(as_blhd(km), dims=("B", "L", "H", "D")),
                                              값=Value(as_blhd(vm), dims=("B", "L", "H", "D")),
                                              까지=Until(L - 1)), ref_l)})

    # ---- Scenario C: chunked prefill, 4 new queries at positions 60..63 over a 64-long cache ----
    Lq = 4
    qc = torch.randn(1, Hq, Lq, D, device=dev, dtype=dt)
    kc = torch.randn(1, Hkv, Lk, D, device=dev, dtype=dt)
    vc = torch.randn(1, Hkv, Lk, D, device=dev, dtype=dt)
    qpos_c = torch.arange(Lk - Lq, Lk, device=dev)
    ref_c = reference(qc, kc, vc, qpos_c, torch.tensor([Lk - 1], device=dev), G)
    cache_c = RoleCache(kc, vc)

    def role_chunk():
        K, V = cache_c.read()
        return 주목(질의=Query(qc, qpos_c), 키=K, 값=V, 까지=Until(Lk - 1), 함께=Share(G))

    res.append({"case": "is_causal with fewer queries than keys (top-left vs bottom-right alignment)",
                "ko": "인과 마스크 정렬 기준 착오",
                "raw": run_raw(lambda: F.scaled_dot_product_attention(qc, kc, vc, is_causal=True, enable_gqa=True),
                               ref_c),
                "rolec": {**run_role(role_chunk, ref_c),
                          "note": "alignment derived from declared absolute positions; the ambiguous form does not exist"}})

    out = {"sanity": sanity, "cases": res}
    print(json.dumps(sanity, ensure_ascii=False))
    print(f"\n{'case':74s} | raw: raised? max|dev| rel     | rolec")
    for r in res:
        raw = r["raw"]
        rs = ("RAISED " + raw["error"][:40]) if raw["raised"] else \
            f"silent  {raw['max_abs']:7.3f}  {raw['rel']*100:6.1f}%" + ("  NaN" if raw.get("nan") else "")
        ro = r["rolec"]["outcome"]
        if ro == "ran":
            ro = f"ran correctly (max|dev| {r['rolec']['max_abs']:.4f})"
        print(f"{r['case'][:74]:74s} | {rs:30s} | {ro[:90]}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "e5_role_errors.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
