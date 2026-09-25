"""rolec: a minimal role-typed front end for attention, a prototype of the particle-role compiler idea.

Every argument states its role; roles are checked before anything runs; lowering uses the roles directly.

    주목(질의=Query(q, positions), 키=Key(k), 값=Value(v), 까지=Until(last), 함께=Share(group))

Design rules (stand-ins for the user's particle grammar):
  - Role-only binding: keyword-only arguments, no positional form, no **kwargs. An argument whose role the
    operation does not take is an error, never silently dropped.
  - Role types: a Key cannot be passed where a Value is expected. The KV cache hands out Key/Value objects,
    so the role is fixed where the data is created, not where it is used.
  - Named dimensions: tensors carry dim names; the front end permutes to its own layout, so a transposed
    tensor cannot be misread even when two dims have the same size.
  - Facts instead of tables: validity is `Until(last)` (inclusive last position, a language rule), head
    sharing is `Share(group)`, causal alignment comes from the queries' absolute positions. There is no
    boolean mask argument, so mask polarity or additive/boolean confusion cannot be expressed at all.
Lowering: one query token -> BACKEND["decode"]:
  "triton"  week-2 hand kernel (valid length + GQA by index, no mask, no copy)
  "flex"    PyTorch FlexAttention; the block metadata is computed by arithmetic from 까지 (no element-wise search)
several query tokens -> SDPA with an attention pattern derived from the declared positions.
The checks here run in Python at call time; in a compiled language they are compile-time checks.
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "week2")]
from triton_decode_attn import decode_attention  # noqa: E402

LAYOUT = ("B", "H", "L", "D")
BACKEND = {"decode": "triton"}
FLEX_BLOCK = 128
_FLEX = {}


class RoleError(TypeError):
    pass


class _Named:
    def __init__(self, t, dims=LAYOUT):
        self.t, self.dims = t, tuple(dims)

    def standard(self):
        if len(self.dims) != 4 or sorted(self.dims) != sorted(LAYOUT):
            raise RoleError(f"{type(self).__name__}: dims {self.dims} must name exactly {LAYOUT}")
        return self.t.permute(*[self.dims.index(d) for d in LAYOUT])


class Query(_Named):
    def __init__(self, t, positions, dims=LAYOUT):
        super().__init__(t, dims)
        self.positions = positions  # absolute position of each query token, shape [L]


class Key(_Named):
    pass


class Value(_Named):
    pass


class Until:
    """까지: the last key position each batch row may attend to, inclusive by language rule."""

    def __init__(self, last):
        self.last = last


class Share:
    """함께: `group` query heads share one key/value head."""

    def __init__(self, group):
        self.group = int(group)


class RoleCache:
    """A KV cache whose read-out is role-typed at the source."""

    def __init__(self, k, v, dims=LAYOUT):
        self._k, self._v, self._dims = k, v, dims

    def read(self):
        return Key(self._k, self._dims), Value(self._v, self._dims)


def 주목(*, 질의, 키, 값, 까지, 함께=None):
    for role, obj, cls in (("질의", 질의, Query), ("키", 키, Key), ("값", 값, Value), ("까지", 까지, Until)):
        if not isinstance(obj, cls):
            raise RoleError(f"role '{role}' takes {cls.__name__}, got {type(obj).__name__}")
    if 함께 is not None and not isinstance(함께, Share):
        raise RoleError(f"role '함께' takes Share, got {type(함께).__name__}")
    q, k, v = 질의.standard(), 키.standard(), 값.standard()
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    if v.shape != k.shape:
        raise RoleError(f"키 {tuple(k.shape)} and 값 {tuple(v.shape)} must have the same shape")
    group = 함께.group if 함께 is not None else 1
    if Hq != Hkv * group:
        raise RoleError(f"{Hq} query heads cannot share {Hkv} key/value heads in groups of {group}")
    last = torch.as_tensor(까지.last, device=q.device).reshape(-1).expand(B)
    if int(last.max()) >= Lk or int(last.min()) < 0:
        raise RoleError(f"까지 {last.tolist()} lies outside the cache of length {Lk}")
    if Lq == 1 and BACKEND["decode"] == "flex":
        return _flex_decode(q, k, v, last, group)
    if Lq == 1 and (D & (D - 1)) == 0:
        return decode_attention(q.contiguous(), k, v, (last + 1).to(torch.int32))
    pos = torch.as_tensor(질의.positions, device=q.device).reshape(-1)
    kv = torch.arange(Lk, device=q.device)
    allowed = (kv[None, :] <= pos[:, None])[None, None] & (kv[None, None, None, :] <= last.view(B, 1, 1, 1))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=group > 1)


def _flex_decode(q, k, v, last, group):
    """까지 lowered for the general compiler: block metadata by arithmetic, then FlexAttention.

    Keys 0..last[b] are valid, so row b has floor((last+1)/BLK) full blocks and at most one partial block right
    after them. No predicate is evaluated over key positions to find this out.
    """
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    if "fn" not in _FLEX:
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
        _FLEX["bound"] = torch.zeros(256, dtype=torch.int64, device=q.device)
    B, blk, Lk = q.shape[0], FLEX_BLOCK, k.shape[2]
    nkv = (Lk + blk - 1) // blk
    valid = (last + 1).to(torch.int32)
    full = torch.div(valid, blk, rounding_mode="floor").to(torch.int32)
    part = (valid % blk != 0).to(torch.int32)
    ar = torch.arange(nkv, device=q.device, dtype=torch.int32)
    full_idx = ar.view(1, 1, 1, nkv).expand(B, 1, 1, nkv).contiguous()
    part_idx = full_idx.clone()
    part_idx[..., 0] = torch.clamp(full, max=nkv - 1).view(B, 1, 1)
    _FLEX["bound"][:B].copy_(last)  # the declared value, copied at call time: later changes cannot leak in
    bm = BlockMask.from_kv_blocks(part.view(B, 1, 1), part_idx, full.view(B, 1, 1), full_idx,
                                  BLOCK_SIZE=blk, mask_mod=_until_mask, seq_lengths=(1, Lk))
    return _FLEX["fn"](q, k, v, block_mask=bm, enable_gqa=group > 1)


def _until_mask(b, h, q_idx, kv_idx):
    return kv_idx <= _FLEX["bound"][b]
