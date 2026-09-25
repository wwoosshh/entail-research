"""E3b: rediscovering block structure versus declaring it, as the sequence grows (no model).

A packed-document causal mask (several documents in one sequence; each token attends to earlier tokens of its own
document) is the standard FlexAttention case where the mask changes with every batch. The programmer knows the
structure: document boundaries and causality. The general API receives only an elementwise predicate, so
create_block_mask evaluates it at all N*N positions to rediscover which 128x128 blocks are empty, partial or full.
Declared: the same block metadata computed from the boundaries at block granularity, O((N/128)^2) work.
Checked: both give the identical block structure and flex_attention outputs.
Reference: one compiled flex_attention forward (32 heads, head dim 128, bf16) with that mask. In a model the mask
is built once per batch and shared by all layers, so the per-forward share is also reported for 36 layers.
"""
import argparse
import os
import statistics
import sys
import time

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week3")]
from ab_common import save_json  # noqa: E402
from common import env_info  # noqa: E402

BLK = 128


def doc_ids(N, avg_len, seed):
    g = torch.Generator().manual_seed(seed)
    ndocs = max(1, N // avg_len)
    cuts = torch.randperm(N - 1, generator=g)[:ndocs - 1] + 1
    d = torch.zeros(N, dtype=torch.int32)
    d[cuts] = 1
    return d.cumsum(0).to(torch.int32).cuda(), ndocs


def make_mask_mod(doc):
    def mask_mod(b, h, q_idx, kv_idx):
        return (doc[q_idx] == doc[kv_idx]) & (kv_idx <= q_idx)
    return mask_mod


def ordered(dense):
    num = dense.sum(-1, dtype=torch.int32)
    idx = torch.argsort(dense.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
    return num[None, None], idx[None, None]


def declared_dense(doc, N):
    """Compilable part of the declared path: which blocks are partial / full, from block-level document ranges."""
    nb = N // BLK
    d = doc.view(nb, BLK)
    lo, hi = d.amin(1), d.amax(1)
    i = torch.arange(nb, device=doc.device)
    lower = i[None, :] <= i[:, None]
    strict = i[None, :] < i[:, None]
    overlap = (lo[None, :] <= hi[:, None]) & (hi[None, :] >= lo[:, None])
    single = lo == hi
    full = strict & single[:, None] & single[None, :] & (lo[None, :] == lo[:, None])
    return lower & overlap & ~full, full


_DECLARED_C = {}


def declared_block_mask_compiled(doc, N, mask_mod):
    """Same work split as create_block_mask(_compile=True): compiled dense part, eager ordering + BlockMask."""
    if "fn" not in _DECLARED_C:
        _DECLARED_C["fn"] = torch.compile(declared_dense, dynamic=False)
    partial, full = _DECLARED_C["fn"](doc, N)
    kn, ki = ordered(partial)
    fn, fi = ordered(full)
    return BlockMask.from_kv_blocks(kn, ki, fn, fi, BLOCK_SIZE=BLK, mask_mod=mask_mod, seq_lengths=(N, N))


def declared_block_mask(doc, N, mask_mod):
    """Block metadata from the declared facts (document ranges per block + causality), no elementwise search."""
    nb = N // BLK
    d = doc.view(nb, BLK)
    lo, hi = d.amin(1), d.amax(1)
    i = torch.arange(nb, device=doc.device)
    lower = i[None, :] <= i[:, None]  # [q block, kv block]
    strict = i[None, :] < i[:, None]
    overlap = (lo[None, :] <= hi[:, None]) & (hi[None, :] >= lo[:, None])
    single = lo == hi
    full = strict & single[:, None] & single[None, :] & (lo[None, :] == lo[:, None])
    partial = lower & overlap & ~full
    kn, ki = ordered(partial)
    fn, fi = ordered(full)
    return BlockMask.from_kv_blocks(kn, ki, fn, fi, BLOCK_SIZE=BLK, mask_mod=mask_mod, seq_lengths=(N, N))


def dense_of(num, idx):
    num = num.reshape(-1)
    idx = idx.reshape(num.numel(), -1).long()
    keep = torch.arange(idx.shape[-1], device=idx.device)[None, :] < num[:, None]
    out = torch.zeros(num.numel(), idx.shape[-1], dtype=torch.int32, device=idx.device)
    out.scatter_add_(1, idx, keep.int())
    return out > 0


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--avg-doc", type=int, default=1000)
    ap.add_argument("--eager-max", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e3b_mask_scaling.json"))
    args = ap.parse_args()
    torch._dynamo.config.cache_size_limit = 64
    flex = torch.compile(flex_attention, dynamic=False)
    res = {"env": env_info(), "args": vars(args), "rows": []}
    for N in args.lengths:
        row = {"N": N}
        doc, ndocs = doc_ids(N, args.avg_doc, seed=N)
        row["documents"] = ndocs
        mm = make_mask_mod(doc)
        try:
            dec = declared_block_mask(doc, N, mm)
            row["declared_ms"] = timed(lambda: declared_block_mask(doc, N, mm), 30)
            decc = declared_block_mask_compiled(doc, N, mm)
            row["declared_compiled_ms"] = timed(lambda: declared_block_mask_compiled(doc, N, mm), 30)
            row["declared_compiled_same_blocks"] = bool(
                torch.equal(dense_of(dec.kv_num_blocks, dec.kv_indices), dense_of(decc.kv_num_blocks, decc.kv_indices))
                and torch.equal(dense_of(dec.full_kv_num_blocks, dec.full_kv_indices),
                                dense_of(decc.full_kv_num_blocks, decc.full_kv_indices)))
            nb_ = N // BLK
            ar_ = torch.arange(nb_, device="cuda", dtype=torch.int32)
            one = torch.ones(1, 1, nb_, dtype=torch.int32, device="cuda")
            idx = ar_.view(1, 1, 1, nb_).expand(1, 1, nb_, nb_).contiguous()
            row["blockmask_construction_only_ms"] = timed(
                lambda: BlockMask.from_kv_blocks(one, idx, one, idx, BLOCK_SIZE=BLK, mask_mod=mm, seq_lengths=(N, N)), 30)
            t0 = time.perf_counter()
            red = create_block_mask(mm, None, None, N, N, device="cuda", _compile=True)
            torch.cuda.synchronize()
            row["rediscover_compiled_first_call_s"] = time.perf_counter() - t0
            row["rediscover_compiled_ms"] = timed(
                lambda: create_block_mask(mm, None, None, N, N, device="cuda", _compile=True), 20)
            if N <= args.eager_max:
                try:
                    row["rediscover_eager_ms"] = timed(
                        lambda: create_block_mask(mm, None, None, N, N, device="cuda"), 5)
                except torch.OutOfMemoryError:
                    row["rediscover_eager_ms"] = "OOM"
                torch.cuda.empty_cache()
            row["same_partial_blocks"] = bool(torch.equal(dense_of(dec.kv_num_blocks, dec.kv_indices),
                                                          dense_of(red.kv_num_blocks, red.kv_indices)))
            row["same_full_blocks"] = bool(torch.equal(dense_of(dec.full_kv_num_blocks, dec.full_kv_indices),
                                                       dense_of(red.full_kv_num_blocks, red.full_kv_indices)))
            nb = N // BLK
            row["blocks_total"] = nb * (nb + 1) // 2
            row["blocks_partial"] = int(dec.kv_num_blocks.sum())
            row["blocks_full"] = int(dec.full_kv_num_blocks.sum())
            q, k, v = (torch.randn(1, args.heads, N, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            o_dec = flex(q, k, v, block_mask=dec)
            o_red = flex(q, k, v, block_mask=red)
            row["outputs_bitwise_equal"] = bool(torch.equal(o_dec, o_red))
            row["outputs_max_abs_diff"] = (o_dec.float() - o_red.float()).abs().max().item()
            row["flex_attention_fwd_ms"] = timed(lambda: flex(q, k, v, block_mask=dec), 10)
            per_fwd = args.layers * row["flex_attention_fwd_ms"]
            row["rediscover_compiled_share_of_attention_in_forward"] = row["rediscover_compiled_ms"] / per_fwd
            row["declared_share_of_attention_in_forward"] = row["declared_ms"] / per_fwd
            row["declared_compiled_share_of_attention_in_forward"] = row["declared_compiled_ms"] / per_fwd
            del q, k, v, o_dec, o_red, dec, red, decc
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        torch.cuda.empty_cache()
        res["rows"].append(row)
        save_json(args.out, res)
        print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
