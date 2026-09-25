"""E3: runtime rediscovery. What does it cost to re-derive, every decode step, a fact the program already knows?

Real Qwen3-4B (int4), prompt 512, decode positions ADVANCE every step (48 steps per round), one manual CUDA graph
per variant, interleaved rounds. The attention fact at step p is "keys 0..p are valid".
  sdpa                 transformers default: boolean mask rebuilt from the position counter every step
  triton_decode        hand kernel reading the valid length directly
  flex_declared        FlexAttention kernel; block metadata computed arithmetically inside the graph from the
                       declared bound (until = position). What the role front end lowers to.
  flex_slice           same kernel; one row of a precomputed causal block mask selected inside the graph
                       (the FlexDecoding expert practice; correct only because the programmer knows it is causal)
  flex_rediscover      same kernel; create_block_mask re-evaluates the elementwise predicate every step to
                       rediscover the block structure (the general API), captured inside the graph
  flex_rediscover_host same, but on the host between graph replays (how an eager user calls it)
  hf_flex              transformers' built-in "flex_attention" implementation (what flipping the flag gives)
The four flex_* variants run the same compiled attention kernel and differ only in how each step's block metadata is
obtained, so their step-time differences are the cost of rediscovering a known fact.
"""
import argparse
import os
import statistics
import sys
import time

import torch
from torch.nn.attention.flex_attention import create_block_mask

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2"), os.path.join(PHASE0, "week3")]
import ab_attention as A  # noqa: E402
import bench_decode_attn_swap as S  # noqa: E402
from ab_common import cuda_kernel_names, interleaved, print_timing, save_json, sync  # noqa: E402
from bench_llm_decode import make_static_cache, quantize_int4  # noqa: E402
from common import env_info  # noqa: E402
from graph_ab import capture  # noqa: E402
from numerics_experiment import load  # noqa: E402

BLK = 128
ATTN = ("fmha", "flash", "_split_kernel", "_combine_kernel", "flex", "tem_")


def bm_tensors(bm):
    return [bm.kv_num_blocks, bm.kv_indices, bm.full_kv_num_blocks, bm.full_kv_indices]


def copy_bm(dst_bm, src_bm):
    for dst, src in zip(bm_tensors(dst_bm), bm_tensors(src_bm)):
        if src.numel() == dst.numel():
            dst.copy_(src.reshape(dst.shape))
        elif dst.dim() == 3:
            dst.copy_(src.reshape(1, 1, 1).expand_as(dst))
        else:
            dst.copy_(src.reshape(1, 1, 1, -1)[..., :dst.shape[-1]].expand_as(dst))


def gpu_cost_us(fn, reps=200):
    """Pure GPU time of fn: capture it alone in a CUDA graph, time replays with CUDA events."""
    g, _ = capture(lambda: (fn(), None)[1])
    for _ in range(10):
        g.replay()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1e3 / reps


def host_cost_ms(fn, state, L, steps, n=100):
    ts = []
    for i in range(n):
        state["p"] = L + (i % steps)
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts[10:])


def run_batch(model, B, args):
    L, steps = args.prompt_len, args.steps
    max_len = ((L + steps + 8 + BLK - 1) // BLK) * BLK
    g = torch.Generator(device="cuda").manual_seed(3000 + B)
    ids = torch.randint(0, model.config.vocab_size, (B, L), device="cuda", generator=g)
    S.set_impl(model, "sdpa")
    cache = make_static_cache(model, B, max_len)
    with torch.no_grad():
        lg = model(ids, past_key_values=cache, use_cache=True, return_dict=False)[0]
    tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
    lens = [layer.cumulative_length for layer in cache.layers]
    cl = lens[0]

    def rewind():
        torch._foreach_zero_(lens)
        torch._foreach_add_(lens, L)

    rewind()
    S.VALID.zero_()
    S.VALID[:B].fill_(L + 1)

    def valid_len_mask(b, h, q_idx, kv_idx):
        return kv_idx < S.VALID[b]

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    bm0 = create_block_mask(valid_len_mask, B=B, H=None, Q_LEN=1, KV_LEN=max_len, device="cuda")
    A.FLEX["bm"] = bm0
    nkv = bm0.full_kv_indices.shape[-1]
    full_causal = create_block_mask(causal, B=None, H=None, Q_LEN=max_len, KV_LEN=max_len, device="cuda")
    causal_rows = [t.clone() for t in bm_tensors(full_causal)]
    arange_kv = torch.arange(nkv, device="cuda", dtype=bm0.full_kv_indices.dtype)

    def pos():
        return cl.reshape(-1)[:1]

    def pre_valid():  # valid length for this step = position counter + 1 (the token written now is attended)
        S.VALID[:B].copy_((pos() + 1).to(torch.int32).expand(B))

    def pre_declared():  # block metadata by arithmetic from the declared bound
        pre_valid()
        v = pos() + 1
        full = torch.div(v, BLK, rounding_mode="floor").to(torch.int32)
        part = (v % BLK != 0).to(torch.int32)
        bm0.full_kv_num_blocks.copy_(full.view(1, 1, 1).expand_as(bm0.full_kv_num_blocks))
        bm0.kv_num_blocks.copy_(part.view(1, 1, 1).expand_as(bm0.kv_num_blocks))
        bm0.full_kv_indices.copy_(arange_kv.view(1, 1, 1, -1).expand_as(bm0.full_kv_indices))
        bm0.kv_indices[..., 0].copy_(full.view(1, 1, 1).expand_as(bm0.kv_indices[..., 0]))

    def pre_slice():  # select the causal row of the query's block, by a GPU index
        pre_valid()
        qb = torch.div(pos(), BLK, rounding_mode="floor")
        for dst, src in zip(bm_tensors(bm0), causal_rows):
            dst.copy_(src.index_select(2, qb).expand_as(dst))

    def pre_rediscover():  # evaluate the predicate over all key positions to rediscover the blocks
        pre_valid()
        copy_bm(bm0, create_block_mask(valid_len_mask, B=B, H=None, Q_LEN=1, KV_LEN=max_len, device="cuda"))

    def pre_rediscover_compiled():  # same search, but compiled (the documented advice), so it can live in the graph
        pre_valid()
        copy_bm(bm0, create_block_mask(valid_len_mask, B=B, H=None, Q_LEN=1, KV_LEN=max_len, device="cuda",
                                       _compile=True))

    state = {"p": L}

    def host_rediscover():
        S.VALID[:B].fill_(state["p"] + 1)
        copy_bm(bm0, create_block_mask(valid_len_mask, B=B, H=None, Q_LEN=1, KV_LEN=max_len, device="cuda"))

    # variant -> (attention impl, in-graph preparation, host work before each replay); risky variants last
    variants = {"sdpa": ("sdpa", pre_valid, None),
                "triton_decode": ("triton_decode", pre_valid, None),
                "flex_declared": ("maskmod_decode", pre_declared, None),
                "flex_slice": ("maskmod_decode", pre_slice, None),
                "flex_rediscover": ("maskmod_decode", pre_rediscover, None),
                "flex_rediscover_host": ("maskmod_decode", pre_valid, host_rediscover),
                "flex_rediscover_compiled": ("maskmod_decode", pre_rediscover_compiled, None),
                "hf_flex": ("flex_attention", pre_valid, None)}
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only or k == "sdpa"}

    def decode_one(t):
        return model(t, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    compiled = torch.compile(decode_one)
    rec = {"batch": B, "prompt_len": L, "steps_per_round": steps, "max_len": max_len, "variants": {}}
    graphs = {}
    for name, (impl, pre, host) in variants.items():
        v = {}
        try:
            S.set_impl(model, impl)
            rewind()
            state["p"] = L
            if host:
                host()
            t0 = time.perf_counter()
            with torch.no_grad():
                pre()
                compiled(tok)
            sync()
            v["first_call_s"] = time.perf_counter() - t0
            rewind()

            def run(pre=pre):
                pre()
                return compiled(tok)

            gph, out = capture(run)
            graphs[name] = (gph, out, host)
            names = cuda_kernel_names(lambda gph=gph: (rewind(), gph.replay()))
            v["attention_kernels"] = [n[:80] for n in names if any(s in n.lower() for s in ATTN)][:6]
            v["n_kernel_types"] = len(names)
        except Exception as e:  # noqa: BLE001
            v["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        rec["variants"][name] = v
        print(f"  B={B} {name}: {v}", flush=True)

    # correctness with advancing positions: 6 steps each, compared with sdpa step by step
    logits = {}
    for name, (gph, out, host) in graphs.items():
        rewind()
        state["p"] = L
        seq = []
        for _ in range(6):
            if host:
                host()
            gph.replay()
            sync()
            seq.append(out.float().clone())
            state["p"] += 1
        logits[name] = torch.stack(seq)
        rec["variants"][name]["position_counter_after_6"] = int(cl.reshape(-1)[0].item())
    if "sdpa" in logits:
        ref = logits["sdpa"]
        for name, lgs in logits.items():
            d = (lgs - ref).abs()
            rec["variants"][name].update({"max_abs_logit_diff_vs_sdpa_6steps": d.max().item(),
                                          "mean_abs_logit_diff_vs_sdpa_6steps": d.mean().item(),
                                          "top1_agree_vs_sdpa_6steps":
                                              (lgs.argmax(-1) == ref.argmax(-1)).float().mean().item()})
        rec["logit_scale"] = ref.abs().max().item()
    flex_names = [n for n in logits if n.startswith("flex_")]
    rec["flex_pairwise_bitwise_equal"] = {f"{a}~{b}": bool(torch.equal(logits[a], logits[b]))
                                          for i, a in enumerate(flex_names) for b in flex_names[i + 1:]}
    # do the declared arithmetic and the rediscovered block mask produce the same metadata at each position?
    meta_same = []
    for p in range(L, L + 6):
        torch._foreach_zero_(lens)
        torch._foreach_add_(lens, p)
        pre_declared()
        a = [t.clone() for t in bm_tensors(bm0)]
        pre_rediscover()
        b = [t.clone() for t in bm_tensors(bm0)]
        nfull = int(a[2].reshape(-1)[0])
        meta_same.append({"p": p, "num_blocks_equal": bool(torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])),
                          "used_indices_equal": bool(torch.equal(a[1][..., :int(a[0].max())], b[1][..., :int(a[0].max())])
                                                     and torch.equal(a[3][..., :nfull], b[3][..., :nfull]))})
    rec["declared_vs_rediscovered_metadata"] = meta_same
    rewind()
    print("  flex pairwise bitwise:", rec["flex_pairwise_bitwise_equal"], "| metadata same:",
          all(m["num_blocks_equal"] and m["used_indices_equal"] for m in meta_same), flush=True)
    print("  correctness vs sdpa over 6 advancing steps (max|diff|, top1 agree):",
          {n: (round(rec['variants'][n].get('max_abs_logit_diff_vs_sdpa_6steps', float('nan')), 3),
               rec['variants'][n].get('top1_agree_vs_sdpa_6steps')) for n in logits}, flush=True)

    # cost of each way of obtaining the block metadata, measured alone
    rewind()
    meta = {}
    for label, fn in (("declared arithmetic", pre_declared), ("slice precomputed causal", pre_slice),
                      ("create_block_mask (rediscover)", pre_rediscover),
                      ("create_block_mask compiled (rediscover)", pre_rediscover_compiled)):
        try:
            meta[label] = {"gpu_us_in_graph": gpu_cost_us(fn), "eager_host_ms": host_cost_ms(fn, state, L, steps)}
        except Exception as e:  # noqa: BLE001
            meta[label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    try:
        meta["create_block_mask (rediscover)"]["host_between_replays_ms"] = host_cost_ms(host_rediscover, state,
                                                                                         L, steps)
    except Exception as e:  # noqa: BLE001
        meta["create_block_mask (rediscover)"]["host_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["metadata_cost_alone"] = meta
    print("  metadata cost alone:", {k: {kk: round(x, 3) for kk, x in d.items() if isinstance(x, float)}
                                     for k, d in meta.items()}, flush=True)

    ok = {n: g for n, g in graphs.items()
          if rec["variants"][n].get("max_abs_logit_diff_vs_sdpa_6steps", 99) < 3.0}
    for n in graphs:
        rec["variants"][n]["timed"] = n in ok
    tv = {}
    for name, (gph, out, host) in ok.items():
        def setup():
            rewind()
            state["p"] = L

        def step(gph=gph, host=host):
            if host:
                host()
            gph.replay()
            state["p"] += 1
        tv[name] = (setup, step)
    if "sdpa" in tv and len(tv) >= 2:
        rec["timing"] = interleaved(tv, rounds=args.rounds, k=steps, warmup_s=args.warmup_s, baseline="sdpa")
        print_timing(f"B={B} {args.dtype} (advancing positions)", rec["timing"])
    del compiled, cache, graphs
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--dtype", default="int4")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--warmup-s", type=float, default=2.0)
    ap.add_argument("--only", nargs="*", default=None, help="subset of variants (sdpa always included)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e3_rediscovery.json"))
    args = ap.parse_args()
    for name, val in (("cache_size_limit", 32), ("recompile_limit", 32), ("automatic_dynamic_shapes", False)):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, val)
    print(S.register_impl(), flush=True)
    A.register_maskmod()
    _, model = load()
    if args.dtype == "int4":
        quantize_int4(model)
    res = {"env": env_info(), "args": vars(args), "runs": []}
    for B in args.batches:
        torch._dynamo.reset()
        res["runs"].append(run_batch(model, B, args))
        save_json(args.out, res)
        torch.cuda.empty_cache()
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
