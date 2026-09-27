"""M19 L3, definitions attached: engine functions held against entail's definitions (entail/definitions.py) with
entail loaded as a user would load it (ENTAIL=load, the start-up shim), one case per process.

  python lowlevel/l2/l3_definitions.py <case> <out.json>

Each case calls the engine function twice on real-sized input (the first call is where entail decides) and once
more unwrapped (the kernel alone), and holds all three outputs against a float64 formula written here, apart from
entail's definition. Cases are the three reproduced defects (vllm#58532, vllm#52576, sglang#21843) and healthy
controls of the same functions; which inputs show a defect is written here, in the measurement, not in entail.
"""
import json
import statistics
import sys
import time

import torch

BOUNDARY = "kernel:definition"


def rel(out, ref):
    o, r = out.double(), ref.double()
    return float((o - r).abs().max() / r.abs().max().clamp_min(1e-30))


def timed(f, n=20):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def capture_on(f, call):
    """Warm up on a side stream and capture one call of f in a CUDA graph (as the engines do); returns the graph and
    the output tensors it writes."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(f)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(f)
    return graph, out


def replayed(f, call, ref):
    """After the decision: capture the wrapped function, replay, and hold the replayed output against the formula."""
    try:
        graph, out = capture_on(f, call)
        graph.replay()
        torch.cuda.synchronize()
        return {"replayed_rel_err": [rel(o, r) for o, r in zip(outputs(out), ref)]}
    except Exception as e:  # noqa: BLE001 - what happened is the result
        return {"error": f"{type(e).__name__}: {e}"[:300]}


# --- vLLM fused_experts (vllm#58532) ---------------------------------------------------------------------------

def moe(kind):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config

    T, E, H, I, K = 256, 8, 512, 256, 2
    torch.manual_seed(0)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    tw = torch.rand(T, K, device="cuda") + 0.5
    ids = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)]).to(torch.int32)
    a1 = a2 = None
    if kind == "bf16":
        w1 = (torch.randn(E, 2 * I, H, device="cuda") * 0.05).to(torch.bfloat16)
        w2 = (torch.randn(E, H, I, device="cuda") * 0.05).to(torch.bfloat16)
        s1 = s2 = None
        qc = None
    else:
        w1 = torch.randint(-8, 8, (E, 2 * I, H), device="cuda", dtype=torch.int8)
        w2 = torch.randint(-8, 8, (E, H, I), device="cuda", dtype=torch.int8)
        if kind in ("int8_channel_static", "int8_channel_token", "int8_channel_static_then_graph"):
            s1 = torch.linspace(0.005, 0.015, E * 2 * I, device="cuda").view(E, 2 * I, 1)
            s2 = torch.linspace(0.005, 0.015, E * H, device="cuda").view(E, H, 1)
        else:
            s1 = torch.linspace(0.005, 0.015, E, device="cuda").view(E, 1, 1)
            s2 = torch.linspace(0.005, 0.015, E, device="cuda").view(E, 1, 1)
        token = kind == "int8_channel_token"
        if not token:
            a1 = a2 = torch.tensor(4.0 / 127, device="cuda")
        qc = int8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, a1_scale=a1, a2_scale=a2,
                                        per_act_token_quant=token)

    def call(f):
        with set_current_vllm_config(VllmConfig()):
            return f(x, w1, w2, tw, ids, quant_config=qc)

    def q8(v, s):
        return torch.clamp(torch.round(v / s), -128, 127) * s

    def formula():
        """float64: per token and slot; activations quantized as the config and scales declare."""
        xd = x.double()
        dq1 = w1.double() * (s1.double() if s1 is not None else 1.0)
        dq2 = w2.double() * (s2.double() if s2 is not None else 1.0)
        if kind == "bf16":
            qa = lambda v: v  # noqa: E731
        elif kind == "int8_channel_token":
            qa = lambda v: q8(v, v.abs().amax(-1, keepdim=True) / 127)  # noqa: E731
        else:
            qa = lambda v: q8(v, a1.double())  # noqa: E731
        out = torch.zeros(T, H, dtype=torch.float64, device="cuda")
        xq = qa(xd)
        for j in range(K):
            e = ids[:, j].long()
            h = torch.einsum("th,tnh->tn", xq, dq1[e])
            act = qa(torch.nn.functional.silu(h[:, :I]) * h[:, I:])
            out += tw[:, j].double().unsqueeze(-1) * torch.einsum("ti,thi->th", act, dq2[e])
        return out

    target = "vllm.model_executor.layers.fused_moe.fused_moe:fused_experts"
    if kind == "int8_channel_static_then_graph":   # repaired first, then captured: the definition cannot be
        return target, fused_experts, call, formula, None, lambda f, ref: replayed(f, call, ref)
    return target, fused_experts, call, formula


# --- vLLM w8a8_triton_block_scaled_mm (vllm#52576) -------------------------------------------------------------

def block(kind):
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    N, K, gn, gk = 512, 1024, 128, 128
    M = 16 if kind == "k256_later" else 256
    torch.manual_seed(0)
    state = {"M": M}

    def inputs(m):
        g = torch.Generator(device="cuda").manual_seed(m)
        return ((torch.randn(m, K, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn),
                torch.rand(m, K // gk, device="cuda", generator=g) * 0.9 + 0.1)

    B = (torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    Bs = torch.rand(N // gn, K // gk, device="cuda") * 0.9 + 0.1
    cfg = lambda bk: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8,  # noqa: E731
                      "num_warps": 4, "num_stages": 3}
    if kind == "k128":
        fp8_utils.get_w8a8_block_fp8_configs = lambda *a, **k: {256: cfg(128)}
    elif kind == "k256":
        fp8_utils.get_w8a8_block_fp8_configs = lambda *a, **k: {256: cfg(256)}
    elif kind == "k256_later":      # a tuned table: small batches get BLOCK_SIZE_K 128, large ones 256
        fp8_utils.get_w8a8_block_fp8_configs = lambda *a, **k: {16: cfg(128), 256: cfg(256)}

    def call(f):
        A, As = inputs(state["M"])
        return f(A, B, As, Bs, [gn, gk], torch.bfloat16)

    def formula():
        A, As = inputs(state["M"])
        a = A.double() * As.double().repeat_interleave(gk, 1)
        b = B.double() * Bs.double().repeat_interleave(gn, 0).repeat_interleave(gk, 1)
        return a @ b.T

    def later():
        state["M"] = 256
    if kind == "k256_warm":         # M19 L3.3a: an engine's warm-ups first (one row over and over, at each size it
        fp8_utils.get_w8a8_block_fp8_configs = lambda *a, **k: {16: cfg(128), 256: cfg(256)}   # will capture),
        for m in (16, 256):                                                  # then the real calls at 256
            A, As = inputs(m)
            fp8_utils.w8a8_triton_block_scaled_mm(A[:1].expand(m, K).contiguous(), B,
                                                  As[:1].expand(m, K // gk).contiguous(), Bs, [gn, gk], torch.bfloat16)
    return ("vllm.model_executor.layers.quantization.utils.fp8_utils:w8a8_triton_block_scaled_mm",
            fp8_utils.w8a8_triton_block_scaled_mm, call, formula, later if kind == "k256_later" else None)


# --- SGLang fused_gdn_gating (sglang#21843) --------------------------------------------------------------------

def gdn(kind):
    from sglang.kernels.ops.attention.fla import fused_gdn_gating as mod

    n, h = 256, 32
    torch.manual_seed(0)
    A_log, dt_bias = torch.randn(h, device="cuda"), torch.randn(h, device="cuda")
    if kind == "contiguous":
        a = torch.randn(n, h, device="cuda").to(torch.bfloat16)
        b = torch.randn(n, h, device="cuda").to(torch.bfloat16)
    elif kind == "rows":            # the engine's layout: two halves of one projection, row stride 2h
        ba = torch.randn(n, 2 * h, device="cuda").to(torch.bfloat16)
        b, a = ba[:, :h], ba[:, h:]
    else:                           # "inner2", "inner2_graph": every other column, inner stride 2
        buf = torch.randn(n, 2 * h, device="cuda").to(torch.bfloat16)
        a, b = buf[:, 0::2], buf[:, 1::2]

    def call(f):
        return f(A_log, a, b, dt_bias)

    def formula():
        g = -torch.exp(A_log.double()) * torch.nn.functional.softplus(a.double() + dt_bias.double())
        return g.unsqueeze(0), torch.sigmoid(b.double()).unsqueeze(0)

    target = "sglang.kernels.ops.attention.fla.fused_gdn_gating:fused_gdn_gating"
    if kind in ("inner2_graph", "inner2_graph_replay"):   # SGLang's order: warm-up and capture on a dummy batch (one
        dummy = torch.empty(n, 2 * h, device="cuda").to(torch.bfloat16)          # repeated row), then the real calls
        dummy[:] = torch.randn(1, 2 * h, device="cuda").to(torch.bfloat16)
        graph, out = capture_on(mod.fused_gdn_gating, lambda f: f(A_log, dummy[:, 0::2], dummy[:, 1::2], dt_bias))
        if kind == "inner2_graph_replay":   # what the captured graph computes once real values sit in its buffer
            def replay_graph(f, ref):
                dummy.copy_(buf)
                graph.replay()
                torch.cuda.synchronize()
                return {"replayed_rel_err": [rel(o, r) for o, r in zip(outputs(out), ref)]}
            return target, mod.fused_gdn_gating, call, formula, None, replay_graph
    if kind == "inner2_then_graph":  # the opposite order: repaired on the first real call, then captured
        return target, mod.fused_gdn_gating, call, formula, None, lambda f, ref: replayed(f, call, ref)
    return target, mod.fused_gdn_gating, call, formula


CASES = {
    "moe_bf16": lambda: moe("bf16"), "moe_int8_tensor_static": lambda: moe("int8_tensor_static"),
    "moe_int8_channel_token": lambda: moe("int8_channel_token"),
    "moe_int8_channel_static": lambda: moe("int8_channel_static"),          # vllm#58532
    "moe_int8_channel_static_then_graph": lambda: moe("int8_channel_static_then_graph"),
    "block_default": lambda: block("default"), "block_k128": lambda: block("k128"),
    "block_k256": lambda: block("k256"),                                     # vllm#52576
    "block_k256_later": lambda: block("k256_later"),                         # the defect on a later call
    "block_k256_warm": lambda: block("k256_warm"),                           # the same, warm-ups first (L3.3a)
    "gdn_contiguous": lambda: gdn("contiguous"), "gdn_rows": lambda: gdn("rows"),
    "gdn_inner2": lambda: gdn("inner2"),                                     # sglang#21843
    "gdn_inner2_graph": lambda: gdn("inner2_graph"), "gdn_inner2_then_graph": lambda: gdn("inner2_then_graph"),
    "gdn_inner2_graph_replay": lambda: gdn("inner2_graph_replay"),          # SGLang's order, the graph replayed
}


def outputs(o):
    return list(o) if isinstance(o, (tuple, list)) else [o]


def main():
    case, out_path = sys.argv[1], sys.argv[2]
    made = CASES[case]()
    target, f, call, formula = made[:4]
    later = made[4] if len(made) > 4 else None
    capture = made[5] if len(made) > 5 else None
    from entail import load
    from entail.adapters import function_reference as fr

    res = {"case": case, "target": target, "wrapped": getattr(f, "__entail_definition__", None) is not None}
    orig = fr._WRAPPED[target][2] if target in fr._WRAPPED else f
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first = outputs(call(f))
    torch.cuda.synchronize()
    res["first_call_ms"] = (time.perf_counter() - t0) * 1e3
    ref_first = outputs(formula())
    if later is not None:
        later()
    ref = outputs(formula()) if later is not None else ref_first
    kern = outputs(call(orig))
    then = outputs(call(f))
    res["rel_err"] = {"first": [rel(o, r) for o, r in zip(first, ref_first)],
                      "later": [rel(o, r) for o, r in zip(then, ref)],
                      "kernel": [rel(o, r) for o, r in zip(kern, ref)]}
    if capture is not None:
        res["captured"] = capture(f, ref)
    res["later_equals_kernel"] = all(torch.equal(a, b) for a, b in zip(then, kern))
    res["ms"] = {"kernel": timed(lambda: call(orig)), "wrapped_after_decision": timed(lambda: call(f))}
    res["decisions"] = [{"verdict": d.verdict.value, "rule": d.rule, "resolution": d.resolution, "note": d.note}
                        for d in load.LEDGER.decisions if d.contract.boundary == BOUNDARY]
    res["stats"] = fr.stats()
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
    v = [d["verdict"] for d in res["decisions"]]
    print(case, "wrapped" if res["wrapped"] else "NOT WRAPPED", v, {k: [f"{e:.3g}" for e in x]
                                                                    for k, x in res["rel_err"].items()})


if __name__ == "__main__":
    main()
