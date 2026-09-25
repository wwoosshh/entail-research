"""Localize why the Triton decode-attention custom op gives wrong results under torch.compile.

Cases (small tensors, no model):
  1. op called directly on inputs, compiled vs eager vs fp32 reference
  2. q produced inside the compiled region with a transposed layout (as in the attention module)
  3. k/v cache updated in place inside the compiled region (index_copy_ on a static-address buffer), then read
  4. same as 3 under mode="reduce-overhead" (CUDA graphs)
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import bench_decode_attn_swap as S  # noqa: E402
from triton_decode_attn import reference  # noqa: E402

torch.manual_seed(0)
B, Hq, Hkv, D, L, POS = 2, 32, 8, 128, 640, 512
scale = D ** -0.5
S.VALID.zero_()
S.VALID[:B].fill_(POS + 1)


def report(name, out, ref):
    d = (out.float() - ref.float()).abs().max().item()
    print(f"  {name:48s} max|diff| vs reference = {d:.5f}", flush=True)


q = torch.randn(B, Hq, 1, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, Hkv, L, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, Hkv, L, D, device="cuda", dtype=torch.bfloat16)
ref = reference(q, k, v, S.VALID[:B], scale)

print("case 1: op on given inputs", flush=True)


def f1(q, k, v):
    return S.decode_attention_op(q, k, v, S.VALID[:B], scale)


report("eager", f1(q, k, v), ref)
torch._dynamo.reset()
report("compiled", torch.compile(f1)(q, k, v), ref)

print("case 2: q made inside the compiled region with a transposed layout", flush=True)
x = torch.randn(B, 1, Hq * D, device="cuda", dtype=torch.bfloat16)
q_from_x = x.view(B, 1, Hq, D).transpose(1, 2)
ref2 = reference(q_from_x, k, v, S.VALID[:B], scale)


def f2(x, k, v):
    qq = (x.view(B, 1, Hq, D) * 1.0).transpose(1, 2)
    return S.decode_attention_op(qq, k, v, S.VALID[:B], scale)


report("eager", f2(x, k, v), ref2)
torch._dynamo.reset()
report("compiled", torch.compile(f2)(x, k, v), ref2)

print("case 3/4: cache updated in place inside the compiled region, then read", flush=True)
for mode in ("default", "reduce-overhead"):
    kc = k.clone()
    vc = v.clone()
    torch._dynamo.mark_static_address(kc)
    torch._dynamo.mark_static_address(vc)
    k_new = torch.randn(B, Hkv, 1, D, device="cuda", dtype=torch.bfloat16)
    v_new = torch.randn(B, Hkv, 1, D, device="cuda", dtype=torch.bfloat16)
    pos = torch.tensor([POS], device="cuda")
    k_ref, v_ref = k.clone(), v.clone()
    k_ref[:, :, POS] = k_new[:, :, 0]
    v_ref[:, :, POS] = v_new[:, :, 0]
    ref3 = reference(q, k_ref, v_ref, S.VALID[:B], scale)

    def f3(q, k_new, v_new, pos):
        kc.index_copy_(2, pos, k_new)
        vc.index_copy_(2, pos, v_new)
        return S.decode_attention_op(q, kc, vc, S.VALID[:B], scale)

    torch._dynamo.reset()
    cf = torch.compile(f3, mode=mode)
    outs = []
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        outs.append(cf(q, k_new, v_new, pos).clone())
    for i, o in enumerate(outs):
        report(f"mode={mode} call {i}", o, ref3)
    print(f"  cache row written correctly: {torch.equal(kc[:, :, POS], k_new[:, :, 0])}", flush=True)
