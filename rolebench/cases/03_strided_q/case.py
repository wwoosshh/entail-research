"""#3 LAYOUT: a kernel assumes packed Q while it receives a strided slice of a fused QKV buffer.

Mechanism of SGLang #31641 (NVFP4 KV cache batched decode corrupted, trtllm_mha on SM120): Q is sliced from the
fused QKV projection output, so its row stride is (H + 2*Hkv) * D, not H * D. A kernel that computes row
addresses as t * H * D reads parts of K and V for every row after the first.
  defect:    Triton kernel called with the assumed packed row stride
  fixed:     the same kernel called with the tensor's real row stride
  reference: float64 torch computation on the real Q
"""
import torch
import triton
import triton.language as tl

META = {
    "id": "03", "title": "strided Q from fused QKV read as packed", "fact": "LAYOUT",
    "issue": "https://github.com/sgl-project/sglang/issues/31641", "engine": "Triton kernel call convention (mechanism)",
    "kind": "mechanism", "boundary": "fused QKV projection (strided view) -> attention kernel (assumes packed)",
    "trigger": {"hardware": "SM120 path in the original report", "feature": "fused QKV + specific backend"},
    "symptom": "garbled", "expected_detection": "runtime (declared stride/packing vs actual tensor)", "compare": "fp32",
}
T, H, HKV, D = 16, 4, 2, 64


@triton.jit
def _rowdot(q_ptr, w_ptr, out_ptr, stride_q, HD: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, HD, BLOCK):
        idx = start + offs
        m = idx < HD
        acc += tl.load(q_ptr + t * stride_q + idx, mask=m, other=0.0) * tl.load(w_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + t, tl.sum(acc, 0))


def kernel(q, w, row_stride):
    out = torch.empty(q.shape[0], device=q.device, dtype=torch.float32)
    _rowdot[(q.shape[0],)](q, w, out, row_stride, HD=q.shape[1], BLOCK=128)
    torch.cuda.synchronize()
    return out


def setup():
    g = torch.Generator("cuda").manual_seed(0)
    qkv = torch.randn(T, (H + 2 * HKV) * D, device="cuda", generator=g)
    q = qkv[:, : H * D]  # a strided view: row stride (H + 2*HKV) * D
    return {"q": q, "w": torch.randn(H * D, device="cuda", generator=g)}


def defect(ctx):
    return kernel(ctx["q"], ctx["w"], H * D)  # assumes packed rows


def fixed(ctx):
    return kernel(ctx["q"], ctx["w"], ctx["q"].stride(0))


def reference(ctx):
    return (ctx["q"].double() @ ctx["w"].double()).float()
