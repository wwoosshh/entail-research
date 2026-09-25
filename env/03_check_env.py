"""End-to-end environment check for GPU compiler research (RTX 4070 Ti, sm_89).
Run inside WSL with the venv active:  python 03_check_env.py
"""
import shutil, sys, time
import torch

def hdr(s): print(f"\n=== {s} ===")

def bench(fn, iters=50, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters

hdr("versions")
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| torch.version.cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
try:
    import triton; print("triton", triton.__version__)
except Exception as e:
    print("triton import failed:", e)
for tool in ["nvcc", "ptxas", "cuobjdump", "nvdisasm", "ncu", "nsys", "compute-sanitizer"]:
    print(f"  {tool:18s}", shutil.which(tool) or "NOT FOUND")

assert torch.cuda.is_available(), "CUDA not available in torch"
p = torch.cuda.get_device_properties(0)
hdr("device")
print(p.name, f"| cc {p.major}.{p.minor} | {p.total_memory/2**30:.1f} GiB | {p.multi_processor_count} SMs")

hdr("Triton JIT kernel (vector add)")
import triton, triton.language as tl
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)
n = 1 << 20
x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda"); out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.cuda.synchronize()
print("max abs err vs eager:", (out - (x + y)).abs().max().item())

hdr("cuBLAS FP16 GEMM (FP32 accumulate; GeForce runs this path at half tensor rate)")
for N in (2048, 4096, 8192):
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    t = bench(lambda: a @ b)
    print(f"  N={N}: {2*N**3/t/1e12:6.1f} TFLOPS  ({t*1e3:.3f} ms)")

hdr("SDPA backends, q/k/v = [1,16,4096,64] fp16")
from torch.nn.attention import sdpa_kernel, SDPBackend
q = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
    try:
        with sdpa_kernel(be):
            t = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), iters=20)
        print(f"  {be.name:20s} {t*1e3:8.3f} ms")
    except Exception as e:
        print(f"  {be.name:20s} unavailable ({type(e).__name__})")

hdr("torch.compile smoke test")
def f(x): return torch.nn.functional.gelu(x * 2 + 1).sum()
cf = torch.compile(f)
x = torch.randn(1 << 16, device="cuda")
t0 = time.perf_counter(); cf(x); torch.cuda.synchronize()
print(f"  first call incl. compile: {time.perf_counter()-t0:.2f}s")
print("  matches eager:", torch.allclose(cf(x), f(x), rtol=1e-4, atol=1e-4))

hdr("device memory copy bandwidth (read+write counted)")
buf = torch.empty(1 << 28, device="cuda", dtype=torch.uint8); dst = torch.empty_like(buf)
t = bench(lambda: dst.copy_(buf))
print(f"  {2*buf.numel()/t/1e9:.0f} GB/s effective (spec ~504 GB/s for 4070 Ti)")
print("\nOK: environment is functional")
