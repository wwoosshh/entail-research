"""Batch-invariant building blocks: a request's output must not depend on which other requests share its batch.

bi_linear   y = x @ w.T. Split-K count is chosen from (N, K) only, never from M; partial sums are reduced in a
            fixed order by a second kernel; the M tile is fixed at 16. So row i is always computed by the same
            instruction sequence, whatever the batch size. cuBLAS instead picks a GEMV for M=1 and a GEMM for M>1.
bi_rmsnorm  one program per row, block size fixed by the row length, fp32 reduction.
Both are exposed as torch custom ops so torch.compile / CUDA graphs treat them as opaque nodes.
Run this file to self-test correctness and invariance.
"""
import torch
import triton
import triton.language as tl

SMS = torch.cuda.get_device_properties(0).multi_processor_count
BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 64


@triton.jit(do_not_specialize=["M"])
def _bi_mm_partial(X, W, P, M, N, K, K_PER_SPLIT,
                   sxm, sxk, swn, swk, sps, spm, spn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_begin = pid_s * K_PER_SPLIT
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_begin + kk + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(W + offs_n[:, None] * swn + offs_k[None, :] * swk,
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(P + pid_s * sps + offs_m[:, None] * spm + offs_n[None, :] * spn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit(do_not_specialize=["M"])
def _bi_mm_reduce(P, Y, M, N, SPLIT, sps, spm, spn, sym, syn, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    m = offs // N
    n = offs % N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, SPLIT):
        acc += tl.load(P + s * sps + m * spm + n * spn, mask=mask, other=0.0)
    tl.store(Y + m * sym + n * syn, acc.to(Y.dtype.element_ty), mask=mask)


_SPLIT = {}


def choose_split(N, K):
    """Split-K from (N, K) only: aim for ~4 programs per SM, keep >= 2 K-blocks per split, split must divide."""
    if (N, K) not in _SPLIT:
        n_tiles = triton.cdiv(N, BLOCK_N)
        kb = triton.cdiv(K, BLOCK_K)
        target = max(1, triton.cdiv(4 * SMS, n_tiles))
        cands = [d for d in range(1, kb + 1) if kb % d == 0 and kb // d >= 2] or [1]
        split = max([d for d in cands if d <= target] or [1])
        _SPLIT[(N, K)] = (split, (kb // split) * BLOCK_K)
    return _SPLIT[(N, K)]


def bi_linear(x, w):
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).contiguous()
    M, K = x2.shape
    N = w.shape[0]
    split, kps = choose_split(N, K)
    part = torch.empty((split, M, N), device=x.device, dtype=torch.float32)
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, BLOCK_N), split, triton.cdiv(M, BLOCK_M))
    _bi_mm_partial[grid](x2, w, part, M, N, K, kps,
                         x2.stride(0), x2.stride(1), w.stride(0), w.stride(1),
                         part.stride(0), part.stride(1), part.stride(2),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    rb = 1024
    _bi_mm_reduce[(triton.cdiv(M * N, rb),)](part, y, M, N, split,
                                             part.stride(0), part.stride(1), part.stride(2),
                                             y.stride(0), y.stride(1), BLOCK=rb, num_warps=4)
    return y.reshape(*shp[:-1], N)


@triton.jit
def _bi_rmsnorm(X, Wt, Y, N, sx, sy, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    y = (x * tl.rsqrt(var + eps)).to(Y.dtype.element_ty)        # hidden_states.to(input_dtype)
    w = tl.load(Wt + offs, mask=mask, other=0.0)
    out = (w.to(tl.float32) * y.to(tl.float32)).to(Y.dtype.element_ty)  # weight * hidden_states
    tl.store(Y + row * sy + offs, out, mask=mask)


def bi_rmsnorm(x, w, eps):
    shp = x.shape
    N = shp[-1]
    x2 = x.reshape(-1, N).contiguous()
    y = torch.empty_like(x2)
    block = triton.next_power_of_2(N)
    _bi_rmsnorm[(x2.shape[0],)](x2, w, y, N, x2.stride(0), y.stride(0), eps, BLOCK=block,
                                num_warps=4 if block <= 1024 else 8)
    return y.reshape(shp)


@torch.library.custom_op("phase0bi::linear", mutates_args=())
def bi_linear_op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return bi_linear(x, w)


@bi_linear_op.register_fake
def _(x, w):
    return x.new_empty(x.shape[:-1] + (w.shape[0],))


@torch.library.custom_op("phase0bi::rmsnorm", mutates_args=())
def bi_rmsnorm_op(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return bi_rmsnorm(x, w, eps)


@bi_rmsnorm_op.register_fake
def _(x, w, eps):
    return torch.empty_like(x)


def rmsnorm_reference(x, w, eps):
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"SMs={SMS}")
    shapes = [(4096, 2560), (1024, 2560), (2560, 4096), (9728, 2560), (2560, 9728), (151936, 2560)]
    for N, K in shapes:
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ w.float().T
        yb = bi_linear(x, w)
        yc = torch.nn.functional.linear(x, w)
        bi_inv = all(torch.equal(bi_linear(x[i:i + 1], w), yb[i:i + 1]) for i in range(8))
        cu_inv = all(torch.equal(torch.nn.functional.linear(x[i:i + 1], w), yc[i:i + 1]) for i in range(8))
        print(f"linear N={N:6d} K={K:5d} split={choose_split(N, K)} | err bi {(yb.float() - ref).abs().max().item():.4f} "
              f"cublas {(yc.float() - ref).abs().max().item():.4f} | row-invariant: bi {bi_inv}, cublas {cu_inv}")
    for N in (2560, 128):
        w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(8, 3, N, device="cuda", dtype=torch.bfloat16) * 3
        yb = bi_rmsnorm(x, w, 1e-6)
        yr = rmsnorm_reference(x, w, 1e-6)
        inv = all(torch.equal(bi_rmsnorm(x[i:i + 1], w, 1e-6), yb[i:i + 1]) for i in range(8))
        ref_inv = all(torch.equal(rmsnorm_reference(x[i:i + 1], w, 1e-6), yr[i:i + 1]) for i in range(8))
        print(f"rmsnorm N={N}: max diff vs eager formula {(yb.float() - yr.float()).abs().max().item():.4f} | "
              f"row-invariant: bi {inv}, eager formula {ref_inv}")
