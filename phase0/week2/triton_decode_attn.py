"""Flash-decoding style attention for ONE query token over a STATIC KV cache, in Triton.

Handles what the compiled StaticCache path in week 1 could not:
  - per-sequence valid lengths (no attention mask materialization),
  - GQA by index mapping (no repeat_kv copy of K/V),
  - fixed shapes regardless of valid length (CUDA-graph friendly).

Two kernels (split-K over the cache, then combine), fp32 accumulation, bf16/fp16 I/O.
This is a measurement artifact, not a production kernel: no autotuning, D must be a power of two.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_kernel(Q, K, V, VALID, MP, LP, OP,
                  sqb, sqh, skb, skh, skn, svb, svh, svn, smb, smh, sms, sob, soh, sos,
                  sm_scale, L_MAX,
                  GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    kvh = h // GROUP
    valid = tl.load(VALID + b)
    per_split = (L_MAX + SPLIT - 1) // SPLIT
    start = s * per_split
    end = tl.minimum(start + per_split, valid)

    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + h * sqh + offs_d).to(tl.float32)
    m_i = tl.full([], float("-inf"), tl.float32)
    l_i = tl.zeros([], tl.float32)
    acc = tl.zeros([D], tl.float32)

    for n0 in range(start, start + per_split, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < end
        k = tl.load(K + b * skb + kvh * skh + offs_n[:, None] * skn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(mask_n, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(scores - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(V + b * svb + kvh * svh + offs_n[:, None] * svn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(MP + b * smb + h * smh + s * sms, m_i)
    tl.store(LP + b * smb + h * smh + s * sms, l_i)
    tl.store(OP + b * sob + h * soh + s * sos + offs_d, acc)


@triton.jit
def _combine_kernel(MP, LP, OP, OUT, smb, smh, sms, sob, soh, sos, soutb, south,
                    D: tl.constexpr, SPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLIT)
    m = tl.load(MP + b * smb + h * smh + offs_s * sms)
    l = tl.load(LP + b * smb + h * smh + offs_s * sms)
    m_max = tl.max(m, axis=0)
    w = tl.exp(m - m_max)
    l_tot = tl.sum(l * w, axis=0)
    offs_d = tl.arange(0, D)
    o = tl.load(OP + b * sob + h * soh + offs_s[:, None] * sos + offs_d[None, :])
    out = tl.sum(o * w[:, None], axis=0) / l_tot
    tl.store(OUT + b * soutb + h * south + offs_d, out.to(OUT.dtype.element_ty))


def decode_attention(q, k_cache, v_cache, valid_len, sm_scale=None, split=8, block_n=64):
    """q: [B, Hq, 1, D]; k_cache, v_cache: [B, Hkv, Lmax, D] (D contiguous); valid_len: [B] int32 CUDA.
    Returns [B, Hq, 1, D] in q.dtype. Positions >= valid_len[b] are ignored exactly (not masked by -inf
    after the fact but never loaded into the softmax)."""
    B, Hq, one, D = q.shape
    assert one == 1 and (D & (D - 1)) == 0, "single query token, power-of-two head dim"
    Hkv, Lmax = k_cache.shape[1], k_cache.shape[2]
    assert Hq % Hkv == 0 and (split & (split - 1)) == 0
    if sm_scale is None:
        sm_scale = D ** -0.5
    q2 = q.reshape(B, Hq, D)
    mp = torch.empty(B, Hq, split, device=q.device, dtype=torch.float32)
    lp = torch.empty_like(mp)
    op = torch.empty(B, Hq, split, D, device=q.device, dtype=torch.float32)
    out = torch.empty(B, Hq, D, device=q.device, dtype=q.dtype)
    _split_kernel[(B, Hq, split)](
        q2, k_cache, v_cache, valid_len, mp, lp, op,
        q2.stride(0), q2.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        mp.stride(0), mp.stride(1), mp.stride(2), op.stride(0), op.stride(1), op.stride(2),
        sm_scale, Lmax,
        GROUP=Hq // Hkv, D=D, BLOCK_N=block_n, SPLIT=split, num_warps=4)
    _combine_kernel[(B, Hq)](
        mp, lp, op, out, mp.stride(0), mp.stride(1), mp.stride(2),
        op.stride(0), op.stride(1), op.stride(2), out.stride(0), out.stride(1),
        D=D, SPLIT=split, num_warps=4)
    return out.reshape(B, Hq, 1, D)


def reference(q, k_cache, v_cache, valid_len, sm_scale=None):
    """Exact fp32 reference over the valid prefix of each sequence (loop over batch)."""
    B, Hq, _, D = q.shape
    Hkv = k_cache.shape[1]
    group = Hq // Hkv
    if sm_scale is None:
        sm_scale = D ** -0.5
    outs = []
    for b in range(B):
        n = int(valid_len[b])
        k = k_cache[b, :, :n].float().repeat_interleave(group, dim=0)      # [Hq, n, D]
        v = v_cache[b, :, :n].float().repeat_interleave(group, dim=0)
        s = torch.einsum("hd,hnd->hn", q[b, :, 0].float(), k) * sm_scale     # [Hq, n]
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hn,hnd->hd", p, v))
    return torch.stack(outs).unsqueeze(2).to(q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, Hq, Hkv, D, Lmax = 4, 32, 8, 128, 640
    q = torch.randn(B, Hq, 1, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Lmax, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Lmax, D, device="cuda", dtype=torch.bfloat16)
    valid = torch.tensor([513, 600, 33, 640], device="cuda", dtype=torch.int32)
    out = decode_attention(q, k, v, valid)
    ref = reference(q, k, v, valid)
    print("max abs err vs fp32 reference:", (out.float() - ref.float()).abs().max().item(),
          "| ref scale:", ref.float().abs().max().item())
