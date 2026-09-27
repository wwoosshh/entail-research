"""M16 case, sgl-project/sglang#21843 (testbed/M16_PROTOCOL.md 5): with num_v_heads == num_k_heads, the a and b
tensors that fix_query_key_value_ordering hands to the fused_gdn_gating Triton kernel are non-contiguous (strides
(2*heads, 2)) and the kernel reads them as flat contiguous rows, so it takes interleaved a/b values. The report's
tensor-level construction (seq 8, 16 k-heads, 16 v-heads, random values), then the 0.5.20 kernel on the
non-contiguous tensors against the same kernel on contiguous copies and against the gating formula in torch.
Run in ~/venvs/sglang: python testbed/m16/cases/sg21843.py <out.json>
"""
import json
import os
import sys


def main():
    import sglang
    import torch
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating

    torch.manual_seed(0)
    seq_len, num_k_heads, num_v_heads = 8, 16, 16
    mixed_ba = torch.randn(seq_len, num_v_heads * 2, device="cuda")
    b, a = torch.split(mixed_ba.view(seq_len, num_k_heads, 2), [1, 1], dim=2)
    a = a.reshape(seq_len, num_v_heads)
    b = b.reshape(seq_len, num_v_heads)
    A_log = torch.randn(num_v_heads, device="cuda")
    dt_bias = torch.randn(num_v_heads, device="cuda")
    g_nc, beta_nc = fused_gdn_gating(A_log, a, b, dt_bias)
    g_c, beta_c = fused_gdn_gating(A_log, a.contiguous(), b.contiguous(), dt_bias)
    g_ref = (-torch.exp(A_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())).view_as(g_c)
    beta_ref = torch.sigmoid(b.float()).view_as(beta_c)
    torch.cuda.synchronize()
    row = {"entail": os.environ.get("ENTAIL", "off"), "sglang": sglang.__version__,
           "a_stride": list(a.stride()), "a_contiguous": bool(a.is_contiguous()),
           "g_noncontig_vs_contig_max_abs": float((g_nc - g_c).abs().max()),
           "beta_noncontig_vs_contig_max_abs": float((beta_nc - beta_c).abs().max()),
           "g_contig_vs_formula_max_abs": float((g_c - g_ref).abs().max()),
           "beta_contig_vs_formula_max_abs": float((beta_c - beta_ref).abs().max())}
    row["reproduced"] = bool(not a.is_contiguous() and (row["g_noncontig_vs_contig_max_abs"] > 1e-4
                                                        or row["beta_noncontig_vs_contig_max_abs"] > 1e-4))
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
