"""#1 LAYOUT: one path reorders quantized weights in place; another path still reads the original layout.

Mechanism of llama.cpp #21589 / #26845 (SYCL "garbage on the second prompt"): for speed the backend rewrites a
Q8_0 weight from block-interleaved layout (per 32 values: fp16 scale then 32 int8) into split layout (all int8
values, then all scales) and records that only in a side flag. The matmul path checks the flag; a second path
(the one taken on the next prompt) dequantizes without checking it and reads the split bytes as interleaved.
  defect:    second path reads the reordered buffer as block-interleaved
  fixed:     second path dispatches on the layout flag (the upstream fix added the missing reader)
  reference: x @ W_q.T with W_q the dequantized original quantization
"""
import torch

META = {
    "id": "01", "title": "reordered quantized weight read with the original layout", "fact": "LAYOUT",
    "issue": "https://github.com/ggml-org/llama.cpp/issues/21589 ; https://github.com/ggml-org/llama.cpp/issues/26845",
    "engine": "llama.cpp SYCL backend (mechanism)", "kind": "mechanism",
    "boundary": "backend reorder (layout flag in tensor extra) -> second-prompt dequantize path",
    "trigger": {"hardware": "Intel GPU (SYCL)", "turn": "second prompt"},
    "symptom": "garbled", "expected_detection": "runtime (closed layout type, exhaustive readers)", "compare": "fp32",
}
N, K, QK = 64, 256, 32  # rows, columns, values per block
BLOCK_BYTES = 2 + QK


def quantize_q8_0(w):
    """Block-interleaved bytes: for each block of 32 values, an fp16 scale (2 bytes) then 32 int8."""
    blocks = w.reshape(-1, QK)
    scale = (blocks.abs().amax(-1) / 127.0).clamp_min(1e-8)
    q = torch.round(blocks / scale[:, None]).clamp(-127, 127).to(torch.int8)
    sb = scale.to(torch.float16).view(torch.uint8).reshape(-1, 2)
    return torch.cat([sb, q.view(torch.uint8)], dim=1).reshape(-1)  # (n_blocks * 34,) uint8


def read_interleaved(buf):
    b = buf.reshape(-1, BLOCK_BYTES)
    scale = b[:, :2].contiguous().view(torch.float16).reshape(-1).float()
    q = b[:, 2:].contiguous().view(torch.int8).float()
    return (q * scale[:, None]).reshape(N, K)


def read_split(buf):
    nb = N * K // QK
    q = buf[: nb * QK].view(torch.int8).float().reshape(nb, QK)
    scale = buf[nb * QK:].contiguous().view(torch.float16).float()
    return (q * scale[:, None]).reshape(N, K)


def reorder_in_place(t):
    """Backend optimisation: rewrite interleaved -> split and record it only in the side flag."""
    b = t["data"].reshape(-1, BLOCK_BYTES)
    split = torch.cat([b[:, 2:].reshape(-1), b[:, :2].reshape(-1)])
    t["data"].copy_(split)
    t["extra"]["reordered"] = True


def setup():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(N, K, generator=g)
    x = torch.randn(8, K, generator=g)
    data = quantize_q8_0(w)
    wq = read_interleaved(data.clone())  # meaning of the weight: the dequantized original quantization
    return {"x": x, "wq": wq, "data": data}


def _fresh_reordered(ctx):
    t = {"data": ctx["data"].clone(), "extra": {}}
    reorder_in_place(t)  # happened during the first prompt's matmul
    return t


def defect(ctx):
    t = _fresh_reordered(ctx)
    return ctx["x"] @ read_interleaved(t["data"]).T  # second path ignores the flag


def fixed(ctx):
    t = _fresh_reordered(ctx)
    reader = read_split if t["extra"].get("reordered") else read_interleaved
    return ctx["x"] @ reader(t["data"]).T


def reference(ctx):
    return ctx["x"] @ ctx["wq"].T
