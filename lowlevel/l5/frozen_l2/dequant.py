"""M19 L2: a checkpoint's declared weight quantization, applied by the format's definition in plain PyTorch.

The layer comparison needs a reference that runs the same model the engine runs. For a quantized checkpoint that is
the quantized model, so this module dequantizes each quantized weight to float32 from the stored codes and scales,
and transformers runs the result as a dense model (layers.py hfq). Formats, read from the checkpoint's
quantization_config:
  awq (version gemm)  int4 codes packed 8 per int32 in AWQ's order, a zero point and an fp16 scale per 128 inputs
  fp8                 e4m3 weights with one scale per 128 x 128 block (weight_scale_inv, multiplied)
  modelopt NVFP4      e2m1 codes packed 2 per byte (low nibble first), an fp8 scale per 16 inputs, a float32 scale
                      per tensor (weight_scale_2)
Weight only: activation quantization that a checkpoint declares is not applied. vLLM 0.30.0 on this GPU (sm_89) runs
all three weight-only through Marlin, and says so in its log for fp8.

  python lowlevel/l2/dequant.py check <quantized dir> <unquantized dir> [n]
      relative error of the first n dequantized weights against the unquantized original's weights of the same name
"""
import json
import os
import sys

import torch

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def quant_config(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    return cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")


def awq_weight(qweight, qzeros, scales, bits, group):
    """AWQ GEMM: int32 codes [in, out * bits / 32] -> float32 weight [out, in]."""
    per = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    mask = (1 << bits) - 1
    order = torch.arange(qweight.shape[1] * per).view(-1, per)[:, AWQ_REVERSE_ORDER].reshape(-1)
    w = ((qweight[:, :, None] >> shifts) & mask).reshape(qweight.shape[0], -1)[:, order]
    z = ((qzeros[:, :, None] >> shifts) & mask).reshape(qzeros.shape[0], -1)[:, order]
    s = scales.float().repeat_interleave(group, dim=0)[: w.shape[0]]
    z = z.repeat_interleave(group, dim=0)[: w.shape[0]]
    return ((w.float() - z.float()) * s).t().contiguous()


def fp8_block_weight(w, scale_inv, block):
    """e4m3 weight [out, in] with a scale per block -> float32 [out, in]."""
    s = scale_inv.float().repeat_interleave(block[0], 0)[: w.shape[0]].repeat_interleave(block[1], 1)[:, : w.shape[1]]
    return w.float() * s


def nvfp4_weight(packed, scale, scale2, group):
    """ModelOpt NVFP4: uint8 [out, in / 2], low nibble first, fp8 scale per group of inputs, float32 tensor scale."""
    codes = torch.stack((E2M1[(packed & 0x0F).long()], E2M1[(packed >> 4).long()]), dim=-1).reshape(packed.shape[0], -1)
    return codes * (scale.float().repeat_interleave(group, dim=1)[:, : codes.shape[1]] * scale2.float())


def iter_state(model_dir):
    """(name, tensor, dequantized) for the dense model: quantized linear weights dequantized to float32, every other
    stored tensor as stored; the quantization's own tensors (scales, zero points, activation scales) are left out."""
    from safetensors import safe_open

    q = quant_config(model_dir) or {}
    method = q.get("quant_method")
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        where = json.load(open(idx))["weight_map"]
    else:
        where = {}
        for f in sorted(x for x in os.listdir(model_dir) if x.endswith(".safetensors")):
            where.update({k: f for k in safe_open(os.path.join(model_dir, f), "pt").keys()})
    handles = {}

    def get(name):
        f = where[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(model_dir, f), "pt")
        return handles[f].get_tensor(name)

    names = set(where)
    groups = [g.get("weights", {}).get("group_size") for g in (q.get("config_groups") or {}).values()]
    nv_group = next((g for g in groups if g), 16)
    aux = set()
    for n in names:
        p, _, leaf = n.rpartition(".")
        if method == "awq" and leaf == "qweight":
            aux |= {f"{p}.qzeros", f"{p}.scales"}
        elif method == "fp8" and leaf == "weight_scale_inv":
            aux.add(n)
        elif method == "modelopt" and leaf in ("weight_scale", "weight_scale_2", "input_scale"):
            aux.add(n)
    for n in sorted(names - aux):
        p, _, leaf = n.rpartition(".")
        if method == "awq" and leaf == "qweight":
            yield f"{p}.weight", awq_weight(get(n), get(f"{p}.qzeros"), get(f"{p}.scales"),
                                            q.get("bits", 4), q.get("group_size", 128)), True
        elif method == "fp8" and leaf == "weight" and f"{p}.weight_scale_inv" in names:
            yield n, fp8_block_weight(get(n), get(f"{p}.weight_scale_inv"), tuple(q.get("weight_block_size", (128, 128)))), True
        elif method == "modelopt" and leaf == "weight" and f"{p}.weight_scale" in names:
            yield n, nvfp4_weight(get(n), get(f"{p}.weight_scale"), get(f"{p}.weight_scale_2"), nv_group), True
        else:
            yield n, get(n), False


def check(qdir, odir, n=12):
    """Relative error of the first n dequantized weights against the unquantized original: a few percent means the
    codes were read as the format defines them, a wrong nibble or packing order gives errors near or above 1."""
    from safetensors import safe_open

    idx = os.path.join(odir, "model.safetensors.index.json")
    if os.path.exists(idx):
        where = json.load(open(idx))["weight_map"]
    else:
        where = {}
        for f in sorted(x for x in os.listdir(odir) if x.endswith(".safetensors")):
            where.update({k: f for k in safe_open(os.path.join(odir, f), "pt").keys()})
    rows = []
    for name, t, deq in iter_state(qdir):
        if not deq or name not in where:
            continue
        o = safe_open(os.path.join(odir, where[name]), "pt").get_tensor(name).float()
        rows.append((name, (t - o).norm().item() / (o.norm().item() or 1.0)))
        print(f"{rows[-1][1]:8.4f}  {name}")
        if len(rows) >= n:
            break
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12)
