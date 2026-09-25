import traceback, torch
from torchao.quantization import quantize_, Int4WeightOnlyConfig
def trial(name, cfg):
    lin = torch.nn.Linear(2560, 9728, bias=False, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(4, 2560, dtype=torch.bfloat16, device="cuda")
    ref = lin(x)
    try:
        quantize_(lin, cfg)
        y = lin(x)
        err = (y - ref).abs().max().item()
        print(f"[OK ] {name}: max abs err {err:.3f}, weight type {type(lin.weight).__name__}")
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[:200]}")
trial("default", Int4WeightOnlyConfig(group_size=128))
for fmt in ["tile_packed_to_4d", "plain", "plain_int32", "preshuffled"]:
    try:
        trial(f"format={fmt}", Int4WeightOnlyConfig(group_size=128, int4_packing_format=fmt))
    except Exception as e:
        print(f"[ERR] building config format={fmt}: {type(e).__name__}: {str(e)[:150]}")
try:
    trial("version=1", Int4WeightOnlyConfig(group_size=128, version=1))
except Exception as e:
    print(f"[ERR] building config version=1: {type(e).__name__}: {str(e)[:150]}")
