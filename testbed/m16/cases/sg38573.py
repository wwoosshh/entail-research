"""M16 case, sgl-project/sglang#38573 (testbed/M16_PROTOCOL.md 5): the compressed-tensors W4AFP8 MoE scheme sizes
its scales by the checkpoint's group_size but the cutlass_w4a8_moe kernel is told a literal chunk of 128, so a
group_size-64 checkpoint dequantizes with the wrong stride, silently. This card is sm89; the scheme declares a
minimum capability of 90 (Hopper) and the CUTLASS W4A8 grouped GEMM is built for it, so the kernel path cannot run
here. The script records the gate and tries the scheme anyway, so the outcome is measured and not assumed.
Run in ~/venvs/sglang: python testbed/m16/cases/sg38573.py <out.json>
"""
import json
import os
import sys


def main():
    import sglang
    import torch

    row = {"entail": os.environ.get("ENTAIL", "off"), "sglang": sglang.__version__,
           "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
    try:
        from sglang.srt.layers.quantization.compressed_tensors.schemes import compressed_tensors_w4a8_fp8_moe as m

        cls = next(getattr(m, n) for n in dir(m) if "W4A8" in n or "W4AFP8" in n)
        row["scheme"] = cls.__name__
        row["scheme_min_capability"] = cls.get_min_capability()
    except Exception as e:  # noqa: BLE001
        row["scheme_error"] = f"{type(e).__name__}: {e}"[:300]
    try:
        import sgl_kernel

        row["kernel_present"] = hasattr(sgl_kernel, "cutlass_w4a8_moe_mm")
    except Exception as e:  # noqa: BLE001
        row["kernel_present"] = f"{type(e).__name__}: {e}"[:200]
    cap = row["capability"][0] * 10 + row["capability"][1]
    row["runnable_here"] = bool(row.get("scheme_min_capability") is not None and cap >= row["scheme_min_capability"])
    row["reproduced"] = False
    row["not_reproduced_reason"] = (None if row["runnable_here"] else
                                    f"the W4AFP8 MoE scheme needs capability {row.get('scheme_min_capability')}, this card is {cap}")
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
