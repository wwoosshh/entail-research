"""Week 2, experiment C: count FLOPs of one SDXL UNet step and compute achieved TFLOPS per kernel class.

Uses torch.utils.flop_counter.FlopCounterMode on the eager fp16 forward (random weights; FLOPs do not
depend on weight values). Combined with week-1 kernel-class times this gives the efficiency of the
cuBLAS / cuDNN / FlashAttention kernels against the GeForce FP32-accumulate ceiling (~80 TFLOPS).
"""
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from bench_sdxl_unet import build_unet, make_inputs  # noqa: E402
from common import time_fn  # noqa: E402

CEILING_TFLOPS = 80.0  # RTX 4070 Ti FP16 tensor, FP32 accumulate (GeForce half rate)


def main():
    unet, src, n_params = build_unet()
    out = {"model_source": src, "params": n_params, "ceiling_tflops": CEILING_TFLOPS, "batches": {}}
    for B in (1, 2):
        inputs = make_inputs(B)
        with torch.no_grad():
            counter = FlopCounterMode(display=False)
            with counter:
                unet(**inputs)
            total = counter.get_total_flops()
            by_op = {str(k): v for k, v in counter.get_flop_counts()["Global"].items()}
            med, mn, mx = time_fn(lambda: unet(**inputs).sample, warmup=3, iters=10)
        groups = {"conv": 0, "matmul": 0, "attention": 0, "other": 0}
        for op, f in by_op.items():
            o = op.lower()
            if "conv" in o:
                groups["conv"] += f
            elif "scaled_dot_product" in o or "attention" in o:
                groups["attention"] += f
            elif "mm" in o or "addmm" in o or "bmm" in o or "linear" in o:
                groups["matmul"] += f
            else:
                groups["other"] += f
        rec = {"total_tflop": total / 1e12, "step_ms": med, "achieved_tflops": total / (med / 1e3) / 1e12,
               "achieved_frac_of_ceiling": total / (med / 1e3) / 1e12 / CEILING_TFLOPS,
               "flops_by_group_tflop": {k: v / 1e12 for k, v in groups.items()},
               "flops_by_op_tflop": {k: v / 1e12 for k, v in sorted(by_op.items(), key=lambda kv: -kv[1])}}
        out["batches"][str(B)] = rec
        print(f"\nB={B}: {total/1e12:.2f} TFLOP per step, {med:.1f} ms -> {rec['achieved_tflops']:.1f} TFLOPS "
              f"({rec['achieved_frac_of_ceiling']*100:.0f}% of the {CEILING_TFLOPS:.0f} TFLOPS ceiling)")
        for k, v in groups.items():
            print(f"   {k:10s} {v/1e12:7.2f} TFLOP")
        del inputs
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "sdxl_flops.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
