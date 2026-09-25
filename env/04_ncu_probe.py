"""Tiny workload for probing Nsight Compute access in WSL2.
Usage (inside WSL, venv active):
  ncu --target-processes all --launch-skip 5 --launch-count 1 \
      --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
      python 04_ncu_probe.py
"""
import torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(8):
    c = a @ b
torch.cuda.synchronize()
print("probe done", c.shape)
