"""Probe: inside a captured CUDA graph, does a device-side assertion stop the graph before its next operation runs?
A pinned host buffer written by the next operation (a captured device-to-host copy of its result) shows whether it
ran; another pinned buffer receives the gate's status before the assertion."""
import sys

import torch

dev = "cuda"
torch.manual_seed(0)
x = torch.randn(256, 256, device=dev)
ok = torch.ones((), dtype=torch.bool, device=dev)
status = torch.zeros(2, dtype=torch.int64, device=dev)
host_status = torch.full((2,), -1, dtype=torch.int64, pin_memory=True)
host_next = torch.full((4,), -7.0, dtype=torch.float32, pin_memory=True)


def body():
    y = x * 2                                           # the gate's output
    status.copy_(torch.stack([ok.to(torch.int64), (~ok).to(torch.int64)]))
    host_status.copy_(status, non_blocking=True)        # what the gate saw, readable after a stop
    torch._assert_async(ok)                             # the stop
    z = y @ y                                           # the graph's next operation (reads the gate's output)
    host_next.copy_(z[0, :4], non_blocking=True)        # the observer: written only if the next operation ran
    return z


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        body()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = body()
print("captured; nodes ok")

host_status.fill_(-1)
host_next.fill_(-7.0)
g.replay()
torch.cuda.current_stream().synchronize()
print("replay ok=1: status", host_status.tolist(), "observer", [round(v, 3) for v in host_next.tolist()])

mode = sys.argv[1] if len(sys.argv) > 1 else "stop"
if mode == "stop":
    ok.fill_(False)
    torch.cuda.synchronize()
    host_status.fill_(-1)
    host_next.fill_(-7.0)
    g.replay()
    try:
        torch.cuda.current_stream().synchronize()
        print("NO ERROR after a failing replay")
    except RuntimeError as e:
        print("stopped at synchronize:", str(e).strip().splitlines()[0])
    print("after the stop: status", host_status.tolist(), "observer", host_next.tolist())
    print("observer untouched (next operation did not run):", host_next.tolist() == [-7.0] * 4)
    try:
        torch.zeros(1, device=dev)
        print("context still usable")
    except Exception as e:  # noqa: BLE001
        print("context lost:", type(e).__name__)
