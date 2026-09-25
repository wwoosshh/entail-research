"""M10 E3, sgl-project/sglang#37606 (testbed/M10_PROTOCOL.md 3.3): the prefill breakable CUDA graph keeps graph-break
inputs as weak references, so a later capture reuses their memory. The issue's model-free reproducer (the mechanism;
its end-to-end model reproduction needs TP8 on B300 and is not run here), on SGLang 0.5.20, entail off or on.
Run in ~/venvs/sglang: python testbed/m10_e3/sg37606.py <out.json>
"""
import json
import os
import sys

N = 64 * 2048


def run(weak):
    import torch
    from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors

    pool = torch.cuda.graph_pool_handle()
    x_a = torch.full((N,), 7.0, device="cuda")
    g_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_a, pool=pool):
        bridge = x_a * 1.0
    addr_a = bridge.data_ptr()
    g_a.replay()
    torch.cuda.synchronize()
    if weak:
        kept = weak_ref_tensors(bridge)
        del bridge
    else:
        kept = bridge
    g_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_b, pool=pool):
        junk = torch.zeros(N, device="cuda")
    reused = junk.data_ptr() == addr_a
    junk.fill_(-999.0)
    torch.cuda.synchronize()
    bridge_corrupted = kept[:4].tolist() != [7.0] * 4
    g_a.replay()
    torch.cuda.synchronize()
    junk_clobbered = junk[:4].tolist() != [-999.0] * 4
    out = {"reused": reused, "bridge_corrupted": bridge_corrupted, "junk_clobbered": junk_clobbered}
    del g_a, g_b, x_a, kept, junk
    torch.cuda.synchronize()
    return out


def main():
    w, s = run(True), run(False)
    ok = (w["reused"] and w["bridge_corrupted"] and w["junk_clobbered"]
          and not (s["reused"] or s["bridge_corrupted"] or s["junk_clobbered"]))
    row = {"entail": os.environ.get("ENTAIL", "off"), "weak": w, "strong": s, "reproduced": bool(ok),
           "scope": "mechanism (model-free); the model output needs TP8 on B300"}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
