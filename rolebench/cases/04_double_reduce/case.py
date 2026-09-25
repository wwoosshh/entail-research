"""#4 REDUCTION: a value that is already all-reduced (replicated) is treated as a partial sum and reduced again.

Mechanism of SGLang #37187 / #31699 (DP attention garbage; dp_gather_partial on replicated data). Two ranks
(gloo, CPU) run a row-parallel linear: each holds half of the input dimension and computes a partial sum,
all_reduce makes it replicated (R). A later step assumes its input is still partial (P) and reduces again,
multiplying the result by the world size.
  defect:    all_reduce, then a second all_reduce on the replicated value
  fixed:     the second step knows the value is replicated and does not reduce
  reference: the same linear in one process
In spmd_types notation: after the first all_reduce the value is R; treating R as P is the violated contract.
"""
import os
import socket
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

META = {
    "id": "04", "title": "replicated value reduced again as if it were a partial sum", "fact": "REDUCTION",
    "issue": "https://github.com/sgl-project/sglang/issues/37187 ; https://github.com/sgl-project/sglang/issues/31699",
    "engine": "SGLang DP attention (mechanism, gloo on CPU)", "kind": "mechanism",
    "boundary": "tensor-parallel linear (replicated output) -> data-parallel gather (expects partial)",
    "trigger": {"feature": "DP attention with tensor parallelism", "world_size": ">= 2"},
    "symptom": "garbled", "expected_detection": "runtime (R/P reduction-state tag, spmd_types notation)",
    "compare": "fp32",
}
WORLD, N, K = 2, 32, 64


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, port, x, w, second_reduce, out_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    k = K // WORLD
    y = x[:, rank * k:(rank + 1) * k] @ w[:, rank * k:(rank + 1) * k].T  # partial sum (P)
    dist.all_reduce(y)  # now replicated (R)
    if second_reduce:
        dist.all_reduce(y)  # the consumer treats R as P
    if rank == 0:
        torch.save(y, out_path)
    dist.destroy_process_group()


def _run(ctx, second_reduce):
    ctx_mp = mp.get_context("fork")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "y.pt")
        port = _free_port()
        procs = [ctx_mp.Process(target=_worker, args=(r, port, ctx["x"], ctx["w"], second_reduce, out))
                 for r in range(WORLD)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        return torch.load(out)


def setup():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(4, K, generator=g), "w": torch.randn(N, K, generator=g)}


def defect(ctx):
    return _run(ctx, True)


def fixed(ctx):
    return _run(ctx, False)


def reference(ctx):
    return ctx["x"] @ ctx["w"].T
