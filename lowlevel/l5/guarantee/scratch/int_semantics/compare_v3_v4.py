"""The L5.4d counterexamples (two address ones, one scale read past the K groups) against kernel_ir of v3 (2a51daf)
and of the working tree: python compare_v3_v4.py"""
import importlib.util
import os
import sys
import time

ROOT = r"<workspace>\entail"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_kernel_ir as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


src = open(os.path.join(T.DATA, "orig.ttir"), encoding="utf-8").read()
cases = [(n, t) for n, t, _v3, _now in T.address_counterexamples(src)]
cases += [(n, t) for n, t, _w in T.counterexamples(src) if n.startswith("one more iteration")]
cases.insert(0, ("vLLM's kernel as compiled", src))
for label, mod in (("v3 2a51daf", load(os.path.join(HERE, "kernel_ir_2a51daf.py"), "kir_v3")),
                   ("v4 working tree", load(os.path.join(ROOT, "entail", "kernel_ir.py"), "kir_v4"))):
    print("==", label)
    T.binding.__globals__["Tensor"] = mod.Tensor
    vals = dict(M=64, N=T.N, K=T.K, group_n=128, group_k=128, stride_am=T.K, stride_bn=T.K, stride_cm=T.N,
                stride_As_m=T.NB, stride_Bs_n=T.NB, ROWS_FROM=8)
    for name, ttir in cases:
        v = mod.check_launch(ttir, T.binding(64), vals, (T.N // 128,))
        print(f"  {name[:70]:70s} {v.verdict:9s} {v.why[:120]}")
    # the largest launch of the engine's profile run: 8192 x 19456 x 2560
    M, N, K = 8192, 19456, 2560
    NB = K // 128
    b = {"A": mod.Tensor("activation", 1, 2, (M, K), (K, 1), (1, 128)),
         "As": mod.Tensor("activation_scale", 2, 1, (M, NB), (NB, 1), (1, 128)),
         "B": mod.Tensor("weight", 3, 4, (N, K), (K, 1), (128, 128)),
         "Bs": mod.Tensor("weight_scale", 4, 3, (N // 128, NB), (NB, 1), (128, 128)),
         "C": mod.Tensor("output", 0, 0, (M, N), (N, 1))}
    vals = dict(M=M, N=N, K=K, group_n=128, group_k=128, stride_am=K, stride_bn=K, stride_cm=N, stride_As_m=NB,
                stride_Bs_n=NB)
    t = time.perf_counter()
    v = mod.check_launch(src, b, vals, (-(-M // 64) * (N // 128),))
    print(f"  largest launch {M}x{N}x{K}: {v.verdict}, {v.programs} programs, {time.perf_counter() - t:.2f} s")
