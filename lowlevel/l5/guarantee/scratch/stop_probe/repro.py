"""The review's two CPU counterexamples and two more, against kernel_ir: python repro.py [kernel_ir.py of a commit]"""
import os, re, sys
ROOT = r"<workspace>\entail"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_kernel_ir as T
if len(sys.argv) > 1:                      # a kernel_ir.py of another commit (git show <commit>:entail/kernel_ir.py)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kernel_ir_under_test", sys.argv[1])
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    check_launch = K.check_launch
    T.binding.__globals__["Tensor"] = K.Tensor
    print("kernel_ir from", sys.argv[1])
else:
    from entail.kernel_ir import check_launch
    print("kernel_ir from the working tree")

src = open(os.path.join(T.DATA, "orig.ttir"), encoding="utf-8").read()
vals = dict(M=64, N=T.N, K=T.K, group_n=128, group_k=128, stride_am=T.K, stride_bn=T.K, stride_cm=T.N,
            stride_As_m=T.NB, stride_Bs_n=T.NB, ROWS_FROM=8)
grid = (-(-64 // 64) * (T.N // 128),)

def run(name, ttir):
    v = check_launch(ttir, T.binding(64), vals, grid)
    print(f"{name:28s} verdict={v.verdict:9s} terms={v.terms} why={v.why!r}")

run("orig", src)
no_store = "\n".join(l for l in src.splitlines() if "tt.store" not in l)
run("no store", no_store)
# loop bound: K tiles = %1 -> one iteration only (first of 20 K groups)
one_iter = src.replace("scf.for %k = %c0_i32 to %1 step", "scf.for %k = %c0_i32 to %c1_i32 step")
assert one_iter != src
run("loop over first K group only", one_iter)
# accumulator overwritten instead of accumulated: last group only
overwrite = src.replace("%accumulator_79 = arith.addf %accumulator_59, %accumulator_78",
                        "%accumulator_79 = arith.addf %cst_3, %accumulator_78")
assert overwrite != src
run("acc overwritten (last group)", overwrite)
# store half the rows: mask rows < M/2 via a shifted compare (M -> 32 const)
half = src.replace("%c_mask = tt.splat %M : i32 -> tensor<64x1xi32>", "%c_mask = tt.splat %c32_i32 : i32 -> tensor<64x1xi32>")
assert half != src
run("store only rows < 32", half)
# store the negated accumulator
neg = src.replace("%c = arith.truncf %accumulator#2", "%negacc = arith.subf %cst_3, %accumulator#2 : tensor<64x128xf32>\n    %c = arith.truncf %negacc")
assert neg != src
run("store -acc", neg)
