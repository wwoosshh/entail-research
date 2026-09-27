
## Summary

PR #30512 and #30645 fixed the *crash/validity* symptoms of threshold-bin tie
overflow in the fused DSA top-k v2 kernel (all output slots written, no `-1`
inside the valid prefix, inf-boundary handling). However the *exactness* gap
remains, on `main` as of `ea27e3d`: when more than `kMaxNumTie = 2048`
elements fall into the threshold coarse bin, the collect pass keeps whichever
2048 arrive first (`atomicAdd` order — scheduling-dependent, value-independent)
and the exact tie-break then runs on that arbitrary subset:

```cpp
// topk_impl.cuh (main @ ea27e3d, L595 / L618; same in v0.5.15–v0.5.17)
const auto count_eq = atomicAdd(&smem->count_eq, 1);
if (count_eq < kMaxNumTie) [[likely]]
  smem->tie.values[count_eq] = {val, idx};       // overflow: value silently dropped
...
const auto tie_count = min(equal_count, kMaxNumTie);  // select from truncated subset
```

Elements in one coarse bin share only the top fp16 bits — they are *not*
fp32-equal — so the dropped candidates can be genuinely larger than the kept
ones. The kernel then returns indices that are not the true top-k, with no
error or flag, and sparse attention reads wrong KV positions. Note
`count_eq` already holds the true bin population, so the overflow is
detectable for free; it is just not acted on.

This is distinct from the ReLU-degenerate exact-tie case discussed in #30645
(thousands of positions on one exact value — there any subset is acceptable).
Our comparison below is against the `torch.topk` **fp32 score multiset**, i.e.
exact ties with different indices are accepted; the mismatches are real value
losses.

## Evidence on real workloads

GLM-5.2-FP8, 8xB200 TP8, official v0.5.17 image, ~1M-token contexts. We dumped
real indexer score rows (~21 top-k layers x 2 adjacent decode steps = 42 rows
per workload) and compared kernel output to `torch.topk` fp32 multisets:

| workload (1M ctx, batch=1) | rows not exact |
|---|---|
| real model logits (synthetic prompt) | 3 / 42 |
| SGLang source-code corpus | 2 / 42 |
| Gutenberg novels corpus | 2 / 42 |

Additional data points:

- Measured threshold-band candidate populations on these rows reach
  **4.7k–31k** vs the 2048 buffer; on the worst rows the output missed
  **1084–1591 of 2048** true top-k members (measured on the same mechanism in
  the v0.5.15 era).
- In a 32K–1M x batch 1–32 microbench matrix, 9 of 21 cells were not exact on
  at least one of five real-score inputs (64K–512K, batch 2–8 included).
- #30512's own device-side diagnostic corroborates trigger frequency: 41
  overflowing rows within a ~9-minute GLM-5.2 bench (`equal_count` up to 1536
  vs the then-1024 cap).

## Reproduction

### A. Suggested minimal repro (synthetic, no model)

Sketch adapted from our validation harness — construct one row where >2048
fp32-distinct values share one fp16 coarse bin:

```python
import torch
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_512_v2

N, TOPK, PAGE = 262_144, 2048, 64
dev = "cuda"

scores = torch.full((1, N), -100.0, dtype=torch.float32, device=dev)
n_narrow = 50_000   # 50k distinct values inside one fp16 bucket
scores[0, :n_narrow] = 1.0 + torch.arange(n_narrow, device=dev, dtype=torch.float32) * 1e-9

seq_lens = torch.tensor([N], dtype=torch.int32, device=dev)
page_table = torch.arange((N + PAGE - 1) // PAGE, dtype=torch.int32, device=dev).unsqueeze(0)
out = torch.full((1, TOPK), -1, dtype=torch.int32, device=dev)

plan = plan_topk_v2(seq_lens)
topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE, plan)

ref_idx = torch.topk(scores[0], TOPK).indices
ref_phys = page_table[0, ref_idx // PAGE] * PAGE + ref_idx % PAGE
missing = set(ref_phys.tolist()) - set(out[0].tolist())
print(f"true top-k members missing from kernel output: {len(missing)} / {TOPK}")
```

The true top-2048 are the *last* 2048 of the narrow band; arrival order keeps
an arbitrary subset, so `missing > 0` whenever the overflow path is taken.
(Post-#30512/#30645 the output contains valid indices — the failure is purely
exactness.) We will follow up with the exact numbers from a fresh run of this
script on B200.

### B. Real rows

We can share the dumped 1M score rows and the multiset-comparison harness used
for the table above on request.

## Suggested fix

1. **Minimal**: after collect, if `count_eq > kMaxNumTie`, take a slow path
   instead of proceeding on the truncated subset. The counter already exists;
   non-overflowing rows pay nothing.
2. **Complete**: progressive refinement — on overflow, subdivide the threshold
   bin by the next bits of the order-preserving fp32 key (12→10→10 covers all
   32 bits) until the candidate set fits; at full depth, remaining ties are
   exactly equal fp32 values and any subset is correct. We run this scheme in a
   production-shaped kernel: 126/126 real rows exact on the three workloads
   above; the refinement branch is rarely taken so the common path is
   unaffected.

A narrow-distribution regression test (shape of Repro A) in
`test/registered/kernels/ops/attention/test_topk_v2.py` would prevent
reintroduction.

