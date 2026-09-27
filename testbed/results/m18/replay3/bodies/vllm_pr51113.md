> ### Credit
>
> **The original work here is @kodek's (#45477) and @yanghui1-arch's (#47861).** They found this root cause and wrote this fix first; both are co-authors of this PR.
>
> Their branches went stale on rebase, not on review. What I added is the same fix re-derived against current `main`, the regression tests, and the measurements below.

## Purpose

Fix prefix-cache poisoning on hybrid Mamba/GDN models in `mamba_cache_mode="align"` when EAGLE/MTP speculative decoding is on — the correctness half of #43559.

`MambaManager.cache_blocks` hashes block-table slot `p` as the recurrent state at exactly `(p + 1) * block_size` tokens. `_mamba_block_aligned_split` only enforced that invariant *below* `last_cache_position`, which EAGLE backs off by one block (and zeroes for prompts shorter than two mamba blocks). Past that point any chunk end was accepted, so:

1. Concurrent prefills share the token budget; one request's chunk ends mid-block, leaving its running-state slot holding e.g. `state@364`.
2. A later chunk crosses `1600`, so `cache_blocks` hashes that same slot as `state@1600`.
3. Every request resuming from that hash silently restores a truncated state. Nothing raises, nothing logs, and the entry persists until restart.

A single request is accidentally safe — its low slots are null blocks, which `cache_full_blocks` skips — which is why this only appears under concurrency and was hard to pin down.

The same hole exists for an unaligned *start*: the mid-block realign stop was gated on `next_block_boundary <= last_cache_position`, so a prefill resuming off-grid (finer `prefix_match_unit`, or unaligned externally computed tokens from a KV connector) could run straight past its block's boundary and poison that slot too.

### Fix

- Gate the end-alignment on `prefill_end` rather than `last_cache_position`. Only the prefill's final chunk may end unaligned; decode completes that slot via the fused postprocess path.
- Drop the `last_cache_position` condition on the mid-block realign stop.

`last_cache_position` is kept as a mandatory stop — it still serves hit rate by materializing a snapshot where a same-length request's eagle-dropped hit actually lands. A chunk the budget cannot fund returns 0, which both call sites already treat as "defer" (`scheduler.py` even lists it as skip reason 4).

### Credit and prior work

This is not an original diagnosis. Two earlier PRs reached the same root cause independently, and both are credited as co-authors on the commit:

- **#45477 (@kodek)** — same diagnosis, and its commit 1 is the same core change: gate the chunk-end alignment on the prefill end rather than `last_cache_position`. It also carries the end-to-end validation this PR cannot reproduce locally (10 identical 2002-token requests on Qwen3.6-27B-FP8, 5/10 corrupted -> clean, decode throughput unchanged). It has been `needs-rebase` since 2026-07-14 and its base is 859 commits behind `main`: `_mamba_block_aligned_split` was since restructured by #47782 (which removed the `num_uncached_common_prefix_tokens` parameter that PR edits) and #50000, so the if/elif chain it rewrites no longer exists. This PR is that fix re-derived against current `main`.
- **#47861 (@yanghui1-arch)** — converged on an equivalent scheduler half plus a coordinator half (`supports_eagle_cache_peek`, false for `MambaSpec`). Closed unmerged on 2026-07-19 after going stale on conflicts.

I am glad to close this in favour of a rebased #45477 if @kodek prefers to carry it; the intent here is to unstick a fix that has died on rebase twice, not to displace it.

One difference worth review: #45477 nests its mid-block realign inside `num_computed_tokens < last_cache_position`, so a mid-block start at or past `last_cache_position` can still run through the boundary. This PR drops that condition, and `test_unaligned_resume_never_runs_past_its_block` covers it. That reading is from the diff — I have not run #45477's branch, so @kodek should confirm.

### Other related PRs

- **#43650 / #48375** prune the final mamba block from cache-hit lookup. That hides the poisoning only when all requests are the same length — the poisoned block is then everyone's pruned final block — and a longer request sharing the prefix still hits the poisoned non-final block. They also cost a full mamba block (560–1600 tokens) of hit length. Removing the poison at the source costs nothing on the hit path. Complementary rather than competing, if a drop is wanted for other reasons.
- **#45614 / #39171** operate on the coordinator + manager hit-length layer, not the chunk split.
- **#48815** (env-gated eagle backoff skip) and **#50409** (`replay_boundary` stop) touch the same function but are hit-rate changes; both compose with this.
- **#48970** adds e2e hybrid-mamba prefix-cache corruption tests; complementary, and a better home for end-to-end coverage than this PR.
- **#50897** is a separate mechanism (lookahead-aware hashing for draft KV) and does not touch this function.

This PR does not close #43559 on its own: the thread also covers CUDA illegal-access crashes (#40756, #50021), the MTP + reasoning off-by-one (#44927), and degenerate loops on deep agentic conversations (#47087).

## Test Plan

```bash
.venv/bin/python -m pytest tests/v1/core/test_mamba_align_chunk_split.py -q
.venv/bin/python -m pytest tests/v1/core/prefix_cache -q
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py tests/v1/core/test_prefix_caching.py \
    tests/v1/core/test_single_type_kv_cache_manager.py tests/v1/core/prefix_cache -q
```

New CPU-only regression tests in `tests/v1/core/test_mamba_align_chunk_split.py` drive the real `KVCacheManager` (full-attn 16 + mamba-align 1600, `num_speculative_blocks=3`, `use_eagle=True`) and the real split helper, with an oracle mirroring the GDN/postprocess state writes, and assert that every hash-cached mamba slot holds its boundary state. `test_unaligned_resume_never_runs_past_its_block` covers `partial_hit` on/off x five resume offsets.

## Test Result

| | pristine `main` | this PR |
|---|---|---|
| `test_mamba_align_chunk_split.py` | **14 failed** / 6 passed | **20 passed** |
| `tests/v1/core/prefix_cache` | 29 passed | 29 passed |
| scheduler + prefix_caching + single_type + prefix_cache | 35 failed / 229 passed | 35 failed / 229 passed |

The 35 failures are pre-existing on this checkout (encoder-cache / EC-connector tests) and identical with and without the patch.

The poisoning shows up on `main` as:

```
AssertionError: mamba slot 0 is hashed as state@1600 but holds state@364
```

The unaligned-resume failures on `main` land exactly where the old gate flips: with `last_cache_position = 1600`, offsets `331` and `1599` pass and `1601`, `2531`, `3011` fail, for both `partial_hit` values.

### Scheduling cost

Enumerated old vs new chunk counts over 85 configs (block size × prompt length × per-step budget share, EAGLE on, `max_num_scheduled_tokens=8192`):

- configs where the new split costs extra forward steps: **0 / 85**
- configs with fewer cacheable boundary snapshots: **0 / 85** (2 gain one)

The affected window is `[last_cache_position, prefill_end]`, whose size is `(P mod block_size) + block_size` — under two blocks. Whenever the step's budget covers it, `end == prefill_end` and the new gate is a no-op, so single-request and light-load prefill are bit-identical to before. Under contention a chunk shrinks to the boundary or the request defers a step; the token budget is shared, so a deferred request does not idle the step.

### Real-workload audit

The invariant can be checked at runtime, so rather than inferring the bug from an accuracy delta I measured it directly. A runtime audit hook records, per request, the token offset each mamba slot actually holds (mirroring the kernel: a prefill chunk leaves the running slot at the chunk end) and flags any slot published under a different offset. Real 20-shot gsm8k prompts, 200 questions, `Qwen/Qwen3.5-0.8B` (hybrid GDN, mamba block 560), MTP `num_speculative_tokens=3`, `--enable-prefix-caching`, `max_num_seqs=32`, `max_num_batched_tokens=8192`:

| | poisoned entries published |
|---|---|
| pristine `main` | **1** |
| this PR | **0** |

```
[UNPATCHED] publish req=11-9496b671 slot=6 hashed as state@3920 but holds state@3662
```

That is the bug happening on ordinary gsm8k traffic: a slot advertised as the state after 3920 tokens while holding the state after 3662, reachable by any later request whose hit lands on it.

### Model evaluation

gsm8k 20-shot, 500 questions, `Qwen/Qwen3.6-27B`, TP=1, `--enforce-eager`, greedy, on unpatched `main`:

| config | accuracy |
|---|---|
| baseline | 0.814 |
| MTP only | 0.814 |
| APC only | 0.804 |
| MTP + APC | 0.802 |

At n=500 the 1σ band is ±1.7 points, so this matrix neither confirms nor refutes an end-to-end effect and I am not claiming an accuracy delta from it. The audit above explains why: on a homogeneous same-length gsm8k workload the poisoning fires rarely and the eagle drop usually keeps the reconciled hit below the poisoned slot, so little of it is consumed. The failure mode reporters describe needs mixed-length requests over a shared prefix, where a longer request's hit reaches the poisoned slot. #45477 reports that end-to-end amplification on Qwen3.6-27B-FP8.

## Note on AI assistance

AI assistance (Claude Code) was used for the investigation, the patch, and the tests. Prior art in #45477 and #47861 is credited above and on the commit. The root cause was derived from the code and confirmed by the unit-level reproduction above before the related PRs were read. All results in this description were produced by running the listed commands on this branch.

