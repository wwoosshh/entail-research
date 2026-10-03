# Raw data of the block FP8 guarantee evaluation: where it is

The published copy of this folder (github.com/wwoosshh/entail-research, `lowlevel/l5/guarantee/`) carries the summaries,
the freeze manifests, the environment records, every run's `result.json` (per step: what was delivered, what the next
operation read, the oracle's verdict, entail's decisions, RNG digests, timings), `run.json`, logs and the small record
files. It leaves out two kinds of files because of their size, listed one by one in the published copy's
`PUBLISH_EXCLUDED.md`:

- `tensors.npz` of every run (the activation as the producer made it, its scales, every observed output, the oracle's
  truth for each step): the inputs and outputs the verdicts were computed from.
- per-call record files of the engine runs over 1 MB (`guarantee.jsonl` of the eager and graph guarantee runs).

They are kept on the researcher's machine, in the research workspace at `lowlevel/l5/guarantee/results/` (the frozen
evaluations, and the development runs of v1 and v2) and `lowlevel/l5/guarantee/scratch/` (the later development runs,
drafts, diagnoses), and are not on GitHub.

## Checking a copy of them

- `results/HASHES.sha256` (published) lists the sha256 of every file under `results/`, the left-out ones included,
  as they are on the researcher's machine. `sha256sum -c HASHES.sha256` inside a copy of `results/` checks it file by
  file. The published text files had personal paths replaced (`/home/<user>`, `<workspace>`), so their sha256 differ
  from the list; the left-out binary files are not touched and match it.
- `scratch/HASHES.sha256` does the same for `scratch/`.
- Each version's per-case verdicts are in `results/holdout_vN/summary.json` (and `VERDICTS.md`); the tensors are
  needed only to compute them again.

The figures of the two hash lists (file count, size, the list's own sha256) are in `SUMMARY.md` and `ROADMAP.md`.
