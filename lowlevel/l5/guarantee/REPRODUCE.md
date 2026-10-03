# Reproducing the block FP8 guarantee evaluation (M19 L5.4a–L5.4d)

What is measured, and what came out, is in `SUMMARY.md`; the per-case verdicts of every holdout are in `VERDICTS.md`.
This file says how each version was run, on what, and how to check a result against its freeze.

## Environment the results were measured in

- GPU: NVIDIA GeForce RTX 4070 Ti (sm_89, 12 GB), Windows 11 driver 616.56, used from WSL2 (Ubuntu 24.04).
- Python 3.12.3 in a virtual environment `~/venvs/vllm`: torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.30.0 (each
  distribution's `RECORD` sha256 is in the environment record of every version). CUDA 13.0 toolkit at
  `/usr/local/cuda` (vLLM's engine needs `nvcc` at start).
- Model for the engine runs: Qwen3-4B-FP8 (block FP8, `weight_block_size` 128 × 128) at `~/models/Qwen3-4B-FP8`.
- Layout: the research workspace at `~/ai_compiler`, the entail repository checked out in `~/ai_compiler/entail`
  (github.com/wwoosshh/Entail, branch `l5.4a/block-fp8-guarantee`).
- The function-level suite forces vLLM's Triton block FP8 kernel; the engine runs set
  `VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel` (vLLM would choose Marlin on
  this GPU). Both choices are recorded in each environment record (`block_fp8_kernel_choice`).
- Measurements ran in a WSL distribution on the internal SSD (`work`); v1's function-level suite ran on the same
  software from a USB HDD distribution (the environment records are identical, `env_freeze_v1.json` =
  `env_work_v1.json`).

## Versions

| version | what | entail commit | freeze manifest (sha256) | environment record (sha256) | run script (`scripts/`) |
|---|---|---|---|---|---|
| v1 | L5.4a guarantee, check output | `05b9950` | `results/freeze_v1.json` (`9da6c1e1…324d`) | `env_freeze_v1.json` (`90963a8a…7314`) | `v1_holdout_cost.sh`, `v1_cost_engine_rerun.sh` |
| v2 | L5.4b check static under ENTAIL=guarantee | `ae4fb02` | `results/freeze_v2.json` (`13bc9f3a…4515`) | `env_freeze_v2.json` (`b6a52523…ae48`) | `v2_freeze.sh`, `v2_holdout_cost_engine.sh` |
| v3 | L5.4c guarantee (output) and structure experiment (static) | `2a51daf` | `results/freeze_v3.json` (`7dd41678…c3c5`) | `env_freeze_v3.json` (`3d4c1fda…0fe5`) | `v3_freeze_holdout_cost_engine.sh`; after the freeze: `v3_post_freeze_memory_and_diagnosis.sh` (tool `3cb07a8`) |
| v4 | L5.4d integer meaning, plan floor, complete runs, freeze check | `ffff858` | `results/freeze_v4.json` (`6d566c72…22ba`) | `env_freeze_v4.json` (`defddaef…bed3`) | `v4_freeze_holdout_cost_engine.sh` |

Development runs (before each freeze, with draft manifests) are `results/dev1`, `results/dev2` (v1),
`results/dev3_static` (v2), `scratch/dev4`, `scratch/dev4_full` (v3) and `scratch/dev5` (v4). The scripts of the last
three are `scripts/v3_dev4_*.sh` and `scripts/v4_dev5.sh`.

## Running a version

```bash
cd ~/ai_compiler/entail && git checkout <entail commit>
source ~/venvs/vllm/bin/activate && export PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$HOME/ai_compiler/entail
bash ../lowlevel/l5/guarantee/scripts/<run script>
```

The run scripts write into `~/ai_compiler/lowlevel/l5/guarantee/results/`. A run of a version that already has
results there writes into the same folders: copy the scripts and point `R` at another folder first.

From v4, every frozen step checks the freeze before it starts (`eval/block_fp8_guarantee/freeze_check.py`): the
sha256 of every code file the manifest lists, the entail commit (and that its tree is clean), and the environment
record, key by key. On another machine the environment record differs (GPU, driver, platform, distribution hashes),
so a reproduction there is a new freeze, not a run of these: `env_record.py`, then `freeze.py` with the same
arguments as the version's script, then the holdout (its seeds come from the new manifest's sha256).

To compare a checkout and environment with a manifest without running anything:

```bash
python eval/block_fp8_guarantee/freeze_check.py ../lowlevel/l5/guarantee/results/freeze_v4.json
```

## Checking a result

- `results/holdout_vN/summary.json` is what `eval/block_fp8_guarantee/aggregate.py results/holdout_vN` wrote. Running
  the aggregator again needs the raw tensors (`tensors.npz`, kept locally, see `RAW_DATA.md`): the normal cases'
  bitwise comparison with off A reads them. Without them the per-case verdicts in `summary.json` are what there is.
- `VERDICTS.md` is made from the summaries by `verdict_table.py`.
- `results/HASHES.sha256` lists the sha256 of every file under `results/` (also the ones not published);
  `scratch/HASHES.sha256` does the same for `scratch/`.
