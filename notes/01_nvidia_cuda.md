# NVIDIA CUDA compiler stack for AI: problems, limitations, and NVIDIA's 2025–2026 response (research date 2026-09-22)

## 1. Structural / lock-in problems

**PTX vs SASS: the real ISA is closed**
- PTX is a virtual ISA (unbounded virtual registers, no scheduling, no control bits); closed `ptxas` lowers to undocumented SASS and, since Volta, writes a 21-bit per-instruction control field (stall counts, barriers) the hardware relies on for correctness. "There is no official SASS assembler". (Jul 27, 2026) https://www.thesoftwarefrontier.com/p/how-the-nvidia-compiler-moat-actually
- "ptxas makes locally optimal scheduling decisions that can be globally suboptimal"; CuAsmRL searches SASS schedules with RL and beats ptxas defaults on fused attention. (Jun 19, 2026) https://vanshverma.com/notes/cuasmrl-sass-scheduling
- Name-conditioned heuristics: ptxas optimizes differently when kernel symbol contains "cutlass" (Dec 15, 2025) https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/ ; CUTLASS issue #3389 (Jul 15, 2026): identical sm_100a PTX → 80 regs / 8,076 B spill stores with "cutlass" in name vs 168 regs / 56 B stack without, +31% kernel time. https://github.com/NVIDIA/cutlass/issues/3389
- Limited control over spilling (2017 forum: -O1 removed spills -O3 produced). https://forums.developer.nvidia.com/t/is-it-possible-to-stop-ptxas-from-spilling-registers/49689

**Inline PTX remains necessary**
- Triton had to revert Blackwell tcgen05 lowering to inline asm because LLVM NVPTX couldn't select tcgen05 intrinsics for sm_103. (Sep 3, 2025) https://github.com/triton-lang/triton/pull/8045
- DeepSeek-V3: "we employ customized PTX (Parallel Thread Execution) instructions and auto-tune the communication chunk size". (Dec 27, 2024) https://arxiv.org/html/2412.19437v1

**Forward-compatibility breaks across generations**
- Blackwell compat guide: `sm_100a`/`compute_100a` binaries "are not forward or backward compatible"; "PTX compiled for compute_90a (Hopper) are not supported on the Blackwell architecture." https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
- CUDA 12.9 family-specific `sm_100f` partial fix; "no forward-compatibility for either PTX or a cubin when using the architecture-specific `a` suffix." (May 1, 2025) https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- Hopper `wgmma.mma_async` deprecated on Blackwell → `tcgen05.mma` with accumulators in Tensor Memory; "tcgen05 operations are now issued by a single thread on behalf of the entire CTA". https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor
- "FA3 (the existing SOTA implementation on Hopper) did not work on Blackwell. WGMMA no longer exists on SM100". (Mar 4, 2026) https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- Intra-generation fragmentation: consumer Blackwell (SM12x, DGX Spark CC 12.1) lacks TMEM and tcgen05; SM100-only kernels (FlashMLA, DeepGEMM) fail. (Feb 19, 2026) https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell ; Triton: "Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'" (Sep 13, 2026) https://github.com/triton-lang/triton/issues/11747
- CUDA 13.0 removed offline compilation for Maxwell/Pascal/Volta. (Aug 6, 2025) https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/

**EULA prohibition on translation layers (ZLUDA)**
- CUDA EULA §1.2 item 8 (v13.4, Jan 26, 2026): "You may not reverse engineer, decompile or disassemble any portion of the output generated using SDK elements for the purpose of translating such output artifacts to target a non-NVIDIA platform." https://docs.nvidia.com/cuda/eula/index.html
- Clause online since 2021, added to installed EULA with CUDA 11.6+, noticed Mar 2024. https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers
- vosen (ZLUDA): EULA matters "as much as Windows EULA affects WINE and Intel EULA affects Apple's Rosetta." https://github.com/vosen/ZLUDA/issues/161
- AMD legal asked Janik to take AMD-funded code down (Aug 9, 2024). https://www.theregister.com/2024/08/09/amd_zluda_take_down/

**"CUDA moat" debate**
- SemiAnalysis (Jan 16, 2023): "The default software stack for machine learning models will no longer be Nvidia's closed-source CUDA." https://newsletter.semianalysis.com/p/nvidiaopenaitritonpytorch
- Lattner (Mar 26, 2025): Triton ~20% loss vs CUDA on H100; "Triton trades performance for productivity".
- Nicholas Wilt on cuTile: "It's hard not to suspect that cuTile was developed directly to counter Triton." (Dec 2025) https://hyper.ai/en/news/47715
- Counterpoint (thesoftwarefrontier): "The closedness is not primarily a business decision. It is downstream of an architectural decision made for area and power reasons around 2012."

## 2. Developer productivity problems

**CUTLASS/CuTe C++ template complexity and compile time**
- CUTLASS 2.2 profiler build "upwards of 45 minutes" (Sep 2020) https://github.com/NVIDIA/cutlass/issues/132 ; "[BUG] Compilation Times too Damn High": 17 min 22 s for two int8 GEMMs (Aug 2023) https://github.com/NVIDIA/cutlass/issues/1042 ; non-deterministic nvcc hangs ~1 in 20 builds https://github.com/NVIDIA/cutlass/issues/1092
- PyTorch team: "anyone who has tried to install FlashAttention knows how painful it can be, with long compile times" (Mar 4, 2026).
- CuTe DSL pitch: "Express compile-time configuration with Python instead of deeply nested C++ templates." FA4: "20-30× faster compile times compared to traditional C++ template-based approaches". (Mar 5, 2026) https://arxiv.org/abs/2603.05451

**Gap between compiler output and hand-tuned kernels**
- FA2 "only 35% utilization of theoretical max FLOPs on the H100"; FA3 hand-written (warp specialization, WGMMA/TMA, pingpong) → 740 TFLOPS / 75%. https://tridao.me/blog/2024/flash3/
- FlexAttention (Triton) ~60% of FA3 on Hopper; Blackwell "What was once a small gap has grown to a chasm!"; CuTeDSL/FA4 backend 1.6–3.2× over Triton on GB200.
- FA4: ~1,613 TFLOPs/s on B200 (71%), 1.3× cuDNN 9.13, 2.7× Triton, via software-emulated exponentials (cubic polynomial exp2) and conditional rescaling to work around SFU-vs-tensor-core asymmetry. https://modal.com/blog/reverse-engineer-flash-attention-4
- Gluon tutorial: compiler "can be beaten by hand-tuned low-level code" and then "there is little the user can do."
- Hand-written async code fragile: FlashInfer CuTe DSL Blackwell kernel released TMEM stage while `tcgen05.ld` in flight → corrupted outputs; missing `fence_view_async_tmem_load()`. (Sep 18, 2026) https://github.com/flashinfer-ai/flashinfer/issues/5334

**DeepSeek: PTX in Jan 2025, DeepGEMM JIT + SASS patching**
- Tom's Hardware (Jan 28, 2025) citing Mirae Asset: DeepSeek used PTX; HN: "Every serious project has at least some parts of their kernels implemented in PTX/AMDGCN." https://news.ycombinator.com/item?id=42859909
- DeepGEMM (Feb 26, 2025): core kernel "~300 lines", Hopper-only, "compiling all kernels at runtime using a lightweight Just-In-Time (JIT) module". FFMA interleaving hack rewrote compiled SASS, flipping yield/reuse bits after noticing NVCC 12.2 vs 12.3 differences; up to 2.7× on small-M. Current README: "As NVCC 12.9 will automatically do the FFMA interleaving, all post optimizations will be no longer supported". https://github.com/deepseek-ai/DeepGEMM

## 3. Compiler lag behind hardware (dates)
Anchors: H100 shipping fall 2022; Blackwell announced Mar 18, 2024; CUDA 12.8 first Blackwell toolkit (Jan 31, 2025).

**Hopper (wgmma, TMA, warp specialization)**
- Triton: NVIDIA "Initial code merge of Hopper support" Aug 7, 2023, GMMA/TMA/auto warp-specialization "experimental for now and turned off by default"; Triton 3.0.0 Jul 9, 2024. https://github.com/triton-lang/triton/pull/2036
- Automated warp specialization only in Triton 3.2 (Jan 22, 2025) / PyTorch 2.6 (Jan 29, 2025), ~2.3 years after H100 shipped, yielding 10–15% on FA and FP8 GEMM. https://pytorch.org/blog/warp-specialization/
- TMA descriptors lost `_experimental` prefix Mar 13, 2025 (PR #6194).
- FA3 (hand-written on CUTLASS) Jul 2024, ~21 months after shipment.

**Blackwell (tcgen05, TMEM, NVFP4, 2-CTA MMA)**
- CUTLASS 3.8 (Jan–Feb 2025): tcgen05 CuTe atoms, TMEM, NVFP4/MXFP4/MXFP6/MXFP8; 2SM MMA in 3.9.0 (Apr 24, 2025); SM103/B300 in 4.2.0 (Sep 15, 2025). https://github.com/NVIDIA/cutlass/releases/tag/v3.8.0
- Triton "Add support for Nvidia Blackwell GPUs" merged Jan 28, 2025 (NVIDIA+OpenAI). https://github.com/triton-lang/triton/pull/5724 ; NVIDIA blog (Feb 5, 2025): warp specialization and MXFP4 ergonomics "ongoing".
- Triton 3.3.0 (Apr 9, 2025) → PyTorch 2.7 (Apr 23, 2025) Blackwell as prototype.
- Explicit tcgen05/TMEM control (Gluon) started May 27–29, 2025; shipped Triton 3.4.0 (Jul 30, 2025).
- 2-CTA and NVFP4×NVFP4 maturity only in Triton 3.8.0 (Aug 28, 2026); initial Rubin SM107. https://github.com/triton-lang/triton/releases/tag/v3.8.0
- TorchInductor: Inductor Triton template for DeepSeek 1×128/128×128 scaled_mm on Blackwell still open PR Sep 1, 2026. https://github.com/pytorch/pytorch/pull/195577 ; stable sm_120 (RTX 50) support still requested Oct 1, 2025. https://github.com/pytorch/pytorch/issues/164342
- Net: CUTLASS ≈ day-0 (Jan 2025); Triton basic +0–3 months; PyTorch prototype +3 months; explicit TMEM/2-CTA/warp-specialized attention in Python DSLs +9–20 months (Gluon, CuTe DSL FA4 Sep 2025–Mar 2026).

## 4. NVIDIA's responses, 2025–2026

**CuTe DSL (Python)**
- CUTLASS 4.0 announced May 14, 2025: "native Python support ... powered by the new CuTe DSL". FAQ: "while the DSL is still in beta, we do not promise any portability"; compiles to PTX then toolkit ptxas. https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html
- 4.4.0 (Feb 14, 2026) AoT + JAX; 4.7.0 (Aug 4, 2026) Primitives API, compile-time register-spill detection, sync-hazard reporting; 4.8.0 (Sep 17, 2026) Rubin SM107, Windows. https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md
- Adoption: FA4, PyTorch Inductor CuTeDSL codegen for FlexAttention, FlashInfer Blackwell kernels.

**cuTile / CUDA Tile IR**
- GTC 2025 (Mar 20, 2025), Mehdi Amini: "The CUDA GPU driver will now include a #MLIR-based JIT compiler!" https://x.com/JokerEph/status/1902758983116657112
- CUDA 13.0 (Aug 6, 2025): "laying the foundation for a second, complementary model: tile-based programming."
- CUDA 13.1 (Dec 4, 2025): cuTile Python, Tile IR (new virtual ISA), `tileiras`; "the largest and most comprehensive update to the CUDA platform since it was invented two decades ago." Release notes: "The initial release targets Blackwell GPUs"; Tile-IR AS "supports only Blackwell-class devices and has limited low-precision support." https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/
- Performance: cuTile GEMM "achieves over 90 percent of the performance of PyTorch calling cuBLAS" on RTX 5080 with hand-picked tiles (Jan 14, 2026); cuTile flash attention 918 TFLOPS at seqlen 16k only after manual tile sizes, `latency=` hints, fast-math, K-loop splitting (Mar 5, 2026). https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/
- Tile IR open-sourced Dec 25, 2025 (Apache 2.0 w/ LLVM exceptions); "does not accept external contributions". https://github.com/NVIDIA/cuda-tile
- Roll-out: 13.2.0 (Mar 24, 2026) Ampere/Ada; 13.3.0 (May 28, 2026) Hopper sm_90 + `mmaf_scaled`, `i4`, `f4E2M1FN`; 13.4.0 (Sep 10, 2026) Rubin sm_107 preview, PDL. Hopper supported last. https://github.com/NVIDIA/cuda-tile/releases
- CUDA 13.3 (May 26, 2026): CUDA Tile C++ (`nvcc --enable-tile`, CC 8.0+) and CompileIQ compiler autotuning (evolutionary search) "up to a 15% speedup on already-optimized Triton attention and CUTLASS GEMM kernels". https://developer.nvidia.com/blog/nvidia-cuda-13-3-enhances-gpu-development-with-tile-programming-in-c-compiler-autotuning-and-python-updates
- Language spread: cuTile.jl (Mar 3, 2026); CUDA Rust (Sep 8, 2026): `cuda-oxide` (SIMT) and `cutile-rs` (Tile; used by Hugging Face Grout, mistral.rs), NVIDIA: "Reach for Tile first." https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/

**Triton: adopted and countered simultaneously**
- Adopted: NVIDIA co-authored Triton Blackwell support; bundles Blackwell ptxas in Triton releases; Triton-to-TileIR backend (blog Jan 30, 2026; `ENABLE_TILE=1`; experimental, Blackwell-validated, requires CUDA 13.4; unsupported inline PTX, some control flow; weak on small GEMMs). https://github.com/triton-lang/Triton-to-tile-IR
- Countered: cuTile occupies Triton's abstraction level with a closed driver JIT; Triton's answer to the perf ceiling is Gluon.
- Tom's Hardware: CUDA Tile as model "for Rubin, Feynman, and beyond". (Dec 31, 2025)

**cuda.core / CUDA Python, Warp**
- CUDA Python 1.0 (Aug 25, 2026): `cuda.core` 1.0, `cuda.compute` 1.0, `nvmath-python` 1.0 under semver: "Python is now a supported way to use the CUDA platform." https://developer.nvidia.com/blog/cuda-python-1-0-stable-apis-one-foundation-full-platform-access
- Warp added tile-based `wp.tile_*`; 1.17.0 (Aug 31, 2026).

**PTX 9.x / CUDA 13**: PTX 9.0 (CUDA 13.0) new targets sm_88, sm_110; `.blocksareclusters`; `enable_smem_spilling`; PTX 9.4 w/ SM_107 (CUDA 13.4, Sep 9, 2026).

**GTC 2026 (Mar 16–19, 2026)**: session "CUDA: New Features and Beyond" (S81859) not retrievable. Documented in GTC window: CUDA 13.2 tile Ampere/Ada (Mar 9), cuTile.jl (Mar 3), cuTile FA tuning (Mar 5), FA4 paper (Mar 5), FlexAttention CuTeDSL backend (Mar 4).

## 5. Known bugs / practitioner complaints
- ptxas hung indefinitely on ~800 lines CUDA 11.8 code (Aug–Sep 2024). https://forums.developer.nvidia.com/t/nvcc-compilation-stuck-at-ptxas/304527
- Triton `tl.dot_scaled` FP4 emulation on sm75: 110 s compile, 40,772-line PTX, 8,466 spills (Jul 17, 2026). https://github.com/triton-lang/triton/issues/10918
- Triton cache breaks on shared network filesystems (Aug 29, 2026). https://github.com/triton-lang/triton/issues/11512
- Miscompilations: ptxas 12.9.86 drops immediate of second `add.s16x2` → wrong results on H100 (Sep 4, 2026) https://github.com/triton-lang/triton/issues/11581 ; ptxas VIMNMX3 fusion drops negated integer-abs operand on GB200 in Inductor FlexAttention kernel (Jul 24, 2026, fixed ptxas 13.4.46) https://github.com/pytorch/pytorch/issues/190973 ; nvcc 13.1 ptxas silently drops `createpolicy` → illegal instruction sm_90 (Marlin, May 1, 2026) https://github.com/IST-DASLab/marlin/issues/44 ; CUDA 13.3 ptxas miscompiles CUB warp-shuffle reduction (Aug 23, 2026) https://github.com/NVIDIA/cccl/issues/10958 ; CuTeDSL ≥4.6 emits PTX ptxas rejects (Aug 31, 2026) https://github.com/NVIDIA/cutlass/issues/3573 ; CUDA 13.2 notes admit fixes for "internal compiler errors, incorrect structure layout, and runtime miscompilations".
- Version matrix: "ptxas : Unsupported .version 8.4; current version is '8.2'" broke AlphaFold 3 users (Dec 9, 2024) https://github.com/jax-ml/jax/issues/25344 ; Meta/NVIDIA negotiating bundled `ptxas-blackwell` (13.3.33 vs 13.4.46) for Triton 3.8 / PyTorch 2.14 (Jul 24, 2026) https://github.com/triton-lang/triton/issues/11038
- JIT PTX latency: driver JIT at context creation; compute cache (256 MiB) invalidated on driver upgrade. https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/

## Fundamental vs. transient
| Problem | Class | Why |
|---|---|---|
| Closed SASS / opaque ptxas scheduling, control bits | Fundamental (architectural) | Correctness depends on per-arch latency tables NVIDIA won't freeze |
| ISA churn per gen (mma.sync → wgmma → tcgen05), `sm_XXa` non-forward-compat | Fundamental, partly mitigated | `sm_100f`, Tile IR are mitigations |
| Consumer vs datacenter fragmentation (SM12x lacks TMEM) | Fundamental (product segmentation) | |
| Compiler vs hand-tuned gap on async, warp-specialized kernels | Fundamental in kind, shrinking in degree | Compilers can't easily discover global pipeline choreography; Gluon/CuTe DSL shift work to humans |
| EULA ban on translation layers | Fundamental (legal/economic) | |
| DSL proliferation (Triton, Gluon, CuTe DSL, cuTile Py/C++/Rust/Julia, Warp) + vendor-controlled compiler | Fundamental (economic) | NVIDIA's incentive to own the abstraction layer |
| Driver/toolkit/ptxas version matrix | Fundamental in structure, transient per instance | |
| nvcc/CUTLASS C++ compile times | Transient | CuTe DSL 20–30× faster |
| Specific ptxas/nvcc miscompiles | Transient, recurring | |
| Triton/Inductor lag per generation | Transient per gen, recurring | ~1–2.5 yrs Hopper; ~3–18 months Blackwell; recurs with Rubin |
| cuTile coverage/perf gap | Transient | Blackwell-only → Ampere/Ada → Hopper → Rubin in 9 months |
| CuTe DSL beta / CUDA Python churn | Transient | CUDA Python 1.0 semver Aug 2026 |
| JIT cold-start | Transient | caches, AoT |

## Key quotes
1. "PTX is a virtual ISA with unbounded virtual registers, no notion of the physical register file, no scheduling, and no control bits." — The Software Frontier, Jul 27, 2026
2. "...for the purpose of translating such output artifacts to target a non-NVIDIA platform." — CUDA EULA §1.2(8)
3. "I expect that CUDA's EULA will affect ZLUDA as much as Windows EULA affects WINE..." — vosen, Mar 5, 2024
4. "The new hardware features (GMMA, TMA, STMATRIX etc.) and automatic warp-specialization are experimental for now and turned off by default." — Triton PR #2036, Aug 2023
5. "When this happens, there is little the user can do to significantly improve performance since all the details are hidden." — Gluon tutorial
6. "What was once a small gap has grown to a chasm!" — PyTorch team, Mar 4, 2026
7. "we employ customized PTX (Parallel Thread Execution) instructions..." — DeepSeek-V3, Dec 27, 2024
8. "It's hard not to suspect that cuTile was developed directly to counter Triton." — Nicholas Wilt, Dec 2025
