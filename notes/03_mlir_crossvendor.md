# The portable AI-compiler ecosystem for GPUs: state of play (as of 2026-09-22)

## 1. MLIR: fragmentation, governance, Lattner's critique, 2026 status
- Fragmentation formally acknowledged upstream 2024-11-01 in RFC "MLIR Project Charter and Restructuring" (Renato Golin, co-signed Stella Laurenzo, Chris Lattner, Alex Zinenko, Jacques Pienaar, Mehdi Amini): overlapping partial solutions (Linalg/TOSA/TCP), absence of single IR + default pipeline like LLVM, broken code-ownership model; proposes area teams. https://discourse.llvm.org/t/rfc-mlir-project-charter-and-restructuring/82896
- "MLIR Organization & Charter" (2025-01-16) split MLIR into usage areas (Core, Tensor Compiler, Language Front-end, Hardware Design); states fragmentation had made it practically impossible to focus on a core pipeline and upstream compiler. https://discourse.llvm.org/t/mlir-organization-charter/84118
- MLIR Tensor Compiler Design Group (TCDG) formed 2025-02-03 with 11 volunteers from AMD, Intel, NVIDIA, Google, Arm, Qualcomm. https://discourse.llvm.org/t/mlir-tensor-compiler-design-group/84386
- First LLVM Area Team elections 2025-02-12 (MLIR: Zinenko, Golin, Pienaar); 2026 election (2026-02-10): Zinenko, Golin, Matthias Springer. https://discourse.llvm.org/t/2026-llvm-area-team-election-results/89778
- No shared upstream end-to-end GPU path; community explicitly chose not to build an "official" one. "MLIR Project Lighthouse" RFC (2025-06-06) = out-of-tree reference project (github.com/llvm/lighthouse). https://discourse.llvm.org/t/rfc-mlir-project-lighthouse/86738
- Lighthouse 2026-09-11 update: PyTorch import via Torch-MLIR → Linalg-on-Tensors, CPU pre-commit CI, Intel GPU (XeGPU) post-commit CI, CPU perf 60–80% of TPP-MLIR reference, no NVIDIA/AMD GPU path yet. https://discourse.llvm.org/t/mlir-project-lighthouse-update/91790
- Lattner "Democratizing AI Compute" series (Parts 1–11, 2025-01-30 to 2025-06-20): https://www.modular.com/democratizing-ai-compute
  - Part 6 (2025-03-12, TVM/XLA): neither displaced CUDA; TVM built for "traditional" AI, fragmented into vendor forks, struggled on tensor cores, slowed as NVIDIA absorbed its leaders; XLA effectively two projects (Google TPU-first XLA vs OpenXLA), fixed StableHLO op set, Google-controlled governance. https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers
  - Part 7 (2025-03-26, Triton): ~20% gap vs tuned CUDA on H100, poor gen-to-gen portability (A100-tuned kernels underperform on H100), lagging AMD, severe degradation on non-SIMT hardware, OpenAI-controlled governance. https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls
  - Part 8 (2025-04-08, MLIR): MLIR scaled before foundations settled; identity crisis between general infra and "AI solution"; early AI dialects not designed for GenAI; core devs dispersed across competing vendors; no reference stack (contrast CUDA); no MLIR-based stack matches CUDA on GenAI. https://www.modular.com/blog/democratizing-ai-compute-part-8-what-about-the-mlir-compiler-infrastructure

## 2. OpenXLA / XLA / StableHLO / PJRT / IREE / Pallas
- openxla.org members: Alibaba, AWS, AMD, Anyscale, Apple, Arm, Cerebras, Google, Graphcore, Hugging Face, Intel, Meta, NVIDIA; components XLA, StableHLO, PJRT, Shardy, XProf, Tokamax. https://openxla.org/
- PJRT plugins (2024-03-13) for Apple Metal, Cloud TPU, NVIDIA CUDA, Intel GPU; AWS Trainium via Neuron PJRT plugin (2024-12-03).
- XLA:GPU codegen: library custom calls (cuBLAS/cuDNN/NCCL), Triton-based emitters for matmul/softmax fusions, native LLVM emitters; docs claim near-roofline on Ampere only. https://openxla.org/xla/gpu_architecture
- AMD weakness: XLA issue #23574 (2025-03-10) Triton kernels via XLA on MI300X ~10x slower than via PyTorch, ThreadsPerWarp hardcoded to 32 instead of 64; fixed PR #23691. https://github.com/openxla/xla/issues/23574
- Static shapes: PyTorch/XLA only bounded dynamic shapes, experimental (XLA_EXPERIMENTAL flag), due to excessive recompilation. https://docs.pytorch.org/xla/master/learn/dynamic_shape.html
- IREE: LF AI & Data sandbox since 2024-05-23; 3.9k stars; backends CPU, CUDA, ROCm/HIP, Vulkan, Metal, experimental AMD AIE, WebGPU. Primary production user AMD: MLPerf Inference v5.0 SDXL submission (2025-04-02) used SHARK (Torch-MLIR, MLIR, IREE, shortfin), while Llama 2 70B used vLLM. https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html
- Pallas / Mosaic GPU: targets NVIDIA Hopper and Blackwell only. Blackwell matmul tutorial reaches 109.6% of cuBLAS in <150 lines by explicitly using TMA, TMEM, tcgen05 MMA, warp specialization, 2-CTA collective MMA, cluster launch control. https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html
- Tokamax (openxla): JAX/Pallas kernel library for NVIDIA GPUs and TPUs with per-backend implementations — Google ships per-backend kernels rather than one portable kernel. https://github.com/openxla/tokamax

## 3. Apache TVM
- ASF board reports: 2025-09-24: 170 commits/38 authors per quarter, 81 committers/29 PMC, releases 0.20.0 (2025-04-27), 0.21.0 (2025-07-16). 2025-12-17: ~100 commits/month, v0.22.0, tvm-ffi split out. 2026-06-17: ~100 commits/month, v0.24 (2026-05-08), tvm-ffi v0.1.11, 84 committers/30 PMC; focus on modularizing per-backend libraries. https://whimsy.apache.org/board/minutes/TVM.html
- Repo: 13.8k stars; TensorIR + Relax; positioned as foundational infra for domain-specific compilers. https://github.com/apache/tvm
- TVM-FFI (2025-10-21): standalone open ABI/FFI for kernels/DSLs/runtimes; ~0.4 µs call overhead; FlashInfer ships with it; PyTorch, JAX, CuPy, TileLang, Triton, Hidet named as integration targets. https://tvm.apache.org/2025/10/21/tvm-ffi
- NVIDIA acquired OctoAI 2024-09-25 (~US$165M reported, possibly >$250M w/ retention); services shut 2024-10-31; CEO Luis Ceze joined NVIDIA. https://www.geekwire.com/2024/chip-giant-nvidia-acquires-octoai-a-seattle-startup-that-helps-companies-run-ai-models/
- Tianqi Chen: CMU Associate Professor and NVIDIA Distinguished Engineer. https://tqchen.com/
- MLC-LLM 23.2k stars. Verdict: role change (infrastructure/glue) more than collapse.

## 4. Modular / Mojo / MAX
- Funding: US$250M Series C 2025-09-24 at US$1.6B post-money (US Innovative Technology Fund, DFJ Growth, GV, General Catalyst, Greylock); US$380M total. Vendor claims: 20–50% over latest vLLM/SGLang on B200 and MI355; 24k+ GitHub stars, 100k+ developers. https://www.modular.com/blog/modular-raises-250m-to-scale-ais-unified-compute-layer
- CUDA/ROCm-free claims (vendor, 2025-06-10): MAX on MI300X/MI325X beats vLLM on same hardware by up to 53% (prefill) / 32% (decode); MI325X+MAX matches/exceeds vLLM on H200. No independent benchmark found. https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus
- **Acquisition: Qualcomm agreed to buy Modular 2026-06-21, closed 2026-07-29, ~US$3.9B in stock; Lattner became EVP, Advanced AI Software and Platforms.** https://www.qualcomm.com/news/releases/2026/06/qualcomm-to-acquire-modular ; https://www.modular.com/blog/qualcomm-completes-acquisition-of-modular
- Open source: Mojo 1.0 shipped 2026-08-11/12 with compiler still closed; compiler/tooling open-sourced 2026-08-18 under Apache 2.0 w/ LLVM exceptions; Modular 26.6 (2026-09-17) opened compiler to external contributions. https://www.modular.com/blog/mojo-open-source
- ModCon (2026-08-18): support for AWS Trainium, Google TPUs, Qualcomm Cloud AI 100 and Dragonfly alongside NVIDIA/AMD GPUs; MiniMax flagship customer. https://www.modular.com/blog/modcon-announcements
- Criticisms: HN — closed compiler non-starter; "Python superset" promise dropped to "Python-like"; vendor-neutrality concerns post-Qualcomm. https://news.ycombinator.com/item?id=49261128

## 5. Emerging kernel DSLs and superoptimizers
- TileLang (arXiv 2025-04-24): tile-level Python DSL on TVM; decouples dataflow from scheduling. Backends CUDA SM70–SM120, ROCm/HIP, Metal; experimental LLVM CPU, CuTe DSL, WebGPU; partners Huawei Ascend, MetaX, Moore Threads, HYGON. 7.5k stars; v0.1.13 (2026-08-03); DeepSeek MLA/V3.2 kernel examples. https://github.com/tile-ai/tilelang
- ThunderKittens (Stanford Hazy): C++ header-only 16x16 register tiles; ~855 TFLOPs (86% peak) H100 matmul <100 lines; v2.0 (2026-01-11) full Blackwell, MXFP8/NVFP4, dropped Ampere; production at Together AI, Jump Trading, Cursor. https://github.com/HazyResearch/ThunderKittens
- HipKittens (2025-11-11): tile abstractions carry to CDNA3/4 but algorithms don't: AMD lacks TMA, wgmma, mbarrier; registers statically partitioned across waves so NVIDIA-style wave specialization caps ~80% peak BF16 GEMM on MI355X; uses 8-wave ping-pong / 4-wave interleave; MFMA layouts non-compositional; XCD-aware scheduling (+19%); hipcc mishandles AGPRs. 1.3–3.0x vs Triton, 1.2–2.4x vs all baselines on some attention shapes. https://arxiv.org/abs/2511.08083
- Hidet (ASPLOS 2023; CentML): NVIDIA acquired CentML (2025-06-30/07-17); hidet repo archived 2026-05-12. Lead author Yaoyao Ding now leads NVIDIA Tilus (tile-level language w/ explicit smem/register control, 1–8-bit types; v0.2.0 Jul 2025 Hopper/Blackwell). https://github.com/NVIDIA/tilus
- Mirage (CMU, Zhihao Jia; OSDI'25): multi-level superoptimizer with μGraphs, probabilistic equivalence verification, up to 3.3x. Mirage Persistent Kernel (MPK; arXiv 2025-12-22, v2 2026-06-10): compiles multi-GPU LLM inference into one megakernel; 1.0–1.7x lower single-batch latency vs vLLM/SGLang on A100/H100/B200; NVIDIA-only. https://arxiv.org/abs/2512.22219
- CuTe DSL: nvidia-cutlass-dsl 4.0.0 2025-06-06 (4.8.0 on 2026-09-21); still "public beta" to graduate end of summer 2026. **FlashAttention-4 (2026-03-05) written entirely in CuTe DSL, compiles 20–30x faster than C++ templates, 1,605 TFLOPs/s (71% B200 BF16 peak), 1.1–1.3x vs cuDNN 9.13, 2.1–2.7x vs Triton; depends on Blackwell-only TMEM, tcgen05, 2-CTA MMA, DSMEM.** https://tridao.me/blog/2026/flash4/
- cuTile / CUDA Tile IR: announced GTC 2025; shipped CUDA 13.1 on 2025-12-04 for Ampere, Ada, Blackwell (Hopper absent at launch); tile-level virtual ISA forward-compat pitch. Independent eval (2026-04-25): cuTile 52–79% of cuBLAS GEMM, 2.5x FA2 on B200 but 53% of FA2 on RTX PRO 6000; Triton 62–101% of cuBLAS across H100/B200/RTX PRO 6000 untuned. https://arxiv.org/abs/2604.23466
- TileIR convergence: NVIDIA Triton→CUDA Tile IR backend (2026-01-30; ENABLE_TILE=1; Blackwell + CUDA 13.1 only). https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/ — Gluon tutorials cover explicit layouts, async copies, TMA, wgmma, tcgen05, warp specialization. — Meta TLX (2026-05-11) warp-group model. Net: tile abstraction converging; hardware-specific escape hatches multiplying.
- Taichi: last release v1.7.4 2024-07-31; team pivoted commercial; 28.4k stars.
- tinygrad: 18,935 lines, 6-person team (2025-12-29); own NVIDIA ("NV") and AMD ("AM") user-space drivers bypassing CUDA/ROCm; new accelerators need ~25 low-level ops; BEAM search; 33.6k stars. "AMD YOLO" (2025-03-08): CUDA = early-ecosystem advantage not durable moat; bets $250k on AMD stock. https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html
- Luminal: US$5.3M seed (2025-11-17), Felicis + Paul Graham; search-based compiler over tiny primitive-op set claims to rediscover FlashAttention-class kernels; monetized as inference cloud. https://techcrunch.com/2025/11/17/luminal-raises-5-3-million-to-build-a-better-gpu-code-framework/
- Tessera: no evidence found.

## 6. Why one IR cannot hit peak on NVIDIA, AMD and TPU at once
- Patrick Toulme, "Portability Is a Myth" (2026-05-16): TPU (VMEM, VLIW bundles, 256x256 MXU, explicit DMA), Blackwell (TMEM, tcgen05, mbarrier, warp specialization), Trainium (SBUF/PSUM, partition tiling) require different algorithms; MoE grouped matmul = 282 lines Pallas on TPU vs ~4M lines generated CUDA on Blackwell with zero shared code; XLA-TPU and XLA-GPU share HLO but are different compilers. "The math is portable. Everything else is different." https://patricktoulme.substack.com/p/portability-is-a-myth-why-the-best
- HipKittens: mechanism-level explanation NVIDIA→AMD: missing async copy/MMA/barrier engines, static register partitioning, non-compositional MFMA layouts, LDS bank phases, chiplet scheduling — change the optimal schedule, not just codegen.
- Romeo et al. (2026-06-10): same OpenMP code ~3x slower on MI250X than A100 at app level, up to 10x per kernel, register spilling up to 47x. https://arxiv.org/abs/2606.12753
- Abraham & Okloki (2026-03-22): across NVIDIA/AMD/Intel/Apple ISAs: 10 hardware-invariant primitives, 6 parameterizable variations, 6 fundamental divergences; parallel reduction only 62.5% of native on NVIDIA. https://arxiv.org/abs/2603.28793
- Within NVIDIA: cuTile 2.5x FA2 on B200 vs 53% of FA2 on RTX PRO 6000; A100-tuned Triton underperforms on H100; FA4 depends on Blackwell-only features.
- Single wrong constant (warp 32 vs 64) cost 10x on MI300X in XLA Triton path.

## Fundamental vs. transient
| Problem | Class | Why |
|---|---|---|
| Divergent explicit memory spaces (TMEM/VMEM/SBUF; SMEM vs LDS) | Fundamental | Different tiling/pipelining algorithms |
| Matrix-unit shape/layout differences (tcgen05 vs MFMA vs MXU) | Fundamental | |
| Async copy/sync engines on NVIDIA absent on AMD (TMA, mbarrier, wgmma) | Fundamental (per gen) | Determines warp-specialization vs ping-pong |
| Register allocation model (static wave partition on AMD) | Fundamental | Caps producer/consumer at ~80% MI355X |
| Chiplet/NUMA topology (XCDs) | Fundamental | |
| Gen-to-gen rewrites within one vendor | Fundamental/recurring | A100→H100; B200 vs RTX PRO 6000; TK dropped Ampere |
| Vendor incentive misalignment (every vendor ships own DSL; NVIDIA acqui-hires OctoAI/CentML) | Structural (economic) | NVIDIA has CUDA C++, CuTe DSL, cuTile, Tilus |
| MLIR lacking upstream end-to-end GPU pipeline | Transient in principle, deliberately unaddressed | Lighthouse = reference, no NVIDIA/AMD path Sept 2026 |
| MLIR governance gaps | Transient, being fixed | |
| XLA static-shape limitation | Transient | |
| cuTile missing Hopper; CuTe DSL beta; TileIR Blackwell-only | Transient (maturity) | |
| Mojo compiler closed | Resolved Aug–Sept 2026 | |
| Sustainability of small DSL projects (Taichi stalled; Hidet archived; OctoAI/CentML absorbed) | Recurring structural risk | Talent flows to NVIDIA |
| ABI/interop fragmentation | Transient, addressed | TVM-FFI |

## Key quotes
1. Patrick Toulme (2026-05-16): "The math is portable. Everything else is different."
2. Lattner (2025-04-08): MLIR faces an identity crisis over whether it is general-purpose infra or an AI solution.
3. Lattner (2025-03-12): XLA = two different projects sharing one brand.
4. Lattner (2025-03-26): ~20% Triton gap = difference between $1B and $800M cloud bill.
5. Golin (2025-01-16): fragmentation made it practically impossible to focus on a core pipeline in MLIR.
6. Lighthouse RFC (2025-06-06): not about creating an official upstream tensor compiler.
7. HipKittens (2025-11-11): AMD statically partitions registers across waves, so producer waves burn registers without computing output.
8. Yadav et al. (2026-04-25): cuTile's effectiveness strongly workload- and architecture-dependent; Triton 62–101% of cuBLAS untuned.
