# Non-NVIDIA AI Compiler Stacks: Problems and the Triton Portability Trend (as of Sept 2026)

## 1. AMD ROCm / HIP

**SemiAnalysis, Dec 22 2024 — "MI300X vs H100 vs H200 Benchmark Part 1: Training"** https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training
- MI300X BF16 GEMM ~620 TFLOP/s vs H100 ~720 (≈14% slower); FP8 ~990 vs ~1,280 (≈22% slower), despite higher paper specs.
- `F.Linear` used unoptimized rocBLAS while `torch.matmul` used hipBLASLt — same math, divergent perf.
- Flash Attention backward <20 TFLOP/s for months. FP8 training segfaulted 3 months before publication; when working, slower than BF16.
- `PYTORCH_TUNABLE_OPS` caused ~25 GB HBM leaks (of 192 GB), 1–2 hours retuning per code change.
- Reaching ~75% of H100/H200 needed a hand-crafted ~60-command Dockerfile from an AMD principal engineer, ~5 h build, unmerged dev branches of hipBLASLt/AOTriton. Public PyTorch 2.5.1 broken for most training.
- AMD's RCCL team had <32 MI300X for R&D vs NVIDIA's 11,000-GPU EOS cluster.
- hipBLASLt/rocBLAS heuristics "pick the wrong algorithm for most shapes out of the box".
- Many AMD libraries are hipified forks of NVIDIA libraries.

**SemiAnalysis follow-ups**
- May 23 2025: SGLang ROCm CI coverage <10% of NVIDIA; ~25% of tested models fail accuracy on ROCm; FP8 DeepSeek V3 broken across vLLM/SGLang/TRT-LLM on AMD; "AMD made this worse by adding environment variables, despite our previous advice to remove them." https://newsletter.semianalysis.com/p/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens
- Oct 9 2025 (InferenceMAX v1): MI325X beat H200 on cost/M tokens for GPT-OSS 120B; B200 significantly outperformed MI355X on FP4; MI355X lagged ~40% on DeepSeek MoE latency; praise for ROCm team responsiveness. https://newsletter.semianalysis.com/p/inferencemax-open-source-inference
- Feb 16 2026 (InferenceX v2): MI355X still shipping on forked vLLM 0.10.1 image; official vLLM 0.15.1 "not yet optimized for the MI355X and runs into hard errors"; zero MI355X tests on vLLM CI; "AMD is more than six months behind on open source distributed inferencing and wide expert parallelism"; AMD's ATOM engine "zero customers in production"; SGLang on MI355X doubled throughput Dec 2025→Jan 2026. https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs

**AMD response — timeline**
- ROCm 6.3.0 (Dec 3 2024), 6.4.0 (Apr 11 2025: driver decoupled from user space), 7.0.0 (Sept 16 2025: MI350X/MI355X, PyTorch 2.7, Triton 3.3.0, OCP FP4/FP6/FP8 in hipBLASLt/CK/rocWMMA, hipBLASLt 1.0.0, LLVM 20). https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html
- AITER: multi-backend operator library using Triton, Composable Kernel, HIP, hand-tuned assembly, FlyDSL; default backend in vLLM and SGLang; claims up to 17x MLA decode. https://github.com/ROCm/aiter
- TheRock: open-source build/release system for HIP and ROCm; official since ROCm 7.14. https://github.com/ROCm/TheRock
- Windows gets PyTorch only (ROCm 6.4.4 preview); full stack still Linux-only.
- Version discontinuity: 7.0–7.8 production, 7.9+ TheRock preview; current "ROCm Core SDK 10.0.0" (Aug 26 2026), PyTorch 2.13.0, JAX 0.11.0, vLLM 0.27.0. https://rocm.docs.amd.com/en/latest/about/release-notes.html
- ROCm 7.2.1 (Mar 25 2026): hipBLASLt MXFP8/MXFP4; ROCTracer/ROCProfiler phased out Q2 2026.

**HIP / LLVM AMDGPU backend technical limitations**
- Register pressure w/ MFMA: "[AMDGPU] Large mfma16x16x16 register tiles plus software pipelining leads to bad register spills" (Mar 19 2025, Modular engineer); gfx942 block sizes >128x256 emit spurious `v_accvgpr_read/write_b32`. https://github.com/llvm/llvm-project/issues/131954
- Full LTO becoming AMDGCN default caused 7.6x slowdown in MIOpen hipconv on gfx950 (260 scratch ops vs 4). https://github.com/ROCm/llvm-project/issues/4434
- MFMA register allocation still being redesigned (Matt Arsenault post-RA AGPR pass merged June 27 2025 as WIP). https://github.com/llvm/llvm-project/pull/145024
- gfx950 issue (Sept 9 2026): inline-asm tied constraint yields ~4% faster code than plain SSA — compiler not finding the better form. https://github.com/llvm/llvm-project/issues/222423
- Chopper (Dec 9 2025) MI300X characterization: DVFS "the single largest contributor to the gap between theoretical and observed performance". https://arxiv.org/pdf/2512.08242
- hipify-clang requires full CUDA install; "additional porting might be required". https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html
- Composable Kernel: long build times flagged in README. https://github.com/ROCm/composable_kernel

**2026 (MI400/Helios)**
- Advancing AI 2026 (July 22–23 2026): MI455X 432 GB HBM4, ~23.3 TB/s; Helios rack 72 MI455X, ~2.9 EFLOPS FP4, $5–5.5M/rack. https://www.servethehome.com/amd-mi400-gpu-at-hot-chips-2026/
- ROCm.ai: AI-assisted dev layer letting coding agents work against ROCm APIs; "Hyperloom" optimizer demoed at 38% throughput gain; explicitly NOT a CUDA-compat layer. https://nand-research.com/amd-launches-helios-rack-scale-platform-mi400-series-and-strong-partnerships/

## 2. Intel
- Falcon Shores cancelled Jan 30 2025: "We plan to leverage Falcon Shores as an internal test chip only"; "We are not yet participating in the cloud-based AI data center market in a meaningful way." https://www.servethehome.com/intel-falcon-shores-gpu-not-coming-to-market-in-an-ai-hit/
- Gaudi failure attributed to software: Gelsinger (Nov 1 2024): uptake "slower than we anticipated as adoption rates were impacted by the product transition from Gaudi 2 to Gaudi 3 and software ease of use"; missed $500M 2024 target; $300M write-down. https://www.constellationr.com/blog-news/insights/intel-defends-gaudi-3-it-misses-2024-sales-targets
- Gaudi software: graph compiler and runtime closed; TPC-C LLVM compiler. Default now Eager + torch.compile; Lazy mode "a legacy fallback that is no longer developed and will be deprecated". https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html
- Triton-XPU out-of-tree; "not compatible with Intel Extension for PyTorch and Intel oneAPI Base Toolkit." https://github.com/intel/intel-xpu-backend-for-triton
- IPEX for XPU wound down: "discontinuing active development on Intel Extension for PyTorch, effective immediately after 2.8 release". https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/releases.html
- NVIDIA $5B investment in Intel (Sept 18 2025, $23.28/share) to co-develop x86 CPUs with NVLink. https://nvidianews.nvidia.com/news/nvidia-and-intel-to-develop-ai-infrastructure-and-personal-computing-products
- Intel cut ~25,000 roles through 2025 (target ~75,000 employees).
- Crescent Island (Xe3P, 160 GB LPDDR5X, 350 W) at Hot Chips 2026; software: Triton, SYCL-TLA, oneCCL, oneDNN, SYCL, Level Zero; commercial launch slipping to 2027. https://www.servethehome.com/intel-crescent-island-160gb-to-480gb-lpddr5x-ai-gpu-at-hot-chips-2026/
- UXL Foundation: modest, library-level (oneAPI Construction Kit 4.0 RISC-V, oneDNN on ARM); 2026 goals documentation, 3 PoCs, 20% star growth — not compiler parity. https://uxlfoundation.org/blog/uxl-foundation-advancing-portable-acceleration-across-architectures-in-2025-and-whats-next-for-2026/

## 3. Other accelerators
**AWS Trainium / Inferentia**
- NKI limits (Nov 1 2024): "NKI kernel development is limited to the operations defined in the NKI library, which are fewer and more constrained than libraries such as Triton and NumPy"; C++ custom op (0.061 ms) beat NKI kernel (0.211 ms). https://towardsdatascience.com/on-the-programmability-of-aws-trainium-and-inferentia-cd455826e26c/
- Neuron SDK 2.27.0 (Dec 22 2025): open-source NKI compiler built on MLIR (private beta), TorchNeuron native PyTorch (private beta), Trainium3. https://aws.amazon.com/about-aws/whats-new/2025/12/announcing-aws-neuron-2-27
- SemiAnalysis Trainium3 deep dive (Dec 4 2025): "massive, multi-phase shift in software strategy" — phase 1 open-sources PyTorch backend, NKI compiler, kernel libs; phase 2 XLA graph compiler and JAX stack. Adoption concentrated among "elite programmers" (Anthropic runs all-custom NKI kernels). https://newsletter.semianalysis.com/p/aws-trainium3-deep-dive-a-potential

**Google TPU**
- Pallas TPU constraints: last two block dims divisible by 8 and 128; no int4; loops fully unrolled; "TPUs are sequential machines with a very wide vector register." https://docs.jax.dev/en/latest/pallas/tpu/details.html
- Pallas GPU-via-Triton lowering deprecated in favor of Mosaic GPU. https://docs.jax.dev/en/latest/pallas/design/design.html

**Groq / Cerebras**
- NVIDIA ~$20B for Groq's assets (Dec 24 2025), non-exclusive license + hiring Jonathan Ross; analyst: structured to keep "fiction of competition alive." https://www.cnbc.com/2025/12/26/nvidia-groq-deal-is-structured-to-keep-fiction-of-competition-alive.html
- Cerebras ~$10B / 750 MW OpenAI inference deal (Jan 15 2026), ~32,768 CS-3. https://www.nextplatform.com/ai/2026/01/15/cerebras-inks-transformative-10-billion-inference-deal-with-openai/4092155

**Tenstorrent**: TT-Forge (MLIR compiler), TT-NN (200+ ops), TT-Metalium, all Apache 2.0. https://github.com/tenstorrent/tt-mlir

**Huawei Ascend / CANN**
- Eric Xu (Sept 18 2025): Huawei will "open interfaces for the compiler and virtual instruction set" and fully open source the rest of CANN by Dec 31 2025 — compiler and virtual ISA implementations stay proprietary behind open interfaces. https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech
- GitCode org hosts ge, metadef, asc-devkit, ops-*, catlass, hcomm, BiSheng compiler repo. https://gitcode.com/cann
- DeepSeek delayed R2 after failing to complete a training run on Ascend (FT, Aug 2025), reverting to NVIDIA for training. https://www.trendforce.com/news/2025/08/14/news-deepseek-r2-model-launch-reportedly-delayed-amid-huawei-ascend-chip-hurdles/

**Chinese GPUs and CUDA-compat layers**
- Moore Threads MUSA: MUSIFY CUDA→MUSA transcompiler; "almost 1:1 compatible with CUDA APIs"; 800,000 registered developers; R&D >80% of 2025 revenue. https://www.tomshardware.com/pc-components/gpus/chinas-moore-threads-polishes-homegrown-cuda-alternative-musa-supports-porting-cuda-code-using-musify-toolkit
- Hygon DCU based on ROCm/HIP; Cambricon QiMeng-Xpiler 95%+ transcompilation accuracy claim; Biren (BIRENSUPA) and Iluvatar custom languages face "adoption headwinds". https://www.machineyearning.io/p/chinas-silicon-vanguard
- ZLUDA: two full-time devs Q2 2025; Q4 2025 ROCm 7, Windows, llama.cpp; funding lapsed with v6, back to hobby status. https://www.tomshardware.com/pc-components/gpu-drivers/cuda-emulator-for-amd-gpus-zluda-loses-funding-with-v6-release-embattled-project-goes-back-to-hobby-status-but-now-includes-32-bit-physx-support
- SCALE (Spectral Compute) v1.4.2 (Oct 2025): "does not support enough CUDA APIs to port significant frameworks like PyTorch". https://www.scaleway.com/en/blog/can-your-cuda-code-run-on-all-gpus/
- chipStar 1.3: "still heavily in development mode". https://github.com/CHIP-SPV/chipStar
- CUDA EULA (since 11.6) bars translating output artifacts to non-NVIDIA platforms.

## 4. Triton as de-facto portability layer
- Upstream Triton: NVIDIA and AMD backends in-tree; Intel out-of-tree.
- Meta MTIA strongest production evidence: "Triton for MTIA" — Triton kernels across 60 model types, 94.5% operator coverage on inference, GEMM >80% of roofline matching expert-tuned C++, 31% faster than replaced handwritten kernels. https://arxiv.org/html/2608.00325v1
- KernelEvolve (Dec 29 2025): Triton "has emerged as the dominant DSL" across NVIDIA, AMD, MTIA. https://arxiv.org/html/2512.23236v1
- AMD: Triton first-class inside AITER alongside CK/HIP/assembly.
- AWS NKI adopts "NumPy and Triton-like syntax".
- Against parity: AMD MI300X tuning guide shows Triton needs AMD-specific knobs (`waves_per_eu`, `matrix_instr_nonkdim`, `num_stages`, `OPTIMIZE_EPILOGUE`); wavefront 64 vs 32; Triton not used when rocBLAS/MIOpen faster. https://rocm.docs.amd.com/en/docs-6.2.4/how-to/tuning-guides/mi300x/workload.html
- 2025 survey: ROCm lacks "the aggressive autotuning maturity found on CUDA"; heuristics tuned for warp-32 "often fail to fully saturate AMD's wavefront size (64 threads)". https://tunguz.github.io/PyTorch_Hardware_2025/
- AMD GEAK agent (Aug 1 2025): 54.89% accuracy / 2.59x on TritonBench-revised, but only 63.33% accuracy / 0.92x on expert-written ROCm benchmark. https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html
- OpenAI–AMD deal (Oct 6 2025): 6 GW, first 1 GW MI450 2H 2026, warrant up to 160M AMD shares; hard deadline for production-grade ROCm. https://openai.com/index/openai-amd-strategic-partnership/

## 5. 2026 developments
- ROCm Core SDK 10.0.0 (Aug 26 2026); MI400/Helios July 2026 w/ ROCm.ai; Intel Crescent Island slipping to 2027.
- CUDA-compat efforts consolidating: SCALE can't run PyTorch; ZLUDA lost funding; AWS and Huawei open-sourcing own compilers.
- Regulatory: China SAMR preliminary finding Sept 15 2025 NVIDIA violated Anti-Monopoly Law (Mellanox conditions); EU questioning bundling; DOJ subpoena Sept 2024. No CUDA-specific antitrust action found. https://www.cnbc.com/2025/09/15/china-nvidia-violated-anti-monopoly-law-will-continue-investigation.html

## Fundamental vs. transient
**Fundamental**: wavefront 64 vs warp 32, MFMA vs WGMMA shapes, AGPR/VGPR banking → portable heuristics not transferable; register-file pressure w/ large tiles + pipelining in AMDGPU backend; non-GPU architectures need explicit memory/pipelining control; building on hipified forks caps AMD at "catching up"; ecosystem scale gaps compound; EULA translation-layer clause; custom-language strategies face adoption resistance.
**Transient**: specific PyTorch/ROCm bugs (fixed); CI capacity; env-var sprawl and VIP Docker images (TheRock, AITER-by-default); full-LTO default regression; fork-vs-upstream drift; IPEX→upstream, Gaudi lazy→torch.compile; Neuron/CANN closedness being opened; Windows/consumer ROCm coverage.
**Uncertain**: whether Triton reaches perf parity on AMD or settles as productivity layer with CK/assembly for peak (AITER's multi-backend design suggests latter).

## Key quotes
1. "AMD software is much better now due to our bug reports; its public software stack still falls short." — SemiAnalysis, Dec 22 2024
2. "AMD hipBLASLt/rocBLAS's heuristic model picks the wrong algorithm for most shapes out of the box." — SemiAnalysis
3. "AMD is more than six months behind on open source distributed inferencing and wide expert parallelism." — SemiAnalysis InferenceX v2, Feb 16 2026
4. "AMD made this worse by adding environment variables, despite our previous advice to remove them." — SemiAnalysis, May 23 2025
5. "adoption rates were impacted by the product transition from Gaudi 2 to Gaudi 3 and software ease of use." — Pat Gelsinger, Nov 1 2024
6. "We plan to leverage Falcon Shores as an internal test chip only without bringing it to market." — Michelle Johnston Holthaus, Jan 30 2025
7. "NKI kernel development is limited to the operations defined in the NKI library..." — Chaim Rand, Nov 1 2024
8. "we will open source and open access with CANN ... by December 31, 2025." — Eric Xu, Huawei, Sept 18 2025
