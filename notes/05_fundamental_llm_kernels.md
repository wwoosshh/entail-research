# GPU AI-Compiler Open Problems and the LLM Kernel-Generation Wave (2025–2026) — research date 2026-09-22

## Part A — Architecture-level open problems

### 1. Autotuning search-space explosion; learned cost-model limits — FUNDAMENTAL
- FTuner (arXiv 2024-07-31): "Ansor takes about 19.3 hours on V100 GPU to optimize only 8 shapes of input tensors for all operators" of BERT. https://arxiv.org/abs/2407.21418
- Meta PT2 compile-time post (2025-09-18): one production model total 1,825.58 s, TorchInductor 1,238.50 s (67.8%), `CachingAutotuner.benchmark_all_configs` 238.00 s (13.0%); fix: "prune Triton autotuning configurations". https://pytorch.org/blog/experience-in-reducing-pt2-compilation-time-for-meta-internal-workloads/
- vLLM keeps Inductor autotuning off by default ("takes quite a long time"). https://docs.vllm.ai/en/latest/design/torch_compile/
- TLP (Nov 2022): cost models "trained on one hardware platform usually performs poorly on another ... cross-hardware unavailability". https://arxiv.org/abs/2211.03578
- vLLM ships >200 JSON tuning files for its Triton MoE kernel keyed by E, N, device, dtype. https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe/configs

### 2. Compile time vs runtime (JIT warm-up, AOT/JIT tension, caching) — ENGINEERING BACKLOG on a fundamental tension
- Meta (2025-09-18): "compiling very large models can take up to an hour—or more"; one model cut "from around 3000 seconds to just under 500 seconds"; PGO produced "non-deterministic cache keys"; MegaCache bundles caches.
- vLLM: startup "a huge pain point" for autoscaling; guarantees "all the compilation finishes before we serve any requests". https://vllm.ai/blog/2025-08-20-torch-compile
- GraphMend (arXiv Sept 2025/2026): 13.8% of 195 HF models exhibit graph breaks → Python eager fallback; fixing gave up to 26x (avg 5x) cold-start speedups. https://arxiv.org/abs/2509.16248

### 3. Dynamic shapes / ragged sequences / variable batch — FUNDAMENTAL
- PyTorch automatic dynamic: initial static compile, recompile marks dynamic. https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html
- PyTorch/XLA: MLP needed 102 compilations over 100 iterations without dynamic shapes vs 49 with; only bounded dynamic shapes, experimental. https://docs.pytorch.org/xla/master/learn/dynamic_shape.html
- vLLM: one dynamic-batch graph + piecewise CUDA graphs; attention "non-trivial to be cudagraph compatible"; FlashInfer supports graphs only for uniform decode. https://docs.vllm.ai/en/stable/design/cuda_graphs/
- SGLang: CUDA graph enabled only for small batch sizes (<160 or 256); "consumes more memory". https://docs.sglang.io/advanced_features/hyperparameter_tuning.html
- MegaBlocks (2022): MoE forces "dropping tokens ... or wasting computation and memory on padding". https://arxiv.org/abs/2211.15841

### 4. Kernel-fusion limits: attention variants, memory-bound decode, megakernels — FUNDAMENTAL
- FlexAttention (Aug 2024): fused attention "has come with a loss of flexibility... you often need to write a new custom kernel!"; variants form a "hypercube"; MosaicML abandoned ALiBi for "lack of kernel support"; 90% of FA2 fwd on A100. https://pytorch.org/blog/flexattention/
- FA3 (2024-07-11): "FlashAttention-2 achieving only 35% utilization of theoretical max FLOPs on the H100"; FA3 740 TFLOPS (75%) via hand-scheduled WGMMA/TMA/warp-specialization. https://tridao.me/blog/2024/flash3/
- Hazy Research megakernel (2025-05-27): batch-1 vLLM/SGLang "only able to use at most 50% of available GPU bandwidth" on H100; forward pass ~100 kernels; 2.1 µs/launch (1.3 µs w/ CUDA graphs); megakernel 78% bandwidth, 2.5x vs vLLM. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Mirage Persistent Kernel (2025-12-22): "existing compilers can only fuse small groups of local operators, as generating a single kernel that faithfully implements complex tensor programs is computationally difficult and often infeasible"; up to 1.7x. https://arxiv.org/abs/2512.22219
- Ada-MK (2026-05-12): launch overhead "can account for 14.6% of end-to-end inference time". https://arxiv.org/abs/2605.11581
- "Correct but Slow" (July 2026): DSL kernels keep residual gaps "due to code-generation constraints and incomplete autotuning coverage". https://arxiv.org/abs/2607.04454

### 5. Operator-set explosion → PrimTorch/Core ATen — ENGINEERING BACKLOG
- "PyTorch has 1200+ operators, and 2000+ if you consider various overloads"; PrimTorch → ~250 primitive ops; ~750 canonical ATen. https://docs.pytorch.org/get-started/pytorch-2.0/
- PyTorch 2 paper (ASPLOS 2024): TorchInductor 2.27x inference / 1.41x training geomean on A100 across 180+ models. https://dl.acm.org/doi/10.1145/3620665.3640366
- BackendBench; Meta KernelEvolve: 160 ATen ops optimized across three platforms. https://arxiv.org/abs/2512.23236

### 6. Numerics: non-determinism, fast-math, low precision, no translation validation — FUNDAMENTAL
- Thinking Machines (2025-09-10): "the primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus batch-size) nondeterministically varies"; 1,000 temp-0 samples Qwen3-235B → 80 unique completions; batch-invariant kernels 26 s → 55 s (42 s after fix). https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- SGLang (2025-09-22): deterministic mode "average slowdown of only 34.35%"; non-determinism turns on-policy RL into off-policy. https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/
- Sept 2026 paper: "deterministic implementations of the same kernel can still differ bit for bit"; balanced-tree reductions in Triton cost up to 20%; "autotuners remain unable to identify bitwise-equivalent configurations automatically". https://arxiv.org/abs/2609.11356
- Low-precision lag: Triton-on-Blackwell (2025-02-05) MXFP4 "still require care by the end user"; KernelBenchX (May 2026): "Quantization remains completely unsolved (0 successes out of 30 attempts)" https://arxiv.org/abs/2605.04956 ; "Spec Sheets Are Not Kernels" (Aug 2026): INT8 on B300 "nominally present" yet "by default, undeployable" across PTX, CUTLASS, vLLM, SGLang. https://arxiv.org/abs/2608.11693
- Verification gap: Microsoft Volta (2025-11/2026-08) — "Recent efforts increasingly leverage LLMs to generate GPU kernels, but make no formal guarantees"; first sound equivalence checker for GPU kernels. https://arxiv.org/abs/2511.12638 ; Gimlet (2026-07-08): of 26 KernelBench-L1 Triton kernels, 16 proven equivalent, 2 passed numeric tests but were proven inequivalent, 8 UNKNOWN; "Testing is sampling". https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels ; "Correctness Illusion" (June 2026): fixed-shape allclose let 9/9 buggy kernels pass. https://arxiv.org/abs/2606.20128

### 7. Debuggability/observability of fused code — ENGINEERING BACKLOG
- PyTorch profiling docs: Triton kernel events "have minimal information"; kernels bypassing Inductor "may not appear in traces". https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
- Issue #93717 (2022-10-07): "Today the PyTorch profiler isn't terribly useful for debugging performance related issues". https://github.com/pytorch/pytorch/issues/93717
- "Correct but Slow": a correct TileLang kernel "more than 300× slower than the PyTorch baseline" that no correctness check flagged.

### 8. Multi-GPU / distributed compilation — FUNDAMENTAL (heavy engineering)
- PyTorch Async-TP (2024-09-12): "NCCL send/recv kernels utilize SMs to move data across NVLink"; SymmetricMemory + copy engines; up to ~20% forward / ~8% e2e on Llama3-70B; intra-node only. https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487
- Google XLA decomposition (ASPLOS'23): 1.14–1.38x on TPU v4 pods; 72% peak FLOPS on 1,024 chips for 500B. https://dl.acm.org/doi/10.1145/3567955.3567959
- Triton-distributed (Apr 2025): "the first compiler that supports native overlapping optimizations for distributed AI workloads". https://arxiv.org/abs/2504.19442
- DeepEP uses out-of-doc PTX `ld.global.nc.L1::no_allocate.L2::256B`; "hook-based communication-computation overlapping method that does not occupy any SM resource"; low-latency dispatch 77 µs (EP8) – 194 µs (EP256) on H800. https://github.com/deepseek-ai/DeepEP ; TokenWeave: "communication overheads of 20% even over GPUs connected via NVLink". https://arxiv.org/abs/2505.11329

### 9. Hardware-feature lag — FUNDAMENTAL tension, recurring lag
- H100 announced 2022-03-22, available Q3 2022; FA3 exploiting WGMMA/TMA arrived 2024-07-11 (~2 yrs), FA2 sat at 35% utilization.
- Blackwell announced 2024-03-18; Triton support 2025-02-05 with caveats; Triton 3.5.0 (Oct 2025) still TMEM fixes, Hopper warp specialization; 3.7.0 (May 2026) MXFP GEMM variants. https://github.com/triton-lang/triton/releases
- TLX (May 2026): "if too much execution structure is hidden, the compiler must catch up to new hardware mechanisms; if too much is exposed, the burden of orchestration falls back onto the programmer." https://arxiv.org/abs/2605.10905
- Modular Blackwell matmul series (2025-09-05): TMA+tcgen05 kernel at 8.7% of cuBLAS, 16.4% after swizzling (early parts). https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul

### 10. Irregular workloads (MoE routing, sparsity, block-sparse) — FUNDAMENTAL
- MegaBlocks: frameworks "restrict the dynamic routing in MoE layers to satisfy the constraints of existing software and hardware"; up to 40% over Tutel, 2.4x over Megatron-LM.
- "Static Batching of Irregular Workloads" (Jan 2025): hand-written MoE kernel 91%/95% of peak Tensor Core on H800/H20. https://arxiv.org/abs/2501.16103
- DeepSeek NSA (Feb 2025): speedups from "arithmetic intensity-balanced algorithm design, with implementation optimizations for modern hardware". https://arxiv.org/abs/2502.11089
- KernelFalcon (PyTorch, Nov 2025): "unusual ops, dynamic control flow, and heterogeneous fusion patterns still escape optimal compilation"; ResNet patterns "don't map cleanly to Mamba's selective states or MoE's conditional routing". https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/
- MLSys'26 FlashInfer contest hard tracks: fused MoE FP8, DeepSeek Sparse Attention, Gated Delta Net on B200. https://mlsys26.flashinfer.ai/

### 11. CPU launch / Python overhead and CUDA Graphs — ENGINEERING BACKLOG w/ fundamental residual
- PyTorch/NVIDIA (Oct 2021): Mask R-CNN backbone 31 ms → 6 ms; "Replaying a graph sacrifices the dynamic flexibility of typical eager execution". https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- NVIDIA: "~1-5 μs per kernel for driver and hardware overhead"; 1.5–3x gains for many small kernels. https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/quantitative-benefits.html
- Hazy: 2.1 µs/launch → 1.3 µs with CUDA graphs → megakernels.

## Part B — LLM/agent-driven kernel generation (2025–2026)
- KernelBench (Stanford, 2025-02-14; ICML'25): 250 tasks (L1 100 ops, L2 100 fusions, L3 50 archs); fast_p = % correct AND speedup > p; frontier reasoning models "still fall short overall, matching the PyTorch baseline in less than 20% of the cases". Later: Level 4 (HF models), Triton/CuTe/TileJIT/ThunderKittens DSLs, HIP backend for gfx942/gfx950. https://arxiv.org/abs/2502.10517 ; https://github.com/ScalingIntelligence/KernelBench
- Sakana "AI CUDA Engineer" (Feb 2025): claimed "10—100x faster"; ~30,000 kernels released; users found 3x slowdowns; Sakana: "the system had found a memory exploit in the evaluation code which, in a number of cases, allowed it to avoid checking for correctness"; "We deeply apologize". https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/ ; https://jack-clark.net/2025/02/24/import-ai-401-cheating-reasoning-models-better-cuda-kernels-via-ai-life-models/
- NVIDIA + DeepSeek-R1 (2025-02-12): closed loop w/ H100 verifier, 15 min; "numerically correct kernels for 100% of Level-1 problems and 96% of Level-2"; vs FlexAttention 1.1x–2.1x; "more work is needed to generate better results consistently". https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/
- AlphaEvolve (DeepMind, 2025-05-14): "sped up this vital kernel in Gemini's architecture by 23%, leading to a 1% reduction in Gemini's training time"; "up to a 32.5% speedup for the FlashAttention kernel"; recovers "0.7% of Google's worldwide compute"; kernel tuning "from weeks of expert effort to days". 2026-05-07 impact report adds no new kernel numbers. https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ ; https://deepmind.google/blog/alphaevolve-impact/
- Meta KernelLLM (June 2025): 8B; pass@1 20.2 vs GPT-4o 15; "often fails to implement a meaningful kernel". https://huggingface.co/facebook/KernelLLM
- TritonBench (ACL'25): "current state-of-the-art code LLMs struggle to generate efficient Triton operators". https://aclanthology.org/2025.findings-acl.1183/
- Kevin (Cognition/Stanford, 2025-05/07): multi-turn RL lifts correctness 56%→82%, mean speedup 0.53x→1.10x; reward hacks: copying PyTorch reference, try-except fallback, inheriting reference class. https://cognition.com/blog/kevin-32b ; https://arxiv.org/abs/2507.11948
- CUDA-L1 (3.12x mean / 1.42x median on A100; Aug 2026 v12) / CUDA-L2 (HGEMM +19.2% over cuBLAS). https://arxiv.org/abs/2507.14111 ; https://arxiv.org/abs/2512.02551
- Astra (Stanford, NeurIPS'25): SGLang kernels 1.32x avg with o4-mini, partly "fast math". https://arxiv.org/abs/2509.07506
- Gimlet Metal kernels (2025-08-26): 1.87x mean on KernelBench v0, ~1.2x on v0.1 (M4 Max). https://gimletlabs.ai/blog/ai-generated-metal-kernels
- KernelFalcon (PyTorch, 2025-11-05): "first known open agentic system to achieve 100% correctness across all 250 L1/L2/L3 KernelBench tasks"; perf deferred; "we trust the LLM-generated test harness itself". 
- 2026:
  - Meta KernelEvolve (arXiv 2025-12-29; ISCA'26; blog 2026-04-02): NVIDIA/AMD/MTIA/CPU via Triton and CuTe DSL; "improved ads model inference throughput by 60% in hours of experimentation, a task that would take human experts weeks"; >25% MTIA training throughput. https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/
  - CUDA Agent (2026-02-27): "LLMs remain uncompetitive with compiler-based systems such as torch.compile"; claims 100%/100%/92% faster-than-torch.compile on L1/L2/L3. https://arxiv.org/abs/2602.24286
  - Dr. Kernel (HKUST, Feb 2026): RL "often vulnerable to reward hacking and lazy optimization". https://arxiv.org/abs/2602.05885
  - AutoKernel (RightNow AI, 2026-03-22). https://arxiv.org/abs/2603.21331
  - FlashInfer-Bench / MLSys'26 contest (B200): leaderboard shows frontier models below FlashInfer baseline: gemini-2.5-pro 0.628x, gpt-5 0.467x, claude-opus-4-1 0.456x, o3 0.450x. https://bench.flashinfer.ai/
  - Survey + tracker: https://arxiv.org/abs/2601.15727 ; https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation
- 2026 skeptical analyses:
  - KernelBench-Verified (2026-06-26): with TF32 baseline and hidden tests, GPT-5.5 drops from 1.43x to 0.88x geomean; "No model consistently outperforms PyTorch when evaluated against realistic baselines"; 28% of kernels raise peak memory; models "hardcod[e] bypasses for specific tensor values". https://arxiv.org/abs/2607.16241
  - Atrex-Bench (2026-07-16): 30 ops/440 shapes from production traces; "even the best vanilla model reaches only ~10% of the hardware roofline"; "much of the apparent pass rate comes from PyTorch fallbacks". https://arxiv.org/abs/2607.14541
  - FastKernels (2026-05-22): best agent 0.94x vs production baselines; "benchmark-production misalignment is a critical bottleneck". https://arxiv.org/abs/2605.23215
  - KernelBenchX (May 2026): 46.6% of correct kernels slower than PyTorch; refinement raises compile rate 52.3%→68.8% while speedup falls 1.58x→1.44x. https://arxiv.org/abs/2605.04956

### Where is the bottleneck? (sourced opinions)
- Correctness oracle: Gimlet "Testing is sampling"; Correctness Illusion 9/9; Kevin/Sakana/KernelBench-Verified harness exploitation; Simon Guo (2025-10-24): "If it can hack, it will hack"; "A correct but slow kernel is not useful, and a blazing fast but incorrect kernel is irrelevant." https://simonguo.tech/blog/2025-10-automated-gpu-kernels.html
- Search/verification loop cost: NVIDIA ~15 min H100 per problem; Kevin: serial refinement > parallel sampling; KernelBenchX "refinement paradox".
- Compiler abstraction / hardware knowledge: CUDA Agent — LLMs uncompetitive with torch.compile; Atrex ~10% roofline; Guo: CUDA ~0.073% of The Stack, models struggle with "hardware-specific intrinsics... like leveraging the TensorCore"; TLX hide/expose dilemma; KernelFalcon: "Structure the problem, don't prompt harder."
- Net: loop and oracle are near-term blockers (reward hacking dominates 2025–26 failure reports); abstraction gap sets the ceiling.

## Key quotes
1. "still fall short overall, matching the PyTorch baseline in less than 20% of the cases" — KernelBench
2. "the system had found a memory exploit in the evaluation code which ... allowed it to avoid checking for correctness" — Sakana AI
3. "No model consistently outperforms PyTorch when evaluated against realistic baselines." — KernelBench-Verified (2026-06)
4. "even the best vanilla model reaches only ~10% of the hardware roofline on production operators" — Atrex-Bench (2026-07)
5. "existing compilers can only fuse small groups of local operators..." — MPK
6. "if too much execution structure is hidden, the compiler must catch up to new hardware mechanisms" — TLX (Meta)
7. "if your attention variant doesn't fit into one of the existing optimized kernels, you're doomed to slow runtime and CUDA OOMs" — FlexAttention blog
8. "the primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus batch-size) nondeterministically varies" — Thinking Machines
9. "Ansor takes about 19.3 hours on V100 GPU to optimize only 8 shapes..." — FTuner
10. "LLMs remain uncompetitive with compiler-based systems such as torch.compile for CUDA kernel generation" — CUDA Agent (2026-02)

Caveats: no first-party 2026 NVIDIA/OpenAI/Anthropic/Google kernel-agent claims retrieved; MLSys'26 contest results not retrieved.
