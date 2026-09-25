# Problems and limitations of the PyTorch 2.x compile stack and OpenAI Triton — evidence review (as of 2026-09-22)

## 1. torch.compile pain points

**Cold compile time**
- Official docs: cold-start compile "typically ranges from seconds to minutes for common models, and can reach tens of minutes to a few hours for very large ones"; "the single most common cause of excessive compile time is recompiling more than necessary." — https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html
- PyTorch 2.14 (Sept 2026) motivates "compile-on-one-rank" with: "Every rank in a distributed job compiles the same model independently, so a single multi-minute compile is paid N times over before training starts." — https://pytorch.org/blog/pytorch-2-14-release-blog/
- Independent measurement (Chaim Rand, 2025-08-18): full compile 196 s; cache pre-loading cut it to 56 s; regional compilation to 80 s. — https://towardsdatascience.com/maximizing-ai-ml-model-performance-with-pytorch-compilation/
- Regional-compilation recipe (toy 64-layer model, PyTorch ≥2.5): 11.67 s full-model vs 0.60 s regional. — https://docs.pytorch.org/tutorials/recipes/regional_compilation.html
- SGLang users (2025-12-29): startup 1:30 without compile vs ~6 min with compile for batch sizes 1,2,4,8,16 (vLLM with compile ~1 min); maintainer reply (2026-08-28) attributes it to per-batch-size compilation. — https://github.com/sgl-project/sglang/discussions/16048

**Recompilation / dynamic shapes / guards**
- Edward Yang (Meta), 2025-08-13: "Most pathological compile times arise from repeated recompilation (often due to dynamic shapes, but sometimes not)"; "Function calls are inlined and loops are unrolled by default," so compile time scales with depth; DTensor + dynamic shapes still open (#159635). — https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
- PyTorch 2.14 adds declarative `@dynamic_spec`; dimensions "become unbacked symbols, so the compiler cannot quietly specialize on the batch size it happened to trace — the trade-off is that shape-dependent branching now surfaces as a data-dependent error rather than a guard and a recompile." — https://pytorch.org/blog/pytorch-2-14-release-blog/
- Meta's Triton grouped-GEMM MoE post (2025-08-19): under expert parallelism "every new token count may require kernel recompilation." — https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/

**Graph breaks and control flow**
- ezyang (2025-09-05): graph-break-generated graphs "can be difficult to reason about"; `torch.cond`/`while_loop` "may not mutate any of their inputs," forcing "functional, JAX-like" rewrites; `torch.while_loop` was "not yet differentiable." — https://blog.ezyang.com/2025/09/so-you-want-to-control-flow-in-pt2/
- PyTorch 2.9 (Oct 2025) added `torch._dynamo.error_on_graph_break()`; 2.8 (Aug 2025) shipped control-flow op library (`cond`, `while_loop`, `scan`, `associative_scan`, `map`) and hierarchical compilation (`nested_compile_region`). — https://pytorch.org/blog/pytorch-2-9/ , https://pytorch.org/blog/pytorch-2-8/

**Caching (FX graph cache, Mega-cache)**
- Mega Cache Beta in PyTorch 2.7 (Apr 2025). — https://pytorch.org/blog/pytorch-2-7/
- Issue #154463 (2025-05-27): `save_cache_artifacts()` circular import failure. — https://github.com/pytorch/pytorch/issues/154463
- vLLM (2025-08-20) stores FX graphs + Triton kernels in `~/.cache/vllm/torch_compile_cache`, but relying on non-public APIs "has led to issues like weird caching issues, or needing to disable vLLM's torch.compile cache for certain models." — https://vllm.ai/blog/2025-08-20-torch-compile
- ezyang (2024-11-05): caches "are not guaranteed to hit." — https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/

**Debuggability**
- PyTorch 2.10 (Jan 2026) promotes `tlparse`/`TORCH_TRACE`, adds DebugMode with tensor hashing "to make it easier to see where subtle numerical errors are introduced"; `torch.compile()` now respects deterministic mode. — https://pytorch.org/blog/pytorch-2-10-release-blog/

**Silent numerical divergence vs eager**
- ezyang (Aug 2025): "when float16/bfloat16 operations are fused together, we do not insert redundant down/up-conversions". Nov 2024: "The compiler does not guarantee exact bitwise equivalence with eager code."
- Empirical study (Li et al., arXiv 2604.08720, Apr 2026): "19.2% of high-priority issues are incorrect outputs of compiled DL models induced by torch.compile bugs, the second-most-common bug category (only behind program crashes at 19.57%)"; AlignGuard found 23 new bugs, 14 high-priority. — https://arxiv.org/abs/2604.08720
- Issue-crawler group (malfet, 2026-09-18): 244 matching issues, 59 verified open, for "different value, dtype, or memory format than eager". — https://github.com/malfet/pytorch_issue_crawler/issues/7
- Umbrella "[PT2] Validation lost" (2026-09-18): "a user adds torch.compile to working code and loses the error that would have caught their bug". — https://github.com/pytorch/pytorch/issues/197554
- Issue #162722 (2025-09-11): Transformer model, 99.9% elements mismatched, max abs diff 3.98; "module: correctness (silent)". — https://github.com/pytorch/pytorch/issues/162722
- PyTorch 2.12 (May 2026) FMA-based `addcdiv` lowering "achieving bitwise numerical parity with eager CUDA execution" because rounding differences "accumulate over thousands of training steps." — https://pytorch.org/blog/pytorch-2-12-release-blog/

**mode="max-autotune" cost**
- Issue #161764 (2025-08-29): DenseNet121 eager 0.340 s vs max-autotune 0.654 s (~1.9× slower), "Not enough SMs to use max_autotune_gemm mode." — https://github.com/pytorch/pytorch/issues/161764

**Distributed training interactions**
- ezyang (Aug 2025): "Distributed collectives and DTensor can be compiled, but are unoptimized by default"; functional collectives "currently do NOT support autograd"; torch.compile "does not assume the program being compiled is SPMD by default"; recommends forking torchtitan.
- Fixes: 2.11 (Mar 2026) "Differentiable Collectives"; 2.13 (Jul 2026) FSDP2 separate reduce-scatter group; 2.14 (Sept 2026) Inductor `simple_overlap` comm/compute overlap "now enabled by default."
- SimpleFSDP (arXiv 2411.00284, Nov 2024): "up to 28.54% memory reduction and 68.67% throughput improvement compared to ... FSDP2 eager" on Llama 3 up to 405B. — https://arxiv.org/abs/2411.00284

**Production adoption rate**
- No quantitative survey found. Proxies: ezyang (Nov 2024) lists vLLM, SGLang, TensorRT-LLM, gpt-fast, HF diffusers, torchtune/torchtitan/torchao. SGLang docs mark `--enable-torch-compile` as "(Note: This feature is out of maintenance and might cause error)". — https://docs.sglang.io/docs/advanced_features/server_arguments
- PyTorch 2.14 frames eager speedups as keeping "PyTorch's default development experience fast without requiring users to reach for torch.compile" — no compile-by-default.

**How inference engines use or avoid it**
- vLLM (Aug 2025): piecewise compilation with attention wrapped as a custom op so "Dynamo will not try to inspect any of the internal operations"; CUDA graphs captured only between attention ops; "vLLM wants fast serving performance and no recompilations during model serving"; custom passes: AllReduce+RMSNorm fusion "up to 15%", attention+quant fusion "up to 7%." — https://vllm.ai/blog/2025-08-20-torch-compile , https://docs.vllm.ai/en/latest/design/torch_compile/
- TensorRT-LLM AutoDeploy is "a prototype" that exports via `torch.export`; "experimental, subject to change." — https://nvidia.github.io/TensorRT-LLM/torch/auto_deploy/auto-deploy.html

## 2. Triton limitations

**Performance gap vs CUTLASS/cuDNN/FlashAttention on Hopper and Blackwell**
- PyTorch FlexAttention team (2026-03-04): Triton FlexAttention was "roughly at 80%" of FA3 on Hopper originally and "achieves roughly 60% of FlashAttention-3's throughput" today; on Blackwell "What was once a small gap has grown to a chasm!"; FA4/CuTeDSL backend gives "1.6–3.2× speedup over Triton" forward and "1.85–2.3× for backward" on GB200. — https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- CUDA Tile evaluation (arXiv 2604.23466, v2 2026-06-05): BF16 GEMM Triton = 98% of cuBLAS on H100 at N=4096 but 76% at N=8192; 62% of cuBLAS on B200 at N=8192 (cuTile 52%); 101% on RTX PRO 6000. — https://arxiv.org/html/2604.23466v1
- Meta autoWS roadmap (2026-01-08): B200 flash-attention forward with autoWS is "1.5-2x of stock Triton (cuDNN still leads by 10-20%)." — https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/
- Tawa (arXiv 2510.14719, Oct 2025; incl. Vinod Grover): "up to 1.1× speedup over highly optimized cuBLAS GEMM kernels", "1.2× speedup over Triton" attention. — https://arxiv.org/abs/2510.14719
- NVIDIA/OpenAI (2025-02-05): Triton on Blackwell "near-optimal performance, comparable to library implementations" for FP8/FP16 GEMM; MXFP4 layouts "still require care by the end user." — https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/
- Counter-evidence (decode): IBM's Triton paged attention went from "19.7% of the performance of FlashAttention3" to "98.6% – 105.9% versus FlashAttention3" on H100 (arXiv 2511.11581, 2025-10-07); vLLM (2026-03-04) "100.7% of the performance of FlashAttention 3" on H100 for long decode. — https://arxiv.org/html/2511.11581v1 , https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive
- Lattner (2025-03-26): "~20% performance loss versus optimized CUDA on H100." — https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls

**Lack of low-level control → Gluon and TLX (2025)**
- Gluon tutorial: "While the Triton compiler does a good job ... it can be beaten by hand-tuned low-level code. When this happens, there is little the user can do." First public issue #7392, 2025-07-04. — https://github.com/triton-lang/triton/issues/7392
- Lei Zhang (2026-02-28): Gluon "is effectively the Python frontend to the `ttg` IR" and "Gluon kernels are *not portable* anymore like Triton." — https://www.lei.chat/posts/gluon-explicit-performance/
- FlexAttention blog: Blackwell needs "a deeply pipelined, warp-specialized kernel. These techniques aren't expressible in our Triton-based implementation."
- TLX (Meta + UCSD, arXiv 2605.10905, 2026-05-11): Triton's SIMB abstraction means "the compiler must discover and implement all of them on the user's behalf"; ~200 lines Python vs thousands of CUDA lines for B200 GEMM; "TLX-authored kernels have been deployed in large-scale training and inference production systems." — https://arxiv.org/html/2605.10905v1
- Meta autoWS (Jan 2026): "Generating optimal warp specialized code to hit hardware roofline performance is a combinatorial problem"; support "is limited and experimental."

**Compile-time and autotuning cost**
- IBM (Oct 2025): "tuning flash attention v2 extensively to achieve best-possible performance took nearly 24 hours for each GPU type."
- Helion: autotuning "typically takes around 10 minutes" per kernel (586.6 s over 1,520 configs). — https://pytorch.org/blog/helion/
- AMD's Origami (PyTorch 2.13/2.14) "replaces brute-force autotuning with analytical GEMM-configuration selection".

**Backend coupling / regressions**
- Issue #11883 (2026-09-20): core Triton headers include NVIDIA/AMD backend trees, out-of-tree backends must build both (625 MB) and download NVIDIA toolchains (662 MB). — https://github.com/triton-lang/triton/issues/11883
- Intel XPU issue #7782 (2026-08-18): Inductor kernel regressed ~2.6× between PyTorch 2.13 (Triton 3.7.2) and 2.14 (Triton 3.8.0). — https://github.com/intel/intel-xpu-backend-for-triton/issues/7782

**Irregular workloads (MoE, ragged, sparse)**
- Mitra (arXiv 2605.23911, 2026-04-07): "BLOCK_M must be fixed (not autotuned)"; "Triton does not support scalar indexing into 2D accumulators"; speedup vs Megablocks falls "from 1.03× to 0.70× as routing skew increases." — https://arxiv.org/html/2605.23911v1
- IBM/vLLM: "Triton does not provide a global barrier"; "Triton kernels always need to be written around one specific problem."
- Tri Dao's SonicMoE (2026): grouped GEMM in CuTeDSL "21% higher forward TFLOPS than triton official example". — https://tridao.me/blog/2026/sonicmoe-blackwell/
- PyTorch 2.14 adds `torch.switch` for MoE branching; ROCm only got Inductor's Triton grouped-GEMM lowering in 2.14.

**Debugging difficulty**
- Lattner: Triton doesn't work with Nsight Compute; developers "are often left guessing what the compiler did."
- Meta (Jan 2026): "Warp specialization can make it harder for model and kernel authors to debug numerics and performance issues".

**Portability reality: AMD / Intel**
- Positive: IBM used "the same Triton kernel source code" on H100 and MI300. Helion on MI350X: 1.44× over hand-written Triton.
- Negative: HipKittens (Stanford, arXiv 2511.08083, 2025-11-11): HK kernels "consistently outperform compiler baselines", "1.2-2.4×" on attention/memory-bound; Triton "struggles with register lifetime tracking and lowering memory accesses to AMD's most performant intrinsics" (secondary); HipKittens "shipped as an official AITER backend in March 2026" (secondary). — https://arxiv.org/abs/2511.08083
- Red Hat cross-vendor benchmark (2026-02-12): 850/852 correctness; "Helion is anomalously low on Consumer GPU A (27 TFLOPS vs 150+ for others)"; "Start with Inductor if you want a low-friction, production-friendly default." — https://next.redhat.com/2026/02/12/from-hand-tuned-to-generated-a-reproducible-triton-gpu-kernel-benchmark-across-different-vendors/
- Lattner: Triton kernels "need to be rewritten for new generations ... to unlock their performance"; non-SIMT accelerators degrade "catastrophically."

**Governance**
- CONTRIBUTING.md: nine core maintainers "as of 09/18/2026"; Phil Tillet lead; core maintainers "have the power to veto any decision made at a Module maintainer level." — https://raw.githubusercontent.com/triton-lang/triton/main/CONTRIBUTING.md
- Triton Developer Conference 2025: 2025-10-21, Microsoft Silicon Valley campus.

## 3. Newer layers and 2025–2026 PyTorch release story

**Helion (Meta)**
- Beta 2025-10-22: "PyTorch with Tiles" DSL → autotuned Triton; attention "30 lines in Helion vs 120 lines in Triton"; B200 geomean speedup over eager 3.27× vs torch.compile(max-autotune) 2.7× vs hand-written Triton 1.76×; "10 minutes" AOT autotune. — https://pytorch.org/blog/helion/
- TAC update (2026-02-02): "Helion adoption is still in its early stages"; users include vLLM (optional), Meta production teams, "another frontier lab"; GA planned PTC Europe 2026-04-07. — https://github.com/pytorch-fdn/tac/issues/26
- Releases v1.0.0–v1.4.0 (CuTe backend, Metal backend, Pallas TPU, "LLM-guided autotuning"). PyTorch 2.14 registers Helion as third native-DSL backend but "No operators are routed through Helion in this release"; "unavailable on ROCm builds."

**BackendBench / KernelLLM / KernelEvolve (Meta)**
- BackendBench: "evaluation suite for testing how well LLMs and humans can write PyTorch backends". — https://github.com/meta-pytorch/BackendBench
- KernelLLM (June 2025): 8B; KernelBench-Triton pass@1 20.2 vs DeepSeek V3 16 / GPT-4o 15; pass@20 57.1. — https://huggingface.co/facebook/KernelLLM
- KernelEvolve (Meta, arXiv 2512.23236, Dec 2025): "reduces development time from weeks to hours." — https://arxiv.org/abs/2512.23236

**Blackwell / next-gen support timeline**
- 2025-02-05 Triton Blackwell support announced. 2.7 (Apr 2025): prototype Blackwell, Triton 3.3, CUDA 12.8. 2.11 (Mar 2026): FA4/CuTeDSL FlexAttention backend. 2.13 (Jul 2026): CuTeDSL "Native DSL" Inductor backend. 2.14 (Sept 2026): NVGEMM with NVFP4 (Blackwell-only), "Inductor targets Rubin (sm_107)."

**2026 releases** (cadence every 2 months)
- 2.10 (Jan 2026): Python 3.14; combo-kernel horizontal fusion; determinism; DebugMode; TorchScript deprecated.
- 2.11 (Mar 2026): differentiable collectives; FA4 Flex backend.
- 2.12 (May 2026): bitwise-parity `addcdiv`; `torch.cond` in CUDA graphs.
- 2.13 (Jul 2026): CuTeDSL backend; kernel compilation "moved from the thread pool to a subprocess pool"; `set_default_backend`.
- 2.14 (Sept 2026): `@dynamic_spec`; compile-on-one-rank; `simple_overlap` default; AOTInductor single-pass codegen; Helion registry.
- Precompile: `torch.compile(...).aot_compile()` "experimental and subject to change"; requires `fullgraph=True`. — https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_aot_compile.html
- No evidence of compile-by-default in any release.

## 4. Critiques and debates (2025–2026)
- Lattner (2025-03-26): "It looks like Python, but it isn't Python"; Triton "has not united the ecosystem or challenged CUDA's dominance"; 20% loss framed as "$1B cloud bill and an $800M". 
- NVIDIA PTC 2025 session (2025-10-22, Jared Roesch): "the challenge remains in balancing algorithmic abstraction with evolving HW"; TileIR backend for TorchInductor. — https://pytorchconference.sched.com/event/27QCV
- cuTile controversy (Dec 2025, CUDA 13.1): Nicholas Wilt: "It's hard not to suspect that cuTile was developed directly to counter Triton." — https://hyper.ai/en/news/47715
- PyTorch's compiler team moving Blackwell-critical paths to CuTeDSL is the strongest implicit statement that Triton alone can't hit Blackwell roofline.
- Meta's layered stack: Helion above Triton, autoWS inside, TLX/Gluon below.
- Pro-Triton: IBM/vLLM — "the paged attention implementation in Triton has roughly 800 lines of code, while FlashAttention3 has around 70'000 lines"; "Maintaining hundreds of kernels across multiple GPU platforms ... quickly becomes impractical."
- NVIDIA engineers (Tawa): manual warp specialization is "labor-intensive, error-prone, and unsustainable."

## Fundamental vs. transient

| Problem | Classification | Why |
|---|---|---|
| Cold compile ∝ graph size; N-rank duplicate compiles | Transient (mostly) | Caching, regional compile, subprocess pools, aot_compile, compile-on-one-rank landed 2025–26; residual JIT cost inherent |
| Recompilation from guards/dynamic shapes | Fundamental | Specialization is the design; `@dynamic_spec` shifts trade-off |
| Graph breaks / control flow → functional rewrites | Fundamental | Consequence of tracing Python |
| Cache bugs, unstable-API reliance | Transient | Engineering backlog |
| Numerics differ from eager | Fundamental | Bitwise equivalence not guaranteed by design |
| Distributed: unoptimized collectives, no SPMD assumption | Mixed | |
| Triton perf ceiling on Hopper/Blackwell | Fundamental | Blocked SIMB abstraction hides warp-level scheduling; Gluon/TLX exist because compiler can't recover it |
| Gluon/TLX non-portability | Fundamental | Explicit hardware control is arch-specific |
| Triton autotuning cost | Fundamental (mitigable) | Movable offline or analytic (Origami) |
| Irregular/MoE | Mostly fundamental | Tile-SPMD vs data-dependent work |
| AMD/Intel parity | Transient | HipKittens shows hardware allows it |
| Backend coupling, version regressions | Transient | |
| Triton debuggability | Mixed | |
| Governance concentration | Fundamental (organizational) | Nine core maintainers w/ veto; OpenAI lead |

## Key quotes
1. "Most pathological compile times arise from repeated recompilation (often due to dynamic shapes, but sometimes not)." — Edward Yang, Meta, 2025-08-13
2. "vLLM wants fast serving performance and no recompilations during model serving." — vLLM team, 2025-08-20
3. "What was once a small gap has grown to a chasm!" — PyTorch FlexAttention team, 2026-03-04
4. "Gluon kernels are not portable anymore like Triton." — Lei Zhang, 2026-02-28
5. "the compiler must discover and implement all of them on the user's behalf." — TLX paper (Meta/UCSD), 2026-05-11
6. "tuning flash attention v2 extensively to achieve best-possible performance took nearly 24 hours for each GPU type" — IBM Research, 2025-10-07
7. "a user adds torch.compile to working code and loses the error that would have caught their bug" — Nikita Shulga (malfet), 2026-09-18
8. "has not united the ecosystem or challenged CUDA's dominance" — Chris Lattner, 2025-03-26

Not retrievable/verified: Tillet's GTC 2025 slides; HipKittens PDF; Triton Conference 2025 keynote content; any quantitative torch.compile production-adoption survey.
