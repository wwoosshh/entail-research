# Prior work for the "role-class defect" problem statement

> **2026-09-23 verification.** Claims were re-checked against primary sources in `reinvestigation/audit/prior_work_audit.json` (partial run; 62 confirmed, 25 partly different, 0 wrong). Corrections are summarized in `THEORY.md` section 4.4.

> **Update 2026-09-23.** A follow-up novelty check with four adversarial searches is in `NOVELTY_CHECK.md` (Korean). It adds sources this file lacks:
> - M2K (arXiv 2603.24595)
> - Emerge (arXiv 2603.21851), HF-vs-vLLM equivalence checking
> - The Foundation Cracks (arXiv 2506.12320)
> - Scalify (arXiv 2509.10694)
> - Meta spmd_types and Megatron-LM #7452
> - the SGLang KV canary
> - the vLLM production-quality blog (2026-07-16)
> - the Artificial Analysis Endpoint Accuracy Index (2026-08-04)
> - vLLM #16801 (Llama 4 INT4 scale bug)
> - the rejected transformers strict-loading PR #48962
> - the history of abandoned semantic tensor typing (named tensors, JAX xmap)
>
> It also revises section 6. The problem's existence and after-the-fact detection are already published. The remaining novelty is narrowed as listed there.

- Compiled: 2026-09-22, for the realworld pilot (70 root-caused wrong-output bugs from vLLM, SGLang, transformers and llama.cpp, 2025-01 to 2026-09; see `REALWORLD_NOTES.md`).
- Method: WebSearch to find sources, then WebFetch on each page. PDFs were saved and converted locally with `pdftotext` so that numbers could be checked against the paper text. All pages were accessed on 2026-09-22.
- Tags:
  - [computed]: my arithmetic from a paper's own table.
  - [secondary]: a third party reporting a primary statement.
  - [not verified]: not stated on the fetched page, or seen only in a search listing.
  - [my reading]: my interpretation, not the source's claim.
- Wording is paraphrased. There are no direct quotations. Category names are given as the papers label them.
- **Comparability warning.** The pilot kept only issues whose titles report wrong output, then classified their root causes. Most studies below classify all bugs, crashes included. Their wrong-output shares are therefore unconditional, while the pilot's 92% silent rate is conditional on wrong output. The closest like-for-like dataset is Ekka's set of silent errors from vLLM and SGLang (section 1.3).

---

## 1. Empirical bug studies

### 1.1 DL programs and DL frameworks

- **Zhang et al. (Yuhao Zhang, Yifan Chen, Shing-Chi Cheung, ...): "An Empirical Study on TensorFlow Program Bugs", ISSTA 2018 (July 2018).** https://xiongyingfei.github.io/papers/ISSTA18b.pdf
  - Sample: 175 bugs in TensorFlow application code, 87 from Stack Overflow and 88 from GitHub.
  - Root causes [computed from Table 3]:
    - TF API change: 44 (25%)
    - Incorrect model parameter or structure (IPS): 38 (22%)
    - TF API misuse: 33 (19%)
    - Unaligned tensor (UT; the tensor shape is not what the API expects): 24 (14%)
    - Confusion with the computation model: 17 (10%)
    - Structure inefficiency: 3
    - Others: 16
  - Symptoms [computed]: error 112 (64%), low effectiveness 40 (23%), low efficiency 9 (5%), unknown 14 (8%).
  - The paper notes that in DL programs, coincidental correctness (a bug fires but no failure is visible) happens at a larger scale and is harder to observe. Users relied on statistics over many runs as their oracle.

- **Islam, Nguyen, Pan, Rajan: "A Comprehensive Study on Deep Learning Bug Characteristics", ESEC/FSE 2019 (arXiv 1906.01388, 3 Jun 2019).** https://arxiv.org/abs/1906.01388
  - Sample: 2,716 Stack Overflow posts (415 bugs) and 500 GitHub fix commits (555 bugs) for Caffe, Keras, TensorFlow, Theano and Torch. That is 970 bugs [computed].
  - Categories closest to the role class:
    - IPS averages 24% of bugs across libraries.
    - Unaligned tensor is 28% of Torch bugs.
    - Absence of type checking is 30% of bugs in Theano, 15% in TensorFlow and 8% in Keras.
    - API change is 9% (TensorFlow) and 7% (Keras).
  - Effects: more than 66% of bugs crash on average. On average 12% cause incorrect functionality, meaning unexpected behaviour without any runtime or compile-time error.

- **Humbatova et al.: "Taxonomy of Real Faults in Deep Learning Systems", ICSE 2020 (arXiv v3 7 Nov 2019).** https://arxiv.org/abs/1910.11015
  - Sample: 1,059 artefacts (477 Stack Overflow discussions, 271 GitHub issues and PRs, 311 commits) plus 20 interviews. A survey of 21 researchers and practitioners validated the taxonomy.
  - Top-level categories, shown as artefact count + interview count:
    - Tensors & Inputs: 53+20
    - Training: 37+160
    - Model: 29+45
    - API: 20+0
    - GPU usage: 10+1
  - The artefact counts sum to 149. Tensors & Inputs is 36% of them [computed].
  - Inside Tensors & Inputs:
    - Wrong tensor shape: 21+5. This is the most frequent single tag from artefacts.
    - Wrong input: 32+15. This splits into wrong shape of input data (22+7), wrong input format (5+5) and wrong type of input data (5+3).
  - The paper has no silent/crash split.

- **Chen, Liang, Shen, Jiang, Li: "Toward Understanding Deep Learning Framework Bugs", ACM TOSEM 32(6) Art. 135, Sept 2023.** https://xgdsmileboy.github.io/files/paper/study-tosem23.pdf
  - Sample: 1,000 bugs, 250 each from TensorFlow, PyTorch, MXNet and DL4J, drawn from 1,250 PRs. The paper uses 13 root-cause categories.
  - Closest categories:
    - Type issue: 142 (14.2%). 100 of these are about tensor types; implicit type conversion is a common trigger.
    - Misconfiguration: 134 (13.4%).
    - API misuse: 118 (11.8%). 84 of these call the wrong API.
    - The four DL-specific causes together are 50.2%: incorrect algorithm implementation (152, 15.2%), type issue, tensor shape misalignment and environment incompatibility.
    - API incompatibility is a separate category; I did not extract its count.
  - Symptoms:
    - Crash: 514 (51.4%)
    - Incorrect functionality: 243 (24.3%). Of these, 115 give wrong intermediate states, 105 wrong predictions and 23 a wrong model structure.
    - Build failure 188, poor performance 21, hang 3, unreported 31.

- **Tambon et al.: "Silent Bugs in Deep Learning Frameworks: An Empirical Study of Keras and TensorFlow", Empirical Software Engineering (DOI 10.1007/s10664-023-10389-6; arXiv v2 1 Sep 2023).** https://arxiv.org/abs/2112.13314
  - Sample: 1,168 closed Keras-related bug issues were screened and 77 reproducible silent bugs kept. Silent means wrong behaviour with no crash, hang or error message.
  - Impact categories:
    - Wrong calculation: 23 (29.9%)
    - Wrong parameter setting: 16 (20.8%). A parameter of a function or component is not applied as set; for example, a learning rate of zero still updated the weights.
    - Wrong displayed message: 15 (19.5%)
    - Wrong save/reload: 10 (13%). The model changes on save or reload; in one example accuracy fell from 100% to 50% after reload.
    - Wrong resulting shape: 6 (7.8%)
    - Performance: 4
    - Wrong structure: 3
  - In a survey of 103 developers, 72.8% judged silent bugs more problematic than ordinary bugs.
  - [my reading] Wrong parameter setting and wrong save/reload have the same shape as the pilot's "fact silently dropped or changed at a boundary".

### 1.2 DL compilers, torch.compile, model converters

- **Shen, Ma, Chen, Tian, Cheung, Chen: "A Comprehensive Study of Deep Learning Compiler Bugs", ESEC/FSE 2021 (Aug 2021).** https://github.com/ShenQingchao/DLCstudy (paper PDF in the repository)
  - Sample: 603 bugs (TVM 318, Glow 145, nGraph 140) from 1,361 bug-fix PRs. The paper uses 12 root causes and 6 symptoms.
  - Closest categories:
    - Type problem: 19.23% (116 bugs). Of these, 62 involve tensor types, 34 conventional types and 20 node types. This is the most common root cause.
    - Tensor shape problem: 13.27%. The definition covers shape and memory layout, including layout transformation.
    - Also present: API misuse; incompatibility (73% of it external, e.g. with DL frameworks); misconfiguration; incorrect assignment.
  - Symptoms:
    - Crash: 59.37%
    - Wrong code (wrong result without a crash): 25.04%
    - Build failure 8.29%, bad performance 1.82%, hang 0.83%.
  - Labelling: two authors. Cohen's kappa reached 85% after calibration and stayed above 95% afterwards.

- **Li, Li, Liu, Cheung: "Demystifying the Silence of Correctness Bugs in PyTorch Compiler", arXiv 2604.08720 (9 Apr 2026).** https://arxiv.org/abs/2604.08720
  - Context: 19.2% of 296 high-priority PyTorch issues are torch.compile correctness bugs, meaning wrong output with no exception, crash or warning. This is the second-largest category after crashes.
  - Sample: 116 confirmed correctness bugs from Apr 2023 to Apr 2025.
  - Categories:
    - In-place operation handling: 21.6%
    - Low-level code generation: 19.8%
    - Operator transformation: 18.1%
    - Graph semantic capturing: 15.5%
    - Memory layout conflicts (layout not preserved): 12.1%
    - Graph caching (wrong reuse of a cached graph): 4.3%
    - Other: 8.6% (precision 3, configuration 5, external library 2)
  - Tool: AlignGuard, which uses LLM-based test mutation. It found 23 new correctness bugs, 14 of them marked high priority.

- **Yuan et al.: "Demystifying Deep Learning Compiler Frontend Bugs: An LLM-Aided Empirical Study", arXiv 2607.25651 (28 Jul 2026).** https://arxiv.org/html/2607.25651
  - Sample: 123 TorchDynamo frontend bugs from PyTorch 2.1 to 2.6.
  - Categories:
    - Execution context and scope: 19.5%
    - Python object modelling: 18.7%
    - Guard overspecialization and deficiencies: 14.6%
    - Container modelling: 12.2%
    - Uncaptured side effects: 12.2%
    - Missing type transformation: 6.5%
    - Iterator-state desynchronization: 3.3%
    - Others: 13.0%
  - Symptoms: crash 71.5%, silent inconsistency 22.8%, timeout 5.7%.

- **Jajal et al.: "Analysis of Failures and Risks in Deep Learning Model Converters: A Case Study in the ONNX Ecosystem", ISSTA 2024 (arXiv 2303.17708).** https://arxiv.org/abs/2303.17708
  - Sample: 200 issues in the PyTorch and TensorFlow ONNX converters, plus a survey of 92 engineers.
  - About 75% of defects are in node conversion. About one third of reported failures produce semantically incorrect models, which convert without error but compute wrong results.

- **Qiu, Wang, Badhe, Limpanukorn, Kim, Zhang: "Finding Compiler-Platform Interaction Bugs in Deep Learning Pipelines via Cross-Layer Constraints" (XCheck), arXiv 2606.18421 (16 Jun 2026).** https://arxiv.org/abs/2606.18421
  - This is a testing paper, not a bug census. It derives constraints that span compilation layers and inserts runtime assertions.
  - It reports 2,034 bug-revealing cases on three DL compilers (TVM and ONNX-MLIR are named), including silent unexpected compilations.
  - [my reading] It is the nearest testing work that treats facts spanning layers as first-class, but it uses them as test oracles, not as declared interface types.

### 1.3 LLM inference, serving and distributed frameworks

- **Liu, Zhong, Bi et al.: "A First Look at Bugs in LLM Inference Engines", arXiv 2506.09713 (v2 Jan 2026); ACM TOSEM, DOI 10.1145/3788873.** https://arxiv.org/html/2506.09713
  - Sample: 929 closed bugs with a bug label, collected as of 1 Dec 2024 from llama.cpp, vLLM, DeepSpeed (inference), MLC-LLM and TensorRT-LLM. There were 1,488 candidates before sampling and filtering.
  - Symptoms (932 instances):
    - Crash: 603 (65%)
    - Unexpected output: 122 (13%)
    - Feature failure: 107 (11%)
    - Abnormal performance: 53 (6%)
    - Hang: 35 (4%)
    - Silent error, defined as an internal anomaly with no error detection: 12 (1%)
  - Root causes (1,041 instances, 28 leaf categories):
    - Configuration: 242 (23%). Misconfiguration 105, incompatible model 65, concurrency 39, mismatched precision 17, documentation 13, job context 3.
    - Functionality: 432 (41%). Incorrect algorithm implementation 184, misused API 75, mismatched shape 53, exception handling 37, conditional logic 24, incorrect type 24, numerical issue 17, incorrect assignment 15, synchronization 3.
    - Environment: 301 (29%). Incompatible backend 138, incompatible version 98, dependent module 65.
    - Resource: 37 (4%), including incorrect cache management 11.
    - Input/output processing: 41 (4%).
  - Unexpected-output bugs trace mainly to functionality (88), then configuration (26) and environment (15). Numerical issue is 1.6% of root-cause instances [computed]. Mismatched shape, type and precision together are 94 of 1,041, about 9% [computed].
  - [my reading] The role class is spread over several leaves (misconfiguration, incompatible model, mismatched precision, shape or type, cache management). It is not a category of its own.

- **Gu, Zhang, Zhu, Fu, Wu, Wang, Kasikci: "Ekka: Automated Diagnosis of Silent Errors in LLM Inference", arXiv 2606.04594 (3 Jun 2026).** https://arxiv.org/html/2606.04594
  - Sample: 90 silent errors (48 vLLM, 42 SGLang). They were found with a bug label or title plus quality-regression keywords such as accuracy, inconsistent and garbage. The 70 closed issues form the empirical study.
  - Root-cause location:
    - Framework implementation: 30.6% (e.g. async engine, CUDA graph compilation)
    - Model implementation: 25.5% (architecture definition, model parameters set wrongly, chat templates)
    - Kernel backend: 24.5% (e.g. FlashAttention)
    - Numerical precision: 19.4% (floating-point instability with no logic defect)
  - The page does not make clear what these percentages are computed over [not verified].
  - 43.8% of issues show up as end-to-end accuracy regressions. Developers toggled configurations to debug in over 60% of issues.
  - Method: Ekka aligns intermediate states with a correct reference implementation (e.g. HuggingFace) and finds where they diverge. It reports 80% pass@1 and 88% pass@5 diagnosis accuracy, and 4 new silent errors confirmed by developers.
  - Relevance: this is the most like-for-like prior dataset (silent errors only, from vLLM and SGLang). It is classified by where the bug lives, not by which fact was lost.
  - Its 19.4% numerical-precision share contrasts with the pilot's 3 of 70 pure numerical kernel defects. The definitions differ, so re-label before claiming a disagreement.

- **"Towards Understanding Bugs in Distributed Training and Inference Frameworks for Large Language Models", arXiv 2506.10426 (June 2025).** https://arxiv.org/html/2506.10426
  - Sample: 308 fixed bugs (DeepSpeed 168, Colossal-AI 131, Megatron-LM 9).
  - The most common symptoms are crash, incorrect functionality and build failure. The most common root causes are API misuse, configuration error, incorrect implementation and missing precondition.
  - Distributed-specific causes include distributed tensor errors. The text gives no per-category counts [not verified].
  - 48% of fixes change 10 lines or fewer.

- **Ma, Zhan, Chen, Li, Keung, Sarro: "A Comprehensive Study of Bugs in Modern Distributed Deep Learning Systems", arXiv 2512.20345 (23 Dec 2025).** https://arxiv.org/html/2512.20345
  - Sample: 849 issues (DeepSpeed 496, Megatron-LM 48, Colossal-AI 305).
  - Inference stage, 10.1% of issues. Root causes:
    - Misconfigured resource or distributed settings: 22.1%
    - Faulty inference logic: 18.6%
    - Dependency or hardware mismatch: 17.4%
    - Misused framework API: 16.3%
    - Framework–model incompatibility: 12.8%
  - Inference symptoms include output inconsistency across devices under tensor parallelism: 11.6%.
  - In the data and model preparation stage, misaligned parameters, tensors or model structure make up 15.7%.

- **Zhao, Zhao, Zhang, Liu, Mazurek: "Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing", arXiv 2605.11202 (11 May 2026).** https://arxiv.org/abs/2605.11202
  - The fuzzer targets vLLM and SGLang. Its oracles cover crashes, hangs, performance pathologies and silent output corruption, confirmed with log-probability checks.
  - It found 15 vulnerabilities (10 confirmed, 2 CVEs), including KV-cache isolation failures and silent cross-request contamination.

### 1.4 GPU kernels and tile programs

- **Wu, Zhou, Zhang, Liu, Zhang: "Characterizing and Detecting CUDA Program Bugs", arXiv 1905.01833 (May 2019).** https://arxiv.org/abs/1905.01833
  - Sample: 319 bugs from 5 CUDA projects.
  - Root causes:
    - Generic error: 63% of kernel-function bugs
    - Improper synchronization: 27 of 217 (12.4%)
    - Non-optimal implementation: 8%
    - Poor portability: 3%
    - Improper resource management (no share extracted)
  - Flaky-test synchronization bugs survived a median of 144 days.

- **Rathnasuriya et al. (author list per search listing): "An Investigation on Numerical Bugs in GPU Programs Towards Automated Bug Detection", ISSTA 2025.** https://youngwei.com/publication/gpubug/
  - Sample: 397 numerical bug samples from GitHub.
  - The GPU-NBDetect tool found 226 bugs in 186 math functions across 4 libraries; 60 were confirmed.
  - These bugs depend on specific input values or types, and reliable oracles are lacking.

- **Rathnasuriya et al.: "Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection", ISSTA 2026 (arXiv 2605.19652).** https://arxiv.org/html/2605.19652v1
  - Sample: 301 code-generation bugs from 401 reports across 8 frameworks: Triton, TileLang, Halide, TVM, XLA, DaCe, NVIDIA Warp and PyTorch. Reports span Jan 2022 to Nov 2025.
  - Categories:
    - Type & operator: 48.84%. This includes loss of numeric meaning through implicit casts or mixed precision (19.27%).
    - Memory: 19.27%. This includes indexing, stride and layout errors (11.63%).
    - IR construction and transformation: 16.28%. This includes lowered nodes with missing metadata.
    - Tile mapping & launch: 6.31%
    - Control flow & scheduling: 5.32%
    - Device-specific: 3.99%
  - Symptoms: crash 58.14%; correctness issues (silent wrong results, numeric pathologies, cross-implementation mismatches) 36.21%; performance 5.65%.

- **Zhou, Lezcano, Goucher et al.: "Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2", ASPLOS 2026 (arXiv 2505.23819).** https://arxiv.org/html/2505.23819v4
  - Reports that 12% of bugs filed in Triton's GitHub repository are layout-related.
  - Hand-written per-layout conversions in the legacy system were error-prone. Linear layouts model every layout as a linear map, so conversions become generic.

### 1.5 Cross-study view

| Study (subject) | Sample | Closest-to-role categories (share) | Wrong-output / silent share |
|---|---|---|---|
| Zhang 2018 (TF programs) | 175 | IPS 22%, API misuse 19%, unaligned tensor 14% [computed] | low effectiveness 23% [computed] |
| Islam 2019 (DL programs) | 970 [computed] | IPS avg 24%; unaligned tensor; no type checking | incorrect functionality 12% |
| Humbatova 2020 (DL programs) | 149 artefact tags | Tensors & Inputs 36% [computed] | not reported |
| Chen 2023 (DL frameworks) | 1,000 | type 14.2%, misconfiguration 13.4%, API misuse 11.8% | incorrect functionality 24.3% |
| Tambon 2023 (Keras) | 77 silent | wrong parameter setting 20.8%, save/reload 13% | 100% (silent by selection) |
| Shen 2021 (DL compilers) | 603 | type 19.23%, shape/layout 13.27% | wrong code 25.04% |
| Li 2026 (torch.compile) | 116 correctness | memory layout 12.1%, graph caching 4.3% | 100% by selection; 19.2% of high-priority issues |
| Yuan 2026 (TorchDynamo) | 123 | guard deficiencies 14.6%, missing type transformation 6.5% | 22.8% |
| Jajal 2024 (ONNX converters) | 200 | node conversion about 75% of defects | about one third semantically incorrect |
| Liu 2025/26 (LLM engines) | 929 | configuration 23%; shape/type/precision mismatch about 9% [computed] | unexpected output 13% + silent 1% |
| Gu 2026, Ekka (vLLM, SGLang) | 70 closed silent | model implementation 25.5% (location-based) | 100% by selection |
| Ma 2025 (distributed, inference slice) | 849 | misconfigured settings 22.1%, framework–model incompatibility 12.8% | cross-device output inconsistency 11.6% of inference issues |
| Rathnasuriya 2026 (tile programs) | 301 | type & operator 48.84%; indexing/stride/layout 11.63% | correctness 36.21% |
| This pilot | 70 root-caused wrong-output bugs | role class 71% (59% conservative) | 92% of role-class bugs silent (conditional) |

---

## 2. Classic software-engineering evidence on interface faults

- **Perry & Evangelist: "An Empirical Study of Software Interface Faults — An Update", Proc. 20th HICSS, Jan 1987, vol. II pp. 113–126.** This updates their 1985 study. https://users.ece.utexas.edu/~perry/work/papers/ie-update.pdf
  - System: the third release of a large real-time system. The faults were those testers reported during integration and system test, over about 350,000 non-comment C source lines.
  - An operational definition (a fix touching two or more C files or a global header) flagged 35.87% of all faults.
  - 51% of single-file faults were also interface faults, which is 32.73% of all faults.
  - Combined, **68.6% of the error population were interface faults.**
  - The paper uses 16 categories: construction, inadequate functionality, disagreements on functionality, changes in functionality, added functionality, misuse of interface, data structure alteration, inadequate error processing, additions to error processing, inadequate postprocessing, inadequate interface support, initialization/value errors, violation of data constraints, timing/performance, coordination of changes, and hardware/software interfaces.
  - Shares of all errors, combined set: error handling 14.1%, interface misuse/support 11.8%, functionality 11.4%, data 10.1%, coordination of changes 7.0%.
  - No single category dominates, so the authors argue no single technique would remove most interface faults. They also judge that methodology problems underlie most of them.
  - The authors' Inscape environment is built on formal module interface specifications. They claim it could potentially prevent a little over 73% of interface faults, which is over 50% of all faults.
  - The paper also summarises earlier rates: Thayer et al. 17% and 22.5% in two projects (narrow definition), Bowen 4.5%, Basili & Perricone 39%.

- **Basili & Perricone: "Software Errors and Complexity: An Empirical Investigation", CACM 27(1), Jan 1984** [secondary: as summarised in Perry & Evangelist 1987 and Perry & Stieg 1993].
  - Medium-scale system.
  - Interface errors were 39% of errors, the largest class. They defined interface errors as errors tied to structures outside a module's local environment that the module uses.
  - They concluded that interfaces appear to be a major problem.

- **Perry & Stieg: "Software Faults in Evolving a Large, Real-Time System: a Case Study", ESEC 1993.** https://users.ece.utexas.edu/~perry/work/papers/esec93.pdf
  - Setting: a very large system (at least 1M NCSL) in evolutionary development. Data came from questionnaires to the owners of each fault report.
  - **Interface faults were about 49% of design and coding faults.** They took about the same effort to find and more effort to fix (fix-weighted 56%).
  - Missing information (incomplete or omitted requirements or design) led to interface faults more often. Ambiguity led to implementation faults more often. For example, incomplete or omitted requirements were 79.6% interface.
  - Formal interface specifications were the means of prevention most associated with interface faults (73.6% interface). Lack of information dominated the underlying causes.
  - The authors note that better programming languages would solve few of the problems: language pitfalls and races together were under 8%.

- **NASA Mars Climate Orbiter Mishap Investigation Board, Phase I Report, 10 Nov 1999.** https://llis.nasa.gov/llis_lib/pdf/1009464main1_0641-mr.pdf
  - Root cause: a ground software file called Small Forces, used in trajectory models, was coded with English units instead of the specified metric units.
  - It is the canonical case of correct components losing a unit fact at an interface.

- **Sculley et al.: "Hidden Technical Debt in Machine Learning Systems", NIPS 2015.** https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
  - Names ML-specific risk factors, including boundary erosion, entanglement, undeclared consumers, data dependencies and configuration issues.

- **What the sources suggest is specific to ML inference** (facts come from the cited sources; the grouping is mine):
  - Weak observability.
    - Coincidental correctness is common and hard to observe in DL programs, and oracles are statistical (Zhang 2018).
    - Anthropic's evaluations did not capture its 2025 degradations, partly because the model often recovers from isolated mistakes (Anthropic, section 3).
    - Wrong-output bugs lack test oracles (Shen 2021; Chen 2023).
  - The relevant facts change per layer and at runtime. vLLM tracks a sliding-window size per layer group and frees KV blocks outside the window; hybrid models mix attention types (vLLM hybrid KV cache doc, section 4.4).
  - Many independent re-implementations of one model diverge.
    - Backend choice accounts for about 39% of out-of-the-box benchmark variability (Masoudian et al. 2026).
    - Model vendors now ship verifiers for provider deployments (Moonshot; OpenAI) (section 3).
  - [my reading] The classic 39–69% figures come from C systems, where the interface is a function signature or a header. In inference stacks the "interface" also includes checkpoint configs, metadata tensors, kernel parameters and chat templates. Those carry the facts the pilot found lost.

---

## 3. Production incidents with silent quality degradation

- **Anthropic: "A postmortem of three recent issues", 17 Sep 2025.** https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues
  - (1) **Context-window routing error.** Introduced 5 Aug 2025, fixed 4 Sep, rollout completed 16–18 Sep.
    - Some short-context Sonnet 4 requests were routed to servers configured for the upcoming 1M-token context window.
    - Routing was sticky, so follow-up requests tended to stay on the wrong server.
    - It started at 0.8% of Sonnet 4 requests on 5 Aug. A load-balancing change on 29 Aug raised it to 16% at the worst hour on 31 Aug.
    - About 30% of Claude Code users had at least one message misrouted.
  - (2) **Output corruption**, 25 Aug to 2 Sep.
    - A misconfiguration on TPU servers, in a runtime performance optimization, occasionally gave high probability to tokens that should be rare. Examples: Thai characters in English replies, syntax errors in code.
    - Opus 4.1 and 4 were affected from 25 to 28 Aug, Sonnet 4 until 2 Sep.
  - (3) **Approximate top-k XLA:TPU miscompilation.** Introduced 25 Aug, partially rolled back 4 Sep, fully rolled back 12 Sep.
    - The 25 Aug change removed a December 2024 workaround, which exposed a deeper bug.
    - Parts of the sampling computation ran at different precisions: the model computes in bf16, while the XLA flag `xla_allow_excess_precision` (default on) let the compiler run some operations in fp32. As a result, operations that should agree on the top token disagreed.
    - Results were wrong only for some batch sizes and model configurations. Haiku 3.5 was confirmed affected, and others possibly.
    - Fix: exact top-k and more fp32 standardisation.
  - Detection: benchmarks and evaluations did not capture the degradation, early reports looked like normal variation, and privacy controls limit engineer access to user interactions.
  - Announced remediations: more sensitive evaluations, continuous evaluations on production systems, and faster debugging tools.
  - [my reading] Issue (1) is a mismatch of a capability or configuration fact between request and server. Issue (3) is a precision-format mismatch between operations meant to agree. Issue (2) is described too briefly to classify.

- **Anthropic: "An update on recent Claude Code quality reports", 23 Apr 2026.** https://www.anthropic.com/engineering/april-23-postmortem
  - Three product-layer changes. The API and inference layer were not affected.
    - The default reasoning effort was lowered from 4 Mar to 7 Apr.
    - A change meant to clear old thinking once, for sessions idle more than an hour, instead cleared it on every later turn (26 Mar to 10 Apr).
    - A verbosity limit in the system prompt (16 to 20 Apr) cost 3% on one evaluation.
  - These were hard to detect because the changes hit different traffic slices. The caching bug needed stale sessions, and internal experiments masked it.
  - [my reading] The thinking-clearing bug is a state fact applied at the wrong time, but in the agent harness, outside the inference stack.

- **OpenAI status incident "Unexpected responses from ChatGPT", 20–21 Feb 2024 (marked resolved 22 Feb).** https://status.openai.com/incidents/ssg8fh7sfyz3
  - An optimization introduced a bug in token selection: inference kernels produced incorrect results in certain GPU configurations, and the output was nonsensical text.
  - The postmortem gives no further mechanism. [my reading] There is not enough detail to classify it as role-class or numerical.

- **Azure deployment of gpt-oss, Aug 2025** [secondary]: Simon Willison, 15 Aug 2025, updated 20 Aug. https://simonwillison.net/2025/Aug/15/inconsistent-performance/
  - Artificial Analysis ran AIME25 with gpt-oss-120b across providers. Initial scores ranged from 36.7% to 93.3%.
  - Microsoft attributed Azure's lower score to old vLLM commits that ignored `reasoning_effort`, so every request ran at medium effort.
  - Azure and Groq reached 93.3% after fixes.

- **OpenAI: "Verifying gpt-oss implementations" (cookbook, undated).** https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations
  - Lists facts providers must get right:
    - the harmony prompt format; wrong formatting degrades generation and function calling
    - returning raw chain of thought so it can be passed back in later turns
    - MXFP4 MoE weights, which need adapted inference code
  - Verification uses a tool-calling smoke test plus AIME, GPQA and HealthBench runs compared against reference implementations.

- **vLLM blog: "Chasing 100% Accuracy: A Deep Dive into Debugging Kimi K2's Tool-Calling on vLLM", 28 Oct 2025.** https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy
  - Three issues:
    - vLLM passes only explicitly declared chat-template arguments. Kimi's tokenizer accepts `add_generation_prompt` only through `**kwargs`, so vLLM silently dropped it and prompts ended without the assistant-turn tokens.
    - vLLM converted empty string content into a list form, which Kimi's template rendered as literal text.
    - A strict tool-call-ID parser raised an IndexError and discarded valid tool calls.
  - Parsed tool calls rose from 218 to 971, about 4.4x.
  - [my reading] The first two are clear role-class defects: an argument dropped at a boundary, and a value-format mismatch.

- **Moonshot AI: K2-Vendor-Verifier (GitHub; results dated 2025-11-15).** https://github.com/MoonshotAI/K2-Vendor-Verifier
  - Measures tool-call trigger F1 and schema accuracy per provider. For example, one provider scored 50.60% F1 on K2-0905.
  - Stated causes: outdated vLLM or SGLang versions, malformed tool-call IDs, and no guided decoding.

- **Moonshot AI: Kimi Vendor Verifier blog (undated; released with K2.6).** https://www.kimi.ai/blog/kimi-vendor-verifier
  - Reports misused decoding parameters, KV-cache bugs, quantization degradation, vision preprocessing errors and tool-calling inconsistencies at third-party providers.
  - Users could not tell implementation failures from model limitations.

- **Meta, Llama 4 launch** [secondary]: TechCrunch, 7 Apr 2025. https://techcrunch.com/2025/04/07/meta-exec-denies-the-company-artificially-boosted-llama-4s-benchmark-scores/
  - Meta's VP of generative AI attributed mixed quality across services to public implementations that still needed several days to stabilise.

- **Masoudian, Shafaei, Swain, Schedl: "What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend", arXiv 2608.04714 (5 Aug 2026).** https://arxiv.org/abs/2608.04714
  - Across five inference frameworks, about 39% of the out-of-the-box score variability comes from the backend: default generation parameters, sampling, and model-dependent effects.

- **Related, but numerical rather than role-class:** He et al. (Thinking Machines Lab), "Defeating Nondeterminism in LLM Inference", 10 Sep 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  - Nondeterminism comes mainly from kernels that are not batch-invariant.
  - With batch-invariant kernels, Qwen3-235B gave 1,000 identical completions out of 1,000, versus 80 unique before, at about 1.6–2x the cost.

---

## 4. Existing solution approaches

Each entry covers:
- the facts the approach can express
- whether it is mandatory or optional
- whether it is checked statically or at runtime
- adoption in inference engines, only where a fetched source showed it

### 4.1 Named axes and dimension typing

- **PyTorch named tensors.**
  - Facts: dimension names, checked and propagated at runtime. Mismatched names in the same position raise an error.
  - Status: a prototype with limited operator coverage (no indexing, nn modules, JIT, distributed or ONNX); optional.
  - Removed entirely in PyTorch 2.13.0; the release date was not verified.
  - Sources: https://docs.w3cub.com/pytorch~2.9/named_tensor.html ; https://github.com/pytorch/pytorch/releases/tag/v2.13.0
- **einops.**
  - Facts: named axes in pattern strings, with runtime checks that axis lengths agree.
  - Optional; works across frameworks. Published at ICLR 2022; the README says more than 10k projects use it.
  - Adoption in engines: not assessed.
  - Source: https://github.com/arogozhnikov/einops
- **jaxtyping.**
  - Facts: dtype plus shape, with named or symbolic dimensions.
  - Checked at runtime through typeguard or beartype with the `jaxtyped` decorator. Dimension sizes must agree across all arguments and the return value of one call.
  - Optional; supports JAX, PyTorch, NumPy, MLX and TensorFlow.
  - Sources: https://docs.kidger.site/jaxtyping/ ; https://docs.kidger.site/jaxtyping/api/runtime-type-checking/
- **torchtyping.**
  - Facts: shape, dtype, names and layout (dense or sparse); runtime checks through typeguard.
  - Its author now recommends jaxtyping instead.
  - Source: https://github.com/patrick-kidger/torchtyping
- **xarray.**
  - Facts: dimension names, coordinate labels and attributes. Broadcasting and alignment go by name and label at runtime.
  - Optional; a scientific-computing tool, not an inference tool.
  - Source: https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html
- **Dex** (Paszke et al., arXiv 2104.05372). Arrays are functions on typed index sets; static typing rules out classes of indexing errors. It is a research language. https://arxiv.org/abs/2104.05372
- **Named Tensor Notation** (Chiang, Rush, Barak; TMLR, Jan 2023). A notation that removes the burden of tracking axis order and meaning. https://arxiv.org/abs/2102.13196
- **Rush, "Tensor Considered Harmful"** (Harvard NLP; undated page). Argues that positional dimensions let axis bugs pass without runtime errors. https://nlp.seas.harvard.edu/NamedTensor
- **Chen, "Typesafe Abstractions for Tensor Operations"** (Scala Symposium 2017). Typed axes, checked at compile time. https://arxiv.org/abs/1710.06892
- [my reading] Everything in this group expresses axis identity, shape and dtype. None expresses the other role facts in the pilot: absolute vs relative position, valid length or window, partial vs replicated, readiness.

### 4.2 Shape and refinement type systems; static analysers

- **PEP 646** (final, Python 3.11). Variadic generics for array shapes. Static checkers only, with no shape arithmetic and no runtime enforcement. https://peps.python.org/pep-0646/
- **Pythia** (Lagouvardos et al., ECOOP 2020). Static shape tracking across TensorFlow library calls. It detected 11 of the 14 shape bugs from Zhang 2018, with 84.62% precision. https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECOOP.2020.15
- **PyTea** (Jhoo et al., Seoul National University). Static, path-sensitive collection of shape constraints for PyTorch code, followed by a satisfiability check; it finds shape errors in seconds on its examples. https://sf.snu.ac.kr/pytea/
- **Gradual Tensor Shape Checking** (Hattori, Kobayashi, Sato; arXiv 2203.08402). Best-effort static refinement-type inference. Where static proof fails, it inserts runtime checks. Annotations are optional. https://arxiv.org/abs/2203.08402
  - [not verified] The ESOP 2023 venue appears only in search listings.
- **Compile-Time Tensor Shape Checking via Staged Shape-Dependent Types** (Suwa & Igarashi, ECOOP 2026). Compile-time assertions guarantee that generated code is shape-consistent. It has an OCaml backend and is demonstrated on ocaml-torch examples. https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECOOP.2026.28
- Adoption in the inference engines studied: none found in the sources read.

### 4.3 Distribution and reduction state (the partial vs replicated fact)

- **PyTorch DTensor.**
  - Facts: placements Shard(dim), Replicate(), and Partial(reduce_op). Partial means a reduction is still pending on that mesh dimension.
  - Placements propagate through operators, and `redistribute` changes them by running collectives.
  - Mixing DTensor with plain Tensor inputs is rejected at runtime.
  - The module is in alpha. Using it is optional.
  - Source: https://docs.pytorch.org/docs/2.14/distributed.tensor.html
- **JAX explicit sharding** (one of three modes: Auto, Explicit, Manual).
  - Shardings are part of the array type and can be queried with `jax.typeof`, even under jit.
  - When the output sharding is ambiguous, propagation raises an error and asks for an annotation instead of guessing. For example, a dot product over sharded contracting dimensions needs `out_sharding`.
  - Source: https://docs.jax.dev/en/latest/parallel.html
- **JAX shard_map `check_vma`.**
  - Facts: for each value, whether it varies or is invariant across each manual mesh axis. The type shows it, e.g. `f32[3]{V:i}`.
  - Checked at trace time and **on by default**.
  - `psum` turns varying into invariant, and `pvary` does the reverse.
  - With the check off, a false replication claim in `out_specs` is silent undefined behaviour.
  - Source: https://docs.jax.dev/en/latest/notebooks/shard_map.html
- **ezyang: "The JAX sharding type system", 28 Jan 2026.**
  - JAX abstract values also track unreduced and reduced axes, i.e. pending reductions.
  - A DTensor developer reportedly said that the need to represent Partial is a main reason DTensor placements are mesh-oriented.
  - Source: https://blog.ezyang.com/2026/01/jax-sharding-type-system/
- Relevance: this is the only mechanism found that puts one pilot fact family, partial vs replicated, into the type and checks it by default before execution.

### 4.4 Layout and format carried with the data

- **torchao.**
  - Quantized weights are `torch.Tensor` subclasses such as Float8Tensor and Int4Tensor.
  - They carry block_size, scale, zero_point, dtype and packing format (how the raw data is laid out).
  - `__torch_function__` dispatches operations to kernels.
  - The page does not document any check for a mismatch between packing format and kernel expectations.
  - Source: https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html
- **MLIR builtin types.**
  - RankedTensorType has an optional encoding attribute, e.g. for sparsity, whose semantics every pass must respect. MemRef has an optional layout (strided or affine map).
  - These are static IR types. I did not verify how uniformly passes check encodings.
  - Source: https://mlir.llvm.org/docs/Dialects/Builtin/
- **Triton linear layouts** (ASPLOS 2026, section 1.4). Layouts become linear maps with generic conversion. Motivated by layout bugs, 12% of Triton issues. Internal to the compiler.
- **vLLM attention backend declarations.**
  - Each backend declares its supported model dtypes, KV-cache dtypes, head sizes, block sizes, attention types, sinks, non-causal support, multimodal prefix, DCP and compute capability.
  - With no explicit choice, vLLM picks the first compatible backend. An explicit choice is validated at startup, and errors list the reasons. The page lists 40+ backends, including FLEX_ATTENTION (no sinks).
  - **Sliding window and logit soft-capping are not declared feature columns in these tables.** The page mentions them only in the text.
  - Source: https://docs.vllm.ai/en/stable/design/attention_backends/
- **vLLM per-layer KV cache specs.**
  - Types include full attention, sliding window with `sliding_window_size`, chunked local and Mamba.
  - The specs drive block allocation, freeing of blocks outside the window, and prefix-cache hit rules.
  - Source: https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/
- Relevance: vLLM's tables are the closest in-engine example of declared and validated facts. They cover hardware and format capabilities, not model semantics such as the window or softcap.

### 4.5 Contracts and runtime checkers for ML code

- **DLContract** (Ahmed et al.; ESEC/FSE 2023).
  - Design by contract for Keras, using "ML variables" for model structure, data and training properties.
  - Checked at runtime through an instrumented library. It detected 259 of 272 buggy programs, with small overhead.
  - Source: https://github.com/shibbirtanvin/DLContract
- **Khairunnesa et al.: "What Kinds of Contracts Do ML APIs Need?"** (EMSE 2023).
  - 413 informal API specifications from Stack Overflow (TensorFlow, scikit-learn, Keras, PyTorch).
  - The most needed contracts constrain single arguments or the order of API calls.
  - Source: https://arxiv.org/abs/2307.14465
- **TensorFlow Data Validation** (Breck, Zinkevich, Polyzotis, Whang, Roy; SysML 2019).
  - Schema-based data validation and training–serving skew detection in TFX.
  - Hundreds of Google product teams use it on petabytes of data per day.
  - Source: https://research.google/pubs/pub47967/
- NeMo Neural Types: see section 5.

### 4.6 Sanitizers

- **NVIDIA compute-sanitizer.** Four tools, runtime and opt-in:
  - memcheck: out-of-bounds and misaligned accesses
  - racecheck: shared-memory hazards
  - initcheck: uninitialized global-memory reads
  - synccheck: invalid use of synchronization primitives
  - Source: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- **PyTorch CUDA Stream Sanitizer** (prototype).
  - Records tensor accesses per stream and reports unsynchronized accesses across streams at runtime.
  - Enabled with a call or an environment variable.
  - It checks one readiness fact: whether cross-stream use is ordered.
  - Source: https://docs.pytorch.org/docs/main/cuda._sanitizer.html
- **Triton-Sanitizer** (Wu, ..., Keren Zhou; ASPLOS 2026).
  - Builds symbolic address and mask expressions and checks them with an SMT solver, falling back to eager simulation for indirect accesses.
  - Found 24 unknown memory errors in 7 Triton kernel repositories. On average it is 1.62x faster than compute-sanitizer.
  - Sources: https://hgpu.org/?p=30696 ; https://www.jokeren.tech/publication/wu-2026-triton-sanitizer/
- No sanitizer found that checks inference-level semantic facts (valid length, position role, window, partial vs replicated at runtime) [scoped to my searches].

### 4.7 Differential testing and fuzzing

- **NNSmith** (Liu et al., arXiv 2207.13066; the venue is not shown on the fetched page).
  - Generates valid, diverse models and searches for inputs that avoid NaN/Inf.
  - Runs differential tests across TVM, TensorRT, ONNXRuntime and PyTorch.
  - Found 72 new bugs (58 confirmed, 51 fixed).
  - Source: https://arxiv.org/abs/2207.13066
- **TitanFuzz** (Deng et al., ISSTA 2023). Fuzzes TensorFlow and PyTorch with LLM-generated programs. Found 65 bugs, 41 previously unknown. https://arxiv.org/abs/2212.14834
- **For LLM stacks:**
  - Ekka: diagnosis against a reference implementation (section 1.3).
  - LLM-serving fuzzing with log-probability oracles (section 1.3).
  - AlignGuard for torch.compile (section 1.2).
  - XCheck, cross-layer assertions (section 1.2).
  - Black-box vendor verifiers: Moonshot and OpenAI (section 3).
- All of these detect or diagnose after the fact. None makes the violated fact explicit in the interface.

### 4.8 Declarative attention specifications

- **FlexAttention.** PyTorch blog (last updated 2025-05-30), https://pytorch.org/blog/flexattention/ ; Dong, Feng, Guessous, Liang, He, arXiv 2412.05496 (Dec 2024), https://arxiv.org/abs/2412.05496
  - Facts are user functions: `score_mod(score, b, h, q_idx, kv_idx)` and `mask_mod(b, h, q_idx, kv_idx)`, compiled by torch.compile into a fused kernel. A BlockMask exploits sparsity.
  - Examples: ALiBi, soft-capping, sliding window, prefix-LM, document masking.
  - The blog's document-masking example converts physical indices to per-document logical indices by hand in user code.
  - The blog describes no check against the model's declared configuration.
  - vLLM lists a FLEX_ATTENTION backend (section 4.4).
- **FlashInfer.** Ye et al., MLSys 2025, arXiv 2501.01005. https://arxiv.org/abs/2501.01005
  - A JIT-compiled attention template. Users write functors for query, key and value transforms, a logits transform, a logits mask and an output transform. Each functor receives batch, query/output, key/value and head indices plus a parameter struct.
  - KV layouts: block-sparse and ragged formats. A CPU-side `plan()` runs before `run()` to stay CUDA-Graph compatible.
  - Integrated into SGLang, vLLM and MLC-Engine.
- **AttentionEngine.** Chen et al., arXiv 2502.15349 (21 Feb 2025). https://arxiv.org/abs/2502.15349
  - Decomposes attention into modular operations with customisable components. Programmable templates plus cross-platform scheduling.
  - Reports up to 10x on configurations existing methods do not cover.
- [my reading, per the fetched pages] These make attention facts declarative and fast. They do not require, or check, that the declared variant matches the model's properties, such as its window, softcap, or position convention.

---

## 5. Semantic or role typing across components; role grammars from natural language

- **NVIDIA NeMo Neural Types.** https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/core/neural_types.html ; typecheck API: https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/core/api.html
  - Every module input and output port has a NeuralType, made of:
    - axis semantics, e.g. batch, time, channel
    - element semantics, e.g. logits, audio signal, embeddings, mel spectrogram
  - Comparing two types gives one of: SAME, GREATER, TRANSPOSE_SAME, SAME_TYPE_INCOMPATIBLE_PARAMS, DIM_INCOMPATIBLE, INCOMPATIBLE.
    - Example of incompatible parameters: an 8 kHz audio signal against a 16 kHz one.
    - Mel and MFCC spectrograms are not interchangeable even when their shapes match.
  - The `typecheck` decorator checks these types at runtime on each call. Checks can be disabled globally or per context, and semantic checks can be disabled on their own.
  - This is the closest prior art found for semantic typing across ML components. Its scope is model and training pipelines (speech, NLP), not inference-engine metadata.
- **fastai** (Howard & Gugger, 2020). A semantic type hierarchy for tensors, plus a type-dispatch system that transforms use. https://arxiv.org/abs/2002.04688
- **F# units of measure.** Units are checked at compile time and erased at runtime; mismatched units are compile errors. https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/units-of-measure
- **Fillmore, "The Case for Case"** (ERIC ED019631, April 1967). Proposes a finite set of case relationships between a verb and its noun phrases. They are primitive in the theory and hold across languages. https://eric.ed.gov/?id=ED019631
- **Perligata** (Damian Conway). A Perl dialect based on Latin, where a word's ending marks its role instead of its position. Source is a Wikiversity translation of Conway's paper. https://beta.wikiversity.org/wiki/Lingua::Romana::Perligata
- **Nadeshiko** (Japanese programming language).
  - Sakatoku, JSSST Computer Software 28(4), Oct 2011, https://www.jstage.jst.go.jp/article/jssst/28/4/28_4_4_23/_article/-char/en
  - IPSJ magazine article by クジラ飛行机 (Kujira Hikōki), 15 Mar 2021, https://note.com/ipsj/n/na3c57ceeffcf. The J-STAGE page lists Sakatoku's affiliation as kujirahand.com.
  - Particles attached to arguments carry their meaning. For example, "from" and "to" in a file-copy command. Swapping the argument order still gives the correct call.
  - [my reading] This is the closest precedent I found for the project's particle-role idea (`../ideas/josa_role_semantics.md`). It applies to general scripting, not to ML component interfaces.
- **Not found** [scoped to my searches]: any work that uses case-grammar or thematic-role marking for the interfaces between ML system components (model definition, engine, kernel, checkpoint).

---

## 6. Gap assessment

### 6.1 What is already known

1. **Interface faults are a known base rate.**
   - In large C systems, interface faults were 68.6% of all faults (Perry & Evangelist 1987) and 39% in Basili & Perricone (1984, as summarised by Perry).
   - They were about 49% of design and coding faults in Perry & Stieg (1993).
   - The pilot's 59–71% falls within this range. **The headline share alone is not new.**
2. **The loss mechanism and its classic remedy are known.**
   - Interface faults tend to come from incomplete or omitted information.
   - Formal interface specifications were the main suggested prevention (Perry & Stieg 1993). Inscape claimed it could potentially prevent about 73% of interface faults (Perry & Evangelist 1987).
   - The Mars Climate Orbiter loss is the canonical silent unit mismatch (MIB 1999).
3. **Type, shape, layout, configuration and API categories are consistently large in ML stacks.**
   - DL compilers: type 19.23%, shape/layout 13.27% (Shen 2021).
   - DL frameworks: type 14.2%, misconfiguration 13.4%, API misuse 11.8% (Chen 2023).
   - LLM engines: configuration 23% (Liu 2025/26).
   - Tile programs: type & operator 48.84% (Rathnasuriya 2026).
   - Triton: 12% of issues are layout-related (Zhou 2026).
4. **Wrong-output bugs are a large minority, hard to catch, and already studied for LLM serving.**
   - Across unconditional studies they are 12–36% of bugs (Islam, Liu, Chen, Shen, Yuan, tile programs), and 19.2% of high-priority torch.compile issues (Li 2026).
   - Test oracles are the stated bottleneck (Shen 2021; Chen 2023).
   - Silent errors in vLLM and SGLang have already been collected and diagnosed by comparison with a reference implementation (Ekka 2026).
5. **Production impact is documented.**
   - Anthropic 2025: a routing and configuration fact, a precision mismatch, and a misconfiguration. Users, not evaluations, surfaced them.
   - OpenAI 2024: kernels wrong on certain GPU configurations.
   - Open-weight models vary by provider: `reasoning_effort` ignored on Azure; a chat-template argument dropped for Kimi K2 in vLLM; Moonshot and OpenAI ship verifiers; about 39% of benchmark variability comes from the backend.
6. **Partial solutions exist for single fact families.**
   - Axes and shape: jaxtyping, einops, PEP 646, Pythia, PyTea, gradual shape types.
   - Partial vs replicated: DTensor placements; JAX `check_vma` and unreduced axes, checked at trace time and on by default.
   - Layout and format: torchao subclasses, MLIR encodings, Triton linear layouts.
   - Hardware capability matching: vLLM backend tables, validated at startup.
   - Semantic port types: NeMo.
   - Readiness: the PyTorch CUDA stream sanitizer and compute-sanitizer.

### 6.2 What appears new in the pilot (relative to the sources read)

1. **A cross-cutting class defined by mechanism.**
   - Every taxonomy read classifies either by the immediate code error (API misuse, type, shape, misconfiguration; Liu, Chen, Shen) or by where the bug lives (framework, model, kernel, numerics; Ekka).
   - None defines a class where each component computes correctly but a fact about a value's meaning is lost, mismatched, or read at the wrong time between them.
   - Existing schemes scatter this class across leaves. In Liu, it would span misconfiguration, incompatible model, mismatched precision, shape and type, and cache management [my reading].
2. **A typology of facts specific to LLM inference.** The facts are: layout and format, partial vs replicated, absolute vs relative position, valid range or window, readiness or staleness, and model property. How well each is covered today:

| Pilot fact kind | Nearest prior bug categories | Nearest existing mechanism | Declared and checked before execution? |
|---|---|---|---|
| Layout / format / scale | shape/layout 13.27% (Shen); memory layout conflicts 12.1% (Li); indexing/stride/layout 11.63% (tile); mismatched precision 17 of 1,041 (Liu) | torchao subclasses; MLIR encoding; Triton linear layouts | Partly (MLIR IR types; torchao mismatch checks not documented) |
| Partial vs replicated | distributed tensor errors (no count); cross-device output inconsistency 11.6% of inference issues (Ma) | DTensor Partial; JAX `check_vma` / unreduced | Yes in JAX (trace time, default on); at runtime in DTensor (alpha) |
| Absolute vs relative position | no dedicated category found | FlexAttention and FlashInfer receive raw indices; conversion is in user code | Not found |
| Valid range / window | no dedicated category found | vLLM sliding-window KV spec; FlexAttention mask_mod; FlashInfer logits mask | Declared for memory management; not checked against kernel masks (not found) |
| Readiness / staleness | graph caching 4.3% (Li); guard deficiencies 14.6% (Yuan); cache management 11, synchronization 3 (Liu); CUDA sync 12.4% (Wu) | PyTorch CUDA stream sanitizer; compute-sanitizer | At runtime and opt-in, for stream and memory ordering only |
| Model property (window, softcap, template arguments) | incompatible model 65 (Liu); model implementation 25.5% (Ekka); wrong parameter setting 20.8% (Tambon) | vLLM backend validation; NeMo parameterised types | vLLM validates hardware capabilities but has no window or softcap columns; NeMo checks at runtime, can be disabled |

   - [my reading] For absolute vs relative position, and for valid range or window as a checked fact, I found neither a dedicated bug category nor a typed representation.
3. **Numerical defects as a minority.**
   - The pilot's 3 of 70 pure numerical kernel defects agrees with small numerical shares elsewhere: 1.6% in Liu [computed] and 3 of 116 precision bugs in Li.
   - It conflicts with Ekka's 19.4% "numerical precision" share of silent errors. Because the definitions differ, re-labelling Ekka's issues with the pilot scheme would settle it.
4. **A conditional silence rate.**
   - The 92% (role-class wrong-output bugs that raised no error or warning) has no direct counterpart. The prior studies report unconditional symptom shares.
5. **An evaluable design claim applied to ML inference bugs.**
   - The claim: declare the facts mandatorily at each boundary (model definition ↔ engine ↔ kernel ↔ checkpoint) and check them before execution, then count how many real bugs would be rejected.
   - Classic SE predicted this remedy (formal interface specifications; Inscape). I found no evaluation of it on ML inference bugs.
6. **Points a reviewer can raise from these sources.**
   - Prior studies used two raters with reported agreement; Shen reports kappa rising from 85% to above 95%. The pilot used one rater per repository with no agreement measure.
   - Keyword sampling selects reports whose titles already name wrong output. Ekka used similar keywords, so the two datasets are comparable on this point.

### 6.3 Closest existing approach

- **In kind: JAX's sharding-in-types**, especially `shard_map`'s `check_vma` (varying vs invariant) and unreduced axes.
  - It puts a role fact into the type, checks it at trace time, is on by default, and raises an error instead of guessing when the answer is ambiguous.
  - It covers only one fact family (distribution and reduction state) and only in JAX. It is not used by the engines studied.
- **Across components: NeMo Neural Types.**
  - Semantic port types, including parameter incompatibilities such as sample rate.
  - Runtime only, can be disabled, coarse (module ports), and not found in inference engines.
- **Inside an inference engine: vLLM.** Its declared backend capabilities are validated at startup, and it keeps per-layer KV cache specs. The capabilities checked are hardware and format support. Model semantics such as sliding window and softcap are not declared columns.
- **For attention variants: FlexAttention, FlashInfer and AttentionEngine.** They make the facts declarative and fast, but nothing checks the declaration against the model's properties.
- **Net:** the sources read contain no mandatory, multi-fact role declaration checked at every component boundary of an LLM inference stack. The pieces exist separately:
  - type-level partial vs replicated in JAX
  - semantic port types in NeMo
  - capability validation in vLLM
  - declarative masks in FlexAttention and FlashInfer
