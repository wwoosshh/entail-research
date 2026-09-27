### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version                : 18.1.3 (1ubuntu1)
CMake version                : version 3.28.3
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.13.0+cu130
Is debug build               : False
CUDA used to build PyTorch   : 13.0
ROCM used to build PyTorch   : N/A
XPU used to build PyTorch    : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0] (64-bit runtime)
Python platform              : Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
    
==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 13.0.88
CUDA_MODULE_LOADING set to   : 
GPU models and configuration : GPU 0: NVIDIA GeForce RTX 4070 Ti
Nvidia driver version        : Could not collect
cuDNN version                : Could not collect
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : False

==============================
          CPU Info
==============================
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  28
On-line CPU(s) list:                     0-27
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Core(TM) i7-14700K
CPU family:                              6
Model:                                   183
Thread(s) per core:                      2
Core(s) per socket:                      14
Socket(s):                               1
Stepping:                                1
BogoMIPS:                                6835.20
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves avx_vnni vnmi umip waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize ibt flush_l1d arch_capabilities
Virtualization:                          VT-x
Hypervisor vendor:                       Microsoft
Virtualization type:                     full
L1d cache:                               672 KiB (14 instances)
L1i cache:                               448 KiB (14 instances)
L2 cache:                                28 MiB (14 instances)
L3 cache:                                33 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-27
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Mitigation; Clear Register File
Vulnerability Retbleed:                  Mitigation; Enhanced IBRS
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected

==============================
Versions of relevant libraries
==============================
[pip3] flashinfer-python==0.6.18.post1
[pip3] nccl4py==0.5.0
[pip3] numpy==2.3.5
[pip3] nvidia-cublas==13.1.1.3
[pip3] nvidia-cuda-cccl==13.3.4.3.1
[pip3] nvidia-cuda-crt==13.4.92
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvcc==13.4.92
[pip3] nvidia-cuda-nvdisasm==13.4.92
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cudnn-cu13==9.20.0.48
[pip3] nvidia-cudnn-frontend==1.29.0
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-cufile==1.15.1.6
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.1
[pip3] nvidia-cutlass-dsl==4.7.1
[pip3] nvidia-cutlass-dsl-libs-base==4.7.1
[pip3] nvidia-cutlass-dsl-libs-core==4.7.1
[pip3] nvidia-cutlass-dsl-libs-cu12==4.7.1
[pip3] nvidia-cutlass-dsl-libs-cu13==4.7.1
[pip3] nvidia-ml-py==13.610.43
[pip3] nvidia-nccl-cu13==2.29.7
[pip3] nvidia-nvjitlink==13.4.92
[pip3] nvidia-nvshmem-cu13==3.4.5
[pip3] nvidia-nvtx==13.0.85
[pip3] nvidia-nvvm==13.4.92
[pip3] pyzmq==27.2.0
[pip3] tokenspeed-triton==3.8.10.post20260920
[pip3] torch==2.13.0
[pip3] torch_c_dlpack_ext==0.1.5
[pip3] torchaudio==2.11.0
[pip3] torchcodec==0.16.0
[pip3] torchvision==0.28.0
[pip3] transformers==5.17.0
[pip3] triton==3.7.1
[conda] Could not collect

==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
vLLM Version                 : 0.30.0
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled; XPU: Disabled
GPU Topology:
  Could not collect

==============================
     Environment Variables
==============================
LD_LIBRARY_PATH=/usr/local/cuda/lib64
CUDA_HOME=/usr/local/cuda
CUDA_HOME=/usr/local/cuda
PYTORCH_NVML_BASED_CUDA_CHECK=1
TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_<user>
```

</details>

### 🐛 Describe the bug

#### Summary

With Transformers v5, `rope_scaling` is a property that replaces `config.rope_parameters` wholesale. When a RoPE
scaling dict is passed at launch through `--hf-overrides`, the resulting `rope_parameters` no longer contains
`rope_theta`. `get_rope` then falls back to `rope_parameters.get("rope_theta", 10000)`. Model files that call
`set_default_rope_theta` (e.g. `qwen2.py`, `qwen3.py`, default 1e6) are unaffected when their checkpoint happens to
use that value; files that pass `config.rope_parameters` straight through (e.g. `llama.py`, `qwen3_moe.py`,
`gpt_oss.py`) run with base 10000. There is no warning.

Passing the same dict in `config.json` works: the config conversion keeps `rope_theta`.

#### Reproduction (vLLM 0.30.0, transformers 5.17.0, one RTX 4070 Ti)

Llama-3.2-3B-Instruct, restating the checkpoint's **own** `rope_scaling` at launch (so the override should be a no-op):

```python
import json
from vllm import LLM
scaling = json.load(open("Llama-3.2-3B-Instruct/config.json"))["rope_scaling"]
llm = LLM("Llama-3.2-3B-Instruct", hf_overrides={"rope_scaling": scaling})
print(llm.llm_engine.model_config.hf_text_config.rope_parameters)   # no 'rope_theta'
```

| run | rope_parameters in the engine | mean NLL (3 documents) | GSM8K (first 500, greedy) |
|---|---|---|---|
| untouched | rope_theta 500000 | 1.128 / 3.685 / 1.253 | 379 |
| same `rope_scaling` via `hf_overrides` | **no rope_theta** | 3.660 / 5.935 / 3.950 | **279** |
| `config.json` with `rope_theta: 10000` | rope_theta 10000 | 3.660 / 5.935 / 3.950 | 279 |

The override run is identical to an explicit base of 10000. Outputs stay fluent; the answers are wrong. Nothing is
logged. A later rerun of the same scripts gave 273 for the override run (379 untouched), so the effect is not a
one-off.

#### How widespread

I checked the 300 most-downloaded text-generation models on Hugging Face at config level, building each config
through vLLM's own `ModelConfig` with the override applied (no weights): the override applies to 180 of them
(registered in vLLM 0.30, RoPE in the model file, config builds). **64 of the 180 silently change their RoPE base**
(36%; 22% of those models' downloads), 8 fail loudly at config build, 108 are unaffected because their model file
carries a per-file default equal to the checkpoint's base. Among the 64: `openai/gpt-oss-20b` and `gpt-oss-120b`
(150000 → 10000 when their own `rope_scaling` is restated), `Qwen/Qwen3-30B-A3B` (1e6 → 1e4), `Qwen/Qwen3-4B-Instruct-2507`
(5e6 → 1e6, the YaRN route its model card documents), `zai-org/GLM-4.7-Flash`, `mistralai/Mistral-7B-Instruct-v0.2`,
`HuggingFaceTB/SmolLM2-135M`, `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (1e7 → 1e4). The full table, the script and
the raw results: https://github.com/wwoosshh/entail-research/blob/main/testbed/results/m10/E1_SUMMARY.md
(`testbed/m10_e1_rope.py`, `testbed/results/m10/e1_llm/rope_on.json`).

#### Why it matters

Qwen3 model cards document this launch-time route for YaRN (`--rope-scaling`, which no longer exists in 0.30, so
users move to `--hf-overrides`). For `Qwen3MoeForCausalLM` (e.g. Qwen3-30B-A3B) the same code path yields base
10000 instead of 1e6 (reproduced at config level with vLLM's own `ModelConfig` and `get_rope`). Per-model defaults
(`set_default_rope_theta(config, 1e6)`) do not cover checkpoints of the same architecture with a different base
(Qwen3-2507: 5e6; Qwen2.5-1M: 1e7).

#### Suggested fix

Apply the override the way the `config.json` conversion does: keep `rope_theta` (and `partial_rotary_factor`) from
the existing `rope_parameters` when the override dict does not carry them, then standardise, instead of relying on
per-model defaults. A warning when `rope_parameters` ends up without `rope_theta` would already have made this
visible. Related: #56066 (BailingMoeV2), #37435 (draft configs dropping overrides).

#### Environment

vLLM 0.30.0, transformers 5.17.0, torch 2.13.0+cu130, Python 3.12, RTX 4070 Ti (12 GB), WSL2 Ubuntu 24.04 (the
full `collect_env` output is in the environment section above).
Scripts, protocol and raw results for the end-to-end runs:
https://github.com/wwoosshh/entail-research/tree/main/issue_track/rope_override

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.
