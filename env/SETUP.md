# GPU 컴파일러 연구 환경 (RTX 4070 Ti, WSL2 Ubuntu 24.04)

작성일: 2026-09-22

## 1. 장비와 선택 사항

| 항목 | 값 / 선택 | 이유 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti, 12 GB, 연산 능력 8.9 (Ada) | 보유 장비 |
| Windows 드라이버 | 616.56, CUDA 13.4까지 지원 | 이미 설치되어 있음. WSL은 이 드라이버를 그대로 사용 |
| 호스트 | Windows 11 Home 26200, RAM 31.8 GB, 논리 CPU 28개 | |
| WSL | 2.7.12, 커널 6.18, 기본 배포판 Ubuntu-24.04 | 리눅스 툴체인이 컴파일러 개발에 훨씬 순탄함 |
| WSL 자원 상한 | `%USERPROFILE%\.wslconfig`: memory=24GB, swap=8GB | LLVM·Triton 소스 빌드 대비 |
| CUDA 툴킷 | cuda-toolkit-13-0 (apt, wsl-ubuntu 저장소) | PyTorch 2.14.0 안정판이 cu130 빌드라 확장 모듈 빌드 호환을 위해 메이저·마이너를 맞춤 |
| PyTorch | 2.14.0+cu130 (download.pytorch.org/whl/cu130) | 조사 시점 최신 안정판 |
| Triton | PyTorch가 고정한 버전이 함께 설치됨 | Triton은 자체 ptxas를 내장하므로 툴킷 버전과 무관 |
| Nsight Compute / Systems | 툴킷에 포함 | 커널 관측 도구 |
| 리눅스 사용자 | <user>, 비밀번호 없는 sudo | 로컬 개발 VM 편의. 되돌리려면 `/etc/sudoers.d/90-<user>` 삭제 후 `passwd` |

## 2. 실행 순서

스크립트는 이 폴더에 있고, WSL에서는 `<workspace>/env/` 로 보인다.

1. Windows PowerShell: `wsl --install -d Ubuntu-24.04 --no-launch` (완료)
2. root로 기본 패키지·사용자·CUDA·Nsight 설치 (완료 시 아래 3절에 결과 기록)
   ```bash
   wsl -d Ubuntu-24.04 -u root -- bash /root/01_wsl_base_cuda.sh <user> cuda-toolkit-13-0
   ```
3. 사용자로 Python 스택 설치
   ```bash
   wsl -d Ubuntu-24.04 -u <user> -- bash -lc "bash <workspace>/env/02_python_stack.sh 2.14.0"
   ```
4. 검증
   ```bash
   wsl -d Ubuntu-24.04 -u <user> -- bash -lc "python <workspace>/env/03_check_env.py"
   ```

## 3. 검증 결과 (2026-09-22, `03_check_env.py`)

| 항목 | 결과 |
|---|---|
| Python / PyTorch / Triton | 3.12.3 / 2.14.0+cu130 (CUDA 13.0, cuDNN 9.24) / 3.8.0 |
| 장치 인식 | NVIDIA GeForce RTX 4070 Ti, cc 8.9, 12.0 GiB, 60 SMs |
| Triton JIT 커널 (vector add) | eager 대비 오차 0.0 |
| cuBLAS FP16 GEMM, FP32 누산 | N=2048: 65.6 TFLOPS, N=4096: 68.2, N=8192: 73.9 (GeForce 절반 속도 상한 약 80의 85~92%) |
| SDPA 백엔드, [1,16,4096,64] fp16 | flash 1.024 ms, efficient 1.278 ms, cuDNN 1.176 ms, math 25.8 ms (모두 사용 가능) |
| torch.compile | 첫 호출 1.05 s, eager와 일치 |
| 장치 내 복사 대역폭 | 376 GB/s (스펙 504 GB/s의 약 75%, 복사 커널의 통상 수준) |
| CUDA 명령줄 도구 | nvcc 13.0.88, ptxas, cuobjdump, nvdisasm, compute-sanitizer, ncu(Nsight Compute 2025.3.1), nsys(Nsight Systems 2025.3.2) 모두 `/usr/local/cuda/bin` |
| 배포판 재시작 후 | 기본 사용자 <user>, systemd 동작, 로그인 셸에서 CUDA_HOME과 PATH 적용 확인 |
| Nsight Systems 추적 (`04_ncu_probe.py`) | 동작. `cuda_gpu_kern_sum` 보고서 생성 |
| Nsight Compute 카운터 (`04_ncu_probe.py`) | 처음엔 ERR_NVGPUCTRPERM. NVIDIA 제어판에서 성능 카운터 접근 허용 후 `wsl --shutdown`을 거치자 **동작**. cuBLAS가 고른 커널 `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn`: 2.07 ms, SM 처리량 47.2%, DRAM 처리량 14.9% (FP32 누산 절반 속도와 부합) |

선택 패키지: cuda-python, cuda-core, nvidia-cutlass-dsl 설치됨. flash-attn은 PyPI에 빌드된 휠이 없어 건너뜀(SDPA flash 백엔드가 FA2 기준선).

주의: `bash -lc`처럼 비대화형 셸에서는 `~/.bashrc`가 초반에 종료되어 가상환경이 자동 활성화되지 않는다. 스크립트에서는 `source ~/venvs/gpu/bin/activate`를 명시한다. 대화형 `wsl` 세션에서는 자동 활성화된다.

## 4. 일상 사용법

- 터미널: `wsl` 또는 `wsl -d Ubuntu-24.04`. 로그인하면 가상환경 `~/venvs/gpu`가 자동 활성화된다.
- VS Code: WSL 확장으로 `Ubuntu-24.04`에 접속해 작업.
- 코드 위치: 성능을 위해 소스와 빌드 산출물은 리눅스 파일시스템(`~/work`)에 두고, 문서와 결과만 `/mnt/c/...`의 프로젝트 폴더에 둔다. `/mnt/c` 경유 I/O는 느리다.
- CUDA 도구: `nvcc`, `ptxas`, `cuobjdump`(생성된 SASS 확인), `nvdisasm`, `compute-sanitizer`, `ncu`(Nsight Compute CLI), `nsys`.

## 5. 주의 사항

- **Nsight Compute 권한.** GeForce에서 `ncu`가 성능 카운터에 접근하려면 Windows의 NVIDIA 제어판 → 개발자 → "GPU 성능 카운터에 대한 액세스를 모든 사용자에게 허용"을 켜야 한다. 이 설정은 GUI에서 사용자가 직접 바꿔야 한다.
- **FP32 누산 절반 속도.** GeForce는 FP16·FP8 텐서 연산에서 FP32 누산을 쓰면 처리량이 절반으로 제한된다. cuBLAS 기본값이 FP32 누산이므로 성능 비교 시 누산 정밀도를 반드시 명시한다. 4070 Ti의 현실적 상한은 FP16 텐서 연산 약 80 TFLOPS(FP32 누산), 메모리 대역폭 약 504 GB/s다.
- **Ada에 없는 기능.** TMA, wgmma, tcgen05, Tensor Memory, 스레드 블록 클러스터, FP4 텐서 연산은 없다. Hopper·데이터센터 Blackwell 전용 코드 경로는 이 카드에서 재현할 수 없고 클라우드 대여가 필요하다.
- **드라이버는 WSL 안에 설치하지 않는다.** Windows 드라이버가 `/usr/lib/wsl/lib`로 노출된다. apt의 `cuda` 메타패키지는 드라이버를 끌어오므로 쓰지 않고 `cuda-toolkit-13-0`만 설치한다.
- **최신 툴킷 병행 설치.** cuTile Python(13.1+), Tile IR, CompileIQ(13.3+)를 쓰려면 `sudo apt-get install cuda-toolkit-13-4`로 `/usr/local/cuda-13.4`를 옆에 두고 `CUDA_HOME`만 바꿔 쓴다. PyTorch 확장 빌드는 13.0을 유지한다.
- **flash-attn.** PyPI에는 소스만 있어 빌드가 오래 걸린다. Ada에서는 PyTorch SDPA의 flash 백엔드가 FlashAttention-2 구현이므로 기본 기준선으로 충분하다. 별도 패키지가 필요하면 GitHub 릴리스에서 torch 2.14·cu13·cp312에 맞는 휠을 골라 설치한다.
- **git 줄바꿈.** 이 폴더의 `.sh`는 LF다. Windows 에디터로 고칠 때 CRLF로 바뀌지 않게 주의한다.
- **Windows에서 `wsl -- bash -lc "..."`로 명령을 넘길 때.** wsl.exe가 문자열을 바깥 셸에서 한 번 더 해석하므로 `$VAR`, `$(cmd)`가 로그인 셸에 도달하기 전에 먼저 확장된다. 변수를 쓰는 작업은 스크립트 파일로 저장해 파일을 실행한다.
- **Nsight Compute 프로브.** `04_ncu_probe.py`는 4096×4096 FP16 행렬곱 8회를 돈다. 권한을 켠 뒤 다음으로 확인한다.
  ```bash
  ncu --target-processes all --launch-skip 5 --launch-count 1 --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed python env/04_ncu_probe.py
  ```

## 6. 추가 기록 (2026-09-22, Phase 0 진행 중 발견)

- **torchao 0.18.0 int4.** `Int4WeightOnlyConfig`의 기본 패킹 포맷(plain, preshuffled)은 외부 커널 라이브러리 `mslk >= 1.0.0`을 요구해 ImportError가 난다. `int4_packing_format="tile_packed_to_4d"`는 PyTorch 내장 tinygemm(`_weight_int4pack_mm`)을 쓰며 이 카드(sm_89)에서 동작한다. `plain_int32`는 CUDA 미지원, `version=1`은 AssertionError. 확인 스크립트: `phase0/probe_torchao_int4.py`.
- **Phase 0 의존성.** 가상환경에 diffusers, transformers, accelerate, safetensors, huggingface_hub, torchao 추가 설치됨(`phase0/run_all.sh`).
- **WSL 경로.** `~/ai_compiler`가 프로젝트 폴더(`<workspace>`)로의 심볼릭 링크다. Windows에서 `wsl -- ...`로 명령을 넘길 때 한글 경로 대신 이 링크를 쓴다.
- **Nsight Systems의 WSL2 제약.** `nsys profile -t cuda`는 CUDA API 호출(cudaLaunchKernel 등)만 기록하고 GPU 커널 실행 트레이스는 수집하지 못한다(`nsys stats --report cuda_gpu_kern_sum`이 "does not contain CUDA kernel data"). 반면 torch.profiler의 커널 타이밍과 Nsight Compute(`ncu`)는 정상 동작한다. GPU 타임라인이 필요하면 리눅스 네이티브 부팅 또는 Windows 네이티브 nsys를 쓴다.
- **WSL `/tmp`는 배포판 재시작 시 비워진다.** 명령 사이에 배포판이 유휴 종료되면 `/tmp`의 산출물이 사라지므로 결과는 `~/` 또는 프로젝트 폴더에 저장한다.

## 7. 서빙 엔진 환경과 평가 도구 (2026-09-23)

기본 `~/venvs/gpu`(torch 2.14)와 섞지 않는다. vLLM 0.30.0과 SGLang 0.5.20이 둘 다 torch 2.13.0을 고정하기 때문이다.

| 환경 | 구성 스크립트 | 내용 | 동작 시험 |
|---|---|---|---|
| `~/venvs/vllm` | `05_serving_engines.sh vllm` | vLLM 0.30.0, torch 2.13.0+cu130, flashinfer 0.6.18.post1, transformers 5.17 | Qwen3-4B 생성 성공 (`08_engine_smoke.py vllm`) |
| `~/venvs/sglang` | `05_serving_engines.sh sglang` | SGLang 0.5.20, torch 2.13.0+cu130, flashinfer 0.6.18, transformers 5.12.1 | Qwen3-4B 생성 성공 (`08_engine_smoke.py sglang`) |
| `~/venvs/gpu` | `07_eval_tools.sh` | lm-eval 0.4.13 추가. torch·transformers는 그대로 | dry-run으로 핵심 패키지 변화 없음을 확인한 뒤 설치 |

- **vLLM 메모리 한계(이 카드).** 12GB 중 약 1.2GB를 시스템이 쓴다. 그래서 `gpu_memory_utilization`은 0.88 이하여야 한다. 0.92는 거부됐다(여유 10.78GiB, 요구 11.03GiB). 또 Qwen3-4B를 bf16으로 올리면 `max_model_len=4096`에서 KV 캐시가 부족하다(여유 0.46GiB, 필요 0.56GiB). 2048로 줄이면 된다.
- **모델.** Gemma 2 2B와 9B를 `~/models/gemma-2-{2b,9b}-it`에 받았다(`06_fetch_gemma2.sh`). 공식 저장소는 로그인이 필요해 Unsloth 사본을 썼다. 9B 가중치 4개 파일의 SHA256은 공식과 같다. 2B는 한 파일로 재포장되어 해시 대조가 불가능하다. 설정의 softcap 값은 기술 보고서와 같다.
- **평가 자료.** GSM8K 시험 세트 1,319문제를 `issue_track/gemma2_softcap/data/`에 받았다(출처와 SHA256은 `SOURCE.txt`).

## 8. 스윕 2회차용 모델 (2026-09-23)

연구자가 여섯 후보 모두의 라이선스에 동의하고 내려받기를 승인했다. 라이선스는 Gemma, Llama 3.2, Qwen 연구용, MIT, Apache다. 내려받기에는 `12_fetch_sweep_models.sh`를 썼다.

- 리비전은 고정했다. 받은 파일은 가중치, 설정, 토크나이저뿐이다. 원격 코드(`.py`)는 받지 않았다. 출처와 리비전은 `~/models/sweep_sources.json`에 적혀 있다.
- 크기는 다음과 같다.

  | 모델 | 크기 |
  |---|---|
  | gemma-3-1b-it (unsloth 미러) | 1.9G |
  | Llama-3.2-3B-Instruct (unsloth 미러) | 6.1G |
  | Qwen2.5-3B-Instruct | 5.8G |
  | Qwen2.5-3B-Instruct-AWQ | 2.6G |
  | Qwen3-4B-FP8 | 4.9G |
  | Phi-3.5-mini-instruct | 7.2G |

- **무결성 검사(`13_verify_downloads.py`, 결과는 `verify_downloads.json`):** 가중치 파일 10개 모두에서 로컬 sha256이 고정 리비전의 허브 기록(LFS oid)과 같았다.
- **미러와 공식본 대조:** 공식 Gemma와 Llama 저장소는 게이트가 걸려 있다. 로그인하지 않으면 API가 oid를 `****`로 가린다. 이 프로젝트는 로그인하지 않으므로 해시로는 대조할 수 없었고, **바이트 크기만 비교했다. 세 파일 모두 공식본과 크기가 같았다.** 파일이 동일하다는 확인은 아니다.
