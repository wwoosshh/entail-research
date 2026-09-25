# Phase 0: 측정 하네스 (1주차)

목적: RTX 4070 Ti에서 두 워크로드를 세 실행 모드로 돌려 "컴파일러가 어디에서 시간을 잃는가"를 숫자로 확인하고, 연구 이점의 축(성능 / 안정성 / 호환성)을 데이터로 정한다.

## 파일

| 파일 | 역할 |
|---|---|
| `common.py` | 공통 측정: 스텝 시간(CUDA 동기화 후 중앙값·최소·최대), 첫 호출 시간(컴파일·그래프 캡처 포함), 피크 메모리, torch.profiler 트레이스에서 커널 수·GPU 점유 시간·유휴 비율·커널 분류·상위 커널 |
| `bench_sdxl_unet.py` | 워크로드 A. SDXL UNet 한 디노이징 스텝, fp16, latent 128×128(1024² 이미지), 배치 1·2. 가중치는 무작위 초기화(구조 측정에는 값이 무관) |
| `bench_llm_decode.py` | 워크로드 B. Qwen3-4B 구조의 LLM 디코드 한 스텝, bf16, 프롬프트 512, 배치 1·4·8. `--int4`로 torchao int4 weight-only 변형도 측정 |
| `make_report.py` | `results/*.json`을 읽어 `RESULTS.md` 표 생성 |
| `nsys_idle.py` | 임의 명령을 nsys로 감싸 실행한 뒤 GPU 유휴 비율과 상위 커널을 계산. **주의: 이 WSL2 환경에서는 nsys가 GPU 커널 트레이스를 수집하지 못해(API 트레이스만 기록) 결과가 비어 있다. 리눅스 네이티브나 Windows 네이티브에서만 유효.** 유휴 비율은 `WEEK1_NOTES.md`의 보정 방식(1 − GPU 점유/스텝 시간)으로 추정하고, 커널별 지표는 `ncu`를 쓴다 |
| `run_all.sh` | 의존성 설치 → A → B(int4 포함) → 보고서. 로그는 `logs/` |

## 실행 모드

| 모드 | 의미 |
|---|---|
| `eager` | 컴파일 없는 PyTorch. LLM은 DynamicCache |
| `compile_default` | `torch.compile` 기본 모드. Inductor가 Triton 커널을 생성하고 융합. LLM은 StaticCache + 디코드 함수 컴파일 |
| `compile_graphs` | `torch.compile(mode="reduce-overhead")`. 위에 CUDA Graphs를 얹어 런치 오버헤드 제거 |

## 지표 읽는 법

- **step ms**: 한 스텝의 벽시계 시간. 세 모드의 차이가 "컴파일러가 이 카드에서 실제로 얻는 것"이다.
- **first call ms**: 첫 호출 시간. 컴파일 모드에서는 컴파일 시간이고, 배치가 바뀔 때의 재컴파일 비용도 여기에 나타난다.
- **kernels/step**: 스텝당 GPU 커널 개수. 융합이 잘 되면 줄어든다.
- **GPU busy ms**: 커널·메모리 복사가 실제로 GPU를 점유한 시간(구간 합집합).
- **idle frac (UB)**: 스텝 구간 안에서 GPU가 놀고 있던 비율. torch.profiler가 CPU 쪽 오버헤드를 더하므로 상한(upper bound)이다. `nsys_idle.py`로 교차 검증한다.
- **커널 분류**: `triton`(Inductor 생성), `attention`(SDPA flash·efficient·cuDNN), `conv_cudnn`, `gemm_lib`(cuBLAS·CUTLASS), `elementwise_reduce`(ATen 요소별·정규화·리덕션), `memcpy_memset`, `other`. 벤더 라이브러리 커널과 컴파일러 생성 커널의 시간 비중이 여기서 갈린다.

## 판정 기준 (5~6주차에 사용)

- 유휴 비율이 크고 `compile_graphs`가 `compile_default`보다 뚜렷히 빠르다 → 런치·스케줄링 병목. 스텝 단위 융합·상주 커널 컴파일러 방향.
- 시간의 대부분이 `gemm_lib`·`attention`에 있고 모드 간 차이가 작다 → 벤더 라이브러리가 지배. 컴파일러로 성능을 더 얻기 어려운 영역이므로 양자화 융합·비표준 어텐션 변형 등 라이브러리가 덜 튠된 곳을 찾는다.
- 컴파일 모드가 실패하거나(status에 error), 배치·shape 변화에 재컴파일이 잦거나, eager와 수치가 다르다 → 안정성·호환성 방향.

## 주의

- GeForce는 FP16·BF16 연산에 FP32 누산을 쓰면 텐서코어 처리량이 절반으로 제한된다. cuBLAS 기본값이 FP32 누산이므로 이 카드의 현실적 상한은 약 80 TFLOPS다.
- 무작위 가중치는 시간·커널 구조 측정에는 영향이 없지만, 수치 비교(2주차)에는 실제 가중치가 필요하다.
- 첫 실행은 Hugging Face Hub에서 모델 설정 파일(수 KB)만 내려받는다. 실패하면 내장된 대체 설정을 쓴다.

## 실행

```bash
wsl
bash ~/ai_compiler/phase0/run_all.sh
```

개별 실행 예:

```bash
source ~/venvs/gpu/bin/activate
python ~/ai_compiler/phase0/bench_sdxl_unet.py --modes eager compile_default --batches 1
python ~/ai_compiler/phase0/bench_llm_decode.py --modes eager --batches 1 4
python ~/ai_compiler/phase0/nsys_idle.py -- python ~/ai_compiler/phase0/bench_llm_decode.py --modes eager --batches 1
python ~/ai_compiler/phase0/make_report.py
```
