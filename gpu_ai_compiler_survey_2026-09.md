# GPU AI 컴파일러의 문제점과 시장 동향 조사

> **2026-09-23 사실 검증.** 이 조사의 사실 146개를 1차 출처로 다시 확인했다(`reinvestigation/audit/survey_audit.json`, 도중 저장본). 확인됨 111, 일부 다름 27, 틀림 4, 확인 못 함 1, 사실 주장 아님 3이다. 틀린 4개와 정정은 `THEORY.md` 4.4절에 있다. 결론(0절, 5절) 점검은 하지 못했다.

- 작성일: 2026-09-22
- 목적: 현재 시장에서 실제로 쓰이는 GPU(및 AI 가속기) AI 컴파일러의 문제점을 "근본적(구조적) 문제"와 "현재 시장 동향"으로 나누어 정리한 연구 배경 조사
- 방법: 7개 주제(NVIDIA CUDA 툴체인 / PyTorch·Triton / MLIR·크로스벤더 생태계 / AMD·Intel·기타 가속기 / 근본 문제·LLM 커널 생성 / 배포용 컴파일러 TensorRT·ONNX / 시장·산업 동향)로 나눠 병렬 웹 조사. 공식 문서·릴리스 노트·GitHub 이슈·논문·벤더 블로그 등 1차 출처를 우선했고, 2차 출처나 미검증 항목은 본문에 표시했다. 참고한 출처는 약 300개이며 7장에 주제별로 정리했다.

---

## 0. 핵심 요약

한 줄 결론: **GPU AI 컴파일러의 가장 큰 문제는 "하드웨어가 세대마다 프로그래밍 모델을 바꾸는 속도"를 "이식성 있는 추상화"가 따라가지 못한다는 구조적 딜레마이며, 2025~2026년 시장은 이 딜레마를 컴파일러로 풀기보다 (1) 수작업 커널 라이브러리 + 얇은 JIT/CUDA Graph, (2) 하드웨어를 노출하는 저수준 파이썬 DSL, (3) LLM 에이전트 기반 커널 탐색이라는 세 갈래 우회로로 대응하고 있다.**

핵심 발견 10가지:

1. **그래프 컴파일러의 후퇴.** NVIDIA 스스로 TensorRT-LLM 1.0(2025-09)에서 PyTorch 백엔드를 기본으로 만들고, 1.2(2026-03)에서 TensorRT 엔진 백엔드를 삭제했다. vLLM·SGLang은 수작업 커널 + CUDA Graph + 부분(piecewise) torch.compile 조합으로 시장을 장악했다. 정적 그래프 컴파일은 엣지(TensorRT Edge-LLM, TensorRT for RTX)와 비-LLM 비전 모델로 후퇴했다.
2. **추상화 딜레마가 수치로 드러났다.** PyTorch 팀은 Triton 기반 FlexAttention이 Hopper에서 FlashAttention-3의 약 60%, Blackwell에서는 "작은 격차가 심연(chasm)이 됐다"고 인정했고, CuTe DSL 기반 FA4 백엔드가 Triton 대비 1.6~3.2배 빠르다. 독립 평가에서 Triton BF16 GEMM은 B200(N=8192)에서 cuBLAS의 62%였다. Meta의 TLX 논문은 "숨기면 컴파일러가 새 하드웨어를 못 따라가고, 노출하면 부담이 프로그래머에게 돌아온다"고 요약한다.
3. **컴파일러는 하드웨어를 1~2년 늦게 따라간다.** H100 출하(2022 가을) 후 Triton의 자동 warp specialization은 2025-01(약 2.3년), FlashAttention-3는 2024-07(약 21개월)에 나왔다. Blackwell도 CUTLASS는 즉시, Triton 기본 지원 0~3개월, PyTorch 프로토타입 3개월, 명시적 TMEM/2-CTA 제어는 9~20개월 걸렸다. Rubin(sm_107)에서 같은 주기가 반복되고 있다.
4. **이식성은 성능과 양립하지 않는다.** "수식은 이식되지만 나머지는 전부 다르다(The math is portable. Everything else is different)." HipKittens 논문은 AMD에 TMA·wgmma·mbarrier가 없고 레지스터가 wave 간 정적으로 분할되어 NVIDIA식 warp specialization이 MI355X에서 피크의 약 80%에 막힌다고 밝혔다. XLA의 Triton 경로는 warp 크기 상수(32 vs 64) 하나가 잘못되어 MI300X에서 10배 느렸다.
5. **동적 shape와 비정형 워크로드(MoE, ragged)는 여전히 컴파일러의 약점이다.** 재컴파일, 패딩, CUDA Graph 버킷팅이 보편적 우회책이고, vLLM은 Triton MoE 커널용 튜닝 JSON을 200개 이상 배포한다. 2026-09 프로덕션 사고에서는 DeepGEMM이 서빙 중 새 shape마다 3~3.6초씩 nvcc JIT를 돌려 3.4초 엔진 정지를 일으켰다.
6. **정확성·결정성·검증이 구조적으로 취약하다.** torch.compile 버그 실증 연구(2026-04)에서 고우선순위 이슈의 19.2%가 "컴파일된 모델의 잘못된 출력"이었고, eager와 값이 다른 이슈만 244건(59건 검증 열림)이다. LLM 추론의 비결정성은 배치 크기 가변성이 주원인이며, 결정성 커널은 34~110% 느리다. 형식 검증 시도(Gimlet, 2026-07)에서는 수치 테스트를 통과한 26개 커널 중 2개가 실제로는 동치가 아니었다.
7. **AMD의 소프트웨어 격차는 줄고 있지만 "6개월 이상" 뒤다.** SemiAnalysis는 2024-12 "CUDA moat 미돌파", 2025-05 "환경변수 남발", 2026-02 "오픈소스 분산 추론에서 6개월 이상 뒤" 순으로 기록했다. 2025~2026년 OpenAI·Meta·Anthropic·Microsoft가 AMD와 GW급 계약을 맺으면서 워런트·지분 조건이 ROCm 개선에 연동되어, 이제 고객이 ROCm을 고치는 인센티브를 보유한다.
8. **LLM 커널 생성은 기대와 검증된 현실의 격차가 크다.** 2025-02 KernelBench에서 프론티어 모델의 PyTorch 대비 승률은 20% 미만이었고, 2026-06 KernelBench-Verified는 현실적 기준선과 숨김 테스트를 쓰면 GPT-5.5의 기하평균이 0.88배로 떨어진다고 보고했다. 반면 Meta KernelEvolve는 광고 모델 처리량 60% 개선, AlphaEvolve는 Gemini 핵심 커널 23% 개선을 실제 운영에서 보고했다. 공통 병목은 보상 해킹을 막는 정확성 오라클이다.
9. **컴파일러 계층은 독립 사업이 아니라 인수 대상이 됐다.** OctoAI(TVM)→NVIDIA(2024), CentML(Hidet)→NVIDIA(2025), Brium→AMD(2025), Modular(Mojo/MAX)→Qualcomm(2026-07 완료), Groq 자산→NVIDIA 약 200억 달러 라이선스(2025-12). MLIR는 2024~2025년 거버넌스를 재편했지만 2026-09 현재 업스트림에 NVIDIA/AMD GPU용 end-to-end 경로가 없다.
10. **한국은 양쪽에 동시에 투자한다.** 2025-10 NVIDIA GPU 26만 장 이상 도입 발표와 함께, 국가성장펀드가 리벨리온에 직접 출자(2026-03, 기업가치 23.4억 달러)하고 국산 NPU를 공공조달 혁신제품으로 지정(2026-08)했다. 국산 NPU 업체는 CUDA식 독자 스택 대신 vLLM·PyTorch·Triton "no forks" 전략을 택했다.

---

## 1. 조사 범위: "시장에서 사용되는 GPU AI 컴파일러"의 계층

| 계층 | 대표 스택 | 역할 |
|---|---|---|
| 벤더 백엔드 컴파일러 | NVIDIA nvcc/ptxas(PTX→SASS), AMD hipcc/LLVM AMDGPU, Intel DPC++/IGC | 모든 상위 스택이 결국 통과하는 최종 코드 생성 단계. 폐쇄성(ptxas)과 세대별 ISA 변동의 근원 |
| 커널 DSL·컴파일러 | OpenAI Triton(+Gluon, Meta TLX), NVIDIA CuTe DSL·cuTile/CUDA Tile IR·Tilus, Meta Helion, TileLang, ThunderKittens/HipKittens, JAX Pallas(Mosaic GPU), AWS NKI | 개별 GPU 커널을 파이썬/C++ 타일 추상화로 작성·자동튜닝 |
| 그래프·프레임워크 컴파일러 | PyTorch 2.x torch.compile(Dynamo/Inductor/AOTInductor), OpenXLA(XLA/StableHLO/PJRT), Apache TVM(Relax/TensorIR), IREE, TensorRT(정적 엔진), ONNX Runtime, OpenVINO | 모델 그래프 전체를 받아 융합·메모리 계획·커널 선택 |
| 서빙 시점 JIT·런타임 융합 | DeepGEMM(DeepJIT), FlashInfer JIT, cuDNN Graph 런타임 융합, TensorRT for RTX(AOT+JIT), CUDA Graphs | 추론 서빙 중 shape·양자화 포맷에 맞춰 커널을 생성·특화 |
| 배포·엣지 | TensorRT Edge-LLM, Core ML/MLX, Qualcomm QNN, LiteRT, WebGPU/WebNN | 엣지 GPU/NPU 배포 |

---

## 2. 근본적 문제 (릴리스로 해결되지 않는 구조적 문제 10가지)

각 항목은 (문제) → (증거) → (왜 근본적인가) → (현재 대응) 순으로 정리했다.

### 2.1 추상화 딜레마: 숨기면 하드웨어를 못 따라가고, 노출하면 이식성을 잃는다

- **증거**
  - PyTorch FlexAttention 팀(2026-03-04): Triton 구현은 Hopper에서 처음 FA3의 약 80%였으나 현재 약 60%이고, Blackwell에서는 "What was once a small gap has grown to a chasm!" 깊은 파이프라이닝과 warp specialization은 "Triton 기반 구현으로는 표현할 수 없다." CuTe DSL 기반 FA4 백엔드가 GB200에서 forward 1.6~3.2배, backward 1.85~2.3배 빠르다.
  - CUDA Tile 독립 평가(arXiv 2604.23466, 2026-06): Triton BF16 GEMM은 H100(N=4096)에서 cuBLAS의 98%지만 N=8192에서 76%, B200(N=8192)에서 62%(cuTile은 52%).
  - Meta autoWS 로드맵(2026-01): B200 flash attention forward에서 자동 warp specialization으로 기본 Triton의 1.5~2배를 얻었지만 "cuDNN이 여전히 10~20% 앞선다." "최적 warp specialized 코드 생성은 조합 폭발 문제."
  - Gluon 튜토리얼(Triton 공식): 컴파일러가 "수작업 저수준 코드에 질 수 있고, 그럴 때 사용자가 할 수 있는 일은 거의 없다."
  - Meta TLX 논문(arXiv 2605.10905, 2026-05): Triton의 SIMB 추상화에서는 "컴파일러가 모든 것을 사용자 대신 발견하고 구현해야 한다"; "실행 구조를 너무 숨기면 컴파일러가 새 하드웨어 메커니즘을 따라잡아야 하고, 너무 노출하면 조율 부담이 프로그래머에게 돌아온다."
- **왜 근본적인가**: Hopper 이후 성능은 스레드 병렬성이 아니라 데이터 이동(TMA)·텐서코어(wgmma/tcgen05)·동기화(mbarrier)의 비동기 조율에서 나온다. 이 전역 파이프라인 안무를 컴파일러가 자동 발견하는 것은 조합 최적화 문제이며, 하드웨어가 세대마다 조율 방식을 바꾼다.
- **현재 대응**: 탈출구(escape hatch) 다층화. Triton 위에 Helion(자동튜닝 상위 DSL), 안에 autoWS, 아래에 Gluon·TLX(명시적 layout/TMA/tcgen05). NVIDIA는 CuTe DSL(파이썬, C++ 대비 컴파일 20~30배 빠름)과 cuTile을 병행. 단, Lei Zhang(2026-02)의 지적처럼 "Gluon 커널은 더 이상 Triton처럼 이식 가능하지 않다."

### 2.2 하드웨어 세대 교체 속도와 컴파일러 지연

- **증거(날짜 기준)**
  - Hopper: H100 2022년 가을 출하 → Triton Hopper 지원 초기 병합 2023-08(GMMA/TMA/자동 warp specialization은 "실험적, 기본 비활성") → FA3(수작업, wgmma/TMA/pingpong) 2024-07(FA2는 H100 이론 FLOPS의 35%만 활용) → Triton 3.2 자동 warp specialization 2025-01(약 2.3년).
  - Blackwell: 2024-03 발표, CUDA 12.8(2025-01-31) 첫 지원 → CUTLASS 3.8 tcgen05/TMEM/NVFP4(2025-01~02) → Triton Blackwell 병합 2025-01-28(NVIDIA·OpenAI 공동; MXFP4 레이아웃은 "사용자 주의 필요") → PyTorch 2.7 프로토타입(2025-04) → Gluon tcgen05/TMEM 명시 제어(Triton 3.4, 2025-07) → 2-CTA·NVFP4×NVFP4 성숙(Triton 3.8, 2026-08) → FA4(CuTe DSL, 2026-03). Inductor의 Blackwell용 DeepSeek 스케일 GEMM 템플릿 PR은 2026-09-01에도 열려 있음.
  - ISA 자체가 바뀐다: Hopper `wgmma.mma_async`는 Blackwell에서 폐기되고 Tensor Memory(TMEM)를 쓰는 `tcgen05.mma`로 대체. "FA3는 Blackwell에서 동작하지 않았다. WGMMA는 SM100에 존재하지 않는다." `sm_100a` 바이너리는 전후방 호환 불가, `compute_90a` PTX는 Blackwell 미지원. 소비자용 Blackwell(SM12x, DGX Spark 포함)은 TMEM·tcgen05가 없어 SM100 전용 커널(FlashMLA, DeepGEMM)이 실행 불가.
  - Modular의 Blackwell matmul 시리즈(2025-09): TMA+tcgen05 첫 커널은 cuBLAS의 8.7%, 스위즐링 후 16.4%에서 시작.
- **왜 근본적인가**: 텐서코어 프로그래밍 모델이 mma.sync → wgmma → tcgen05로 세대마다 바뀌고, 세대 내에서도 제품군(데이터센터 vs 소비자)으로 분화한다. 컴파일러가 하드웨어와 함께 설계되지 않는 한 지연은 반복된다(Rubin sm_107이 2026-08~09 Triton 3.8·CUDA 13.4·CUTLASS 4.8에 "초기 지원"으로 들어오며 같은 사이클 시작).
- **현재 대응**: NVIDIA의 `sm_100f` 패밀리 타깃(12.9), CUDA Tile IR(타일 수준 가상 ISA로 전방 호환 약속), 벤더가 DSL 컴파일러 개발에 직접 참여(Triton Blackwell PR).

### 2.3 이식성–성능 트레이드오프

- **증거**
  - Patrick Toulme "Portability Is a Myth"(2026-05): TPU(VMEM, VLIW, 256×256 MXU, 명시적 DMA), Blackwell(TMEM, tcgen05, mbarrier, warp specialization), Trainium(SBUF/PSUM)은 다른 알고리즘을 요구. MoE grouped matmul이 TPU Pallas 282줄 vs Blackwell 생성 CUDA 약 400만 줄, 공유 코드 0. "The math is portable. Everything else is different."
  - HipKittens(arXiv 2511.08083, 2025-11): 타일 추상화는 CDNA3/4로 이식되지만 알고리즘은 아니다. AMD는 TMA·wgmma·mbarrier가 없고, 레지스터가 wave 간 정적 분할되어 NVIDIA식 wave specialization은 MI355X BF16 GEMM에서 피크의 약 80%에 캡; 대신 8-wave ping-pong/4-wave interleave 필요. MFMA 레지스터 레이아웃은 비합성적(non-compositional), XCD(칩렛) 인지 스케줄링으로 +19%. hipcc는 AGPR 처리 오류. 결과: Triton 대비 1.3~3.0배, 전 기준선 대비 1.2~2.4배.
  - XLA 이슈 #23574(2025-03): ThreadsPerWarp가 32로 하드코딩되어 MI300X에서 Triton 커널이 PyTorch 경로 대비 약 10배 느림.
  - Pallas Blackwell matmul 튜토리얼: 150줄 미만으로 cuBLAS의 109.6%를 달성하지만 TMA·TMEM·tcgen05·2-CTA MMA·cluster launch control을 명시적으로 사용(Mosaic GPU는 Hopper/Blackwell 전용). Google의 Tokamax 커널 라이브러리는 op마다 GPU/TPU 별도 구현을 배포.
  - 벤더 내에서도: cuTile attention은 B200에서 FA2의 2.5배지만 RTX PRO 6000에서 FA2의 53%; A100용으로 튠된 Triton 커널은 H100에서 저성능(Lattner); ThunderKittens v2.0(2026-01)은 Blackwell 지원과 함께 Ampere 지원을 제거.
  - Romeo et al.(2026-06): 동일 OpenMP 코드가 MI250X에서 A100 대비 앱 수준 약 3배, 커널 수준 최대 10배 느리고 레지스터 스필 최대 47배. Abraham & Okloki(2026-03): NVIDIA/AMD/Intel/Apple ISA 전반에서 하드웨어 불변 프리미티브 10개, 매개변수화 가능 6개, 근본적 분기 6개; 벤더 중립 모델은 6개 벤치마크 중 5개에서 네이티브 수준이지만 병렬 리덕션은 62.5%.
- **왜 근본적인가**: 메모리 공간(TMEM/VMEM/SBUF, SMEM vs LDS), 행렬 유닛 형태(tcgen05 vs MFMA vs MXU), 비동기 엔진 유무, 레지스터 할당 모델, 칩렛 토폴로지가 다르면 최적 스케줄 자체가 달라진다. 코드 생성이 아니라 알고리즘 선택의 문제다.
- **현재 대응**: "타일" 추상화로의 수렴(Triton, cuTile, TileLang, Helion, NKI, TK)과 동시에 하드웨어별 탈출구 증가. 기업은 이식성을 포기하고 백엔드별 커널을 유지(Tokamax, AITER의 Triton/CK/HIP/어셈블리 다중 백엔드).

### 2.4 정적 특화 vs 동적 shape·비정형 워크로드

- **증거**
  - torch.compile: 기본은 정적 shape로 컴파일 후 재컴파일 시 동적 표시(automatic dynamic). Edward Yang(Meta, 2025-08): "병적인 컴파일 시간의 대부분은 반복 재컴파일(주로 동적 shape)에서 나온다." PyTorch 2.14(2026-09)의 `@dynamic_spec`은 재컴파일 대신 데이터 의존 오류를 내는 방식으로 트레이드오프를 옮겼을 뿐이다.
  - PyTorch/XLA: 동적 shape 미지원 시 MLP가 100 iteration에 102회 컴파일; 지원은 "bounded dynamic shape만, 실험 단계".
  - vLLM: 배치 크기만 동적인 그래프 하나 + piecewise CUDA Graph(어텐션은 CUDA Graph 호환이 "non-trivial"); SGLang은 CUDA Graph를 작은 배치(160 또는 256 미만)에서만 기본 활성. TensorRT는 min/opt/max 최적화 프로파일을 벗어나면 서빙 중 엔진 재생성. Qualcomm QNN EP는 "동적 shape 모델 미지원". MLX `mx.compile`은 shape·dtype·입력 수가 바뀌면 재컴파일.
  - MoE: MegaBlocks(2022) "토큰 드롭 또는 패딩 낭비" 강제; Mitra(2026-04)에서 Triton은 "BLOCK_M을 고정해야 하고(오토튠 불가) 2D 누산기에 스칼라 인덱싱이 안 되며", 라우팅 편향이 커지면 Megablocks 대비 1.03배→0.70배로 하락. IBM/vLLM: "Triton은 전역 배리어를 제공하지 않아" 리덕션에 2차 런치 필요, "Triton 커널은 항상 특정 문제 하나에 맞춰 써야 한다."
  - 서빙 JIT 사고: vLLM 이슈 #56684(2026-09-13, DeepSeek-V4.1-Flash, 8×B200): DeepGEMM이 "처음 보는 구성마다 새 sm100_fp8_gemm_1d1d 커널을 JIT 컴파일"해 nvcc 3~3.6초씩, 엔진 3.4초 정지; 해결책은 1024 토큰 버킷 패딩과 워밍업 프리컴파일.
  - 수작업 튜닝의 잔존: vLLM은 Triton MoE 커널용 튜닝 JSON을 200개 이상(E, N, device, dtype별) 배포.
- **왜 근본적인가**: 컴파일러의 속도는 정적 특화에서 나온다. 동적 shape·데이터 의존 라우팅은 특화를 무효화하며, 패딩·버킷팅·재컴파일은 모두 그 특화를 일부 포기하는 것이다.
- **현재 대응**: piecewise 컴파일, CUDA Graph 버킷, `torch.switch`(MoE 분기, 2.14), grouped GEMM 라이브러리(cuBLASLt), 정적 배칭 커널(H800에서 텐서코어 피크 91~95%), 하이브리드(정적 부분 CUDA Graph + 동적 부분 JIT; arXiv 2604.23467, TTFT 최대 66% 감소).

### 2.5 커널 융합의 한계와 메가커널

- **증거**
  - FlexAttention 블로그(2024-08): 융합 어텐션은 "유연성 상실"을 대가로 하며 변형이 "하이퍼큐브"를 이뤄 "지원이 드문드문"; MosaicML은 "커널 지원 부재"로 ALiBi를 포기. "어텐션 변형이 기존 최적화 커널에 맞지 않으면 느린 실행과 CUDA OOM이 운명이다."
  - FA3(2024-07): FA2는 H100 이론 FLOPS의 35%; FA3는 수작업 warp specialization/WGMMA/TMA/pingpong으로 740 TFLOPS(75%).
  - Hazy Research 메가커널(2025-05): 배치 1에서 vLLM/SGLang은 H100 대역폭의 최대 50%만 사용; forward pass가 약 100개 커널; 런치당 2.1µs(CUDA Graph로도 1.3µs); 메가커널은 대역폭 78%, vLLM 대비 2.5배. Mirage Persistent Kernel(2025-12): "기존 컴파일러는 소규모 지역 연산자 그룹만 융합할 수 있다. 복잡한 텐서 프로그램을 충실히 구현하는 단일 커널 생성은 계산적으로 어렵고 종종 불가능하다"; vLLM/SGLang 대비 최대 1.7배. Ada-MK(2026-05): 런치 오버헤드가 end-to-end 추론 시간의 14.6%.
  - "Correct but Slow"(2026-07): DSL 커널들은 "코드 생성 제약과 불완전한 오토튠 커버리지"로 잔여 격차 유지.
- **왜 근본적인가**: 커널 경계를 넘는 융합은 전역 스케줄링(SM 단위 태스크 그래프)을 요구하며, 어텐션 변형은 조합적으로 늘어난다.
- **현재 대응**: 메가커널/persistent kernel(MPK, Hazy), FlexAttention 같은 템플릿 기반 부분 일반화(정적 템플릿의 부분집합만 지원), Inductor combo-kernel 수평 융합(2.10).

### 2.6 오토튜닝 탐색 비용

- **증거**: Ansor는 BERT 8개 shape 최적화에 V100에서 19.3시간(FTuner, 2024); IBM은 Triton FA2 극한 튜닝에 GPU 종류당 약 24시간(2025-10); Helion은 커널당 약 10분(1,520개 config 586초); Meta 프로덕션 모델 컴파일 1,825초 중 오토튠 벤치마크 238초(13%), Inductor 전체 1,238초(67.8%); PyTorch 자체 문구 "max-autotune은 가장 빠른 모델을 만들지만 컴파일이 매우 오래 걸린다"; DenseNet121에서 max-autotune이 eager보다 1.9배 느린 사례(#161764, SM 수 부족). 학습 기반 비용 모델은 "한 하드웨어에서 학습하면 다른 하드웨어에서 성능이 나쁘다(cross-hardware unavailability)"(TLP).
- **왜 근본적인가**: 탐색 공간은 조합적이고 비용 모델은 하드웨어 종속적이다. 캐싱·가지치기·오프라인화는 비용을 상환할 뿐 없애지 못한다.
- **현재 대응**: AMD Origami(PyTorch 2.13/2.14, 해석적 GEMM 구성 선택), Helion의 AOT 오토튠+config 고정, NVIDIA CompileIQ(CUDA 13.3, 진화 탐색으로 최적화된 Triton/CUTLASS 커널에 최대 15%), LLM 유도 오토튜닝(Helion 1.4).

### 2.7 수치 정확성·결정성·검증의 부재

- **증거**
  - torch.compile 침묵 오류: 실증 연구(Li et al., arXiv 2604.08720, 2026-04) "고우선순위 이슈의 19.2%가 torch.compile 버그로 인한 잘못된 출력(2위, 1위는 크래시 19.57%)"; 이슈 크롤러(2026-09-18) eager와 값·dtype·메모리 포맷이 다른 이슈 244건(59건 검증 열림); "[PT2] Validation lost" 우산 이슈: "동작하던 코드에 torch.compile을 추가하면 버그를 잡아줄 오류가 사라진다"; Transformer 99.9% 원소 불일치(#162722). 설계상 "컴파일러는 eager와 비트 단위 동치를 보장하지 않는다"(fp16/bf16 융합 시 다운/업캐스트 생략). PyTorch 2.12(2026-05)는 `addcdiv`를 FMA로 낮춰 비트 동치를 회복했는데, 이유는 반올림 차이가 "수천 스텝에 걸쳐 누적"되기 때문.
  - 비결정성: Thinking Machines(2025-09): "거의 모든 LLM 추론 엔드포인트가 비결정적인 주된 이유는 부하(배치 크기)가 비결정적으로 변하기 때문"; Qwen3-235B temperature 0 샘플 1,000개에서 80가지 완성; 배치 불변 커널은 26초→55초. SGLang 결정성 모드 평균 34.35% 감속. 2026-09 논문: "같은 커널의 결정적 구현들도 비트 단위로 다를 수 있고", Triton에서 균형 트리 리덕션 강제 시 최대 20% 비용, "오토튜너는 비트 동치 구성을 자동 식별하지 못한다."
  - 저정밀 포맷: Triton Blackwell(2025-02) "MXFP4 레이아웃·패킹은 사용자 주의 필요"; KernelBenchX(2026-05) "양자화는 완전히 미해결(30회 중 0 성공)"; "Spec Sheets Are Not Kernels"(2026-08) B300 INT8은 "명목상 존재하지만 기본적으로 배포 불가".
  - 검증 부재: Microsoft Volta(2025-11) "LLM 커널 생성은 형식적 보장이 없다", GPU 커널용 최초 sound 동치 검사기 주장; Gimlet(2026-07) KernelBench L1 Triton 커널 26개 중 16개 동치 증명, **2개는 수치 테스트를 통과했지만 비동치로 증명**, 8개 미확정 — "테스트는 샘플링이다"; "Correctness Illusion"(2026-06) 고정 shape allclose가 버그 커널 9/9를 통과시킴.
- **왜 근본적인가**: 부동소수점 비결합성 + 데이터 병렬 리덕션 때문에 비트 재현성은 성능과 교환되고, 형식 동치 검사는 제한된 커널 클래스에서만 가능하다.
- **현재 대응**: PyTorch 2.10 DebugMode(텐서 해시), 결정성 모드 존중; 배치 불변 커널; 커널 동치 검사기 연구(Volta, Gimlet).

### 2.8 분산(멀티 GPU) 컴파일

- **증거**: PyTorch Async-TP(2024-09) "NCCL send/recv 커널이 SM을 사용해 matmul 오버랩용 SM을 줄인다" → SymmetricMemory+copy engine, Llama3-70B forward 최대 약 20%, 노드 내 한정. Edward Yang(2025-08): "분산 collective와 DTensor는 컴파일되지만 기본적으로 최적화되지 않는다"; functional collective는 autograd 미지원(2.11에서 해결); torch.compile은 SPMD를 가정하지 않음. 2.14(2026-09)에서 Inductor 통신·계산 오버랩 `simple_overlap`이 기본 활성. DeepEP는 문서에 없는 PTX `ld.global.nc.L1::no_allocate.L2::256B`와 "SM을 점유하지 않는 hook 기반 오버랩"을 수작업으로 사용; TokenWeave "NVLink 연결에서도 통신 오버헤드 20%". Triton-distributed(2025-04)가 "분산 오버랩을 네이티브 지원하는 최초의 컴파일러" 주장. SemiAnalysis(2026-07): "단일 노드 최적화는 한계에 도달", 분산·분리형(disaggregated) 추론이 결정적.
- **왜 근본적인가**: 오버랩은 SM vs copy engine 자원, 단방향 메모리, 클러스터 스케줄링을 모델링해야 하는데 단일 디바이스 데이터플로 컴파일러는 이를 표현하지 않는다.

### 2.9 폐쇄 ISA·라이선스·벤더 인센티브

- **증거**
  - PTX는 가상 ISA(무한 가상 레지스터, 스케줄링·제어 비트 없음)이고, 폐쇄된 ptxas가 문서화되지 않은 SASS로 낮추며 Volta 이후 명령당 21비트 제어 필드(stall count, barrier)를 기록하는데 하드웨어는 이를 **정확성**에 의존한다. "공식 SASS 어셈블러는 없다"(2026-07). ptxas는 지역 최적 스케줄링(CuAsmRL이 RL로 SASS 스케줄 탐색해 개선). 커널 심볼에 "cutlass"가 포함되면 다르게 최적화(CUTLASS #3389, 2026-07: 동일 sm_100a PTX가 이름에 따라 80 reg/8,076B 스필 vs 168 reg/56B, +31% 시간).
  - inline PTX 필요 잔존: Triton은 LLVM NVPTX가 sm_103 tcgen05 intrinsic을 선택하지 못해 inline asm으로 회귀(2025-09); DeepSeek-V3 "맞춤 PTX 명령으로 통신 청크 자동튠"; DeepGEMM은 NVCC 12.2/12.3 차이를 보고 컴파일된 SASS의 yield/reuse 비트를 직접 패치(FFMA 인터리빙, 소형 M에서 최대 2.7배; NVCC 12.9가 자동 수행하게 되어 폐기).
  - EULA: CUDA EULA §1.2(8)(v13.4, 2026-01) "SDK 출력물을 비-NVIDIA 플랫폼을 타깃하도록 번역할 목적으로 역공학·디컴파일·디스어셈블할 수 없다." ZLUDA는 AMD 법무팀 요청으로 AMD 자금 코드 철회(2024-08), 자금 상실로 취미 프로젝트 복귀(v6). SCALE은 "PyTorch 같은 대형 프레임워크를 포팅할 만큼 CUDA API를 지원하지 못함"(2025-10).
  - DSL 난립과 컴파일러 소유권: NVIDIA는 CUDA C++·CuTe DSL·cuTile(Python/C++/Rust/Julia)·Tilus·Warp를 동시에 운영. Tile IR은 Apache 2.0으로 공개했지만 "외부 기여 불허". cuTile은 GTC 2025에서 "드라이버에 MLIR 기반 JIT 컴파일러 포함"으로 발표(Blackwell 우선, Hopper는 2026-05 마지막 지원). CUDA 원팀 Nicholas Wilt: "cuTile이 Triton에 대항하기 위해 개발됐다고 의심하지 않기 어렵다."
  - 흡수: TVM 상용화 주체 OctoAI(2024-09), Hidet 개발사 CentML(2025-06, 저장소 2026-05 아카이브)이 NVIDIA에 인수, Hidet 저자는 NVIDIA Tilus를 이끈다. Taichi는 2024-07 이후 릴리스 정지.
  - 반론(The Software Frontier, 2026-07): "폐쇄성은 주로 사업 결정이 아니라 2012년경 면적·전력 이유로 내린 아키텍처 결정의 결과다."
- **왜 근본적인가**: NVIDIA의 인센티브는 추상화 계층을 소유하는 것이고, 폐쇄 SASS는 아키텍처(면적·전력)와 얽혀 있다. 법적·경제적 제약은 릴리스로 풀리지 않는다.

### 2.10 디버깅·관측성

- **증거**: PyTorch 프로파일러 문서 "Triton 커널 이벤트는 최소 정보만"; Inductor를 우회하는 커널은 "수동 주석 없으면 트레이스에 나타나지 않을 수 있음"; 2022년 이슈 "PyTorch 프로파일러는 성능 디버깅에 별로 유용하지 않다" 잔존. Lattner: Triton은 Nsight Compute와 맞지 않아 "컴파일러가 무엇을 했는지 추측하게 된다." Meta autoWS: warp specialization은 "수치·성능 디버깅을 더 어렵게 한다." "Correct but Slow": 정확하지만 PyTorch 기준선보다 300배 느린 TileLang 커널을 어떤 정확성 검사도 잡지 못함. TensorRT는 폐쇄 코어라 Polygraphy 이분 탐색이 유일한 경로. Meta는 동적 차원 표시가 "수많은 실험이 필요한 매우 복잡한" 작업이라고 기록.
- **분류**: 도구 투자로 개선 가능(엔지니어링 백로그)하지만, 융합 커널은 소스 op와 1:1 대응이 없다는 점은 추상화에 내재.

---

## 3. 스택별 현재 문제점

### 3.1 NVIDIA CUDA 툴체인 (nvcc / ptxas / CUTLASS / cuTile)

| 문제 | 증거 | 분류 |
|---|---|---|
| 세대 간·제품군 간 비호환 | `sm_100a` 전후방 비호환; SM12x(소비자 Blackwell)에 TMEM 없음; CUDA 13.0에서 Maxwell/Pascal/Volta 오프라인 컴파일 제거 | 근본(제품 분할) |
| ptxas 오컴파일 재발 | ptxas 12.9.86이 `add.s16x2` 즉시값 누락(H100, 2026-09); VIMNMX3 융합 오류로 GB200 FlexAttention 마스크 행 수 오류(2026-07, 13.4.46에서 수정); nvcc 13.1이 `createpolicy` 누락→sm_90 illegal instruction(Marlin, 2026-05); CUDA 13.3 ptxas가 CUB warp-shuffle 리덕션 오컴파일(2026-08); CUDA 13.2 노트 자체가 "런타임 오컴파일" 수정 인정 | 일시적이나 매 릴리스 재발 |
| 버전 매트릭스 | "ptxas: Unsupported .version 8.4"로 AlphaFold 3 사용자 중단(2024-12); Triton 3.8/PyTorch 2.14에 번들할 ptxas-blackwell 버전(13.3.33 vs 13.4.46) 협상(2026-07); RTX 50 안정 지원 2025-10에도 요청 중 | 구조적(드라이버 JIT vs 툴킷 분리) |
| C++ 템플릿 컴파일 시간 | CUTLASS int8 GEMM 2개 17분 22초(2023); nvcc 비결정적 행(1/20 빌드); PyTorch 팀 "FlashAttention 설치의 고통" | 일시적(CuTe DSL 20~30배 개선) |
| cuTile 성숙도 | 13.1(2025-12) Blackwell 전용·저정밀 제한; GEMM은 수동 타일로 cuBLAS 90%+; flash attention 918 TFLOPS는 수동 타일·latency 힌트·fast-math·K-loop 분할 후; 독립 평가 cuBLAS 52~79% | 일시적(9개월 내 Ampere/Ada→Hopper→Rubin 확대) |
| CuTe DSL 베타 | FAQ "베타 동안 이식성 약속 없음"; CuTeDSL ≥4.6이 ptxas가 거부하는 PTX 생성(2026-08); FlashInfer Blackwell 커널의 TMEM fence 누락으로 출력 손상(2026-09) | 일시적 |
| NVIDIA 2025~2026 대응 | CuTe DSL(CUTLASS 4.0, 2025-05→4.8 Rubin 2026-09), cuTile/Tile IR(13.1), Tile C++(`nvcc --enable-tile`, 13.3), CompileIQ 오토튠, Triton→Tile IR 백엔드(2026-01, Blackwell 전용), CUDA Python 1.0 semver(2026-08), CUDA Rust(2026-09) | — |

### 3.2 PyTorch 2.x (torch.compile: Dynamo / Inductor / AOTInductor)

| 문제 | 증거 | 분류 |
|---|---|---|
| 콜드 컴파일 시간 | 공식 문서 "수초~수분, 초대형은 수십 분~수 시간"; 측정 196초(캐시로 56초, regional로 80초); SGLang 시작 1분30초→약 6분(배치 크기별 컴파일); 분산 학습에서 N개 rank가 각자 컴파일(2.14 compile-on-one-rank로 대응) | 대부분 일시적 |
| 재컴파일·guard | 동적 shape·DTensor 조합 미해결(#159635); 전문가 병렬에서 "새 토큰 수마다 재컴파일 가능" | 근본(특화가 설계) |
| graph break·제어 흐름 | 13.8%의 HF 모델(195개 중)에 graph break→Python eager 폴백(GraphMend); `torch.cond`/`while_loop`는 입력 변경 불가, 함수형 재작성 강제 | 근본(파이썬 트레이싱) |
| 캐시 불안정 | Mega Cache 순환 import 실패(#154463); vLLM "비공개 API 의존으로 이상한 캐싱 이슈"; "캐시 히트 보장 없음" | 일시적 |
| 침묵 수치 오류 | 2.7절 참고(19.2%, 244건) | 근본 |
| 분산 | collective 기본 비최적화, SPMD 비가정(2.11~2.14에서 부분 해결) | 혼합 |
| 프로덕션 채택률 | 정량 조사 없음. SGLang 문서는 `--enable-torch-compile`을 "유지보수 중단, 오류 가능"으로 표기; 2.14 릴리스도 "torch.compile 없이도 빠른 기본 경험"을 목표로 명시(compile-by-default 아님) | — |
| 2026 변화 | 2개월 릴리스 주기; 2.10 DebugMode·결정성; 2.11 미분 가능 collective·FA4 백엔드; 2.12 `addcdiv` 비트 동치; 2.13 CuTeDSL 네이티브 백엔드·서브프로세스 컴파일; 2.14 `@dynamic_spec`·오버랩 기본·Helion 등록(단, ROCm 미지원, 라우팅 op 0) ; `aot_compile()`은 "실험적" | — |

### 3.3 OpenAI Triton

| 문제 | 증거 | 분류 |
|---|---|---|
| Hopper/Blackwell 성능 상한 | 2.1절(FA3의 60%, B200 GEMM 62%, cuDNN 10~20% 우위); Lattner "H100에서 약 20% 손실" | 근본(SIMB 추상화) |
| 저수준 제어 부재→Gluon/TLX | Gluon(2025-07~)은 "ttg IR의 파이썬 프론트엔드", 비이식적; TLX(Meta) 약 200줄로 B200 GEMM, 프로덕션 배포 | 근본 |
| 오토튠·컴파일 비용 | FA2 튠 GPU당 24시간; FP4 에뮬레이션 커널 110초 컴파일·40,772줄 PTX·8,466 스필(#10918) | 근본(완화 가능) |
| 비정형 워크로드 | 전역 배리어 없음; 2D 누산기 스칼라 인덱싱 불가; BLOCK_M 고정; Tri Dao SonicMoE는 CuTeDSL grouped GEMM이 Triton 예제보다 21% 높은 TFLOPS | 대부분 근본 |
| 백엔드 결합·회귀 | 코어 헤더가 NVIDIA/AMD 트리를 포함해 out-of-tree 백엔드가 625MB 빌드+662MB 툴체인 다운로드(#11883, 2026-09); Intel XPU에서 Triton 3.7.2→3.8.0 간 2.6배 회귀(#7782) | 일시적 |
| AMD/Intel 이식성 | 긍정: IBM은 동일 소스로 H100/MI300, decode paged attention FA3의 98.6~105.9%. 부정: HipKittens가 Triton 대비 1.3~3.0배; AMD 튜닝 가이드는 AMD 전용 knob(`waves_per_eu`, `matrix_instr_nonkdim`) 요구, rocBLAS가 빠르면 Triton 미사용; Red Hat 크로스벤더 벤치에서 Helion이 소비자 GPU A에서 27 TFLOPS(타사 150+) 이상치 | 일시적(하드웨어는 허용) |
| 거버넌스 | 코어 메인테이너 9명(2026-09-18 기준)에 거부권; 리드 Phil Tillet(OpenAI); 재단 없음 | 근본(조직) |
| 디버깅 | Nsight 비호환; warp specialization으로 수치·성능 디버깅 난이도 상승 | 혼합 |

### 3.4 TensorRT / TensorRT-LLM / ONNX Runtime (배포용)

| 문제 | 증거 | 분류 |
|---|---|---|
| 엔진 비이식성 | "특정 GPU·설정용으로 생성되며 이식 불가"; TensorRT 버전 간 역직렬화 실패(NVIDIA 자체 Maxine SDK 사례); 엔진 빌드 384초(캐시 없음) vs 9초(엔진 캐시) | AOT 오토튠에 근본 |
| 동적 shape 프로파일 | 범위 벗어나면 서빙 중 엔진 재생성; TensorRT-RTX 1.2 "CUDA Graph 캡처 데이터 레이스로 동적 shape 성능 저하 가능" | 근본 |
| API 파괴·플러그인 | TensorRT 11.0(2026-05) `IPluginV2` 전부 제거, 번들 플러그인 제거, 암시적 INT8 캘리브레이션(`IInt8Calibrator`) 제거→Q/DQ 명시 양자화 강제; 11.3(2026-09)에도 DeBERTa FP16 16~20% 회귀, SDXL/FLUX 빌드 실패 잔존; DLA는 10.7이 마지막 | 일시적(2~3년 주기 반복) |
| TRT-LLM의 방향 전환 | 0.17(2025-01) PyTorch 백엔드 도입 → 1.0(2025-09) 기본 → 1.2(2026-03) TensorRT 백엔드 삭제("PyTorch is now the sole execution backend"). 이유: "모듈화·수정 용이", "새 아키텍처마다 수작업 재구현"의 한계; 최고 성능 커널을 FlashInfer로 방출. AutoDeploy(torch.export 기반 패턴 매칭)는 베타, "컴파일러 주도 워크플로로의 전환" 표방 | 시장 신호 |
| TRT-LLM 잔여 불만 | torch.compile 활성 시 시작 53초→115초(piecewise 한정으로 93초, 2026-09 PR); 개발 이미지 63GB; CUDA 13.1·PyTorch 핀 충돌 | 일시적 |
| ONNX opset 지연·익스포터 | ONNX 1.21(opset 26, 2026-03)→ORT 1.28(2026-07) 약 4개월; PyTorch 2.9(2025-10) dynamo 익스포터 기본화 후 배치 차원을 1로 하드코딩하는 버그(#170172, TensorRT 배포 중단), LSTM 디코더 실패(#168969) | 일시적·반복 |
| 실행 프로바이더 재편 | ROCm EP 제거→MIGraphX(1.23, 2025-09); DirectML "sustained engineering"→Windows ML(2026-08 "신규 연산자·최적화 계획 없음"); WebGL 폐기→WebGPU; ORT GenAI는 여전히 0.x(0.16.0, 2026-09) | 일시적 이행 |
| LLM에서의 ONNX 위상 | 데이터센터 LLM 스택(TRT-LLM, vLLM, SGLang)은 HF 체크포인트를 직접 로드, ONNX 미경유; ONNX는 엣지/클라이언트 AOT 입력 포맷으로 잔존 | 구조적 |

### 3.5 XLA / MLIR / TVM / IREE

| 스택 | 문제 | 증거 |
|---|---|---|
| MLIR | 업스트림에 end-to-end GPU 경로 없음; 방언 분열; 거버넌스 재편 | 2024-11 RFC "Project Charter and Restructuring"(Linalg/TOSA/TCP 중복, LLVM 같은 기본 파이프라인 부재); 2025-01 헌장·영역 팀, Tensor Compiler Design Group(AMD·Intel·NVIDIA·Google·Arm·Qualcomm 11명); Lighthouse(2025-06)는 "공식 텐서 컴파일러가 아닌" 참조 프로젝트, 2026-09 현재 CPU·Intel GPU CI만, NVIDIA/AMD 경로 없음. Lattner Part 8: "범용 인프라와 AI 솔루션 사이의 정체성 위기", "CUDA 같은 참조 스택 부재" |
| OpenXLA/XLA | 정적 shape 중심; GPU 성능 주장은 Ampere 기준; AMD 경로 미성숙 | PyTorch/XLA bounded dynamic shape 실험적; XLA:GPU 문서는 Ampere near-roofline만 언급; MI300X warp-size 10배 버그; Lattner Part 6: "XLA는 한 브랜드를 공유하는 두 프로젝트(Google TPU-first vs OpenXLA)"; StableHLO 고정 연산자 집합 |
| TVM | 역할 축소(end-to-end 컴파일러→인프라·글루) | ASF 보고서상 월 약 100 커밋, 84 committer로 "건강"; TVM-FFI(2025-10, 0.4µs 호출 오버헤드, FlashInfer 채택); OctoAI→NVIDIA, Tianqi Chen은 NVIDIA Distinguished Engineer 겸직; TileLang이 TVM 위에 구축; Lattner "GenAI 이전 워크로드용 설계, 벤더 포크로 분열" |
| IREE | 사용자 집중(AMD) | LF AI 샌드박스, 3.9k stars; AMD MLPerf v5.0 SDXL은 SHARK/IREE, Llama 2 70B는 vLLM |
| Pallas/Mosaic GPU | Hopper/Blackwell 전용; Triton 경로 폐기 | Blackwell 튜토리얼 cuBLAS 109.6%는 Blackwell 전용 기능 명시 사용; Tokamax는 백엔드별 커널 |
| Modular(Mojo/MAX) | 폐쇄 컴파일러 비판→2026-08 오픈소스; 독립 벤치마크 부재 | 2025-09 $250M(기업가치 $1.6B); 2026-06 Qualcomm 인수 발표, 07-29 완료(Lattner는 Qualcomm EVP; 금액은 공식 발표 미공개, 약 39억 달러 보도); Mojo 1.0(2026-08-11) 시점 컴파일러 폐쇄→08-18 Apache 2.0, 26.6(2026-09-17)부터 외부 기여 수용; "Python superset" 약속은 "Python-like"로 후퇴; AMD 대비 vLLM 성능 우위 주장은 벤더 자료만 |
| 신흥 DSL | 지속 가능성 위험 | TileLang(7.5k stars, DeepSeek MLA 예제, Ascend·MetaX·Moore Threads 파트너), ThunderKittens(Together·Jump·Cursor 프로덕션), Mirage/MPK(NVIDIA 전용), Tilus(NVIDIA, Hidet 후속), tinygrad(6인, 자체 NV/AM 드라이버로 CUDA·ROCm 우회), Luminal($5.3M). Taichi 정지, Hidet 아카이브 |

### 3.6 AMD ROCm / HIP

| 문제 | 증거 | 분류 |
|---|---|---|
| 기본 설정 성능·QA | SemiAnalysis(2024-12): MI300X BF16 GEMM 약 620 vs H100 약 720 TFLOP/s(스펙은 더 높음); `F.Linear`는 rocBLAS·`torch.matmul`은 hipBLASLt로 갈림; FA backward 수개월간 20 TFLOP/s 미만; FP8 학습 segfault; `PYTORCH_TUNABLE_OPS` 25GB 누수·1~2시간 재튠; H100 75% 도달에 AMD 수석 엔지니어의 60개 명령 Dockerfile·미병합 브랜치 필요; hipBLASLt 휴리스틱 "대부분 shape에서 잘못된 알고리즘 선택"; RCCL 팀 GPU 32장 미만 vs NVIDIA EOS 11,000장 | 일시적(대부분 수정) + 구조적(hipify 포크 의존) |
| 추론 스택 지연 | 2025-05: SGLang ROCm CI 커버리지 NVIDIA의 10% 미만, 모델 25% 정확도 실패, FP8 DeepSeek V3 전 엔진 고장, 환경변수 남발; 2026-02: MI355X가 포크 vLLM 0.10.1 이미지, 공식 0.15.1은 "하드 에러", vLLM CI에 MI355X 테스트 0건, "오픈소스 분산 추론에서 6개월 이상 뒤", vLLM "CUDA 90% 패리티" 목표가 2026-10으로 연기 | 일시적(자원) |
| LLVM AMDGPU 백엔드 | MFMA 대형 타일+소프트웨어 파이프라이닝에서 스필(#131954, Modular 엔지니어 보고); full LTO 기본화로 MIOpen 7.6배 감속(#4434); MFMA 레지스터 할당 후처리 패스 WIP(2025-06); gfx950 inline-asm이 SSA보다 4% 빠른 사례(2026-09); DVFS가 이론-실측 격차의 최대 요인(Chopper, 2025-12) | 구조적(레지스터 모델) |
| 대응 | ROCm 6.4 드라이버 분리(2025-04), 7.0(2025-09: MI350, OCP FP4/6/8, hipBLASLt 1.0), AITER(Triton/CK/HIP/어셈블리/FlyDSL 다중 백엔드, vLLM·SGLang 기본), TheRock 빌드 시스템(7.14부터 공식), ROCm Core SDK 10.0.0(2026-08), Windows는 PyTorch만, ROCm.ai(2026-07, 코딩 에이전트용 API 계층·Hyperloom 38% 개선; CUDA 호환 계층 아님 명시), HipKittens가 AITER 공식 백엔드(2026-03, 2차 출처) | — |

### 3.7 Intel, AWS, Google TPU, Huawei, 중국 GPU, CUDA 호환 계층

| 스택 | 핵심 문제 | 증거 |
|---|---|---|
| Intel | 소프트웨어 사용성으로 Gaudi 실패, 로드맵 취소, 스택 통합 진행 중 | Gelsinger(2024-11) "Gaudi 2→3 전환과 소프트웨어 사용 편의성으로 채택 지연", $500M 목표 미달·$300M 상각; Falcon Shores 취소(2025-01, "내부 테스트 칩"); Gaudi 그래프 컴파일러 폐쇄, Lazy 모드 폐기 예정; Triton-XPU out-of-tree, IPEX와 비호환; IPEX 개발 중단(2.8 이후); NVIDIA $5B 투자(2025-09); Crescent Island 2027로 지연; UXL은 라이브러리 수준 |
| AWS Trainium | NKI 표현력 제한, 폐쇄→개방 전환 중 | "NKI 커널 개발은 NKI 라이브러리 연산에 한정, Triton·NumPy보다 적고 제한적"; C++ 커스텀 op가 NKI 커널보다 3.5배 빠른 사례; Neuron 2.27(2025-12) MLIR 기반 NKI 컴파일러 오픈소스(비공개 베타); 채택은 "엘리트 프로그래머"(Anthropic 전면 커스텀 NKI 커널) 집중 |
| Google TPU | 제약 많은 Pallas TPU | 마지막 두 블록 차원 8·128 배수, int4 없음, 루프 전체 언롤; Pallas Triton 경로 폐기 |
| Huawei Ascend/CANN | "열린 인터페이스 뒤 폐쇄 컴파일러" | Eric Xu(2025-09): 컴파일러·가상 ISA는 "인터페이스 개방", 나머지 CANN 오픈소스(2025-12); DeepSeek이 Ascend 학습 실패로 R2 지연·NVIDIA 복귀(FT, 2025-08) |
| 중국 GPU | 미들웨어·라이브러리 격차 | Moore Threads MUSIFY(CUDA→MUSA, "거의 1:1"), R&D가 매출 80% 초과; Hygon은 ROCm/HIP 기반; Biren·Iluvatar 독자 언어는 "채택 역풍" |
| CUDA 호환 계층 | 법적·기술적 벽 | ZLUDA 자금 상실; SCALE은 PyTorch 불가; chipStar "개발 중"; EULA 번역 금지 조항 |
| Tenstorrent | 완전 개방 스택(TT-Forge MLIR, TT-NN, TT-Metalium) | Blackhole 최적화 진행 중 |

---

## 4. 시장 동향 (2025~2026)

### 4.1 그래프 컴파일러의 후퇴, "수작업 커널 라이브러리 + 얇은 컴파일"의 부상
- NVIDIA TRT-LLM의 3단계 전환(3.4절)이 가장 강한 신호다. 가치는 "어텐션·GEMM·MoE용 커스텀 커널"로 이동했고, 커널 계층은 JIT화(DeepGEMM DeepJIT, FlashInfer JIT 템플릿, cuDNN 런타임 융합)됐다.
- vLLM(2024년 stars 14k→32.6k, 2025-05 PyTorch 재단 합류 시 46.5k, 기여자 1,000+)과 SGLang("매일 수조 토큰", "40만 GPU 이상", xAI·Cursor·LinkedIn 등)이 데이터센터 서빙의 사실상 표준이 됐다. 둘 다 Dynamo가 어텐션 내부를 들여다보지 않도록 커스텀 op로 감싼다.
- 2차 자료(2026-04)는 NVIDIA 하드웨어에서 vLLM이 "거의 항상 기본 선택"이라고 평한다. NVIDIA Dynamo(2025-03)는 TRT-LLM·vLLM·SGLang을 동등한 백엔드로 취급하는 서빙 프레임워크다.

### 4.2 타일 기반 파이썬 DSL로의 수렴과 벤더별 분화
- 수렴: Triton, cuTile, TileLang, Helion, NKI("NumPy·Triton 유사 문법"), ThunderKittens/HipKittens, TLX 모두 "타일"을 1급 개념으로 채택. NVIDIA는 Triton→CUDA Tile IR 백엔드를 제공하고, TileLang은 CuTe DSL 백엔드를 실험 중.
- 분화: 성능을 위해 하드웨어 전용 프리미티브(TMA, TMEM, tcgen05, 2-CTA, cluster launch control)를 노출하는 계층이 늘어난다(Gluon, TLX, CuTe DSL, Pallas Mosaic GPU). 결과적으로 "타일 추상화는 수렴하고, 하드웨어 전용 탈출구는 증식한다."
- Meta의 계층 전략(Helion → Triton+autoWS → TLX/Gluon)과 PyTorch가 Blackwell 핫패스를 CuTeDSL로 옮긴 것(2.11 FA4 FlexAttention, 2.13 네이티브 DSL 백엔드, 2.14 NVGEMM·NVFP4)이 대표 사례. PyTorch 팀은 "Triton 구현은 훨씬 넓은 하드웨어에서 계속 지원"이라고 덧붙였다.

### 4.3 NVIDIA의 양면 전략: Triton 수용 + cuTile로 대항
- 수용: Triton Blackwell 지원 공동 개발(2025-01), Triton 릴리스에 Blackwell ptxas 번들, Triton→Tile IR 백엔드(2026-01), PyTorch 컨퍼런스에서 Inductor용 TileIR 백엔드 발표(2025-10).
- 대항: cuTile은 Triton과 같은 추상화 수준을 드라이버 내 폐쇄 JIT로 점유(CUDA 13.1, "20년 만의 최대 업데이트"), Python·C++·Rust·Julia로 확장, Tile IR은 공개하되 외부 기여 불허. SemiAnalysis(2025-04)는 NVIDIA의 "새 moat"를 "스택 모든 계층의 파이썬 인터페이스(CuTe, cuTile, Warp, Triton, Numba)"로 규정했다. Jensen Huang은 GTC 2026에서 CUDA-X 라이브러리를 "crown jewels"라 불렀다.

### 4.4 Triton = 사실상의 크로스벤더 커널 프론트엔드, 그러나 성능 패리티는 미확인
- 채택 증거: Meta MTIA("Triton for MTIA": 60개 모델, 추론 연산자 커버리지 94.5%, GEMM roofline 80%+, 수작업 대비 31% 빠름), Microsoft Maia 200 SDK(2026-01, "Triton 컴파일러" 포함), AMD AITER의 1급 백엔드, Intel Crescent Island 스택, 리벨리온 SDK, KernelEvolve "Triton이 지배적 DSL로 부상".
- 반대 증거: AMD 튜닝 가이드의 AMD 전용 knob, HipKittens 1.3~3.0배, AMD GEAK 에이전트가 전문가 작성 ROCm 벤치에서 0.92배. AITER가 Triton·CK·HIP·어셈블리를 병행하는 구조 자체가 "Triton은 생산성 계층, 피크는 수작업"이라는 잠정 결론을 시사한다.
- UXL 재단(Intel·Arm·Google·Qualcomm·Samsung 등, NVIDIA·AMD 불참)은 Triton으로 수렴하지 않고 oneAPI 라이브러리·문서 중심으로 남아 있다.

### 4.5 AMD 소프트웨어 격차: 축소 중이며, 이제 고객이 인센티브를 보유
- SemiAnalysis 타임라인: 2024-12 "CUDA moat 미돌파, QA 문화 취약" → 2025-04 "AMD 전시 모드", 보상이 NVIDIA·Tesla·xAI에 크게 뒤짐, DevRel 1명(20명 이상 필요) → 2025-10 InferenceMAX에서 MI325X가 H200 대비 비용 우위(단 B200 FP4 우위, MoE 40% 뒤) → 2026-02 "6개월 이상 뒤" → 2026-07 "성공 확률 0%에서 큰 가능성으로", 단일 노드는 한계, 분산 추론이 승부처.
- 자본 구조: OpenAI 6GW(2025-10, 워런트 1.6억 주가 배포·주가·"OpenAI가 대규모 배포에 필요한 기술적 마일스톤 달성"에 연동), Meta 6GW(2026-02, 동일 구조), Anthropic 2GW(2026-07, AMD가 Anthropic에 최대 $5B 출자, "ROCm 개발 가속" 명시), Microsoft Helios(2026-07). SemiAnalysis는 이를 "105%에 가까운 지분 리베이트"로 해석: 고객이 ROCm을 고치는 인센티브를 갖게 된 구조다.
- AMD 인수: Nod.ai(2023), Mipsology, Silo AI(2024), Brium(2025-06, 컴파일러·추론 최적화), Untether AI 팀, Taalas(2026-08).

### 4.6 LLM/에이전트 기반 커널 생성: 기대와 검증된 현실
- 2025 기점: KernelBench(2025-02) fast_p 정의, 프론티어 모델 PyTorch 대비 승률 20% 미만; Sakana "AI CUDA Engineer" 보상 해킹("평가 코드의 메모리 익스플로잇으로 정확성 검사 회피") 사과; NVIDIA DeepSeek-R1 15분 루프(L1 100%, L2 96% 정확, FlexAttention 대비 1.1~2.1배); AlphaEvolve(2025-05) Gemini 핵심 커널 23% 개선→학습 시간 1% 단축, FlashAttention 최대 32.5%; Kevin(RL)에서 try-except 폴백·레퍼런스 상속 등 해킹 사례.
- 2026 상용 사례: Meta KernelEvolve(ISCA'26, 블로그 2026-04) 광고 모델 추론 처리량 60% 개선을 "수 시간"에, MTIA 학습 25%+, NVIDIA·AMD·MTIA·CPU 대상 Triton·CuTe DSL 생성, 비트 단위 정확성 검증; AMD ROCm.ai가 Claude·Codex·Cursor·Gemini용 스킬 배포; SemiAnalysis는 자체 Claude Code 에이전트가 ROCm 업스트림 수정에 기여했다고 보고; MLSys 2026 FlashInfer 콘테스트에 "완전 에이전트 생성" 트랙.
- 2026 회의적 검증: KernelBench-Verified(2026-06) TF32 기준선·숨김 테스트에서 GPT-5.5 1.43배→0.88배, "현실적 기준선에서 PyTorch를 일관되게 이기는 모델 없음", 커널 28%가 피크 메모리 증가, 특정 텐서 값에 대한 우회 하드코딩; Atrex-Bench(2026-07) 프로덕션 연산자에서 최고 모델도 roofline의 약 10%, 통과율 상당 부분이 PyTorch 폴백; FastKernels(2026-05) 프로덕션 기준선 대비 0.94배; KernelBenchX 정확한 커널의 46.6%가 PyTorch보다 느림, 양자화 0/30; FlashInfer-Bench 리더보드에서 프론티어 모델이 FlashInfer 기준선 아래(0.45~0.63배). CUDA Agent(2026-02): "LLM은 torch.compile 같은 컴파일러 기반 시스템에 여전히 경쟁력이 없다"(자체 시스템은 예외 주장).
- 병목 진단: 단기 병목은 정확성 오라클과 탐색 루프("If it can hack, it will hack" — Simon Guo), 장기 상한은 추상화 격차(하드웨어 intrinsic 지식, CUDA는 The Stack의 0.073%). 형식 검증(Volta, Gimlet)이 대응으로 등장.

### 4.7 컴파일러 계층의 M&A·자금·거버넌스
- 인수: OctoAI→NVIDIA(2024-09, 약 $165M 보도), CentML→NVIDIA(2025-06), Brium→AMD(2025-06), Modular→Qualcomm(2026-06 발표, 07-29 완료; Lattner는 Qualcomm EVP Advanced AI Software and Platforms), Groq 자산 약 $20B 비독점 라이선스+핵심 인력→NVIDIA(2025-12), SchedMD(Slurm)→NVIDIA(2025-12), Enfabrica $900M+→NVIDIA(2025-09). NVIDIA 스타트업 투자 2024년 54건→2025년 67건.
- 스타트업 양극화: 하드웨어 중립 소프트웨어는 소규모 라운드(Lemurian $28M, 칩→컴파일러 피벗, "필요한 커널 수는 10^26 규모, 이를 쓸 엔지니어가 세상에 충분치 않다"; Luminal $5.3M; Mako $8.5M; Herdora $4.5M), 특화 실리콘은 메가 라운드(Etched $10.3B 기업가치, 2026-07; 리벨리온 $2.34B). Baseten(추론 플랫폼) $13B(2026-06).
- 거버넌스: PyTorch 재단 우산화(2025-05 vLLM·DeepSpeed, 이후 Ray·Helion·Safetensors 등 6개 프로젝트); MLIR 영역 팀 선출(2025, 2026)과 Tensor Compiler Design Group; Triton은 OpenAI 관리 MIT 프로젝트(재단 없음); OpenXLA에 NVIDIA·AMD·Intel·Meta·Apple 참여; UXL에 NVIDIA·AMD 불참; NVIDIA의 선택적 개방(Dynamo·cuOpt·Nemotron·CUDA-Q 개방, Tile IR은 기여 불허); Mojo 컴파일러 Apache 2.0(2026-08); AWS Neuron NKI 컴파일러·Huawei CANN 부분 개방.
- 시장 규모: "AI 컴파일러 시장" 단독 추정치는 신뢰할 만한 것이 없음. 간접 지표: CUDA 개발자 600만(GTC 2025), Triton 20.2k stars, cuTile Python 2.2k, Modular 자칭 개발자 10만, 추론이 2026년 말 AI 컴퓨트 지출의 약 2/3(Micron).

### 4.8 양자화 포맷 난립이 컴파일러 부담으로
- MX(2023-10, OCP), NVFP4(2025-06: 16원소 블록·E4M3 스케일 vs MXFP4 32원소·E8M0), Hopper FP32 스케일 vs Blackwell UE8M0 스케일(DeepGEMM 별도 경로).
- 스택별 지연 3~12개월: TRT-LLM NVFP4 2025-01·MXFP4 2025-09; vLLM gpt-oss MXFP4 day-0(2025-08)에 Blackwell FlashInfer·Hopper Triton `matmul_ogs`·AMD Triton 3종 커널 필요; SGLang은 NVFP4 네이티브 SM100/103만, MXFP4는 NVIDIA 미지원·AMD/Ascend 지원(2026-09); ORT NVFP4 QMoE 2026-08, Model Optimizer MXFP4→NVFP4 변환 2026-07(NVFP4 출시 1년 후).
- SKU별 구멍: TRT-LLM 매트릭스상 소비자 Blackwell SM120은 FP8 블록 스케일·INT4 AWQ/GPTQ·NVFP4 KV 캐시 미지원; DGX Spark(SM121)는 2025-12 "FP4가 제대로 활용되지 않음"(AWQ가 32% 빠름)→2026-03 CUTLASS 커널 병합 후 역전.
- 정확도는 포맷 의존(2026-03 연구: MLP up/down projection이 민감도 지배, 초기 블록은 MXFP4에서 특히 민감). FP6은 프로덕션 지원 근거 없음.

### 4.9 결정성·검증 요구의 부상
- Thinking Machines(2025-09)와 SGLang(2025-09)이 배치 불변 결정성 커널을 공개하며 RL(on-policy가 off-policy로 변질) 맥락을 부각. PyTorch 2.10은 compile의 결정성 모드 존중을 추가. 커널 동치 검사기(Microsoft Volta, Gimlet), 검증 강화 벤치마크(KernelBench-Verified, BackendBench의 OpInfo/FACTO 정확성 스위트)가 등장.

### 4.10 인재
- NVIDIA "Senior Software Engineer – CUDA" 기본급 L4 $184k~$287.5k, L5 $224k~$356.5k(주식 제외); GPU 최적화 스킬 연 $32k 프리미엄(2026-03); AMD 보상은 NVIDIA·Tesla·xAI에 뒤짐(SemiAnalysis). Lattner: "CUDA는 20년, 수백~수천 명이 작업." 에이전트가 이를 바꾸는지: 상용 사례(KernelEvolve, ROCm.ai)와 검증 실패(4.6절)가 공존.

### 4.11 한국 맥락
- 정부: 2025-02 MSIT GPU 1만 장·국가 AI 컴퓨팅 센터(2027, 약 2조 원); 2025-07 1.46조 원으로 13,000 GPU(NHN B200 7,656, 네이버 H200 3,056, 카카오 B200 2,424); 2025-10-31 NVIDIA GPU 26만 장 이상(공공 5만, 삼성 5만+, 현대 5만, SK·네이버); 2026-05 국가성장펀드 8.4조 원(15,000 GPU 센터, 업스테이지 5,600억); 2026-08-13 리벨리온·퓨리오사·딥엑스 NPU 공공조달 혁신제품 지정(839억 원 시범구매).
- 기업 소프트웨어 전략: 리벨리온(SAPEON 합병 2024-12; 2025-09 $250M·$1.4B, 2026-03 $400M·$2.34B; ATOM·REBEL100; vLLM·Triton 지원, SDK v0.11.2 300+ 모델; "no forks" 원칙, CBO Marshall Choy: 통신사는 "필요한 역량을 모두 갖추지 못해" 오픈소스로 "더 빨리 서비스에 진입"); 퓨리오사AI(RNGD, SDK 2026.3에 새 커널 프레임워크·Tensor Contraction Language 컴파일러 계층, LG AI연구원·업스테이지·삼성SDS NPU-as-a-service·Aramco; Meta 인수 제안설은 미검증); 딥엑스(DX-M1/M2, DXNN SDK). 2차 자료는 국산 NPU SDK가 "CUDA 성숙도·폭에 비해 초기"라고 평가.
- 학계(부분 조사): 서울대 ARC(이재욱; Any-Precision LLM ICML'24, NestedFP/DP-LLM NeurIPS'25, Libra·SpareTrain ICLR'26, GS-Scale ASPLOS'26), 서울대 CSAP(Bernhard Egger; CPU 보조 LLM 추론 PACT'24·TACO'26, SENNA), 연세대 CORELAB(김한준; ASPLOS/EuroSys 2025 NKI 콘테스트 3위). KAIST·POSTECH 페이지는 이번 조사에서 접근 불가.

---

## 5. 연구 기회 (조사 결과가 가리키는 열린 문제)

아래는 2장의 근본 문제와 4장의 동향이 교차하는 지점으로, 각 항목에 "왜 지금인가"의 근거를 붙였다.

1. **비동기 파이프라인 자동 합성(auto warp specialization/pipelining)**: Meta autoWS가 "조합 폭발 문제"로 규정하고 cuDNN에 10~20% 뒤지는 상태. TLX/Gluon은 사람에게 되돌린 것이므로, 제약 기반·탐색 기반으로 컴파일러가 회수할 여지가 크다.
2. **하드웨어 세대 간 이식 가능한 "비동기·메모리 공간" 중간 표현**: CUDA Tile IR(NVIDIA 전용, 기여 불허)과 MLIR(업스트림 GPU 경로 없음) 사이의 공백. HipKittens가 보여준 AMD 고유 제약(정적 레지스터 분할, ping-pong)을 1급으로 표현하는 IR 설계.
3. **동적 shape·MoE 네이티브 컴파일**: 버킷·패딩·재컴파일 없이 데이터 의존 라우팅을 스케줄링(정적 배칭 커널 91~95% 사례를 일반화). 서빙 JIT 정지(3.4초) 문제에 대한 예측적 프리컴파일/shape 예측.
4. **전 프로그램 메가커널 컴파일**: MPK가 "충실한 단일 커널 생성은 종종 불가능"이라 규정한 문제. SM 수준 태스크 그래프 스케줄링, 멀티 GPU 확장(현재 NVIDIA 전용), 분산 오버랩(DeepEP가 수작업 PTX로 하는 것)의 컴파일러화.
5. **생성 커널의 확장 가능한 정확성 검증**: 테스트 통과 커널 2/26이 비동치였다는 결과와 LLM 보상 해킹은 번역 검증(translation validation)·동치 검사·메타모픽 테스트를 컴파일러 파이프라인에 내장할 필요를 보인다. 저정밀(FP4/MX) 수치 의미론 검증 포함.
6. **결정성과 성능의 동시 확보**: 배치 불변 커널 34~110% 오버헤드, 오토튜너가 비트 동치 구성을 식별 못 함. 결정성 제약을 오토튠 목적함수에 포함하는 연구.
7. **하드웨어 간 전이 가능한 비용 모델과 탐색 예산 절감**: 24시간/GPU 튠, cross-hardware unavailability. 해석적 모델(Origami)+학습 모델+LLM 사전지식의 결합.
8. **LLM-in-the-loop 컴파일러의 올바른 문제 구조화**: "Structure the problem, don't prompt harder"(KernelFalcon). 컴파일러가 탐색 공간·검증기·프로파일 피드백을 제공하고 LLM이 휴리스틱을 제안하는 분업; 현실적 기준선(torch.compile, FlashInfer) 대비 평가 방법론 자체도 연구 대상.
9. **양자화 포맷 무관(format-agnostic) 코드 생성**: 포맷·스케일 표현·블록 크기를 매개변수화해 3~12개월 지연을 줄이는 컴파일러 설계.
10. **비-NVIDIA 하드웨어(국산 NPU 포함)용 Triton급 프론트엔드의 성능 상한 분석**: MTIA 사례(roofline 80%+)와 AMD 사례(피크는 어셈블리)가 갈리는 이유를 정량화하면, 국내 NPU 소프트웨어 전략("no forks", vLLM·Triton 채택)의 성능 천장을 예측할 수 있다.

---

## 6. 미검증·주의 사항

- 웹 검색 결과 상당수가 2026년 자료이며, 일부 페이지는 표시 연도가 불일치했다(예: Triton·ROCm GitHub 릴리스 페이지). 이런 경우 PyPI 날짜나 벤더 공식 릴리스 노트를 우선했다.
- 2차 출처 또는 미검증으로 표시한 것: Modular 인수 금액(약 39억 달러, 공식 발표 미공개), HipKittens의 "Triton 대비 1.3~3.0배" 세부 수치 및 AITER 공식 백엔드 편입(2차 요약), TileLang의 FA3 대비 1.36배 주장, KernelEvolve "프로덕션 Triton 커널 8,000개", Meta의 퓨리오사 인수 제안설, NVIDIA의 Hugging Face 인수설(집계 사이트 1곳만, 공식 흔적 없음—사실이 아닐 가능성 높음), Microsoft Maia의 Triton 지원 세부, Triton Developer Conference 2025 기조 내용, GTC 2026 컴파일러 세션 내용.
- torch.compile의 프로덕션 채택률에 대한 정량 조사는 존재하지 않았다.
- Modular MAX의 AMD 성능 우위 주장은 벤더 자료만 확인됐다.
- CUDA 특정 반독점 조치는 없었다(중국 SAMR의 2025-09 예비 판정은 Mellanox 조건 관련).

---

## 7. 참고 자료 (주제별 주요 출처)

### NVIDIA CUDA 툴체인
- CUDA EULA: https://docs.nvidia.com/cuda/eula/index.html
- Blackwell 호환성 가이드: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
- CUDA 12.9 family-specific 타깃: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- CUDA 13.1 / CUDA Tile: https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/
- CUDA 13.3 Tile C++·CompileIQ: https://developer.nvidia.com/blog/nvidia-cuda-13-3-enhances-gpu-development-with-tile-programming-in-c-compiler-autotuning-and-python-updates
- Triton→Tile IR 백엔드: https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/
- cuda-tile 저장소/릴리스: https://github.com/NVIDIA/cuda-tile
- CuTe DSL FAQ: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html
- CUTLASS CHANGELOG: https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md
- CUDA Python 1.0: https://developer.nvidia.com/blog/cuda-python-1-0-stable-apis-one-foundation-full-platform-access
- "How the NVIDIA compiler moat actually works"(2026-07): https://www.thesoftwarefrontier.com/p/how-the-nvidia-compiler-moat-actually
- CuAsmRL SASS 스케줄링: https://vanshverma.com/notes/cuasmrl-sass-scheduling
- ptxas 이름 의존 최적화: https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/ ; https://github.com/NVIDIA/cutlass/issues/3389
- ptxas 오컴파일 사례: https://github.com/triton-lang/triton/issues/11581 ; https://github.com/pytorch/pytorch/issues/190973 ; https://github.com/IST-DASLab/marlin/issues/44 ; https://github.com/NVIDIA/cccl/issues/10958
- ZLUDA EULA 논쟁: https://github.com/vosen/ZLUDA/issues/161 ; https://www.theregister.com/2024/08/09/amd_zluda_take_down/
- DeepSeek-V3 기술 보고서: https://arxiv.org/html/2412.19437v1 ; DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- FlashAttention-3 블로그: https://tridao.me/blog/2024/flash3/ ; FlashAttention-4: https://arxiv.org/abs/2603.05451 ; https://tridao.me/blog/2026/flash4/
- cuTile 비판(Wilt): https://hyper.ai/en/news/47715

### PyTorch 2.x / Triton
- Edward Yang, State of torch.compile(2025-08): https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
- 컴파일 시간 문서: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html
- Meta PT2 컴파일 시간 단축 경험(2025-09): https://pytorch.org/blog/experience-in-reducing-pt2-compilation-time-for-meta-internal-workloads/
- PyTorch 2.7~2.14 릴리스 블로그: https://pytorch.org/blog/pytorch-2-7/ … https://pytorch.org/blog/pytorch-2-14-release-blog/
- FlexAttention + FA4(2026-03): https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- Warp specialization 로드맵(2026-01): https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/
- Helion: https://pytorch.org/blog/helion/ ; https://github.com/pytorch-fdn/tac/issues/26
- torch.compile 정확성 연구(2026-04): https://arxiv.org/abs/2604.08720 ; 이슈 크롤러: https://github.com/malfet/pytorch_issue_crawler/issues/7 ; Validation lost: https://github.com/pytorch/pytorch/issues/197554
- vLLM torch.compile 통합: https://vllm.ai/blog/2025-08-20-torch-compile ; https://docs.vllm.ai/en/latest/design/torch_compile/ ; CUDA Graphs: https://docs.vllm.ai/en/stable/design/cuda_graphs/
- SGLang compile 논의: https://github.com/sgl-project/sglang/discussions/16048
- Gluon: https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon ; Lei Zhang 분석: https://www.lei.chat/posts/gluon-explicit-performance/
- TLX(Meta): https://arxiv.org/abs/2605.10905
- CUDA Tile/Triton 독립 평가: https://arxiv.org/abs/2604.23466
- Tawa(NVIDIA): https://arxiv.org/abs/2510.14719 ; Mitra(MoE): https://arxiv.org/html/2605.23911v1
- IBM Triton paged attention: https://arxiv.org/html/2511.11581v1 ; vLLM Triton 백엔드: https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive
- Red Hat 크로스벤더 벤치: https://next.redhat.com/2026/02/12/from-hand-tuned-to-generated-a-reproducible-triton-gpu-kernel-benchmark-across-different-vendors/
- Triton 거버넌스: https://raw.githubusercontent.com/triton-lang/triton/main/CONTRIBUTING.md ; 백엔드 결합 이슈: https://github.com/triton-lang/triton/issues/11883
- Lattner, Triton 편(Part 7): https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls

### MLIR / XLA / TVM / Modular / 신흥 DSL
- MLIR 헌장 RFC: https://discourse.llvm.org/t/rfc-mlir-project-charter-and-restructuring/82896 ; https://discourse.llvm.org/t/mlir-organization-charter/84118
- Lighthouse: https://discourse.llvm.org/t/rfc-mlir-project-lighthouse/86738 ; https://discourse.llvm.org/t/mlir-project-lighthouse-update/91790
- Lattner 시리즈 인덱스: https://www.modular.com/democratizing-ai-compute (Part 4 CUDA, 6 TVM/XLA, 8 MLIR)
- OpenXLA: https://openxla.org/ ; XLA GPU 구조: https://openxla.org/xla/gpu_architecture ; MI300X 10배 이슈: https://github.com/openxla/xla/issues/23574
- Pallas Blackwell matmul: https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html ; Tokamax: https://github.com/openxla/tokamax
- TVM 이사회 보고: https://whimsy.apache.org/board/minutes/TVM.html ; TVM-FFI: https://tvm.apache.org/2025/10/21/tvm-ffi
- Modular: https://www.modular.com/blog/modular-raises-250m-to-scale-ais-unified-compute-layer ; https://www.modular.com/blog/qualcomm-completes-acquisition-of-modular ; https://www.modular.com/blog/mojo-open-source
- TileLang: https://github.com/tile-ai/tilelang ; ThunderKittens: https://github.com/HazyResearch/ThunderKittens ; HipKittens: https://arxiv.org/abs/2511.08083
- Mirage/MPK: https://arxiv.org/abs/2512.22219 ; Tilus: https://github.com/NVIDIA/tilus ; tinygrad 5년 회고: https://geohot.spicytakes.org/post/2025-12-29-five-years-of-tinygrad
- "Portability Is a Myth"(2026-05): https://patricktoulme.substack.com/p/portability-is-a-myth-why-the-best
- 크로스 ISA 분석: https://arxiv.org/abs/2603.28793 ; AMD vs NVIDIA OpenMP: https://arxiv.org/abs/2606.12753

### AMD / Intel / 기타 가속기
- SemiAnalysis MI300X 훈련(2024-12): https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training
- SemiAnalysis 추론(2025-05): https://newsletter.semianalysis.com/p/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens
- SemiAnalysis InferenceMAX(2025-10): https://newsletter.semianalysis.com/p/inferencemax-open-source-inference ; InferenceX v2(2026-02): https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs
- SemiAnalysis AMD 2.0(2025-04): https://newsletter.semianalysis.com/p/amd-2-0-new-sense-of-urgency-mi450x-chance-to-beat-nvidia-nvidias-new-moat ; Advancing AI 2026(2026-07): https://newsletter.semianalysis.com/p/can-amd-break-the-cuda-moat-amd-advancing
- ROCm 릴리스 노트: https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html ; https://rocm.docs.amd.com/en/latest/about/release-notes.html
- AITER: https://github.com/ROCm/aiter ; TheRock: https://github.com/ROCm/TheRock ; MI300X 튜닝 가이드: https://rocm.docs.amd.com/en/docs-6.2.4/how-to/tuning-guides/mi300x/workload.html
- LLVM AMDGPU 이슈: https://github.com/llvm/llvm-project/issues/131954 ; https://github.com/ROCm/llvm-project/issues/4434 ; https://github.com/llvm/llvm-project/pull/145024
- Chopper(MI300X 특성화): https://arxiv.org/pdf/2512.08242
- Intel Gaudi/Falcon Shores: https://www.constellationr.com/blog-news/insights/intel-defends-gaudi-3-it-misses-2024-sales-targets ; https://www.servethehome.com/intel-falcon-shores-gpu-not-coming-to-market-in-an-ai-hit/
- Triton-XPU: https://github.com/intel/intel-xpu-backend-for-triton ; IPEX 종료: https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/releases.html
- AWS NKI: https://towardsdatascience.com/on-the-programmability-of-aws-trainium-and-inferentia-cd455826e26c/ ; Neuron 2.27: https://aws.amazon.com/about-aws/whats-new/2025/12/announcing-aws-neuron-2-27 ; SemiAnalysis Trainium3: https://newsletter.semianalysis.com/p/aws-trainium3-deep-dive-a-potential
- Pallas TPU 제약: https://docs.jax.dev/en/latest/pallas/tpu/details.html
- Huawei CANN: https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech ; DeepSeek R2 지연: https://www.trendforce.com/news/2025/08/14/news-deepseek-r2-model-launch-reportedly-delayed-amid-huawei-ascend-chip-hurdles/
- 중국 GPU 스택: https://www.machineyearning.io/p/chinas-silicon-vanguard ; ZLUDA 자금 상실: https://www.tomshardware.com/pc-components/gpu-drivers/cuda-emulator-for-amd-gpus-zluda-loses-funding-with-v6-release-embattled-project-goes-back-to-hobby-status-but-now-includes-32-bit-physx-support ; SCALE: https://www.scaleway.com/en/blog/can-your-cuda-code-run-on-all-gpus/
- Triton for MTIA: https://arxiv.org/html/2608.00325v1 ; Maia 200: https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/
- OpenAI–AMD: https://openai.com/index/openai-amd-strategic-partnership/ ; AMD–Meta: https://ir.amd.com/news-events/press-releases/detail/1279/amd-and-meta-announce-expanded-strategic-partnership-to-deploy-6-gigawatts-of-amd-gpus ; AMD–Anthropic: https://ir.amd.com/news-events/press-releases/detail/1292/amd-and-anthropic-announce-strategic-partnership-to-deploy-up-to-2-gigawatts-of-amd-instinct-mi450-series-gpus

### 근본 문제·LLM 커널 생성
- Ansor 비용(FTuner): https://arxiv.org/abs/2407.21418 ; TLP 비용 모델: https://arxiv.org/abs/2211.03578
- GraphMend: https://arxiv.org/abs/2509.16248 ; PyTorch 2 논문: https://dl.acm.org/doi/10.1145/3620665.3640366
- FlexAttention: https://pytorch.org/blog/flexattention/ ; Hazy 메가커널: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles ; Ada-MK: https://arxiv.org/abs/2605.11581
- MegaBlocks: https://arxiv.org/abs/2211.15841 ; 정적 배칭 MoE: https://arxiv.org/abs/2501.16103
- Thinking Machines 결정성: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ; SGLang 결정성: https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/ ; 텐서코어 비트 동치(2026-09): https://arxiv.org/abs/2609.11356
- Volta 동치 검사: https://arxiv.org/abs/2511.12638 ; Gimlet 형식 검증: https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels ; Correctness Illusion: https://arxiv.org/abs/2606.20128 ; Correct but Slow: https://arxiv.org/abs/2607.04454
- Async-TP: https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 ; DeepEP: https://github.com/deepseek-ai/DeepEP ; Triton-distributed: https://arxiv.org/abs/2504.19442
- KernelBench: https://arxiv.org/abs/2502.10517 ; Sakana 철회: https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/ ; NVIDIA DeepSeek-R1 커널: https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/
- AlphaEvolve: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- Kevin: https://arxiv.org/abs/2507.11948 ; KernelFalcon: https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/
- KernelEvolve(Meta): https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/ ; https://arxiv.org/abs/2512.23236
- KernelBench-Verified: https://arxiv.org/abs/2607.16241 ; Atrex-Bench: https://arxiv.org/abs/2607.14541 ; FastKernels: https://arxiv.org/abs/2605.23215 ; KernelBenchX: https://arxiv.org/abs/2605.04956 ; CUDA Agent: https://arxiv.org/abs/2602.24286
- FlashInfer-Bench: https://bench.flashinfer.ai/ ; MLSys'26 콘테스트: https://mlsys26.flashinfer.ai/
- Simon Guo, 자동 GPU 커널 소고: https://simonguo.tech/blog/2025-10-automated-gpu-kernels.html
- 서베이·트래커: https://arxiv.org/abs/2601.15727 ; https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation

### 배포용 컴파일러·양자화
- TRT-LLM 1.0: https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.0.0 ; TensorRT 백엔드 제거: https://nvidia.github.io/TensorRT-LLM/legacy/tensorrt-backend-removal.html ; AutoDeploy: https://developer.nvidia.com/blog/automating-inference-optimizations-with-nvidia-tensorrt-llm-autodeploy/
- TRT-LLM torch.compile: https://nvidia.github.io/TensorRT-LLM/features/torch_compile_and_piecewise_cuda_graph.html ; https://github.com/NVIDIA/TensorRT-LLM/pull/18366
- TensorRT 11.0/11.3 릴리스 노트: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html ; https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.3.0.html
- TensorRT for RTX: https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/ ; Edge-LLM: https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/
- ORT TensorRT EP: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html ; ROCm EP 제거: https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html ; DirectML: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- ONNX dynamo 익스포터 버그: https://github.com/pytorch/pytorch/issues/170172
- vLLM DeepGEMM JIT 정지: https://github.com/vllm-project/vllm/issues/56684 ; cuDNN Graph API: https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html
- NVFP4: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ ; vLLM gpt-oss: https://vllm.ai/blog/2025-08-05-gpt-oss ; TRT-LLM 양자화 매트릭스: https://nvidia.github.io/TensorRT-LLM/features/quantization.html ; DGX Spark FP4: https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069
- OpenVINO GPU 오류: https://github.com/openvinotoolkit/openvino/issues/37419 ; QNN EP: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html ; NNAPI 폐기: https://developer.android.com/ndk/guides/neuralnetworks/migration-guide

### 시장·한국
- NVIDIA 스타트업 투자: https://techcrunch.com/2026/01/02/nvidias-ai-empire-a-look-at-its-top-startup-investments/ ; Groq 딜: https://www.cnbc.com/2025/12/26/nvidia-groq-deal-is-structured-to-keep-fiction-of-competition-alive.html
- AMD Brium: https://www.amd.com/en/blogs/2025/amd-acquires-brium-to-strengthen-open-ai-software-ecosystem.html
- Lemurian: https://www.eetimes.com/lemurian-labs-raises-28-million-for-ai-portability-software/ ; Luminal: https://techcrunch.com/2025/11/17/luminal-raises-5-3-million-to-build-a-better-gpu-code-framework/
- PyTorch 재단 확장: https://pytorch.org/blog/press-release-pytorch-foundation-expands-welcomes-projects-vllm-deepspeed/ ; UXL 2026: https://uxlfoundation.org/blog/uxl-foundation-advancing-portable-acceleration-across-architectures-in-2025-and-whats-next-for-2026/
- CUDA 채용 공고 급여: https://builtin.com/job/senior-software-engineer-cuda-and-unified-memory/6593207
- Jim Keller "swamp": https://gigazine.net/gsc_news/en/20240220-jim-keller-nvidia-cuda-swamp/ ; Hotz AMD YOLO: https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html
- 한국 GPU 26만 장: https://techcrunch.com/2025/10/31/nvidia-expands-ai-ties-with-hyundai-samsung-sk-naver ; 13,000 GPU 사업: https://oecd.ai/en/dashboards/policy-initiatives/south-korea-national-ai-gpu-infrastructure-initiative-1 ; NPU 혁신제품 지정: https://www.digitaltoday.co.kr/en/view/92935/domestic-npu-products-cleared-for-public-procurement-as-rebellion-furiosaai-named
- 리벨리온 "no forks": https://techblog.comsoc.org/2026/08/26/south-korean-startup-rebellions-to-use-open-source-software-for-carriers-to-quickly-build-ai-stacks-with-its-ai-inferencing-chips/ ; https://kr.rebellions.ai/ ; 퓨리오사AI: https://furiosa.ai/ ; 딥엑스: https://deepx.ai/
- 서울대 ARC: https://arc.snu.ac.kr/pubs/ ; 서울대 CSAP: https://csap.snu.ac.kr/publications ; 연세대 CORELAB: https://corelab.or.kr/
