# 대조군 재연: 비역할로 분류된 버그 20건에 역할 기반 접근을 대 보면

- 작성일: 2026-09-22
- 대상: 1차 조사(`bugs_vllm.md`, `bugs_sglang.md`, `bugs_transformers.md`, `bugs_llamacpp.md`)가 역할 계열이 아니라고 본 20건. 1차 분류는 N1 3건, N2 7건, N3 8건, N5 2건이다.
- 방법: GitHub 읽기 전용(`gh api` GET만 사용). 이슈 본문·댓글·타임라인을 새로 받았고, 수정 PR의 본문과 diff를 읽었다. 기전이 글에 없으면 수정 전후 소스를 직접 봤다. 예: vLLM `dd127d82`/`2bd89576`의 XPU FP8 커널 배정, vllm-xpu-kernels `v0.1.10.1...v0.1.11` 비교, llama.cpp b8277의 SYCL L2_NORM과 `qwen35.cpp`.
- 질문: (1) 기전 한 줄, (2) 비역할 판정에 동의하는가, (3) 다섯 접근 S1–S5가 막는가(P), 실행 중 잡는가(D), 출시 전 테스트로 드러내는가(T), 도움이 안 되는가(–).

## 1. 판정 기준

표시는 모두 "그 접근을 도입했다면"이라는 가상 판단이다. 넉넉하게 주지 않으려고 아래 기준을 고정했다.

- **S1 P**: 경계를 넘는 값에 관한 사실을 어휘(LAYOUT, REDUCTION, POSITION-FRAME, RANGE, TIME, PROPERTY, MAPPING, DTYPE)로 선언할 수 있어야 한다. 또 추론을 돌리기 전(그래프 구성, 적재, 변환 시점)에 양쪽 선언을 비교해 거부할 수 있어야 한다.
- **S2 D**: 소비 지점에서 그림자 태그가 서로 어긋나 보여야 한다. 한 구성 요소 안에서 값이 손상되는 경우는 해당하지 않는다.
- **S3 T**: 역할에서 나온 경계 사례가 원래 테스트에 없던 요소였고, 생성된 테스트가 스펙에서 유도한 기준과 비교될 때만 준다. 해당 하드웨어에서 기존 테스트만 돌려도 잡혔을 버그와, 역할과 무관한 수치 경계는 –로 둔다.
- **S4 D**: 선언되거나 조회된 속성을 받는 쪽이 지원하지 않으면서 조용히 버리거나 다른 것으로 바꾼 경우다. 적재나 디스패치 시점의 오류도 D로 센다. 플랫폼이 기능을 거짓으로 알린 경우와 선언된 적 없는 전제는 –로 둔다.
- **S5 P**: 커널과 메타데이터(배정, 스케일 모양, 지원 신고, 파이프라인)를 따로 손으로 맞춘 것이 원인이고, 한 선언에서 둘 다 생성하면 불일치가 생길 수 없을 때 준다. 성능 조정 값과 한 커널 안의 산술은 –로 둔다.
- **(부분)**: 한 이슈에 결함이 둘이고 그중 하나만 막을 때 붙인다.

## 2. 결과 표

접근별 근거는 3절에 한 줄씩 적었다.

| # | 버그 | 1차 | 기전 (한 줄) | 비역할 판정에 동의? | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [vLLM #48058](https://github.com/vllm-project/vllm/issues/48058) | N2 | XPU는 W8A8 FP8 체크포인트를 기본으로 W8A16 커널에 배정했다. vLLM은 채널별 스케일을 `[1,N]`으로 넘겼고, vllm-xpu-kernels 0.1.10.1은 `dim()==2 && numel()>1`이면 블록 양자화로 추론해 `{k,k}` 블록 스케일로 곱했다. 0.1.11의 #440이 이 판정을 고쳤다. | 비동의: **LAYOUT** (스케일 입도를 모양에서 추론, 부차 PROPERTY) | P | D | T | D | P |
| 2 | [SGLang #31011](https://github.com/sgl-project/sglang/issues/31011) | N2 | xccl/oneCCL은 정수 MIN/MAX all_reduce를 지원하지 않으면서 조용히 SUM을 계산했다. 그래서 TP 토큰 id 동기화가 `token_id × tp_size`를 돌려줬고, oneCCL 갱신으로 풀렸다. | 대체로 동의. 경계 사례: 모양은 REDUCTION 누락이지만 결함이 벤더 라이브러리 안에 있고 단독으로도 틀린다 | – | – | T | D | – |
| 3 | [SGLang #28685](https://github.com/sgl-project/sglang/issues/28685) | N2 | 결함이 둘이다. ROCm 7.2 LLVM이 없앤 `-amdgpu-coerce-illegal-types`를 aiter가 "skip it"으로 빼서 CK bpreshuffle GEMM이 잘못 컴파일됐다(rocm-libraries#8639). 또 같은 플래그가 켠 전치 활성 스케일의 물리 배치를 생산자와 소비 GEMM이 다르게 읽었다(SGLang #29275). | 부분 비동의: **LAYOUT** + N2 혼합. #8639만 넣었을 때 GSM8K는 0/82였다 | P(부분) | D(부분) | T | D(부분) | P(부분) |
| 4 | [transformers #46489](https://github.com/huggingface/transformers/issues/46489) | N3 | DeepSeek-Coder v1은 `model_type: llama`, `tokenizer_class: LlamaTokenizerFast`여서 v5가 LlamaTokenizer로 보냈다. 이 클래스가 tokenizer.json의 ByteLevel 파이프라인을 SentencePiece식으로 대체해 공백이 사라졌다. | 비동의: **PROPERTY** | P | – | T | D | P |
| 5 | [transformers #46710](https://github.com/huggingface/transformers/issues/46710) | N3 | DeepSeek-R1-Distill-Llama-8B도 같은 기전이다. LlamaTokenizerFast가 없어져 LlamaTokenizer로 떨어졌고, 출력에 `Ġ`·`Ċ`가 그대로 남았다. | 비동의: **PROPERTY** | P | – | T | D | P |
| 6 | [transformers #45920](https://github.com/huggingface/transformers/issues/45920) | N3 | OLMo2, HyperClovaX, Yi, ERNIE 등에서 GPT2Tokenizer/LlamaTokenizer의 `__init__`이 tokenizer.json의 pre-tokenizer를 하드코딩 값으로 바꿔 토큰 id가 틀렸다. | 비동의: **PROPERTY** | P | – | T | D | P |
| 7 | [transformers #45812](https://github.com/huggingface/transformers/issues/45812) | N3 | Granite의 tokenizer.json은 `Split(\p{N}{1,3}) + ByteLevel(use_regex=False)`인데, GPT2Tokenizer `__init__`이 `ByteLevel(use_regex=True)`로 덮어써 숫자 분할이 달라졌다. | 비동의: **PROPERTY** | P | – | T | D | P |
| 8 | [transformers #45356](https://github.com/huggingface/transformers/issues/45356) | N3 | v5.4가 Kimi-K2.5를 TokenizersBackend로 강제했다. `add_tokens()`가 체크포인트의 비연속 명시 id를 차례로 다시 매겨 id 163588 이후가 모두 어긋났고, `</think>`가 빈 문자열로 디코딩됐다. | 비동의: **MAPPING** | P | – | T | D | – |
| 9 | [transformers #44779](https://github.com/huggingface/transformers/issues/44779) | N3 | DeepSeek-R1/V3: Hub의 틀린 tokenizer_class 때문에 v5가 ByteLevel pre-tokenizer와 decoder를 Metaspace 계열로 바꿔 공백이 사라졌다. | 비동의: **PROPERTY** | P | – | T | D | P |
| 10 | [transformers #44448](https://github.com/huggingface/transformers/issues/44448) | N3 | v5 변환 경로에 Pegasus 전용 `convert_from_spm`이 없어 특수 토큰과 예약 id(offset 103) 배치가 모델 임베딩과 어긋났다. | 비동의(중간 확신): **MAPPING** | – | – | T | – | – |
| 11 | [llama.cpp #25734](https://github.com/ggml-org/llama.cpp/issues/25734) | N1 | Vulkan warptile이 WM을 subgroup 크기로 잡았다. subgroup 128 장치에서 셰이더의 암묵 불변식 WM ≤ BM이 깨져 공유 메모리를 넘어 읽고 일부 열을 계산하지 않았다. | 동의 (RANGE로 볼 여지가 있는 숨은 커널 전제) | – | – | – | – | – |
| 12 | [llama.cpp #25027](https://github.com/ggml-org/llama.cpp/issues/25027) | N1 | Vulkan step 셰이더가 `>` 대신 `>=`를 써서 x == 0에서 CPU와 다른 값을 냈다. | 동의 | – | – | – | – | – |
| 13 | [llama.cpp #23574](https://github.com/ggml-org/llama.cpp/issues/23574) | N1 | FA MMA 커널의 `j_vram*stride_mask`가 KQ 마스크 원소 수가 INT32_MAX를 넘자 int32로 넘쳤다(`int64_t` 캐스트로 수정). | 동의 | – | – | – | – | – |
| 14 | [llama.cpp #23850](https://github.com/ggml-org/llama.cpp/issues/23850) | N2 | MoltenVK(AMD)가 지원한다고 알린 subgroup shuffle이 실제로는 깨져 있어 FWHT 셰이더가 틀렸다. 그 기능을 끄는 것으로 해결했다. | 동의 | – | – | – | – | – |
| 15 | [llama.cpp #21887](https://github.com/ggml-org/llama.cpp/issues/21887) | N2 | #19378이 CUDA P2P를 `cudaDeviceCanAccessPeer`만 보고 기본으로 켰고, 일부 메인보드·BIOS·IOMMU 구성에서 GPU 간 복사가 손상됐다. #21910이 opt-in으로 바꿨다(보고자 미확인). | 동의 | – | – | – | – | – |
| 16 | [llama.cpp #21855](https://github.com/ggml-org/llama.cpp/issues/21855) | N2 | #21887과 원인과 수정이 같다(보고자 확인). | 동의 | – | – | – | – | – |
| 17 | [llama.cpp #21648](https://github.com/ggml-org/llama.cpp/issues/21648) | N2 | Windows HIP에서 peer 접근이 없을 때 `cudaMemcpyPeerAsync`가 데이터를 손상했다. llama.cpp는 수정하지 않았고, 보고자가 HIP 버그로 보고 닫았다. | 동의 (수정이 없어 규칙상 N4일 수 있음) | – | – | – | – | – |
| 18 | [llama.cpp #18452](https://github.com/ggml-org/llama.cpp/issues/18452) | N3 | 변환기가 Jina의 RobertaTokenizer를 바이트 수준 BPE(`gpt2`)로 옮기면서 tokenizer.json의 `Whitespace` pre-tokenizer와 `Lowercase` normalizer를 읽지 않았다. llama.cpp BPE는 항상 바이트 인코딩을 적용했다. | 비동의: **PROPERTY** | P | – | T | D | P |
| 19 | [llama.cpp #21893](https://github.com/ggml-org/llama.cpp/issues/21893) | N5 | SYCL B70에서 `GGML_SYCL_DISABLE_OPT=1`(가중치 재배치 끔)일 때만 정상이었다. 빌드(2026-04-13)가 Q8_0 재배치 도입(#21527)과 재배치 인식 dequantizer 보강(#21638, Q8_0·Q4_K·Q6_K) 사이에 있었다. | 추정 비동의: **LAYOUT** (확인 안 됨) | P | D | T | D | P |
| 20 | [llama.cpp #20423](https://github.com/ggml-org/llama.cpp/issues/20423) | N5 | b8277의 SYCL L2_NORM은 stride를 무시하고 연속 행으로 읽었는데 `supports_op`는 무조건 true였다. Qwen3.5 GDN은 q/k의 strided view에 l2_norm을 건다. 보고자는 #20283(stride 지원) 뒤 gibberish가 사라졌다고 했고, 이후 불안정(HTTP 500)은 원인 불명이다. | 추정 비동의: **LAYOUT** (1단계 증상) | P | D | T | D | P |

## 3. 접근별 근거 (한 줄씩)

**1. vLLM #48058**
- S1 P: 스케일 입도(채널별/블록)를 경계의 역할로 선언하면, 채널별 `[1,N]` 스케일을 블록으로 해석하는 결합이 적재 시점에 거부된다.
- S2 D: `weight_scale`에 "채널별" 태그를 달면 op가 블록 분기로 들어가는 첫 GEMM에서 불일치가 잡힌다.
- S3 T: 허용되는 스케일 모양(`[N]`, `[N,1]`, `[1,N]`)마다 역양자화 후 matmul 기준과 비교하면 드러난다. #440이 추가한 `scale_layout` 테스트가 바로 이것이다.
- S4 D: 선언된 동적 토큰별 FP8 활성 양자화를 W8A16 배정이 조용히 버리는 것이 오류가 되면 스케일 rank를 제대로 읽는 W8A8 경로로 간다. 다만 op 결함 자체는 남는다.
- S5 P: 변형(W8A8, 채널별)을 한 번 선언하고 배정과 스케일 메타데이터를 함께 생성하면, 손으로 맞춘 모양 규약이 없어진다.

**2. SGLang #31011**
- S1 –: 호출부의 MIN 선언은 맞다. 불일치는 선언된 역할 둘 사이가 아니라 백엔드가 숨긴 미지원에 있다.
- S2 –: 태그는 요청한 MIN을 그대로 적고, SUM 결과도 랭크마다 같아서 복제본 검사까지 통과한다.
- S3 T: REDUCTION 축(SUM/MIN/MAX × dtype)으로 collective 테스트를 만들면 드러난다. 보고자의 10줄 재현이 이 형태다. XPU 2장 이상의 CI가 필요하다.
- S4 D: 지원하지 않는 정수 MIN을 SUM으로 바꾸지 않고 오류를 내면 막힌다. 단, 규칙이 xccl/oneCCL 경계 안에서 지켜져야 한다.
- S5 –: 커널과 메타데이터를 맞추는 문제가 아니다.

**3. SGLang #28685**
- S1 P(부분): 활성 스케일의 물리 배치를 생산자→GEMM 경계의 역할로 선언하면 배치 결함은 거부된다. 컴파일러 오작동은 남는다.
- S2 D(부분): 소비 GEMM이 `x_scale`의 배치 태그를 검사하면 배치 결함이 잡힌다. 잘못 컴파일된 값은 태그로 보이지 않는다.
- S3 T: 스케일 배치 × 배정 변형(triton/CK bpreshuffle, 튜닝 목록 밖 모양)을 기준 GEMM과 비교하면 두 결함이 모두 드러난다. gfx950 + ROCm 7.2 CI가 필요한데, 그 워크플로는 당시 건너뛰어졌다.
- S4 D(부분): aiter가 지원되지 않는 필수 플래그를 "skip it"으로 넘기지 않고 오류를 내면 잘못 컴파일된 모듈이 쓰이지 않는다. 배치 결함은 못 잡는다.
- S5 P(부분): bpreshuffle 변형의 커널과 스케일 배치 구체화를 한 선언에서 만들면 배치 결함이 없어진다. 컴파일러 오작동은 남는다.

**4·5·6·7·9. transformers #46489, #46710, #45920, #45812, #44779 (클래스 라우팅이 tokenizer.json을 덮어씀)**

다섯 건 모두에 같은 근거가 적용된다. 건마다 다른 점은 덮어쓴 요소뿐이다. #45812와 #45920은 pre-tokenizer의 숫자 분할, #46489·#46710·#44779는 ByteLevel pre-tokenizer와 decoder다.
- S1 P: tokenizer.json이 선언한 pre-tokenizer와 decoder를 역할로 두면, 클래스가 구현하는 파이프라인과 묶는 `from_pretrained` 시점에 불일치가 거부된다.
- S2 –: 토큰 id에는 모델이 확인할 태그가 없고, 불일치는 적재 시점에 이미 정해진다.
- S3 T: 선언된 요소에서 나온 경계 입력(숫자열, 공백·개행, CJK)을 tokenizers 라이브러리의 tokenizer.json 실행 결과와 비교하면 드러난다. 보고자들이 쓴 `PreTrainedTokenizerFast` 비교가 그 형태다.
- S4 D: 클래스 `__init__`이 선언된 pre-tokenizer를 조용히 덮어쓰지 않고 오류를 내면 막힌다.
- S5 P: 토크나이저를 선언(tokenizer.json)에서 생성하면 손으로 쓴 클래스의 하드코딩이 끼어들 수 없다. 실제 수정(#44801, #45813, #46091)도 TokenizersBackend로 보내는 것이었다.

**8. transformers #45356**
- S1 P: `added_tokens_decoder`의 명시 id를 MAPPING 역할로 두면, 차례로 다시 매긴 id와의 불일치가 적재 시점에 거부된다.
- S2 –: 적재 시점에 이미 정해지고, 값에 붙은 태그가 없다.
- S3 T: 선언된 특수 토큰마다 원래 TikToken 구현과 encode/decode를 비교하면 `</think>`에서 바로 드러난다.
- S4 D: `add_tokens()`가 명시 id를 재현할 수 없을 때 오류를 내면 막힌다.
- S5 –: 오히려 선언에서 생성하는 범용 백엔드로 강제한 것이 원인이었다. 생성기가 선언을 표현하지 못하면 S4가 함께 있어야 한다.

**10. transformers #44448**
- S1 –: 예약 id 103칸 규약은 클래스 코드의 기본값이다. 체크포인트에 비교할 선언 필드가 있는지는 확인하지 못했다.
- S2 –: 적재 시점에 정해지고, 값에 붙은 태그가 없다.
- S3 T: 선언된 특수 토큰(mask, pad, eos)의 id를 원 SentencePiece와 Pegasus 규약으로 만든 기준과 비교하면 드러난다. 실제로는 v4와의 bisect로 찾았다.
- S4 –: 변환기가 모르는 규약은 "지원하지 않는 선언"으로 보이지 않는다. offset이 선언된 필드라면 D가 된다.
- S5 –: v5의 범용 변환(선언 → 토크나이저)이 모델 고유 규약을 잃은 것이 원인이다.

**11. llama.cpp #25734**
- S1 –: WM ≤ BM은 커널 내부의 타일 불변식이지, 경계를 넘는 값의 역할이 아니다.
- S2 –: 태그를 붙일 값이 없다.
- S3 –: 기존 test-backend-ops MUL_MAT이 subgroup 128 장치에서 이미 229건 중 2건 실패한다. 빠진 것은 역할 사례가 아니라 하드웨어다.
- S4 –: "warp ≤ 64" 전제가 선언된 적이 없어 규칙이 걸릴 곳이 없다. 지원 범위가 선언되어 있었다면 D다.
- S5 –: 타일 크기는 성능 조정 값이라 변형 선언에 들어가지 않는다.

**12. llama.cpp #25027**
- S1 –: 경계에서 어긋난 역할이 없다.
- S2 –: 한 셰이더 안의 값 오류라 태그로 보이지 않는다.
- S3 –: x == 0은 step 정의의 수치 경계이지 역할 경계가 아니다. 일반 경계값 테스트면 잡혔을 것이고, 기존 균등분포 입력은 0을 뽑지 않았다.
- S4 –: 버려진 선언 속성이 없다.
- S5 –: 한 셰이더 안의 비교 연산이다.

**13. llama.cpp #23574**
- S1 –: 인덱스 폭 넘침은 한 커널 안의 산술이다.
- S2 –: 어떤 역할 태그도 바뀌지 않는다.
- S3 –: 원소 2^31개를 넘는 마스크(긴 문맥 × 큰 ubatch)가 필요하다. RANGE 역할은 유효 길이를 다루지 인덱스 폭을 다루지 않는다. 극단 크기 스트레스 테스트라면 잡혔을 것이다.
- S4 –: 버려진 선언이 없다. 커널이 최대 크기를 선언했다면 D다.
- S5 –: 손으로 쓴 인덱스 계산이다.

**14. llama.cpp #23850**
- S1 –: 드라이버가 기능 지원을 잘못 알린 것이지, 선언끼리 어긋난 것이 아니다.
- S2 –: 태그로 보이지 않는 값 손상이다.
- S3 –: 기존 MUL_MAT_HADAMARD 테스트가 해당 장치에서 6건 모두 실패했다. 빠진 것은 플랫폼이다.
- S4 –: 선언된 기능이 버려진 것이 아니라 거짓이었다.
- S5 –: 생성된 커널도 같은 subgroup 명령을 쓴다.

**15·16. llama.cpp #21887, #21855**
- S1 –: 플랫폼이 가능하다고 답한 P2P가 실제로 데이터를 손상했다.
- S2 –: 역할 태그는 손상된 바이트를 보지 못한다.
- S3 –: 특정 메인보드·BIOS·IOMMU 구성에서만 나타나 CI 테스트 대상이 아니다.
- S4 –: 버려진 선언이 없다. 관리자도 검출할 방법이 없다고 했다.
- S5 –: 해당 없음.

**17. llama.cpp #21648**
- S1 –: 드라이버 API 구현 결함이다.
- S2 –: 복사 결과의 바이트 손상은 태그로 보이지 않는다.
- S3 –: Windows HIP와 peer 미지원 다중 GPU 구성에서만 나타난다.
- S4 –: 문서상 peer 접근 없이도 동작해야 하는 API가 틀렸다. 버려진 선언이 없다.
- S5 –: 해당 없음.

**18. llama.cpp #18452**
- S1 P: tokenizer.json의 `Whitespace` pre-tokenizer와 `Lowercase` normalizer를 역할로 두면, 바이트 수준 BPE 소비자와 묶는 변환 시점에 거부된다.
- S2 –: 변환 시점에 이미 정해지고 id에 태그가 없다.
- S3 T: 선언에서 나온 입력(한자, 대문자)을 HF 토크나이저와 비교하면 드러난다.
- S4 D: 지원하지 않는 pre-tokenizer를 `gpt2`로 조용히 바꾸지 않고 변환 오류를 내면 막힌다. 관리자도 "지원하지 않는 토크나이저"라고 했다.
- S5 P: 선언된 파이프라인에서 토크나이저를 생성하면 막힌다. 수정(#18756)도 선언 사실을 GGUF 메타데이터로 옮기는 형태였다.

**19. llama.cpp #21893**
- S1 P: 제자리 재배치를 가중치 배치 역할의 전이로 선언하면, 재배치 인식 판독기가 없는 GEMM 경로가 정적으로 거부된다.
- S2 D: GEMM dequantizer가 읽을 때 가중치 버퍼의 "재배치됨" 태그를 검사하면 첫 잘못된 읽기에서 잡힌다.
- S3 T: 같은 가중치에 tg 다음 pp를 돌리는 순서를 양자화 형식마다 CPU 기준과 비교하면 드러난다. Intel GPU CI가 필요하다.
- S4 D: reorder 표지를 모르는 판독기가 오류를 내면 막힌다.
- S5 P: 재배치 변형을 한 번 선언하고 모든 판독기를 생성하면 빠진 판독기가 생길 수 없다.

**20. llama.cpp #20423**
- S1 P: view의 stride 역할과 커널의 "연속 행" 입력 역할이 그래프를 구성할 때 어긋난다.
- S2 D: SYCL l2_norm이 비연속 태그를 검사하면 잡힌다.
- S3 T: LAYOUT 역할에서 비연속 입력 사례를 만들면 드러난다. 실제로 상류가 이런 사례를 추가하자 SYCL이 바로 실패했고, 그것이 #20283의 출발점이었다.
- S4 D: `supports_op`가 비연속 입력을 확인하지 않고 true를 돌렸다. 규칙이 있었다면 오류나 CPU 배정이 된다.
- S5 P: 커널이 실제로 지원하는 배치와 `supports_op` 메타데이터를 한 선언에서 생성하면 거짓 지원 신고가 없어진다.

## 4. 접근별 합계

| 접근 | 20건 전체 | 재분류된 12건 중 | 비역할로 남은 8건 중 (N1 3, N2 5) |
|---|---:|---:|---:|
| S1 정적 역할 타입 (P) | 11 (부분 1) | 11 | 0 |
| S2 실행 중 역할 검사기 (D) | 4 (부분 1) | 4 | 0 |
| S3 역할 기반 테스트 생성 (T) | 13 | 12 | 1 (#31011) |
| S4 조용한 누락 금지 (D) | 12 (부분 1) | 11 | 1 (#31011) |
| S5 선언형 변형 명세 (P) | 10 (부분 1) | 10 | 0 |
| 하나라도 도움 | 13 | 12 | 1 |

- #28685에서 S4가 잡는 것은 역할 쪽 결함이 아니라 컴파일러 플래그 누락(비역할)이다.
- 기준을 넓혀 커널의 능력 한계(최대 인덱스 폭, 지원 subgroup 크기)까지 선언한다고 보면 세 곳이 바뀐다. #23574(S3 T, S4 D), #25734(S4 D), #44448(offset이 선언 필드라면 S1 P, S4 D)이다. 이때 S1은 12, S3는 14, S4는 15다.
- S1과 S4는 PROPERTY 계열에서 같은 사실을 같은 시점(적재)에 잡으므로 겹친다.

## 5. 재분류: 20건 중 12건을 역할 계열로 본다

| 확신 | 이슈 | 어휘 | 이유 |
|---|---|---|---|
| 확실 (9) | transformers #45812, #45920, #46489, #46710, #44779 | PROPERTY | 체크포인트가 tokenizer.json에 선언한 파이프라인을 손으로 쓴 클래스가 조용히 덮어썼다. 수정 PR과 보고자가 이를 명시했다. vLLM #51063(설정이 체크포인트의 실제 lm_head를 이김), transformers #44671(설정의 tie 누락)을 R1로 센 것과 같은 모양이다. |
| | transformers #45356 | MAPPING | 체크포인트의 명시 id가 범용 백엔드의 순번으로 바뀌었다(PR #45359 본문). |
| | transformers #44448 | MAPPING | 관리자는 "added tokens" 매핑 문제라고 했고, 수정은 Pegasus의 예약 id 배치를 변환에 넣었다. 확신은 중간이다. |
| | llama.cpp #18452 | PROPERTY | 선언된 `Whitespace`/`Lowercase`를 변환기가 읽지 않았고, 수정은 그 사실을 GGUF로 옮겼다. 1차 노트도 "경계 R1"이라고 적었다. |
| | SGLang #28685 | LAYOUT (+N2) | 수정 작성자가 "SGLang-side scale-layout contract"라고 불렀고, 컴파일러 수정만으로는 GSM8K가 0/82였다. 역할 결함이 필요한 공동 원인이다. |
| 추정 (3) | vLLM #48058 | LAYOUT | 스레드에는 "0.1.11에서 해결"만 있다. 다만 reporter 명령은 compressed-tensors의 XPU 기본값("otherwise default to W8A16")에 따라 W8A16 op로 갔고, 0.1.10.1→0.1.11에서 2차원 FP8 GEMM 동작을 바꾼 변경은 #440 하나뿐이다. #440 본문은 같은 증상(RedHatAI FP8-dynamic 체크포인트)을 적었다. 수정은 의존 패키지에 있지만 그 패키지는 vLLM 프로젝트의 자체 커널이고, 결함은 두 쪽 규약의 불일치다. |
| | llama.cpp #20423 | LAYOUT | 보고자가 #20283 전후를 비교했고, b8277 코드에서 stride를 무시하는 L2_NORM과 Qwen3.5의 strided q/k view를 확인했다. 후속 불안정은 원인 불명이다. |
| | llama.cpp #21893 | LAYOUT | 빌드 날짜, 재배치를 끄는 우회책, 관리자가 가리킨 #21638, 같은 Gemma 4 계열이 #21527 뒤 깨졌다는 보고가 모두 같은 방향이다. 보고자 확인은 없다. |

- 재분류하지 않은 경계 사례: **SGLang #31011**. 선언된 reduce op가 조용히 바뀐 것은 R1 모양이다(역할로 센다면 REDUCTION). 그러나 결함 전체가 벤더 collective 안에 있고, 그 구성 요소만 떼어 테스트해도 틀린다. 그래서 N2로 둔다. vLLM 조사처럼 의존성 안의 기전도 R(§)로 센다면 13건이 된다.
- 유지: N1 #25734, #25027, #23574, N2 #23850, #21887, #21855, #21648. #21648은 어디에도 수정이 없으므로 1차 규칙상 N4로 옮길 여지가 있다.
- 같은 결함을 한 번씩 세면 20건은 15개 결함이다. 토크나이저 라우팅 5건이 하나, P2P 2건이 하나다. 이렇게 세면 역할 계열 8개, 비역할 7개다.
- 재분류가 생긴 이유는 두 가지다. N3은 기전이 아니라 영역(토크나이저)으로 정해졌다. N2는 수정이 놓인 위치(패키지 갱신, 툴체인)로 정해졌는데, 그 수정 안에 규약 변경이 들어 있었다(vllm-xpu-kernels #440, SGLang #29275).

## 6. 관찰

검토 뒤에도 비역할로 남은 8건 중 다섯 접근이 조금이라도 돕는 것은 #31011(S3, S4) 하나뿐이다. 반면 #25734, #23850, #31011은 해당 하드웨어에서 기존 테스트나 10줄 재현만 돌려도 바로 드러났으므로, 접근들은 역할 계열에 특이적이고 비역할 버그의 병목은 역할 정보가 아니라 플랫폼 커버리지다. 비역할로 분류된 20건 중 12건(결함 단위로는 15개 중 8개)이 기전상 역할 계열이어서 1차 비율은 분류 규칙에 따라 과소 추정되어 있다. 그러나 "선언된 설정이 조용히 무시됨"을 모두 R1로 세면 정의가 거의 모든 설정 버그를 삼킬 수 있으므로, 비율을 다시 내기 전에 네 저장소에 같은 규칙(기전 우선, 의존성 안의 기전 처리)을 먼저 정해야 한다. S3와 S4만 역할 계열 밖(oneCCL의 조용한 SUM, aiter가 빼 버린 컴파일 플래그)까지 닿고, S2는 배치형 결함 4건에서만 쓸모가 있으며, S5는 선언이 완전하면 막지만 생성기가 선언을 표현하지 못하면 오히려 원인이 되므로(#45356, #44448) S4와 짝지어야 한다.

## 7. 한계와 원자료

- 한 에이전트가 판정했고 교차 검증은 없다. 표시는 접근을 도입했다고 가정한 판단이며, 토크나이저 건의 S1·S4는 역할 규율이 체크포인트→토크나이저 경계까지 확장된다고 가정했다.
- #48058과 #440의 연결은 소스 비교로 추론한 것이다. #21893은 보고자 확인이 없다. #20423은 1단계 증상만 설명한다. #44448의 offset이 체크포인트 선언 필드인지는 GitHub 밖(Hugging Face Hub)이라 확인하지 않았다.
- 원자료(이슈·PR JSON, 비교 결과, 소스 사본)는 `C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Desktop------ai-compiler\27a9157c-9e06-4d5e-8615-c84cc90eb8bf\scratchpad\replay_control\`에 있다. 세션이 끝나면 지워질 수 있다.
