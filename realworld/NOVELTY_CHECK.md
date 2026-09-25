# 선행 확인: 이미 보고되거나 해결된 문제인가

> **2026-09-23 검증.** 인용한 사실 88개를 다시 확인했다(`reinvestigation/audit/prior_work_audit.json`, 도중 저장본). 확인됨 62, 일부 다름 25, 틀림 0이다. 중요한 정정은 `THEORY.md` 4.4절에 있다. Anthropic 사후 분석의 두 사건을 섞었고, gpt-oss 36.7%는 압축 모델의 점수이며, '선언 방식은 합산 상태 하나이고 2026년에 시작'은 틀렸다. 계약 제안도 일부는 채택됐다(vLLM #28454).

- 날짜: 2026-09-23
- 연구자의 질문:
  - 이 문제의식과 해법이 이미 보고되었거나, 다른 사람이 이미 해결한 것은 아닌가?
  - 이미 해결된 논쟁이라면 새로 풀어도 아무도 원하지 않는다.
  - 이슈성도 있어야 한다.
- 방법:
  - 네 갈래를 동시에 조사했다.
    - 학술 연구
    - 텐서 의미 타입의 역사
    - 공개 사건과 업계 대응
    - 엔진 내부 기능과 제안(RFC)
  - 각 갈래에는 반대 입장을 지시했다. "새롭지 않다는 증거를 가장 강하게 찾으라"는 것이다. 새롭다는 결론 쪽으로 기우는 것을 막기 위해서다.
  - 판정을 좌우하는 출처는 직접 다시 열어 확인했다.
- 표시:
  - [직접]: 2026-09-23에 직접 열어 확인한 출처
  - [에이전트]: 조사 에이전트가 열어 보고했고, 내가 다시 열지는 않은 출처
  - [전날]: 2026-09-22 `prior_work.md` 작성 때 확인한 출처
- 인용문은 쓰지 않고 모두 풀어 썼다. 조사 에이전트가 받아 둔 소스 파일은 세션 임시 폴더에 있었고, 영구 보관하지 않았다.

## 0. 판정

1. **문제가 존재한다는 것은 이미 알려져 있다.** 2025~2026년에 학계와 업계가 모두 다룬다.
   - 모델과 커널 사이의 가정이 명시되지 않는다는 진단: M2K
   - HF와 vLLM 구현의 차이를 자동으로 찾는 도구: Emerge
   - 조용한 오류의 진단: Ekka
   - 업체별 정확도 지표: Artificial Analysis

   "이런 결함이 있다"는 주장만으로는 새롭지 않다.
2. **그러나 해결된 논쟁은 아니다.**
   - 대응은 거의 모두 틀린 뒤에 찾는 사후 탐지다. 평가 점수, 참조 구현과의 비교, 업체 검증기, 품질 라우팅이 여기에 속한다.
   - 선언해서 미리 막는 방식은 합산 상태라는 사실 하나에만 있다. 그것도 학습 코드에서 2026년에 막 시작됐다(JAX, Meta spmd_types).
   - 추론 엔진의 배관 경계 전체에 필수 선언과 검사를 두는 시도는 찾지 못했다.
   - 엔진 안의 검사는 조각나 있고, 대부분 켜야 동작하거나 CI에서만 돈다.
3. **새롭다고 말할 수 있는 부분은 좁아졌다.**
   - (a) 사실 종류로 나눈 분류와 그 비중
   - (b) 아무도 담지 않는 사실: 위치 기준, 유효 범위와 커널 마스크의 대조, 모델 성질과 커널 능력의 대조, 버퍼 시점, 무시된 인자
   - (c) 적재부터 커널까지 이어지는 필수 선언
   - (d) 같은 실제 버그 묶음에서 사후 탐지와 나란히 잰 평가
4. **위험 두 가지가 커졌다.**
   - 참조 구현과 비교하는 도구가 선언 없이도 같은 부류를 상당수 잡는다. Emerge는 알려진 13건 중 10건을 잡았고, 새로 8건을 찾았다.
   - 선택형 의미 타입은 채택되지 않은 역사가 있다. transformers의 엄격 적재 제안도 2026-09-22에 닫혔다.
5. **이슈성은 조건부로 있다.**
   - "모델이 멍청해졌다"는 사건에는 반응이 크다.
   - 원인이 된 결함은 PR 안에서 조용히 고쳐져 대중에게 보이지 않는다. 둘을 잇는 사람이 없다.
   - 큰 전후 수치가 있는 실제 결함을 새로 찾아내는 것이 가장 강한 이슈가 된다.

## 1. 문제 쪽: 이미 알려진 것

| 출처 | 내용 | 이 연구와의 관계 | 표시 |
|---|---|---|---|
| M2K, arXiv 2603.24595 (v1 2026-03-06, v2 2026-08-29) | 모델과 CUDA 커널 사이 인터페이스가 암묵적이라 양쪽의 가정이 어긋난다고 진단한다. 실행을 추적해 인터페이스를 추론하고 커널을 기호 실행한다. 새 버그 181건, 오탐 9건 | 진단이 같다(가정이 명시되지 않는다). 대상은 커널의 메모리 안전이고, 방법은 선언이 아니라 추론이다. 에이전트는 SOSP'26 채택이라고 했지만 초록 페이지에서는 확인하지 못했다 | [직접] |
| Emerge, arXiv 2603.21851 (2026-03-23) | HF Transformers와 vLLM의 계산 그래프가 같은 계산인지 검증한다. 관계를 추론하고 SMT와 무작위 시험으로 확인한다. 알려진 결함 13건 중 10건을 잡고, 새로 8건을 찾았다(개발자 확인). 사용자 선언은 필요 없다 | 이 연구의 결함 부류를 찾는다. 에이전트가 본 예: 무시된 설정 필드, RoPE 방식과 우선순위, 슬라이딩 윈도, TP 가중치 분할. 가장 강한 경쟁 수단이다 | [직접] 초록, 결함 예는 [에이전트] |
| Ekka, arXiv 2606.04594 | vLLM·SGLang의 조용한 오류 90건. 참조 구현과 비교해 진단한다 | 이미 알던 연구다. 예방 설계는 제안하지 않는다 | [전날] |
| The Foundation Cracks, arXiv 2506.12320 (2025-06-14) | Transformers·vLLM의 버그 수정 313건. 최다 원인은 API 오용(32.17~48.19%)이고, 테스트 부족으로 빠져나간다 | 인터페이스 쪽 결함이 주류라는, 같은 방향의 근거 | [직접] |
| Megatron-LM 이슈 #7452 (2026-09-17, ezyang) | 병렬화 조합에서 조용한 수치 오류 13건. spmd_types의 R/P/S 표기로, 어긴 계약을 적었다. 보고자는 전부를 직접 검증하지는 않았다고 밝혔다 | 합산 상태 사실의 결함이 학습 쪽에서 타입 검사로 발견되었다 | [직접] |
| vLLM 블로그 (2026-07-16) | v0.20.0의 회귀가 CI를 통과해 사용자에게 갔다. 정확도 회귀는 거의 충돌하지 않고, 응답은 정상인데 답이 틀린다고 설명한다 | 조용함을 업계가 확인한 사례 | [직접] |
| Artificial Analysis Endpoint Accuracy Index (2026-08-04) | 자체 참조 배포와 비교하는 지표. gpt-oss-120b의 도구 호출(BFCL)이 업체별로 22~37%. 업체는 양자화, 자체 커널, 설정 조정을 하고, 때로는 버그를 낸다고 적었다 | 업체별 차이가 공개 지표가 되었다 | [직접] |
| vLLM PR #16801 (2025-04-18 병합) | Llama 4 INT4의 스케일을 안전하지 않게 형 변환했다. INT4 Scout의 MMLU-Pro가 0.0286이었고, 수정 뒤 0.7168 | 저장·스케일 형식 사실의 조용한 결함. 전후 차이가 크다 | [직접] |

**찾지 못한 것:** 결함을 "어떤 사실이 부품 사이에서 사라졌는가"로 나누고 비중을 잰 연구. 기존 연구는 코드상의 직접 원인(API 오용, 설정, 타입)이나 결함이 있는 위치(프레임워크, 모델, 커널)로 나눈다.

## 2. 해법 쪽: 누가 무엇을 하고 있나

### 2.1 사후 탐지: 붐비는 영역

| 수단 | 누가 | 방식 | 표시 |
|---|---|---|---|
| 업체 검증기 | Moonshot의 K2VV와 KVV | 업체별 도구 호출 정확도와 순위를 공개 | [에이전트] |
| 품질 라우팅 | OpenRouter Exacto (2025-10). Auto Exacto는 2026-03부터 기본값 | 품질이 나쁜 업체를 피해 요청을 보냄 | [에이전트] |
| 업체 정확도 지표 | Artificial Analysis (2026-08) | 참조 배포와 비교 | [직접] |
| 엔진 CI 정확도 평가 | vLLM (2026-07) | 17개 모델·장비 조합에 GSM8K, GPQA, AIME, BFCL을 매일 밤 실행 | [직접] |
| 제출 정확도 기준 | MLPerf Inference | 참조 정확도의 99% 이상 등 | [에이전트] |
| 동등성 검증 | Emerge. Scalify(arXiv 2509.10694: 분산 그래프 검증, Amazon 프레임워크에서 새 버그 5건). GraphGuard | 참조 구현이나 단일 장치 그래프와 비교 | Emerge·Scalify [직접], GraphGuard [에이전트] |
| 커널 검증 | M2K, Triton-Sanitizer, Kernel Contracts(arXiv 2604.22032) | 커널 수준 | M2K [직접], 나머지 [에이전트] |
| 진단 | Ekka | 참조 구현과 중간값을 비교 | [전날] |

- **공통점:** 모두 틀린 뒤에 찾는다. 참조 구현이 있어야 하거나, 결함이 평가 데이터에서 드러나야 한다.
- **에이전트가 읽은 한계:**
  - Emerge는 데이터에 따라 달라지는 경로(MoE 라우팅 등)와 일부 윈도 결함을 놓쳤다.
  - Scalify는 그래프 밖의 결함(KV 캐시 자르기, 로짓 배치)을 보지 못한다.

### 2.2 선언으로 막기: 합산 상태 하나, 학습 쪽

- **JAX 샤딩 타입:** 아직 합산되지 않은 값(unreduced)을 타입에 넣고, 추적 시점에 검사한다. [전날]
- **Meta spmd_types** (저장소 생성 2026-03-30):
  - JAX의 방식을 PyTorch 학습 코드로 옮긴 타입 시스템이다. 표기는 R(복제), P(부분합), S(분할) 등이다.
  - 선택형으로 쓰는 것이 설계 목표다.
  - Megatron에서 결함 13건을 찾았다. [직접]
- **PyTorch DTensor:** Partial과 Replicate 표기가 있다. 아직 alpha이고, 즉시 실행 부담이 크다. transformers의 TP가 사용한다. [에이전트]
- **결론:** 이 연구의 REDUCTION 사실(실제 버그 50건 중 3건)은 이미 다른 팀들이 가는 길이다. 새로 만들지 않고 표기를 맞춘다.

### 2.3 엔진 안의 조각: 켜야 동작하거나 한 부분만

| 엔진 | 있는 것 | 한계 | 표시 |
|---|---|---|---|
| vLLM | 백엔드를 고를 때 sliding window, sinks 등을 검증한다. 층별 KV 명세 타입이 있다. 가중치 누락을 검사한다 | softcap은 선택 기준에 없다(일부 백엔드는 생성할 때 거부한다). 누락 검사는 양자화하지 않은 모델에만 한다. 묶음 가중치 불일치는 경고만 한다. 양자화 파라미터의 형식 정보는 재등록할 때 사라진다 | [에이전트, 코드 읽기] |
| SGLang | KV canary. KV 칸마다 토큰, 위치, 범위 태그를 달고 매 forward에서 검사한다. 실제로 SWA 위치 오염을 잡았다(#33656) | 켜야 동작한다(모드 NONE이면 꺼짐). KV 캐시만 다룬다 | 존재와 모드 [직접], 세부 [에이전트] |
| TensorRT-LLM | FP4 스케일 형식 태그(스위즐 여부)를 소비하는 쪽에서 assert한다 | 그 자료형에 한정된다 | [에이전트] |
| transformers | 적재 보고(누락 키, 예상 밖 키) | 경고만 하고, 엄격 적재 옵션이 없다. Gemma 2의 sdpa 경로는 softcap 인자를 받기만 하고 읽지 않는다 | softcap은 [직접](2026-09 main 코드), 나머지 [에이전트] |
| llama.cpp | test-backend-ops(연산마다 CPU 결과와 비교), GGUF 구조 검사 | 연산 단위 시험이다. SYCL 재배치 플래그는 검증하지 않는다 | [에이전트] |
| PyTorch | custom_op의 `mutates_args`가 필수다 | 틀리게 적어도 검사하지 않고, opcheck 시험에서만 확인한다 | [전날] |

### 2.4 제안과 그 결과: 논쟁의 현재 상태

| 제안 | 결과 | 표시 |
|---|---|---|
| transformers PR #48962: 선택형 엄격 적재 | 2026-09-22 닫힘. 유지보수자는 2년 된 요청이라 원하는지 모르겠다는 취지로 답했다 | [직접] |
| vLLM RFC #24384: 설정 스키마, 쓰인 필드와 안 쓰인 필드 분리 | 비활성으로 자동 종료(not_planned). 반대 논거는 없었다 | [에이전트] |
| vLLM RFC #32613: 정확도 시험 | 자동 종료 | [에이전트] |
| vLLM RFC #48312: 가중치 재적재 정확성 (2026-07-11) | 열려 있다. 일부 수정만 병합됐다 | 제목·상태 [직접] |
| PyTorch RFC #190792: 분산 집합 연산의 의미 정체성 (2026-07-22) | 열려 있고 검토가 없다 | 제목·상태 [직접] |
| SGLang #32432: CUDA Graph 메타데이터 계약 | 열려 있다. 댓글은 작성자 것뿐이다 | [에이전트] |

계약을 넣자는 제안은 계속 나오지만, 엔진들은 받아들이지 않았다. 적힌 이유는 호환성, 로그 비용, 필요성에 대한 의문이었다. 내용을 반박한 기록은 찾지 못했다.

### 2.5 시도되고 버려진 것

| 시도 | 결과와 이유 | 표시 |
|---|---|---|
| PyTorch named tensors | 2021년에 개발이 멈췄다(설계의 모호성, 인력 부족). 연산 지원이 부족했고, 이름 있는 텐서와 없는 텐서를 섞기 어려웠다. 2.13에서 제거됐다(부담과 코드 크기) | [에이전트], 제거 사실은 [전날] |
| JAX xmap | 2024년에 제거되고 shard_map로 대체됐다. 이름 축과 자동미분이 얽히는 문제를 피하려는 것이었다 | [에이전트] |
| Mesh TensorFlow | 저장소 보관 처리 | [에이전트] |
| ONNX 타입·차원 표기(denotation) | 권고 수준이고 실험 단계다. 검사기가 없다 | [에이전트] |
| NeMo Neural Types | 음성 쪽에만 남았다. LLM 쪽 컬렉션은 제거됐다 | [에이전트] |

- **버려진 것은 편의를 위한 축 이름이었다.** 버린 이유는 모호성, 지원 범위, 섞어 쓰기, 부담이었다. "버그를 못 잡아서"라는 이유는 없었다.
- **반대로 정확성을 위한 합산 상태 타입은 2024~2026년에 늘고 있다.**
- **경고 신호도 있다.** [에이전트]
  - vLLM의 TPU 백엔드는 JAX의 검사를 24개 파일에서 끈다.
  - 사람이 물리 단위를 맞게 적은 비율은 절반 정도였다(ASE'18 연구).

## 3. 이슈성

| 사건 | 사용자가 본 것 | 반응 | 표시 |
|---|---|---|---|
| ChatGPT, 2024-02 | 뜻 없는 문장. 원인은 특정 GPU 설정에서 커널이 틀린 결과를 낸 것 | HN 293점 | [에이전트] |
| Unsloth의 Gemma 결함 공개, 2024-03 | 미세조정 불안정. 원인은 GELU 근사, RoPE 정밀도 등 | HN 166점, 호평 | [에이전트] |
| Gemma 2 출시, 2024-06 | 27B 출력 이상. 원인은 softcap 누락 | HN 2점 | [에이전트] |
| vLLM의 Llama 4 INT4, 2025-04 | MMLU-Pro 2.9% | HN 글 없음 | 수치 [직접] |
| Anthropic 사후 분석, 2025-09 | 품질 저하 | HN 381점, InfoQ 기사 | [에이전트] |
| Moonshot KVV 블로그, 2026-04 | 업체별 차이 | HN 310점 | [에이전트] |
| Anthropic Claude Code 사후 분석, 2026-04 | 품질 저하(추론 계층이 아니라 제품 계층) | HN 942점, The Register, VentureBeat | [에이전트] |
| Artificial Analysis 지표, 2026-08 | 업체별 22~37% | HN 2점 | [에이전트] |

- 폐쇄형 모델 회사가 품질이 나빠졌다고 인정하면 반응이 크다. 다만 관심은 정직성 문제로 소비된다.
- 원인이 되는 결함(템플릿, 파서, 스케일, softcap, 정규화)은 공개 모델이 나올 때마다 반복된다. 그런데 PR 안에서 조용히 고쳐진다.
- 업계의 2025~2026년 대응은 탐지다. 검증기, 라우팅, 지표, CI 평가가 그것이다. 해결되었다고 주장하는 출처는 없었다.
- 가장 강한 이슈의 형태는 인기 엔진에서 새 결함을 찾아 큰 전후 수치로 보이는 것이다. Unsloth가 그 예다.

## 4. 이 연구의 주장 수정

**버릴 주장** (`RESEARCH_PLAN.md` 3.4절에 추가):
- 이런 결함이 존재한다는 것 자체가 새롭다.
- 텐서에 의미 타입을 붙이는 발상이 새롭다. ONNX(2017), NeMo, named tensors가 이미 있었다.
- 합산 상태를 타입으로 검사하는 것이 새롭다. JAX와 spmd_types가 이미 한다.

**남는 새로움:**
1. **분류와 비중.** 결함을 잃어버린 사실의 종류로 나누고 비중을 쟀다. 실제 버그 70건 중 역할 계열이 71%이고, 그중 92%가 조용했다.
2. **아무도 선언하거나 검사하지 않는 사실.**
   - 위치 기준
   - 유효 범위와 커널 마스크의 대조
   - 모델 성질과 커널 능력의 대조(softcap 등)
   - 버퍼 시점
   - 무시된 인자
   - 저장·스케일 형식을 적재부터 커널까지 잇는 것
3. **필수 선언과 실제 데이터 대조.** 선택이 아닌 필수 선언을 경계에서 실제 데이터와 대조한다.
4. **사후 탐지와 나란히 잰 평가.** 같은 실제 버그 묶음으로 잰다.
   - 어떤 결함은 평가 점수로 드러난다(Llama 4 INT4).
   - 어떤 결함은 참조 구현과의 비교로 드러난다(Emerge).
   - 어떤 결함은 둘 다 놓친다.
   - 선언 방식의 몫은 셋째 묶음과, 배포 전에 막는다는 시점에 있다. 이 몫의 크기가 이 연구의 핵심 수치가 된다.

**역사가 요구하는 답:**
- **지원 범위와 섞어 쓰기:** 모든 연산이 아니라 경계에서만 검사한다.
- **부담:** 끄면 비용이 0이어야 하고, 켜 둘 수 있을 만큼 싸야 한다.
- **틀린 선언:** 선언을 실제 텐서, 설정과 대조한다.
- **검사기의 건전성:** 검사기 자체의 오류를 시험한다.
- **켜 둘 이유:** 실제 결함을 찾아 보인다.
- **재사용:** 합산 상태는 spmd_types의 표기와 맞춘다.

## 5. 계획 변경

- **1단계:** 사례마다 기존 탐지 수단이 잡는지 함께 기록한다. 종단 평가 점수의 변화, 참조 구현과의 비교, SGLang KV canary, spmd_types가 대상이다.
- **2단계:** 합산 상태의 표기를 spmd_types와 맞추고, 새로 만들지 않는다.
- **평가 계획(`RESEARCH_PLAN.md` 6절):** 비교 대상에 네 가지를 더한다. Emerge·Ekka 방식의 참조 비교, vLLM 방식의 정확도 CI, KV canary, spmd_types다.
- **이슈성:** 새 공개 모델이 나온 직후, 업체 간 차이가 클 때 도구를 돌려 실제 결함을 찾는 시험을 둔다. 찾은 것은 연구자의 허락을 받은 뒤에만 상류에 보고한다.
- **상류 보고 후보:**
  - transformers에서 Gemma 2의 sdpa 경로가 softcap을 읽지 않는 문제가 2026-09 main 코드에도 남아 있다. sdpa 함수의 최근 수정은 2026-08-20이다.
  - 관련 이슈를 검색했지만 같은 보고는 찾지 못했다.
  - 실제 모델에서 영향을 잰 뒤 보고할지는 연구자가 정한다.

## 출처

- 논문
  - M2K: https://arxiv.org/abs/2603.24595
  - Emerge: https://arxiv.org/abs/2603.21851
  - The Foundation Cracks: https://arxiv.org/abs/2506.12320
  - Scalify(분산 그래프 검증. Graphcore의 동명 도구와는 다르다): https://arxiv.org/abs/2509.10694
  - GraphGuard: https://arxiv.org/abs/2508.09505
  - Kernel Contracts: https://arxiv.org/abs/2604.22032
  - The Silent Hyperparameter: https://arxiv.org/abs/2605.19537
  - TrainCheck: https://arxiv.org/abs/2506.14813
  - TrainVerify: https://arxiv.org/abs/2506.15961
  - TTrace: https://arxiv.org/abs/2506.09280
- 이슈, PR, 저장소
  - Megatron-LM #7452: https://github.com/NVIDIA/Megatron-LM/issues/7452
  - spmd_types: https://github.com/meta-pytorch/spmd_types
  - vLLM #16801: https://github.com/vllm-project/vllm/pull/16801
  - SGLang #33656: https://github.com/sgl-project/sglang/issues/33656
  - SGLang KV canary 코드: https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/kv_canary
  - transformers PR #48962: https://github.com/huggingface/transformers/pull/48962
  - vLLM RFC #24384, #32613, #48312: https://github.com/vllm-project/vllm/issues/24384 , https://github.com/vllm-project/vllm/issues/32613 , https://github.com/vllm-project/vllm/issues/48312
  - PyTorch #190792: https://github.com/pytorch/pytorch/issues/190792
  - SGLang #32432: https://github.com/sgl-project/sglang/issues/32432
  - transformers sdpa 경로: https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py
  - transformers Gemma 2 모델: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
- 블로그와 지표
  - vLLM 블로그: https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality
  - Artificial Analysis 지표: https://artificialanalysis.ai/articles/endpoint-accuracy-index
  - Moonshot KVV: https://www.kimi.ai/blog/kimi-vendor-verifier
  - OpenRouter Auto Exacto: https://openrouter.ai/blog/announcements/auto-exacto/
  - MLPerf 규칙: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
  - ezyang, DTensor erasure: https://blog.ezyang.com/2026/02/dtensor-erasure/
- 버려진 시도
  - named tensors: https://github.com/pytorch/pytorch/issues/60832 , https://github.com/pytorch/pytorch/pull/173895
  - shard_map 설계 문서: https://docs.jax.dev/en/latest/jep/14273-shard-map.html
  - ONNX 표기: https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md
  - 단위 표기 연구(ASE'18): https://jpwco.com/pdf/ase18main-p15-p-bcc79e2-37685-final.pdf
- HN 점수는 에이전트가 Algolia API(https://hn.algolia.com/api/v1/search)로 2026-09-23에 조회한 값이다.
