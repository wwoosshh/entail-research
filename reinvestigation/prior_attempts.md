# 선행 시도 재조사: 값의 의미를 타입처럼 선언·보존해 조용한 결함을 막는다는 목표

- 작성일: 2026-09-23
- 조사 대상 이론: 연구자 원문(2026-09-22~23)의 명제 1~5. 의미의 중립성, 손실, 필수 선언과 강제·보존, 라이브러리와 컴파일러 형태, 보조 진단.
- 방법
  - 공식 문서, 논문, 저장소, 공식 발표를 직접 열었다. GitHub은 `gh api -X GET`으로 읽기만 했다.
  - 이론을 지지하는 근거와 반박하는 근거를 같은 무게로 찾았다. 항목마다 두 칸을 따로 두었다.
  - 찾지 못한 것은 '확인 못 함'으로 적었다.
  - 인용은 15단어 이하, 원문 그대로다.
- 표기
  - 검사: 정적 / 실행 중 / 실제 데이터와 대조 / 없음
  - 경계 보존: '부품 안'은 한 함수나 한 프로그램 안에서만 뜻이 유지된다는 뜻이다. '파일로 이동'은 체크포인트나 설정 파일에 실려 부품 사이를 건넌다는 뜻이다. '끝까지'는 적재부터 커널까지 이어진다는 뜻이다.

## 0. 종합

(작성 중. 모든 범주를 마친 뒤 채운다.)

---

## 1. 텐서 타입과 의미 주석

### 1.1 표

| 이름 | 무엇의 의미 | 선언 방식 | 필수 | 검사 | 경계 보존 | 대상 | 상태 |
|---|---|---|---|---|---|---|---|
| jaxtyping | dtype, 축 이름과 크기 | 타입 주석 | 선택 | 실행 중(외부 검사기 위임) | 한 함수 호출 안 | 학습·연구 일반 | 활발 |
| torchtyping | dtype, shape, 축 이름 | 타입 주석 | 선택 | 실행 중 | 한 함수 호출 안 | PyTorch | 저자가 jaxtyping으로 이전 권고 |
| TensorAnnotations (DeepMind) | dtype, 축의 의미 표지(Time, Batch) | 타입 주석과 타입 스텁 | 선택 | 정적(pytype, mypy) | 부품 안 | JAX, TF | 2023-06 폐기, 보관 |
| PyTorch named tensors | 축 이름 | 텐서 속성 | 선택 | 실행 중(이름 대조와 전파) | 부품 안 | PyTorch | 2021 개발 중단, 2.13(2026-07)에서 제거 |
| functorch.dim (torchdim) | 축을 객체로 | 차원 객체 | 선택 | 실행 중 | 부품 안 | PyTorch | 실험 단계, 2025 파이썬 이식 |
| harvardnlp namedtensor | 축 이름 | 래퍼 | 선택 | 실행 중 | 부품 안 | PyTorch | 초안 종료 |
| TF labeled_tensor | 축 이름과 좌표 라벨 | 래퍼 | 선택 | 실행 중 | 부품 안 | TF 1.x | tf.contrib 정리 때 삭제 |
| xarray | 축 이름, 좌표, 속성 | 라벨 배열 | 선택 | 실행 중 | 부품 안(속성은 모호하면 버림) | 과학 데이터 | 활발 |
| einops, einx | 연산마다 축의 뜻 | 문자열 패턴 | 선택 | 실행 중(차원 수, 지정한 크기) | 연산 하나 | 학습·추론 코드 | 활발, 널리 쓰임 |
| Haliax | 축 이름(샤딩과 연결) | 이름 축 텐서 | Haliax 코드 안에서는 필수 | 실행 중(추적 시점) | 부품 안 | LLM 학습(JAX) | 활발(marin 저장소로 이동) |
| Penzai named axes | 축 이름 | 이름 축 배열 | 선택 | 실행 중 | 부품 안 | JAX 연구 | 저장소 보관 |
| NeMo Neural Types | 값의 뜻(LogitsType, AudioSignal 등)과 축 종류 | 모듈의 input_types/output_types | 모듈 안에서는 기본 켜짐, 끌 수 있음 | 실행 중(모듈 경계) | 모듈 경계 | 음성(ASR, TTS) | 음성 쪽만 남음. LLM 경로는 쓰지 않음 |
| RETURNN dim tags | 축 종류, 배치별 시퀀스 길이 | 차원 객체 | 권장 | 실행 중 | 부품 안 | 음성·번역 학습 | 활발(소규모) |
| Keras InputSpec | 입력의 차원 수, dtype, shape | 층의 속성 | 층마다 선택 | 실행 중(층 호출) | 층 경계 | 학습·추론 | 활발 |
| Dex | 인덱스 집합 타입 | 언어의 타입 | 필수(언어) | 정적 | 프로그램 안 | 연구 | 2024-01 이후 main 커밋 없음 |
| Hasktorch | 타입 수준 shape | 하스켈 타입 | 선택(두 API) | 정적 | 프로그램 안 | 연구 | 활발(소규모) |
| PyTea, ShapeFlow, TFP, tsalib | shape | 주석 없음 또는 주석 | 해당 없음 | 정적 분석, 추상 해석 | 프로그램 안 | 학습 코드 | 연구 시제품, 정지·보관 |
| PEP 646 | 배열 shape를 타입 인자로 | 언어 문법 | 선택 | 정적(검사기 몫) | 해당 없음 | 파이썬 일반 | Final(3.11) |
| Swift for TensorFlow | 언어 차원의 미분·정적 분석 | 새 언어 통합 | 해당 없음 | 정적 | 해당 없음 | 학습 | 2021-02 보관 |

### 1.2 항목별 설명

**jaxtyping**
- 무엇: 배열의 dtype과 shape를 `Float[Tensor, "batch channel"]` 같은 주석으로 적는다. 한 함수 호출 안에서 같은 이름의 차원이 같은 크기인지 확인한다. 이름은 isinstance 검사로 결속된다. https://docs.kidger.site/jaxtyping/api/runtime-type-checking/
- 검사: 실행 중. beartype이나 typeguard에 맡긴다. `@jaxtyped` 장식자나 import hook으로 켠다. JAX jit 안에서는 추적할 때만 검사하므로 실행 비용이 없다고 적혀 있다. 정적 shape 검사는 하지 않는다. (같은 출처)
- 상태: 활발하다. 별 1,873개, 마지막 push 2026-09-04. https://github.com/patrick-kidger/jaxtyping
- 빈틈: 축의 크기와 이름만 다룬다. 저장 형식, 합산 상태, 위치 기준, 유효 범위, 읽는 시점, 모델 성질은 담지 않는다. 주석은 코드에만 있고 체크포인트나 설정 파일로 건너가지 않는다.
- 지지 근거: 선언을 실제 값에 대조하는 설계다. 이론의 "선언을 실제 데이터로 검증"과 같은 방향이다.
- 반박 근거: 이 계열에서 살아남은 도구는 선택형이고 실행 중에만 검사한다. 필수 선언이나 정적 강제 쪽은 아래처럼 대부분 멈췄다.

**torchtyping**
- 무엇: shape, dtype, 축 이름을 주석으로 적고 실행 중에 검사한다. https://github.com/patrick-kidger/torchtyping
- 상태: README 첫 줄이 "Please use jaxtyping instead"이다. 이유로 정적 타입 검사기와 호환되지 않는다는 점을 든다. 마지막 push는 2025-05-02다. (같은 출처)
- 교훈: 개념이 실패한 것이 아니라 후계 도구로 옮겨 갔다.

**TensorAnnotations (Google DeepMind)**
- 무엇: dtype과 축의 의미 표지(Time, Batch, Height 등)를 타입으로 적고 pytype·mypy로 정적 검사한다. "semantic shape information"을 보존하는 타입 스텁을 제공했다. https://github.com/google-deepmind/tensor_annotations
- 상태: 저장소 보관(마지막 push 2023-07-07). README에 "no longer being maintained"라고 적혀 있다. (같은 출처)
- 버린 이유: 폐기 문서(Matthew Rahtz, 2023-06)에 네 가지가 적혀 있다. https://docs.google.com/document/d/1AAP-wq06j1TQwJPtrlky4lfyPHyl7-itgN5S47oZO98/edit
  - 공식 라이브러리 타입과 어울리지 않는다("Poor interoperability").
  - 기존 타입 스텁과 충돌하고, 라이브러리 전체의 스텁을 유지하는 부담이 크다.
  - 모노레포에서 간접 의존을 통해 원치 않는 곳까지 퍼진다("Cancerous behaviour in monorepos").
  - API 범위가 불완전하고 오버로드 조합이 폭발한다("Poor coverage").
  - 대안으로 jaxtyping을 권한다. 정적 검사를 버리고 실행 중 검사를 택하는 교환을 받아들였다.
- 이론에 주는 뜻: 정적·전면 적용형 의미 타입이 실패한 가장 분명한 1차 기록이다. 실패 이유는 "버그를 못 잡아서"가 아니라 상호운용, 유지 부담, 범위, 전염성이었다.

**PyTorch named tensors**
- 무엇: 축에 이름을 붙이고, 연산마다 이름을 대조하고 전파한다. 문서는 "The named tensor API is a prototype feature and subject to change."라고 적었다. 인덱싱, NN 모듈, JIT, 분산, ONNX 내보내기는 지원하지 않았다. https://docs.pytorch.org/docs/2.12/named_tensor.html
- 개발 중단: 2021-06-28 유지보수자(zou3519)가 설계의 모호성을 풀 인력이 없어 개발하지 않는다고 답했다. 사용자들은 기본 연산에서 오류가 나 "basically not usable"이라고 적었다. https://github.com/pytorch/pytorch/issues/60832
- 제거: PR #173895가 2026-06-01에 한 번 되돌려진 뒤 2026-06-04에 다시 병합됐다. PyTorch 2.13.0 릴리스 노트(2026-07-08)는 "long-deprecated prototype"을 "overhead and code bloat"를 줄이려고 완전히 제거했다고 적는다. https://github.com/pytorch/pytorch/pull/173895 , https://github.com/pytorch/pytorch/releases/tag/v2.13.0
- 이론에 주는 뜻: 선택형 축 이름은 연산 지원 범위가 좁고 이름 없는 텐서와 섞기 어려워 쓰이지 않았다. 없애는 이유로는 부담과 코드 크기가 적혔다. 버그를 못 잡았다는 기록은 없다.

**functorch.dim (torchdim)**
- 무엇: 차원을 문자열이 아니라 `Dim` 객체로 다룬다. einsum, 배치, 인덱싱을 같은 표기로 한다. https://github.com/pytorch/pytorch/blob/main/functorch/dim/README.md
- 상태: 2025-09 ezyang이 C++ 구현을 파이썬으로 옮겼다(PR #160236, 병합). https://github.com/pytorch/pytorch/pull/160236
- 빈틈: 축의 정체성만 다룬다.

**harvardnlp namedtensor ("Tensor Considered Harmful")**
- 무엇: 이름 기반 축 접근, 집합 연산식 브로드캐스트, 연산의 이름 명세(`.spec`)로 실행 중 검사를 한다. https://github.com/harvardnlp/namedtensor
- 상태: README가 초안 구현은 끝났다고 하고, PyTorch 코어 구현을 쓰라고 안내한다. 그 코어 구현은 위처럼 제거됐다. 마지막 push 2022-07-29.

**TensorFlow labeled_tensor**
- 무엇: "semantically meaningful dimension and coordinate labels"를 텐서에 붙이는 라이브러리. https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/labeled_tensor/README.md
- 결과: tf.contrib 정리 RFC(Accepted, 2019-04-09 갱신)에서 labeled_tensor의 처리는 "delete"다. https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md

**xarray**
- 무엇: 차원 이름, 좌표, 속성을 배열에 붙인다. 문서는 이것이 "less error-prone" 경험을 준다고 쓴다. https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html
- 속성 보존: `keep_attrs`의 기본값은 "attrs should only be kept in unambiguous circumstances"다. 연산 결과의 뜻이 모호하면 메타데이터를 버린다. https://docs.xarray.dev/en/stable/generated/xarray.set_options.html
- 이론에 주는 뜻: 메타데이터로 달아 둔 의미가 연산을 거치며 사라지는 실제 설계 사례다. 명제 1을 지지한다.

**einops, einx**
- 무엇: 연산마다 입력과 출력의 축 구조를 패턴으로 적는다. README는 "einops focuses on interface: what is the input and output"라고 쓰고, 주석과 달리 패턴은 검사된다고 설명한다. 지정한 축 크기가 맞지 않으면 실행 중에 실패한다. https://github.com/arogozhnikov/einops
- 상태: einops 별 9,604개. einx는 ICLR 2026 oral 논문. https://github.com/fferflo/einx
- 빈틈: 연산 하나의 축 구조만 다룬다. 값의 뜻은 없다.
- 이론에 주는 뜻: 널리 채택된 이유가 검사 자체보다 표기의 편의라는 점이 README 구성에서 보인다. 검사를 편의에 묶어 파는 방식이 채택에 유리하다는 간접 근거다.

**Haliax**
- 무엇: JAX에서 이름 축 텐서로 신경망을 짠다. 이름 축을 샤딩(FSDP, TP)과 연결한다. Levanter의 기반이고 70B 파라미터 학습까지 썼다고 README가 적는다. https://github.com/marin-community/haliax
- 상태: 개발이 marin 모노레포로 옮겨졌다(같은 출처).
- 이론에 주는 뜻: 이름 축을 LLM 규모 학습에 실제로 쓴 사례다. 다만 추론 엔진과 체크포인트 경계로는 확장되지 않았다.

**Penzai named axes**
- 무엇: JAX 함수를 이름 축으로 벡터화하는 경량 체계(`pz.nx`). https://github.com/google-deepmind/penzai
- 상태: 저장소 보관. 마지막 커밋 2025-06-22. 보관 이유는 확인 못 함.

**NeMo Neural Types**
- 무엇: 값의 뜻(로짓, 로그 확률, 오디오 신호, 임베딩 등)과 축의 종류, 선택적으로 차원 수를 타입으로 적는다. 목적은 "catch semantic and dimensionality errors during model creation"이다. https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/core/neural_types.html
- 검사: `@typecheck()` 장식자가 모듈의 입력·출력 타입을 실행 중에 검사한다. 이 장식자가 붙은 함수는 모든 인자를 키워드로만 받는다. 전역 스위치 `_TYPECHECK_ENABLED = True`가 기본이고 `disable_checks()`로 끌 수 있다. https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/core/classes/common.py
- 쓰임: 분리 전 마지막 릴리스 v2.7.3에서 ASR의 `ctc_models.py`에는 typecheck·NeuralType이 13번 나오고, LLM의 `gpt/model/base.py`에는 한 번도 나오지 않는다. https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/asr/models/ctc_models.py , https://github.com/NVIDIA-NeMo/Speech/blob/v2.7.3/nemo/collections/llm/gpt/model/base.py
- 상태: 저장소가 2026년에 음성 중심으로 바뀌었다(README "pivoted to focus on audio, speech"). https://github.com/NVIDIA-NeMo/Speech
- 이론에 주는 뜻: 이론의 형태(값의 뜻을 타입으로, 모듈 경계에서, 키워드 인자 필수, 기본 켜짐)와 가장 닮은 기존 시도다. 같은 회사의 LLM 경로로는 옮겨지지 않았다. 옮기지 않은 이유를 적은 공식 문서는 찾지 못했다(확인 못 함).

**RETURNN dimension tags**
- 무엇: `Dim` 객체가 축 종류(배치, 공간, 특징)와 동적 크기를 가진다. 가변 길이 축은 배치 항목별 시퀀스 길이를 함께 가진다. 연산은 위치가 아니라 논리 축으로 맞춘다. 문서는 명시적 `Dim`을 "the recommended way"라고 적는다. https://returnn.readthedocs.io/en/latest/getting_started/data.html
- 상태: 활발(마지막 push 2026-09-22, 별 377개). https://github.com/rwth-i6/returnn
- 이론에 주는 뜻: 유효 범위(시퀀스 길이)를 축과 함께 들고 다니는 설계가 이미 있다. 이 연구가 "아무도 담지 않는 사실"로 꼽은 유효 범위에 대한 반례다. 다만 학습 프레임워크 안에 한정된다.
- 참고: PyTorch 이슈 #60832에서 RETURNN 개발자가 이 설계를 named tensor의 대안으로 소개했다. https://github.com/pytorch/pytorch/issues/60832

**Keras InputSpec**
- 무엇: 층이 입력의 차원 수, dtype, shape를 선언하고, 층을 호출할 때 첫 인자를 검사한다. https://github.com/keras-team/keras/blob/master/keras/src/layers/input_spec.py
- 빈틈: 형태만 다룬다. 값의 뜻은 없다.

**Dex**
- 무엇: 배열 인덱스 집합을 타입으로 다루는 연구 언어. https://github.com/google-research/dex-lang
- 상태: main의 마지막 커밋이 2024-01-11이다(같은 저장소 커밋 기록).

**Hasktorch**
- 무엇: 하스켈에서 텐서와 신경망을 다룬다. 타입 수준 shape를 쓰는 API가 있다. https://github.com/hasktorch/hasktorch
- 상태: 활발(마지막 push 2026-08-19), 별 1,215개.

**정적 shape 검사기: PyTea, ShapeFlow, Tensors Fitting Perfectly, tsalib**
- PyTea: 모든 실행 경로를 따라 shape 제약을 모아 위반을 찾는다. 공식 PyTorch 예제와 StackOverflow 코드에서 몇 초 안에 찾았다고 한다. https://arxiv.org/abs/2112.09037 . 저장소의 마지막 push는 2022-04-26. https://github.com/ropas/pytea
- ShapeFlow: shape만 계산하는 추상 해석기. 52개 프로그램에서 오탐 0, 미탐 1. https://arxiv.org/abs/2011.13452
- Tensors Fitting Perfectly: Swift 프로그램의 shape 오류를 SMT로 찾는다. README가 "highly experimental"이라 적고, 저장소는 보관됐다. https://github.com/google-research/swift-tfp
- tsalib: shape 주석 라이브러리. 마지막 push 2020-05-18. https://github.com/ofnote/tsalib
- 이론에 주는 뜻: shape 오류는 흔하다는 동기로 여러 도구가 나왔지만 연구 시제품에 머물렀다.

**PEP 646 (Variadic Generics)**
- 무엇: 배열 shape를 타입 인자로 쓸 수 있게 하는 문법. 동기에 "The shape of variables is often just as important" 문장이 있다. Status Final, Python 3.11. 저자에 TensorAnnotations의 Matthew Rahtz가 있다. https://peps.python.org/pep-0646/
- 한계: 차원 산술(예: N에서 2*N)은 범위 밖이라고 명시한다(같은 출처).

**Swift for TensorFlow**
- 무엇: 미분 가능 프로그래밍과 컴파일러 분석을 언어에 넣은 차세대 ML 플랫폼 실험. https://github.com/tensorflow/swift
- 상태: 2021-02 보관. 미분 기능은 공식 Swift 컴파일러에 남았다고 README가 적는다. 공식 중단 이유는 README에 없다(확인 못 함).
- 이론에 주는 뜻: 새 언어로 ML 스택을 바꾸려는 대형 시도도 채택되지 못했다. 로드맵 4단계(새 문법)의 위험을 보여 준다.

### 1.3 범주 소결

- 이 범주의 시도는 거의 모두 축(shape, 축 이름)만 다룬다. 값의 뜻을 타입으로 적은 것은 NeMo Neural Types와 TensorAnnotations의 의미 표지 정도다.
- 살아남은 것(jaxtyping, einops, xarray, Keras InputSpec)은 선택형이고 실행 중에 검사한다. 정적·전면 적용형(TensorAnnotations, named tensors, labeled_tensor)은 폐기되거나 제거됐다.
- 버린 이유로 적힌 것은 상호운용, 연산 지원 범위, 이름 없는 텐서와 섞기, 유지 부담, 코드 크기, 설계 모호성이다. 버그를 못 잡아서 버렸다는 기록은 찾지 못했다.
- 부품 경계(체크포인트, 설정, 엔진, 커널)를 건너 끝까지 보존하는 것은 이 범주에 없다.

---

## 2. 분산·합산 상태 타입

### 2.1 표

| 이름 | 무엇의 의미 | 선언 방식 | 필수 | 검사 | 경계 보존 | 대상 | 상태 |
|---|---|---|---|---|---|---|---|
| JAX explicit sharding | 배열의 샤딩, 합산 대기(unreduced) | 배열 타입의 일부 | set_mesh를 쓰면 기본값이 Explicit | 추적 시점(정적) | jit 프로그램 안 | 학습·추론(JAX) | 활발 |
| JAX shard_map check_vma | 값이 메시 축마다 달라지는지(varying) | 중간값의 타입 | 기본 켜짐, 끌 수 있음 | 추적 시점 | shard_map 안 | 학습·추론(JAX) | 활발 |
| JAX xmap | 이름 축과 병렬화 | 이름 축 | 선택 | 추적 시점 | 부품 안 | JAX | 0.4.31(2024-07-29)에서 삭제 |
| PyTorch DTensor | 배치 상태 Shard, Replicate, Partial | 텐서 하위 클래스 | 선택 | 실행 중(연산마다 전파) | 부품 안(체크포인트로 이어짐) | 학습, 일부 추론(TP) | alpha. 즉시 실행 부담이 큼 |
| Meta spmd_types | R/P/S, 기울기 합산 대기 | 타입 단언, 경계 계약 | 선택(설계 목표) | 실행 중 타입 검사(가짜 프로세스 그룹 가능) | 모듈 경계 계약 권장 | LLM 학습 | 2026-03 생성, 활발 |
| OneFlow SBP | 분할, 복제, 부분합 | 텐서의 SBP 서명 | 전역 텐서에서 필수 | 실행 중 자동 변환(boxing) | 프레임워크 안 | 학습 | 활동 감소(main 마지막 커밋 2025-08) |
| GSPMD, Shardy | 샤딩 | 컴파일러 IR 주석과 전파 | 일부 텐서만 | 컴파일 시점 | 컴파일러 IR 안 | 학습·추론(XLA) | Shardy가 GSPMD를 대체 중 |
| Mesh TensorFlow | 이름 차원과 메시 배치 | 이름 차원 | 필수(라이브러리 안) | 그래프 구성 시점 | 부품 안 | 학습(TF) | 저장소 보관 |
| PyTorch RFC-0057 (이슈 #190792) | 집합 연산의 목적(FSDP 등) | 생산자가 준 메타데이터 | 제안 | 해당 없음 | 컴파일러 변환을 건너 보존하자는 제안 | 학습(Inductor) | 2026-07 열림, 검토 없음 |
| vLLM TPU 백엔드 | (반대 사례) | shard_map check_vma=False | 해당 없음 | 검사 끔 | 해당 없음 | LLM 추론 | 24개 파일에서 끔 |

### 2.2 항목별 설명

**JAX explicit sharding ("sharding in types")**
- 무엇: 배열의 샤딩이 타입의 일부다. `f32[8@X,4@Y]`처럼 적히고 jit 안에서도 `jax.typeof`로 물을 수 있다. 결과 샤딩이 모호한 연산은 조용히 기본값을 쓰지 않고 오류를 내 사용자에게 `out_sharding`을 요구한다. https://docs.jax.dev/en/latest/parallel.html
- 기본값: 문서 원문 "By default, all mesh axis types are `AxisType.Explicit`." https://github.com/jax-ml/jax/blob/main/docs/parallel.md
- 합산 상태: 배열이 메시 축에 대해 unreduced일 수 있다. 문서는 논리값이 "equals the distributed sum of the physical shards' values"라고 정의한다(같은 출처).
- 빈틈: 합산 상태와 샤딩만 다룬다. 한 jit 프로그램 안의 타입이고, 체크포인트나 다른 엔진으로 건너가지 않는다.
- 지지 근거: 합산 상태라는 역할 사실을 타입에 넣고 모호하면 오류를 내는 설계가 주류 프레임워크의 기본값이 됐다. 명제 3의 방식이 한 사실에서는 채택됐다는 뜻이다.
- 반박 근거: 새로움 주장에 대해서는 반박이다. 합산 상태 타입은 이미 있다.

**JAX shard_map check_vma**
- 무엇: shard_map 안의 모든 중간값에 "varying manual axes" 타입을 단다. `out_specs`가 복제를 약속했는데 코드가 보장하지 않으면 예외를 낸다. 끄면 정의되지 않은 동작이 조용히 생긴다고 문서가 적는다. 기본값은 켜짐이다. https://docs.jax.dev/en/latest/notebooks/shard_map.html
- 자동미분: psum의 전치가 pvary라는 관계를 써서 불필요한 방어적 psum을 없앤다(같은 출처). 타입이 정확성과 성능을 함께 준다는 사례다.

**JAX xmap (삭제)**
- 무엇: 이름 축으로 벡터화와 병렬화를 표현했다.
- 결과: 0.4.31(2024-07-29)에서 삭제됐고 shard_map이 대체했다. https://github.com/jax-ml/jax/blob/main/CHANGELOG.md
- 이유: shard_map 설계 문서는 xmap이 "too powerful"하고 기능 조합을 추론하기 어렵다고 쓴다. 이름 축과 자동미분의 상호작용, 논리 shape와 장치별 버퍼 shape의 불일치도 든다. https://docs.jax.dev/en/latest/jep/14273-shard-map.html
- 교훈: 이름 축 자체보다 기능이 많아 생긴 복잡도가 문제였다. 기능을 좁힌 후계가 살아남았다.

**PyTorch DTensor**
- 무엇: Shard(dim), Replicate(), Partial(reduce_op)을 텐서에 붙인다. Partial은 "pending reduction"이다. 연산마다 배치가 전파되고 `redistribute()`로 바꾼다. 문서는 "currently in alpha state and under development"라고 적는다. https://docs.pytorch.org/docs/2.14/distributed.tensor.html
- 부담: ezyang은 DTensor의 즉시 실행 성능이 나쁘다고 쓰고, 종단 학습에서 35~60% 느려진 측정을 든다. 대안으로 타입을 검사에만 쓰고 실행은 일반 텐서로 하는 "DTensor erasure"를 제안했다(2026-02-01). https://blog.ezyang.com/2026/02/dtensor-erasure/
- 현장 반응: Megatron의 FSDP v2 설계 이슈는 DTensor를 빼고 일반 텐서에 spmd_types 주석을 달자고 제안한다. 이유로 "DTensor does not compose with torch.Tensor. Mixing the two in one op raises."를 든다. https://github.com/NVIDIA/Megatron-LM/issues/6901 . 같은 작성자가 v2가 v1보다 스텝당 약 18% 느리다고 보고했다. https://github.com/NVIDIA/Megatron-LM/issues/7264
- 이론에 주는 뜻: 뜻을 텐서 객체에 실어 끝까지 들고 다니는 방식(하위 클래스)은 섞어 쓰기와 실행 부담 때문에 현장에서 밀려난다. 지우고 실행할 수 있는 주석과 경계 검사 쪽으로 옮겨 가는 중이다. 이론 명제 4(성능을 크게 떨어뜨리지 않음)와 경계 검사 설계를 지지하고, 값에 뜻을 계속 달고 다니는 설계에는 경고가 된다.

**Meta spmd_types**
- 존재 확인: 저장소 생성 2026-03-30, 마지막 push 2026-09-18, 별 45개. 설명은 "based off of JAX's sharding in types, but adapted for the PyTorch ecosystem"이다. https://github.com/meta-pytorch/spmd_types
- 무엇: 지역 SPMD 타입(역전파 기울기가 합산 대기인지)과 전역 SPMD 타입(단일 장치와 같은 뜻)을 둔다. 가짜 프로세스 그룹으로 GPU 없이 검사할 수 있다(README 예제).
- 설계 목표: "Be optional, so that code can run without any types at runtime." 기존 Megatron류 코드에는 전역 타입 주석을 모듈 경계에서 입출력 계약으로만 달라고 권한다. 이유로 모듈 사이 샤딩·부분합 계약이 학습 코드의 가장 큰 오류 원천이라고 쓴다. https://github.com/meta-pytorch/spmd_types/blob/main/docs/design.md
- 실적: Megatron-LM 이슈 #7452(2026-09-17, ezyang)에서 조용한 수치 결함 13건을 보고했다. Astra와 spmd_types로 찾았고, 보고자는 전부를 직접 검증하지는 않았다고 밝혔다. 원문 "None of these raise an error." 이후 기여자들이 일부를 재현하고 수정 PR을 냈다(예: 11번 #7613, 10번은 main에서 재현). https://github.com/NVIDIA/Megatron-LM/issues/7452
- 이론에 주는 뜻: 경계 계약, 선택형, 실행 가능한 검사, 실제 결함 발견이라는 점에서 이론과 가장 가까운 최신 시도다. 다만 합산 상태 한 가지 사실만, 학습 코드만 다룬다.

**OneFlow SBP**
- 무엇: 분할(split), 복제(broadcast), 부분합(partial-value)을 전역 텐서에 달고, 호환되지 않는 상태 사이를 자동으로 변환(boxing)한다. https://arxiv.org/abs/2110.15032
- 상태: 별 9,437개. main의 최근 커밋은 2025-08-20이고 활동이 줄었다. https://github.com/Oneflow-Inc/oneflow
- 이론에 주는 뜻: 부분합 상태를 타입처럼 다루는 발상은 2021년에 이미 논문과 제품으로 있었다.

**GSPMD, Shardy**
- GSPMD: 사용자가 일부 텐서에 분배 주석을 달면 컴파일러가 나머지를 추론한다. 2048 TPUv3 코어에서 50~62% 활용률을 보고했다. https://arxiv.org/abs/2105.04663
- Shardy: GSPMD와 PartIR 팀이 만든 MLIR 기반 분할 체계. 축 기반 샤딩 표현, 전파, 디버깅 기능을 둔다. README는 "work in progress"라고 적는다. https://github.com/openxla/shardy . JAX는 Shardy를 기본으로 켜는 이전을 진행했다. https://github.com/jax-ml/jax/blob/main/docs/shardy_jax_migration.md
- 이론에 주는 뜻: 컴파일러 IR 안에서 샤딩 뜻을 보존하는 일은 이미 성숙했다. 다만 IR 밖(체크포인트, 설정, 엔진 배관)은 다루지 않는다.

**Mesh TensorFlow**
- 무엇: 이름 차원을 메시에 배치해 모델 병렬화를 쉽게 하려던 라이브러리. https://github.com/tensorflow/mesh
- 상태: 저장소 보관(마지막 push 2023-11-17). 보관 이유는 확인 못 함.

**PyTorch RFC-0057: Inductor 집합 연산의 의미 정체성**
- 무엇: Inductor가 집합 연산이 FSDP 파라미터 모으기인지, 기울기 동기화인지를 구조 휴리스틱으로 추측하고, 혼합 병렬 그래프에서 이 추측이 모호해진다고 적는다. 생산자가 목적을 메타데이터로 주고 그래프 변환을 건너 보존하자고 제안한다. https://github.com/pytorch/pytorch/issues/190792 , https://github.com/pytorch/rfcs/pull/103
- 상태: 2026-07-22 열림. 이슈의 댓글은 참조 요청 하나뿐이고, RFC PR에는 CLA 봇 댓글만 있다(2026-09-23 기준).
- 이론에 주는 뜻: 명제 1~3(컴파일러가 목적을 추측한다, 생산자가 선언하고 보존해야 한다)을 컴파일러 개발자 스스로 적은 1차 기록이다. 아직 받아들여지지 않았다.

**vLLM TPU 백엔드(tpu-inference)의 검사 끄기**
- 사실: GitHub 코드 검색으로 `check_vma=False`가 24개 파일에서 나온다. 어텐션, 선형층, MoE, 샘플링, KV 전송 경로가 포함된다. https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/layers/common/attention_interface.py , https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/layers/common/linear.py
- 이유: 코드 주석에서 이유를 찾지 못했다(확인 못 함).
- 이론에 주는 뜻: 추론 엔진 개발자는 켜져 있는 합산 상태 검사도 커널 경로에서 끈다. 필수 검사가 성능 커널과 부딪히면 꺼진다는 반박 근거다.

### 2.3 범주 소결

- 합산 상태라는 역할 사실은 선언·검사 방식이 이미 넓게 있다. OneFlow(2021), GSPMD(2021), JAX(기본값 Explicit, unreduced), DTensor(alpha), spmd_types(2026)가 있다.
- 방향은 "값에 뜻을 달고 다니는 객체"(DTensor)에서 "지우고 실행할 수 있는 주석과 경계 계약"(spmd_types, DTensor erasure)으로 옮겨 가는 중이다. 이유는 섞어 쓰기 실패와 실행 부담이다.
- 추론 쪽에서는 켜진 검사도 끈다(vLLM TPU). 추론 엔진에 합산 상태 계약을 필수로 둔 사례는 찾지 못했다.
- 컴파일러가 목적을 추측한다는 문제는 PyTorch 개발 문서(RFC-0057)에 스스로 적혀 있다. 선언으로 바꾸자는 제안은 검토되지 않았다.

---

## 3. 높은 수준의 의미를 보존하는 컴파일러 IR

### 3.1 표

| 이름 | 무엇의 의미 | 선언 방식 | 필수 | 검사 | 경계 보존 | 대상 | 상태 |
|---|---|---|---|---|---|---|---|
| MLIR 다단계 낮춤 | 높은 수준 구조와 의미 | 방언(dialect)의 연산·타입 | IR 안에서는 구조상 필수 | 정적(검증기) | 컴파일러 IR 안 | 컴파일러 일반 | 활발 |
| MLIR quant 방언 | 저장 타입, 표현 타입, scale, zero point, 축, 블록 | `!quant.uniform` 타입 | IR 안에서 필수 | 정적(검증기) | 컴파일러 IR 안 | 양자화 추론 | 활발 |
| MLIR sparse tensor 방언 | 희소 저장 형식 | 텐서 타입의 encoding 속성 | IR 안에서 필수 | 정적 | 컴파일러 IR 안 | 희소 연산 | 활발 |
| StableHLO | 연산 의미, 양자화 타입 제약 C1~C12 | 명세와 직렬화 형식 | 사용 시 필수 | 정적(명세 제약) | 직렬화 산출물로 건너감(하위 호환 5년) | 학습·추론(XLA 계열) | 활발 |
| TVM Relax StructInfo | 기호 shape, 구조 정보 | IR 주석 | IR 안에서 필수 | 정적·실행 중 일부 | 그래프에서 커널·외부 라이브러리 호출까지 | 추론(LLM 포함, MLC) | 활발 |
| Triton 레이아웃 인코딩 | 스레드·메모리 배치 | 텐서 타입의 인코딩 속성 | 컴파일러 내부 | 정적 | 컴파일러 안 | 커널 | 활발 |
| Halide | 알고리즘과 스케줄의 분리 | 언어 구조 | 필수(언어) | 정적 | 프로그램 안 | 이미지 처리 | 활발 |
| FlexAttention | 어텐션 변형(score_mod, mask_mod), 블록 희소 마스크 | 파이썬 함수와 BlockMask | 선택(API) | 컴파일 시점, 길이 불일치는 실행 중 오류 | 호출 하나 | LLM 학습·추론 | prototype |
| torch.export 동적 shape | 차원의 범위와 관계 | Dim(min, max, 선형 관계) | 선택. AUTO·DYNAMIC은 느슨함 | 추적 시점, 범위 제약은 산출물에 저장 | 내보낸 프로그램으로 건너감 | 배포 | 활발 |
| torch.library custom_op | 인자 변경(mutates_args), 별칭 | 연산 스키마 | mutates_args는 필수 인자 | 선언 검사 없음, opcheck 시험에서만 | 컴파일러 안 | 커스텀 커널 | 활발 |

### 3.2 항목별 설명

**MLIR의 "높은 수준 의미 유지" 원칙**
- 원 논문(2020, arXiv 2002.11054)의 설계 원칙 절 "Maintain higher-level semantics"는 분석과 성능 최적화에 필요한 높은 수준 의미를 유지해야 한다고 쓴다. 원문 "Attempts to raise semantics once lowered are fragile". 구조의 상실은 "conscious"해야 하고 더는 필요 없는 곳에서만 일어나야 한다고 쓴다. https://arxiv.org/abs/2002.11054 (PDF 본문에서 직접 추출해 확인)
- 이론에 주는 뜻: 명제 1~2(한번 잃은 의미를 다시 찾는 것은 깨지기 쉽다)는 컴파일러 학계의 주류 설계 원칙으로 이미 쓰였다. 이 원칙은 컴파일러 IR 안의 이야기다. 부품 사이 배관(체크포인트, 설정, 엔진)은 다루지 않는다.

**MLIR quant 방언**
- 무엇: `!quant.uniform` 타입이 저장 타입, 표현 타입, scale, zero point를 담는다. 층별, 채널별, 블록별 양자화를 지원한다. 검증기가 채널 축과 scale 개수 일치, 블록 크기의 나눠떨어짐 같은 규칙을 강제한다. https://mlir.llvm.org/docs/Dialects/QuantDialect/
- 이론에 주는 뜻: 저장·스케일 형식을 타입에 넣고 검증하는 일은 컴파일러 IR에서는 이미 된다. 이 연구가 "저장·스케일 형식을 적재부터 커널까지 잇는 것"을 새롭다고 할 때, 새로움은 IR 밖(체크포인트 파일과 엔진 적재기)에 있어야 한다.

**MLIR sparse tensor 방언**
- 무엇: 텐서 타입에 `#sparse_tensor.encoding`을 달아 저장 형식을 적는다. 문서의 원칙은 "treating sparsity as a property, not a tedious implementation detail"이다. TACO 계보다. https://mlir.llvm.org/docs/Dialects/SparseTensorOps/
- 이론에 주는 뜻: 저장 형식을 값의 타입 속성으로 두고 코드를 생성하는 설계의 선례다.

**StableHLO**
- 무엇: 양자화 텐서 원소 타입에 저장 타입, 표현 타입, 양자화 차원과 제약 C1~C12를 명세한다. https://github.com/openxla/stablehlo/blob/main/docs/spec.md
- 경계: 직렬화한 이식 산출물에 5년 하위 호환, 2년 상위 호환을 보장한다. https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md
- 이론에 주는 뜻: 형식 정보가 파일로 건너가 다른 도구에서 같은 뜻으로 읽히는 예다. 다만 컴파일러 산출물의 이야기이고, 파이썬 추론 엔진의 적재 배관은 아니다.

**TVM Relax (StructInfo)**
- 무엇: 1급 기호 shape 주석이 "a cross-level abstraction"이 되어 계산 그래프, 루프 수준 텐서 프로그램, 외부 라이브러리 호출을 한 표현에 담는다. ASPLOS 2025. https://arxiv.org/abs/2311.02103
- 상태: 활발(v0.27.0.rc1 2026-09-22, 별 13,777). https://github.com/apache/tvm
- 이론에 주는 뜻: 그래프에서 커널까지 사실(여기서는 기호 shape)을 이어서 보존하는 설계가 LLM 배포에도 쓰인다. 담는 사실은 shape이고 역할은 아니다.

**Triton 레이아웃 인코딩**
- 무엇: TritonGPU 방언이 텐서 타입에 레이아웃 인코딩 속성(공유 메모리 swizzle, MMA, dot 피연산자 등)을 붙인다. https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
- 이론에 주는 뜻: 커널 컴파일러 안에서는 배치의 뜻이 타입으로 보존된다. 커널 밖(파이썬 호출자)과의 경계에는 이런 타입이 없다.

**Halide**
- 무엇: 알고리즘과 스케줄을 떼어 놓는다("decoupling algorithms from schedules"). https://halide-lang.org/
- 상태: 활발(별 6,610). https://github.com/halide/Halide
- 이론에 주는 뜻: 뜻(알고리즘)과 실행 방법(스케줄)을 분리해 최적화가 뜻을 건드리지 못하게 하는 설계의 대표 선례다.

**FlexAttention**
- 무엇: 어텐션 변형을 `score_mod`와 `mask_mod` 함수로 선언하면 컴파일러가 커널을 만든다. 논문은 FlashAttention의 "monolithic nature"가 새 변형을 막는 "software lottery" 문제를 든다. https://arxiv.org/abs/2412.05496
- 검사: `flex_attention`은 BlockMask가 만들어진 길이와 실제 q, kv 길이가 다르면 오류를 낸다. 오류문은 "block_mask was created for a smaller length than you're using it for"이다. `_adjust`로 자르면 "does not work for all mask_mods!"라고 경고한다. https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py
- 상태: 문서가 "prototype feature"라고 적는다. https://docs.pytorch.org/docs/2.14/nn.attention.flex_attention.html
- 이론에 주는 뜻: 어텐션 변형의 뜻을 선언으로 넘기면 커널이 추측할 필요가 없어진다. 명제 3의 가장 성공한 추론 쪽 사례다. 다만 길이만 대조하고, mask_mod의 위치 기준과 캐시 위치가 맞는지는 대조하지 않는다.

**torch.export 동적 shape 제약**
- 무엇: `Dim("dx", min=4, max=256)`처럼 범위와 선형 관계를 선언한다. 추적 중 선언과 충돌하는 가드가 나오면 `ConstraintViolationError`를 낸다. 범위 제약은 `range_constraints`로 내보낸 프로그램에 남는다. https://github.com/pytorch/pytorch/blob/main/docs/source/user_guide/torch_compiler/export.md
- 느슨한 모드: `Dim.DYNAMIC`은 범위가 다르면 "automatically update the range without raising an error"이고, `Dim.AUTO`는 정적으로 추론돼도 오류를 내지 않는 "best effort" 방식이다(같은 출처).
- 이론에 주는 뜻: 엄격한 선언을 먼저 두고, 편의를 위해 조용히 넘어가는 모드를 나중에 더했다. 필수·엄격 선언이 사용성 압력을 받는다는 사례다.

**torch.library custom_op의 mutates_args와 opcheck**
- 무엇: 문서 원문 "This MUST be accurate, otherwise, the behavior is undefined." 선언이 맞는지는 실행 중에 검사하지 않고, `opcheck`라는 시험 도구가 스키마, 가짜 텐서, 자동미분 등록, 컴파일 결과를 확인한다. https://docs.pytorch.org/docs/2.14/library.html
- 현장: PyTorch 2.13 릴리스 노트는 oneDNN 연산이 옛 스키마로 "silently bypassed aliasing checks"였다고 적고, 입력과 별칭인 출력을 내는 커스텀 연산을 폐기 예정으로 돌렸다. https://github.com/pytorch/pytorch/releases/tag/v2.13.0
- 이론에 주는 뜻: 선언을 필수로 해도 실제 동작과 대조하지 않으면 틀린 선언이 조용히 남는다. "선언을 실제 데이터로 검증한다"는 이론의 요소를 지지한다.

### 3.3 범주 소결

- "낮추면 잃고, 다시 찾는 것은 깨지기 쉽다"는 명제는 MLIR 원 논문의 설계 원칙으로 이미 쓰였다. 컴파일러 IR 안에서는 형식(quant, sparse), 배치(Triton, Shardy), shape(Relax)를 타입으로 보존하는 일이 성숙했다.
- 빈틈은 IR 밖이다. 체크포인트와 설정 파일, 파이썬 엔진의 적재기, 커널 호출 인자 사이에는 이런 타입이 없다. 이 범주의 어떤 시도도 그 배관을 다루지 않는다.
- 선언형 API(FlexAttention, torch.export)는 길이·범위 같은 수치 사실만 대조한다. 위치 기준 같은 역할 사실은 대조하지 않는다.
- 엄격 선언은 사용성 압력으로 느슨한 모드가 붙는다(torch.export AUTO). 선언만 있고 대조가 없으면 틀린 선언이 조용히 남는다(custom_op).

---

## 4. 데이터와 함께 다니는 형식과 메타데이터

### 4.1 표

| 이름 | 무엇의 의미 | 선언 방식 | 필수 | 검사 | 경계 보존 | 대상 | 상태 |
|---|---|---|---|---|---|---|---|
| torchao 텐서 하위 클래스 | 파생 dtype, 패킹 형식, block_size, scale | 텐서 하위 클래스 | torchao 경로에서는 구조상 필수 | 실행 중(디스패치) | 텐서 객체와 함께 저장·적재 | 추론(양자화) | 활발 |
| compressed-tensors | 압축 형식(닫힌 enum), 양자화 인자 | config.json의 quantization_config, pydantic 모델 | 이 형식을 쓰면 필수 | 적재 시 스키마 검증(`extra="forbid"`) | 파일로 이동(체크포인트) | LLM 추론(vLLM, SGLang) | 활발 |
| GGUF | 아키텍처, RoPE, 토크나이저, 채팅 템플릿, 양자화 버전 | 타입이 있는 키-값 | 몇 개만 필수, 나머지 권장 | 적재기마다 다름(없으면 기본값 또는 오류) | 한 파일에 모두 담음 | LLM 추론(llama.cpp 계열) | 활발 |
| safetensors `__metadata__` | 임의 문자열 | 문자열→문자열 맵 | 선택 | 없음 | 파일로 이동 | 전 분야 | 활발 |
| Stability AI ModelSpec | 아키텍처, 구현, prediction_type, 데이터 형식 | safetensors 메타데이터의 `modelspec.*` 키 | MUST/SHOULD/CAN 3단계 | 없음(소비자 재량) | 파일로 이동 | 이미지 생성, 일부 LLM | 2024-06 이후 갱신 없음, 채택 부분적 |
| diffusers scheduler 설정 | prediction_type 등 | scheduler_config.json | 선택(기본값 epsilon) | 없음 | 별도 파일 | 이미지 생성 | 활발. 단일 파일 적재는 추측 |
| ONNX 타입·차원 표기 | 입력의 의미(이미지, 색 순서), 축의 의미 | TypeProto의 denotation | 선택 | 설계상 검증, 검사기에는 없음 | 파일로 이동 | 추론 | 2018년 이후 실험 단계 |
| HF quantization_config | 양자화 방법(quant_method) | config.json | 양자화 모델에서 필수 | 적재 시 닫힌 집합 확인 | 파일로 이동 | LLM 추론 | 활발 |
| Core ML ImageType | 색 순서, scale, bias | 모델 안의 입력 정의 | 이미지 입력으로 선언하면 필수 | 실행 중 자동 적용 | 모델 파일 안 | 모바일 추론 | 활발 |
| LiteRT(TFLite) Metadata | 정규화 mean/std, 색 공간, 라벨 | 모델 파일의 메타데이터 | 선택 | 코드 생성기가 소비 | 모델 파일 안 | 모바일 추론 | 활발 |

### 4.2 항목별 설명

**torchao 텐서 하위 클래스**
- 무엇: 양자화 텐서를 `torch.Tensor` 하위 클래스로 만들고, "derived dtype"과 "packing format"(메모리 배치) 두 축으로 나눈다. 예: `Int4Tensor`(두 int4를 int8 하나에), `Int4PreshuffledTensor`(적재 최적화 형식). https://github.com/pytorch/ao/blob/main/docs/source/contributing/quantization_overview.rst
- 경계: 하위 클래스 객체가 state_dict로 저장되고 다시 적재된다(직렬화 튜토리얼). https://github.com/pytorch/ao/blob/main/docs/source/eager_tutorials/serialization.rst
- 이론에 주는 뜻: 저장 형식을 값의 타입에 넣어 커널이 추측하지 않게 하는 설계가 PyTorch 공식 양자화 라이브러리에 있다. 다만 torchao 텐서 안에서만이다. 다른 엔진은 통합 코드가 따로 있어야 한다.

**compressed-tensors**
- 무엇: safetensors를 확장한 통합 압축 체크포인트 형식. GPTQ, AWQ, FP8, NVFP4, MXFP4 등을 한 형식으로 담는다. https://github.com/vllm-project/compressed-tensors
- 선언: `CompressionFormat`이 닫힌 enum이다(dense, pack-quantized, nvfp4-pack-quantized 등). `QuantizationArgs`는 pydantic 모델이고 필드 검증기와 `extra="forbid"`를 둔다. 예: group_size는 strategy가 group일 때만 허용. https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/config/base.py , https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_args.py
- 이론에 주는 뜻: 체크포인트 경계에서 저장 형식을 닫힌 집합으로 선언하고 적재 때 검증하는 것은 이미 LLM 추론 생태계에 있다. 이 연구의 "형식을 닫힌 집합으로"라는 요소와 겹친다. 남는 빈틈은 선언을 실제 텐서(shape, dtype, 값 범위)와 대조하는지, 그리고 커널 선택까지 이어지는지다. 이번 조사에서 그 부분은 확인하지 못했다.

**GGUF**
- 무엇: "designed to be unambiguous by containing all the information needed to load a model"이라는 목표를 가진 단일 파일 형식. 이전 형식의 타입 없는 값 목록을 타입이 있는 키-값으로 바꿨다. RoPE 기준값과 스케일링 종류, `tokenizer.chat_template`까지 담는다. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- 필수 여부: `general.architecture`, `general.quantization_version`(양자화 시), `general.alignment` 정도만 필수다. 나머지는 권장이고, 빠지면 읽는 쪽이 "default or error as appropriate"로 처리하라고 적는다(같은 출처).
- 이론에 주는 뜻: 의미를 데이터와 함께 보낸다는 목표는 GGUF가 이미 명시적으로 내세웠다. 그러나 대부분 선택이고, 없을 때 기본값을 쓰는 것을 허용한다. 이론의 "필수 선언, 없으면 시끄럽게 실패"와 다른 지점이다.

**safetensors `__metadata__`**
- 무엇: 헤더의 `__metadata__`는 "free form string-to-string map"이고 임의 JSON은 허용하지 않는다. https://github.com/huggingface/safetensors
- 이론에 주는 뜻: 가장 널리 쓰이는 가중치 형식이 타입도 스키마도 없는 문자열 칸 하나만 둔다. 명제 1(형태만 전달한다)의 구조적 근거다.

**Stability AI ModelSpec**
- 무엇: safetensors 메타데이터에 `modelspec.` 접두 키를 정의한다. 목적은 추론 엔진이 "how to load it correctly"를 판단하게 하는 것이다. 키를 MUST, SHOULD, CAN으로 나눈다. https://github.com/Stability-AI/ModelSpec
- 내용: `architecture`는 MUST이고, SDv2-512와 SDv2-768-v처럼 추론 코드가 달라야 하면 구분해야 한다고 적는다. `prediction_type`(v 또는 epsilon)은 CAN(선택)이다. 텍스트 모델의 `data_format`은 MUST이며, 이유로 형식이 "often not accurately reflected in tensor data type"이라고 쓴다(같은 출처).
- 상태: 별 71개, 마지막 push 2024-06-04. 소비자에게 형식이 맞지 않아도 관대하라("be lenient")고 권한다.
- 채택: kohya-ss/sd-scripts는 이 형식을 쓴다(코드 검색 15개 파일). ComfyUI는 병합 노드에서 `modelspec.architecture`를 쓰지만 예측 방식은 명세에 없는 `modelspec.predict_key`로 적고, 빈 텐서 `v_pred`를 state dict에 넣는다. https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_model_merging.py
- ComfyUI의 적재 쪽: SDXL의 예측 방식을 state dict에 `v_pred` 키가 있는지로 판단하고, 없으면 `ModelType.EPS`를 돌려준다. https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/supported_models.py
- 이론에 주는 뜻: 모델 성질(예측 방식)을 전달하는 표준이 있어도 선택 항목이고, 도구마다 다른 이름과 표시 방식을 쓴다. 표시가 없으면 기본값으로 조용히 추측한다. 명제 1~2를 이미지 생성 쪽에서 직접 지지한다. 이 연구의 ComfyUI 현장 시험(표지 없는 v 모델을 놓침)과 같은 구조다.

**diffusers의 prediction_type**
- 무엇: 스케줄러 설정의 `prediction_type`은 기본값이 "epsilon"이다. https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py
- 추측: 단일 파일 적재기는 SD2 모델에서 `global_step == 875000`이면 epsilon, 아니면 v_prediction으로 정한다. 코드 주석이 이 방식이 "brittle global step parameter"에 기댄다고 스스로 적는다. https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py
- 이론에 주는 뜻: 명제 2(받는 쪽이 의미를 추측한다)의 1차 증거다. 추측이 깨지기 쉽다는 것도 개발자가 적었다.

**ONNX 타입 표기와 차원 표기**
- 무엇: 타입 표기는 입력이 이미지인지, 색 순서가 bgr인지 같은 "semantic information"을 적는다. 차원 표기는 축에 DATA_BATCH, DATA_CHANNEL 같은 뜻을 주고 전파와 검증을 하려던 설계다. 문서는 이것을 "an experimental attempt"라고 부르고, 잘못된 전치를 "no existing infrastructure will report an error"라고 동기를 적는다. https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md , https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md
- 역사: 차원 표기는 2018-04(#443), 타입 표기와 이미지 메타데이터는 2018-05(#879)에 들어왔다. 그 뒤 문서 변경은 라이선스·공백 수정뿐이다(파일 커밋 기록).
- 검사: 설계 문서에는 불일치 시 오류를 보고해야 한다고 적혀 있다. 그러나 `onnx/checker.cc`에는 "denotation"이라는 문자열이 없다(2026-09-23 main). https://github.com/onnx/onnx/blob/main/onnx/checker.cc
- 이론에 주는 뜻: 이론과 같은 문제의식(축의 의미가 틀려도 아무도 오류를 내지 않는다)으로 2018년에 표준 안에 선택형 의미 표기가 들어갔다. 검증은 구현되지 않은 채 8년간 실험 단계에 머물렀다. 선택형 의미 표기가 쓰이지 않는다는 가장 오래된 사례다.

**HF quantization_config**
- 무엇: config.json의 `quantization_config.quant_method`로 양자화 방법을 고른다. 없거나 모르는 값이면 "Unknown quantization type" 오류를 낸다. https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/auto.py
- 이론에 주는 뜻: 방법 이름은 닫힌 집합으로 검사한다. 방법 안의 세부(스케일 배치, 패킹)는 각 양자화기에 맡긴다.

**Core ML ImageType**
- 무엇: 이미지 입력의 색 순서(RGB, BGR, 흑백), scale, bias를 모델에 저장한다. 문서 원문 "Scale and biases are stored in the model and, at runtime, are applied". https://apple.github.io/coremltools/docs-guides/source/image-inputs.html
- 이론에 주는 뜻: 전처리의 뜻을 모델 파일 안에 넣어 호출자가 틀릴 수 없게 하는 설계가 모바일 추론에서는 표준이다. 이미지 생성 파이프라인(색 공간, 정규화 범위)에는 이런 장치가 없다는 대비가 된다.

**LiteRT(TFLite) Model Metadata**
- 무엇: 정규화 mean/std, 색 공간, 라벨 파일, 양자화 파라미터를 모델 파일에 담는다. 코드 생성기와 Task Library가 소비한다. 문서 원문 "Passing in a model without metadata is allowed." https://developers.google.com/edge/litert/models/metadata
- 이론에 주는 뜻: 뜻을 담는 칸은 있지만 선택이다.

### 4.3 범주 소결

- "의미를 데이터와 함께 보낸다"는 목표는 여러 형식이 이미 명시적으로 내세웠다. GGUF("unambiguous"), ModelSpec("how to load it correctly"), ONNX 표기, Core ML, LiteRT가 그렇다.
- 대부분 선택이고, 없으면 기본값을 쓰라고 허용한다. 필수로 한 곳(Core ML 이미지 입력, compressed-tensors 스키마, quant_method)은 좁은 범위에서 잘 작동한다.
- 추측이 실제 코드에 남아 있다. diffusers는 SD2 예측 방식을 학습 스텝 수로 추측하고 스스로 "brittle"이라 적었다. ComfyUI는 빈 텐서 키로 예측 방식을 표시하고, 없으면 기본값으로 간다.
- 같은 뜻을 도구마다 다른 이름으로 적는다(ModelSpec `prediction_type`, ComfyUI `modelspec.predict_key`, `v_pred` 텐서). 표준이 있어도 이름과 필수 여부가 맞지 않으면 뜻이 건너가지 않는다.
- 선택형 의미 표기는 검증기 없이 오래 방치된다(ONNX 8년).

---

## 5. 설정과 메타데이터 검증

### 5.1 표

| 이름 | 무엇의 의미 | 선언 방식 | 필수 | 검사 | 경계 보존 | 대상 | 상태 |
|---|---|---|---|---|---|---|---|
| huggingface_hub `@strict` + transformers PreTrainedConfig | 설정 필드의 타입, 필드 간 일관성 | strict dataclass와 `validate_*` 메서드 | 기본 적용, 단 모르는 키는 허용 | 생성·대입 시 검증 | 설정 객체 | LLM 전반 | 활발(transformers main) |
| transformers RoPE 검증 | rope_type별 필수·선택 키 | 설정 검증 함수 | 필수 키 누락만 오류 | 설정 생성 시 | 설정 객체 | LLM | 활발 |
| transformers 생성 설정·인자 검증 | 무시되는 플래그, 모델이 쓰지 않는 인자 | GenerationConfig.validate, `_validate_model_kwargs` | 기본은 경고, strict=True면 오류. 쓰지 않는 model_kwargs는 오류 | 호출 시 | generate와 모델 사이 | LLM | 활발 |
| transformers 엄격 적재 PR #48962 | 누락·예상 밖 키 | `from_pretrained(strict=True)` 제안 | 선택형 제안 | 해당 없음 | 적재 경계 | LLM | 2026-09-22 닫힘 |
| vLLM 가중치 적재 추적 | 체크포인트에서 초기화되지 않은 가중치 | 적재기 검사 | 양자화하지 않은 모델에서만 기본 켜짐 | 적재 시 오류 | 적재 경계 | LLM 추론 | 활발 |
| vLLM 어텐션 백엔드 validate_configuration | 모델 요구와 백엔드 능력 | 능력 선언(supports_*) | 백엔드 선택 시 자동 | 선택 시 대조 | 모델과 커널 사이 | LLM 추론 | 활발 |
| vLLM RFC #24384 | 엔진이 쓰는 설정 필드의 계약 | 통합 설정 스키마 제안 | 제안 | 해당 없음 | 설정 경계 | LLM 추론 | 2026-03 비활성 자동 종료 |
| vLLM RFC #48312, #48478 | 가중치 재적재의 불변식, 그래프가 잡은 저장소 | 불변식 분류, 명시 등록과 fail-closed 검사 | 제안 | 재적재 완료 시 검사 제안 | 적재와 CUDA Graph 사이 | LLM 추론(RL) | 열림 |
| SGLang KV canary | KV 칸마다 토큰, 위치, 해시 | 실행 중 태그 | 기본 꺼짐(none), log·raise 선택 | 매 forward 대조 | KV 캐시 | LLM 추론 | 활발 |
| SGLang RFC #32432 | CUDA Graph 재생의 메타데이터·작업공간·스트림 소유 계약 | 계약 제안 | 제안 | 해당 없음 | 그래프 재생 경계 | LLM 추론 | 열림, 작성자 댓글뿐 |

### 5.2 항목별 설명

**huggingface_hub strict dataclass와 transformers PreTrainedConfig**
- 무엇: `@strict`는 타입 힌트로 필드를 생성과 대입 때 검증하고, `validate_*` 메서드로 필드 간 일관성을 검사한다. 문서는 흔한 쓰임을 "validating a model configuration"이라 적는다. https://huggingface.co/docs/huggingface_hub/package_reference/dataclasses
- transformers 적용: main의 `PreTrainedConfig`가 `@strict(accept_kwargs=True)`로 장식돼 있다. 모르는 키도 받는다는 뜻이다. `layer_types`는 허용 목록 밖이면 오류를 낸다. https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py
- 한계의 이유: 특수 토큰 id 검사는 경고로만 한다. 코드 주석 원문 "Can't be an exception until we can load configs that fail validation". Hub의 여러 설정이 `pad_token_id=-1` 같은 잘못된 값을 가진다고 적는다(같은 출처).
- 이론에 주는 뜻: 설정 경계의 타입 검증은 2026년 transformers 기본값이 됐다. 다만 이미 배포된 체크포인트가 검증을 통과하지 못해 엄격하게 할 수 없다는 제약을 개발자가 직접 적었다. 필수 선언을 기존 모델에 붙일 때 가장 큰 장애물이 기존 데이터라는 1차 근거다.

**transformers RoPE 설정 검증**
- 무엇: rope_type마다 필수·선택 키를 정하고, 필수 키가 없으면 `KeyError`를 낸다. 값 범위 위반과 모르는 rope_type은 경고만 한다. https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
- 이론에 주는 뜻: 위치 인코딩 설정의 뜻을 검증하는 장치가 있다. 대부분 경고라서 조용한 실패를 막지는 못한다.

**transformers 생성 설정과 인자 검증**
- 무엇: `GenerationConfig.validate(strict=False)`는 `do_sample`이 꺼졌는데 샘플링 플래그가 설정된 경우처럼 무시될 설정을 찾는다. 기본은 기록만 하고, `strict=True`면 오류를 낸다. 모델 기본 설정에서 물려받은 값이면 경고를 줄인다는 설명이 있다. https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
- 인자 검증: `generate()`는 모델이 쓰지 않는 `model_kwargs`가 있으면 "are not used by the model" 오류를 낸다. https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
- 이론에 주는 뜻: "무시된 인자"라는 사실은 generate 경계에서 이미 검사된다. 이 연구가 아무도 검사하지 않는 사실로 꼽은 "무시된 인자"에 대한 부분 반례다. 다만 모델 forward가 받기만 하고 쓰지 않는 인자(예: 이 연구가 찾은 `cache_position`)는 이 검사로 잡히지 않는다.

**transformers 엄격 적재 제안(PR #48962)**
- 무엇: `from_pretrained(strict=True)`를 넣어 누락·예상 밖 키가 있으면 오류를 내자는 선택형 제안. 2년 된 요청(#32067)에서 왔다. https://github.com/huggingface/transformers/pull/48962
- 결과: 유지보수자가 "2 years old feature request, so not sure we actually want it"라고 답했고, 2026-09-22 닫혔다(같은 출처).
- 이론에 주는 뜻: 선택형 엄격 모드조차 수요가 확인되지 않아 들어가지 못했다. 반대 논거는 내용이 아니라 수요였다.

**vLLM 가중치 적재 추적**
- 무엇: 체크포인트에서 초기화되지 않은 가중치가 있으면 "Following weights were not initialized from checkpoint" 오류를 낸다. 기본으로 켜지는 조건은 `model_config.quantization is None`이다. https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/default_loader.py
- 이론에 주는 뜻: 시끄러운 적재 검사가 있지만, 저장 형식 사실이 가장 중요한 양자화 모델에서는 기본으로 꺼진다.

**vLLM 어텐션 백엔드 능력 검증**
- 무엇: 백엔드가 `supports_head_size`, `supports_sink`, `supports_sliding_window`, `supports_non_causal` 등을 선언하고, `validate_configuration`이 모델 요구와 대조해 맞지 않는 이유 목록을 만든다. https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py
- 빈틈: 같은 파일의 검사 목록에 logits softcap은 없다(2026-09-23 main에서 확인).
- 이론에 주는 뜻: "모델 성질과 커널 능력의 대조"는 vLLM에 이미 부분적으로 있다. 이 연구가 새롭다고 할 수 있는 몫은 빠진 항목(softcap 등)과 다른 엔진으로의 일반화다.

**vLLM RFC #24384: 설정을 HF에서 떼어 내기**
- 무엇: vLLM이 쓰는 설정 필드와 쓰지 않는 필드를 나누고, 중요한 필드의 이름과 뜻을 명시하자는 제안. https://github.com/vllm-project/vllm/issues/24384
- 논의: HF 쪽 유지보수자는 이것이 vLLM이 풀 문제가 아니라 Transformers에서 이름을 표준화해 풀 문제라고 반대했다. vLLM 쪽은 엔진이 필요로 하는 정보의 "constrained interface"가 필요하다고 답했다(같은 출처 댓글).
- 결과: 2026-03-10 비활성으로 자동 종료(not_planned).
- 이론에 주는 뜻: 반대는 계약의 필요성이 아니라 누가 소유하느냐였다. 계약을 어디에 둘지 정하지 못하면 제안이 멈춘다.

**vLLM RFC #48312과 #48478: 가중치 재적재 정확성**
- 무엇: RL에서 가중치를 다시 적재할 때의 실패를 증상이 아니라 근본 원인으로 분류한다. 저장소 정체성, 파생값 갱신, 적재기 수명, 상태 보존, 파라미터 배정·분할, 이름 대응, 캐시 일관성의 7개 불변식을 둔다. 저장소 정체성 범주에는 "fail explicitly instead of silently"를 요구한다. https://github.com/vllm-project/vllm/issues/48312
- 후속: #48478은 CUDA Graph가 잡은 저장소를 명시 등록하고, 재적재가 끝날 때 검사해 실패하면 워커를 멈추는 "Fail-Closed Graph Storage Contract"를 제안한다. https://github.com/vllm-project/vllm/issues/48478
- 상태: 둘 다 열려 있다(#48312 댓글 19개, #48478 댓글 1개).
- 이론에 주는 뜻: 엔진 안에서 "사실의 종류로 결함을 나누고 경계에 계약을 두자"는 접근이 2026-07에 독립적으로 나왔다. 이 연구의 "읽는 시점·버퍼 시점" 사실과 겹친다. 이 연구의 분류가 새롭다는 주장은 이 RFC와 비교해 다시 좁혀야 한다.

**SGLang KV canary**
- 무엇: KV 칸마다 토큰, 위치, 연쇄 해시를 기록하고 forward마다 대조한다. 모드는 none, log, raise이고 기본값은 none이다. 도움말은 log를 "production-safe", raise를 CI용으로 설명한다. https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/kv_canary/config.py , https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/observability.py
- 실적: DeepSeek-V4와 계층 캐시 조합에서 SWA KV의 위치 오염을 잡았다(`fail_reason=write_position`). 증상은 샘플링 확률의 NaN이었다. https://github.com/sgl-project/sglang/issues/33656
- 이론에 주는 뜻: KV 캐시 안에서는 "위치 기준" 사실을 실제 데이터에 태그로 달고 대조하는 도구가 이미 있다. 이 연구가 "아무도 담지 않는 사실"로 꼽은 위치 기준에 대한 부분 반례다. 기본으로 꺼져 있고 KV 캐시만 다룬다는 점이 남는 빈틈이다.

**SGLang RFC #32432: CUDA Graph 재생 계약**
- 무엇: 그래프 재생에서 재사용하는 메타데이터, 작업공간, 요청·KV 칸, 스트림의 소유와 수명을 명시 계약으로 만들자는 제안. https://github.com/sgl-project/sglang/issues/32432
- 상태: 열려 있다. 댓글 9개가 모두 작성자 것이다.

### 5.3 범주 소결

- 설정 경계의 검증은 2025~2026년에 빠르게 늘었다. transformers의 strict 설정, RoPE 검증, 생성 설정 검증, vLLM의 백엔드 능력 대조와 적재 추적, SGLang의 KV canary가 있다.
- 거의 모두 기본값이 느슨하다. 모르는 키 허용, 경고만, 양자화 모델 제외, 기본 꺼짐. 이유로 적힌 것은 기존 Hub 설정과의 호환, 경고 소음, 비용이다.
- 계약형 제안(#24384, #48312, #48478, #32432, #48962)은 계속 나오지만 받아들여진 것이 없다. 반대 논거는 소유권과 수요였고 내용 반박은 찾지 못했다.
- 이 연구가 새롭다고 꼽은 사실 가운데 무시된 인자(generate 경계), 위치 기준(KV canary), 모델 성질과 커널 능력의 대조(vLLM validate_configuration), 버퍼 시점(#48478)은 각각 부분적으로 다른 곳에 이미 있다. 한 곳에 모아 필수로 두고 적재부터 커널까지 잇는 것은 찾지 못했다.

---

## 6. 사후 탐지와 검증

### 6.1 표

| 이름 | 무엇을 잡나 | 방식 | 선언 필요 | 검사 시점 | 대상 | 상태 |
|---|---|---|---|---|---|---|
| Emerge | HF와 vLLM 구현의 비동등 | e-graph, 관계 추론, SMT, 무작위 시험 | 없음 | 배포 전(참조 필요) | LLM 추론 | 논문 2026-03 |
| Ekka | 조용한 오류의 근본 원인 | 참조 구현과 중간 상태 비교 | 없음 | 증상 발견 뒤 | LLM 추론 | 논문 2026-06 |
| M2K | 모델과 CUDA 커널 사이 가정 불일치(메모리) | 실행 추적으로 인터페이스 추론, 커널 기호 실행 | 없음(추론) | 배포 전 | LLM 추론 커널 | 논문 v2 2026-08 |
| GRIEF | 동시성·캐시에서 생기는 결함과 조용한 출력 오염 | 다중 요청 퍼징, log-prob 재생 | 없음 | 배포 전 | vLLM, SGLang | 논문 2026-05 |
| infer-check | 양자화·서빙·KV 캐시의 조용한 오류 | 백엔드·양자화·동시성 차분 시험 | 없음 | 배포 전 | LLM 추론 | 개인 프로젝트(별 2) |
| Kimi Vendor Verifier | 업체별 정확도 차이 | 공개 벤치마크, 사전 검증 | 없음 | 배포 뒤·출시 전 | 특정 모델 API | 활발 |
| Artificial Analysis Endpoint Accuracy Index | 업체 엔드포인트와 참조 배포의 차이 | 벤치마크 비교 | 없음 | 배포 뒤 | 공개 가중치 모델 API | 2026-08 시작 |
| vLLM 야간 정확도 평가 | 엔진 회귀 | lm-eval, BFCL, 17개 조합 | 없음 | 출시 후보마다 | vLLM | 2026-07 도입 |
| lm-evaluation-harness | 모델 성능 | 벤치마크 | 없음 | 임의 | LLM | 활발 |
| OpenRouter Auto Exacto | 품질 나쁜 업체 | 원격 측정과 벤치마크로 경로 선택 | 없음 | 운영 중 | LLM API | 2026-03 기본값(도구 요청) |
| MLPerf Inference | 제출 정확도 | 참조 대비 99%·99.9% | 없음 | 제출 시 | 벤치마크 | 활발 |
| NVIDIA Polygraphy | 백엔드 간 결과 차이 | 여러 백엔드 실행 비교 | 없음 | 개발 중 | 추론(TensorRT, ONNX) | 활발 |
| llama.cpp perplexity KLD, test-backend-ops | 양자화 품질 손실, 백엔드 연산 불일치 | 로짓 분포 KL, 연산별 교차 비교 | 없음 | 개발 중 | LLM 추론 | 활발 |
| The Silent Hyperparameter | 백엔드 선택에 따른 점수 차이 | 5개 엔진 비교 연구 | 없음 | 연구 | LLM 평가 | 논문 2026-05 |
| Kernel Contracts | 커널의 정밀도·순서·예외값 약속 | 명세 언어(참조 오라클 포함) | 명세 작성 | 적합성 평가 | ML 커널 | 논문 2026-04(1인) |
| TrainCheck, Scalify, GraphGuard | 학습의 조용한 오류, 분산 그래프 비동등 | 불변식 추론, 등식 포화, 관계 추론 | 없음 | 학습 중·배포 전 | 학습 | 논문 2025 |

### 6.2 항목별 설명

**Emerge (Verify Implementation Equivalence of Large Models)**
- 무엇: HF Transformers와 vLLM의 추론 계산 그래프가 같은지 검증한다. 실행 값에서 후보 관계를 추론하고, 기호로 다룰 수 있으면 SMT로, 불투명한 커널은 제약을 지킨 무작위 시험으로 확인한다. 알려진 결함 13건 중 10건을 잡았고, 개발자가 확인한 새 문제 8건을 찾았다. https://arxiv.org/abs/2603.21851
- 이론에 주는 뜻: 선언 없이 참조 구현과 비교해 같은 부류의 결함을 상당수 잡는다. 이론의 가장 강한 대안이다. 참조 구현이 없는 새 모델, 참조와 같은 가정을 공유한 결함, 참조가 없는 설정(양자화 형식 등)은 이 방식의 약점이다. 이 약점을 수치로 적은 1차 자료는 초록에서 찾지 못했다.

**Ekka**
- 무엇: 조용한 오류를 "differential debugging problem"으로 보고, 참조 구현과 중간 상태를 맞춰 비교해 원인을 짚는다. 진단 정확도 pass@1 80%, pass@5 88%. 새 조용한 오류 4건을 찾아 확인받았다. https://arxiv.org/abs/2606.04594
- 이론에 주는 뜻: 이론이 보조로 둔 "진단"은 이미 자동화 연구가 있다.

**M2K**
- 무엇: 제목이 "Making the Model-Kernel Interface Explicit"이다. 모델과 CUDA 커널 사이 인터페이스가 암묵적이라 shape와 크기 가정이 어긋나 메모리 결함이 생긴다고 진단한다. GPU 없이 모델 실행을 추적해 기호 제약을 내고(HFProbe), 커널을 기호 실행한다(cuKLEE). 새 버그 181건, 오탐 9건. https://arxiv.org/abs/2603.24595
- 이론에 주는 뜻: "모델과 커널 사이 가정을 명시해야 한다"는 문제의식이 같다. 방법은 선언이 아니라 추론이고, 대상은 메모리 안전이다.

**GRIEF**
- 무엇: vLLM과 SGLang을 시간 순서가 있는 다중 요청으로 퍼징한다. 충돌, 멈춤, 성능 병리, 조용한 출력 오염을 오라클로 쓰고 log-prob 재생으로 확인한다. 취약점 15건, 개발자 확인 10건, CVE 2건. https://arxiv.org/abs/2605.11202
- 이론에 주는 뜻: 동시성·캐시 재사용에서 오는 조용한 오염은 선언 방식이 닿기 어려운 영역이다. 이 부류에는 퍼징이 맞는다.

**infer-check**
- 무엇: "Catches the correctness bugs that benchmarks miss in LLM inference engines"를 표방하는 CLI. 백엔드, 양자화 수준, 동시성 조건 사이 차분 시험을 한다. README는 vLLM의 FP8 KV 양자화가 "repeated garbage output"을 낸 사례를 든다. https://github.com/NullPointerDepressiveDisorder/infer-check
- 상태: 2026-03-09 생성, 마지막 push 2026-04-20, 별 2개. 개인 프로젝트다.
- 이론에 주는 뜻: 존재는 확인했다. 반복 출력 같은 증상이 엔진 결함에서 올 수 있다는 현장 기록이 된다.

**Kimi Vendor Verifier (K2VV → KVV)**
- 무엇: Kimi K2 출시 뒤 업체별 도구 호출 정확도 차이가 커서 만든 검증기. https://github.com/MoonshotAI/K2-Vendor-Verifier (별 594)
- 개편: 블로그는 원인으로 "misuse of Decoding parameters"(temperature, top_p), KV 캐시 결함, 양자화 저하, 비전 전처리, 도구 호출 불일치를 든다. 대응으로 vLLM·SGLang·KTransformers의 상류 수정, 출시 전 검증, 공개 순위표를 적는다. 원문 "If users cannot distinguish between model capability defects and engineering implementation deviations, trust will collapse." https://www.kimi.ai/blog/kimi-vendor-verifier
- 이론에 주는 뜻: 모델 회사가 "모델 실력"과 "배관 결함"을 구분해야 한다고 공개적으로 말했다. 이론의 범위 설정(모델이 아니라 버그)을 지지한다. 대응은 사후 측정과 상류 수정이다.

**Artificial Analysis Endpoint Accuracy Index**
- 무엇: 서버리스 API 엔드포인트를 자체 참조 배포와 비교한다(2026-08-04). gpt-oss-120b의 도구 호출은 업체별 22~37%였다. 업체들이 "Quantize weights, write custom kernels and tune their inference stacks"한다고 쓰고, "Serving configuration changes what the model does"라고 적는다. https://artificialanalysis.ai/articles/endpoint-accuracy-index
- 이론에 주는 뜻: 명제 2(배관에서 조용한 품질 손실)의 업계 측정. 대응은 사후 지표다.

**vLLM 야간 정확도 평가**
- 무엇: v0.20.0 회귀가 CI를 빠져나간 뒤 도입했다. 원문 "The model returns a valid response, but the answer is wrong". GSM8K, GPQA, AIME(lm-eval)와 BFCL을 17개 모델·장비 조합에서 매일 밤 돌리고, 출시 후보마다 게이트로 쓴다. https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality
- 앞선 RFC: 정확도 시험을 넓히자는 RFC #32613은 2026-06-20 비활성으로 닫혔다. 본문은 정확도 문제가 "difficult to detect, hard to debug"라고 적었다. https://github.com/vllm-project/vllm/issues/32613
- 이론에 주는 뜻: 엔진 개발자가 조용함을 인정했고, 대응은 종단 평가로 갔다. 평가는 17개 조합 밖의 모델·설정 조합을 보지 못한다.

**lm-evaluation-harness**
- 무엇: 언어 모델 few-shot 평가 프레임워크. 별 14,060개. https://github.com/EleutherAI/lm-evaluation-harness
- 이론에 주는 뜻: 사후 탐지의 사실상 표준 도구다. 결함이 벤치마크에 드러나야만 잡는다.

**OpenRouter Auto Exacto**
- 무엇: 도구 호출 요청을 품질이 좋은 업체로 보낸다. 약 5분마다 업체를 다시 평가하고, JSON 유효성, 스키마 준수, 도구 이름 정확도와 벤치마크 점수를 쓴다. 2026-03-12부터 도구 요청에 기본 적용. https://openrouter.ai/blog/announcements/auto-exacto/
- 이론에 주는 뜻: 시장은 결함을 고치기보다 피해 가는 방식을 택했다.

**MLPerf Inference 정확도 기준**
- 무엇: Llama2-70B는 "99.9% of FP32", Llama3.1-405B·DeepSeek-r1은 "99% of FP16" 같은 목표를 요구하고, LoadGen 정확도 실행으로 확인한다. https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- 이론에 주는 뜻: 벤치마크 제출에는 정확도 하한이 있다. 제출된 설정 밖에서는 보장하지 않는다.

**NVIDIA Polygraphy**
- 무엇: TensorRT, ONNX Runtime 등 여러 백엔드에서 추론을 돌려 결과를 비교하고, 문제가 있는 TensorRT tactic을 분리한다. https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/README.md
- 이론에 주는 뜻: 추론 배포 현장의 표준 차분 도구다. 변환 과정에서 뜻이 어긋나는 결함을 사후에 잡는다.

**llama.cpp perplexity KL-divergence와 test-backend-ops**
- 무엇: 양자화 모델과 FP16의 로짓 분포 KL, 최고 확률 토큰 일치율 등을 잰다. https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md . `test-backend-ops`는 여러 백엔드의 연산 결과가 일치하는지 확인한다. https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp
- 이론에 주는 뜻: 연산 단위와 분포 단위의 사후 대조는 이미 흔하다.

**The Silent Hyperparameter**
- 무엇: 추론 백엔드를 "silent hyperparameter"로 본다. 35,000편의 ML 논문에서 추론 스택 보고 여부를 조사하고 5개 엔진을 비교했다. 백엔드 선택만으로 점수가 최대 16.6%p 달라졌다. 원인으로 prefix caching, CUDA graph, 커스텀 커널, 엔진별 로짓 처리 기본값을 든다. 권고는 추론 스택의 표준 보고다. https://arxiv.org/abs/2605.19537
- 이론에 주는 뜻: 엔진 기본값과 최적화가 결과를 조용히 바꾼다는 명제 2의 정량 근거다. 이 논문의 해법은 보고(문서화)이지 강제가 아니다.

**Kernel Contracts**
- 무엇: ML 커널의 암묵적 약속을 명세하는 언어. 식별자, 범위, 전제, 결과, 허용 오차, 참조 오라클, 측정 절차, 위반 표지의 8요소를 둔다. 정밀도, 순서, 컴파일러 유발, 예외값의 12개 계약 부류를 다룬다. 계약마다 참조를 따르는 구현과 어기는 구현을 하나씩 두는 보정 절차가 있다. https://arxiv.org/abs/2604.22032
- 상태: 1인 저자 논문(2026-04-23). 채택 사례는 확인 못 함.
- 이론에 주는 뜻: 커널 수준의 계약 명세라는 점에서 이론과 가깝다. 다만 하드웨어 사이 적합성 평가용이고, 부품 경계의 역할 사실을 다루지는 않는다.

**학습 쪽 검증기: TrainCheck, Scalify, GraphGuard**
- TrainCheck(OSDI'25): 학습 불변식을 자동으로 추론해 조용한 학습 오류를 찾는다. 재현한 실제 오류 20건 중 18건을 한 반복 안에 잡았고, 새 버그 6건을 찾았다. https://arxiv.org/abs/2506.14813
- Scalify: 최적화·병렬화된 그래프가 원래 그래프와 같은지 등식 포화와 Datalog식 추론으로 검증한다. Llama-3.1-405B를 수 분 안에 검증했고, Amazon 프레임워크에서 새 버그 5건을 찾았다. https://arxiv.org/abs/2509.10694
- GraphGuard: 분산 모델의 출력으로 순차 모델의 출력을 재구성할 수 있는지 반복 재작성으로 증명한다. https://arxiv.org/abs/2508.09505
- 이론에 주는 뜻: 불변식을 사람이 선언하지 않고 추론하는 방향이 학계의 주류다. TrainCheck는 추론한 불변식을 "proactive checks"로 쓴다. 이론의 "필수 선언"과 대비되는 강한 대안이다.

### 6.3 범주 소결

- 사후 탐지는 붐빈다. 학계(Emerge, Ekka, M2K, GRIEF, TrainCheck, Scalify, GraphGuard), 업계(KVV, Artificial Analysis, OpenRouter, vLLM 야간 평가), 도구(Polygraphy, llama.cpp KLD, infer-check)가 모두 있다.
- 공통점은 선언 없이 참조 구현, 벤치마크, 추론한 불변식에 기대는 것이다. 선언 방식이 이들보다 나은 곳은 참조가 없거나 참조와 같은 가정을 공유하는 결함, 그리고 배포 전 시점이다. 이 몫이 얼마인지 잰 1차 자료는 찾지 못했다.
- 모델 회사(Moonshot)와 연구(The Silent Hyperparameter)가 모두 "모델 실력과 배관 결함을 구분해야 한다"고 말한다. 이론의 범위 설정은 업계 문제의식과 맞는다.
- 해법으로 제시된 것은 측정, 보고, 경로 회피, 상류 수정이다. 경계에서 필수 선언을 강제하자는 해법은 이 범주에서 찾지 못했다.
