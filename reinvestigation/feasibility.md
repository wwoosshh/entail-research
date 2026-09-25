# 실현 가능성 평가: 값의 의미를 타입처럼 선언하고 강제해 조용한 오답을 구조적으로 막을 수 있는가

- 작성: 2026-09-23. 평가자 역할로 썼다. 가능하다는 근거와 어렵다는 근거를 같은 무게로 찾았다.
- 대상 이론: 연구자 원문(2026-09-22~23)과 그것을 구조화한 명제 1~5. 이 문서 1절 앞머리에 다시 적는다.
- 표시 규칙
  - 모든 사실에 URL이나 파일 경로를 단다. 로컬 경로는 `<workspace>\`를 뺀 상대 경로로 적는다.
  - [측정]은 이 프로젝트가 잰 값, [문서]는 공식 문서나 논문이 적은 값, [판단]은 이 평가자의 추론이다. 판단은 측정과 섞지 않는다.
  - 찾지 못한 것은 "확인 못 함"으로 적는다.
- 작성 중(절마다 중간 저장).

## 1. 평가 대상과 방법

### 1.1 대상

연구자 원문(2026-09-22~23, 이 평가를 맡긴 지시문에 인용됨)을 다섯 명제로 읽었다.

1. **의미의 중립성.** 부품 사이를 오가는 값과 명령은 형태(dtype, shape, 키 이름)만 전하고 목적, 역할, 의미는 전하지 않는다.
2. **손실.** 받는 쪽이 의미를 다시 찾거나 추측하거나 잘못 해석해서 성능 손실이나 조용한 오답이 생긴다.
3. **해법.** 값과 인자의 의미를 타입이나 역할 표지로 필수 선언하고, 검사기가 강제하고, 부품을 건너는 동안 보존하면 손실을 구조적으로 막는다.
4. **형태.** 기존 모델·엔진에 붙이는 라이브러리와 새 코드를 위한 컴파일러. 성능을 크게 떨어뜨리지 않는다.
5. **진단.** 그래도 의미를 잃는 곳을 짚는 것은 보조 기능이다.

범위는 "모델 자체가 아닌 버그"다. 환각이나 지식 같은 모델의 실력은 뺀다. 목표는 셋이다: 구조적 방어, 대부분의 모델에 붙임, 성능을 크게 떨어뜨리지 않음.

### 1.2 방법

- 1차 출처(공식 문서, 논문, 저장소 코드)를 직접 열었다. PDF는 내려받아 텍스트로 바꿔 읽었다. 내려받은 사본은 세션 임시 폴더에만 두었다.
- 코드 사실은 두 곳에서 읽었다.
  - GitHub 저장소: `gh api -X GET`으로만 읽었다. 인용한 파일은 가능한 한 커밋 SHA를 붙였다.
  - 이 PC의 WSL 가상환경에 설치된 소스: torch 2.14.0+cu130, torchao 0.18.0. 경로는 `~/venvs/gpu/lib/python3.12/site-packages/`다. torch의 `v2.14.0` 태그가 GitHub에 있음을 확인했다(https://github.com/pytorch/pytorch/tree/v2.14.0, 커밋 2b3ec348).
- 이 평가자가 새로 확인한 것은 소스 읽기와 셈뿐이다. 성능은 새로 재지 않았다.
  - 설치된 torch 2.14.0에 named tensor API가 있는가(없다, 6.1절)
  - DTensor 전파 규칙 코드의 줄 수와 연산 이름 수(4.1절)
  - torchao int4 텐서가 구현한 연산 수(4.4절)
- 로컬 수치는 결과 파일(JSON)의 값을 우선했다. 문서(md)에 적힌 값이 결과 파일과 다르면 둘 다 적고 어느 쪽인지 밝혔다(8절).

## 2. 이론적 한계와 가능성

### 2.1 명시적 의미 타입이 보장할 수 있는 것

- 타입 건전성의 고전적 표어는 "잘 타입된 프로그램은 잘못되지 않는다"(Milner 1978)다. 여기서 "잘못"은 언어가 정의한 오류 범주에 한정된다. Wadler와 Findler는 이 계열을 정리하면서, 더 타입된 부분과 덜 타입된 부분이 섞인 프로그램이 잘못되면 책임은 덜 타입된 쪽에 있음을 증명했다. [문서] (https://homepages.inf.ed.ac.uk/wadler/papers/blame/blame.pdf, 1절. 저자 사본은 ICFP 2008 제출본이고, 출판은 ESOP 2009다.)
- 이 이론에 옮기면 보장할 수 있는 것은 넷이다. [판단]
  1. **선언끼리의 불일치.** 생산자가 A라고 선언하고 소비자가 B를 요구하면 빠짐없이 잡는다. 단 두 선언이 모두 참이라는 전제에서다. 이 프로젝트의 가상 평가도 정적 역할 타입(S1)을 "양쪽 선언이 참이라고 가정"하고 판정했다(`realworld/replay_vllm_sglang.md` 21행).
  2. **선언의 부재.** 필수 선언이면 선언이 없는 곳이 "모름"으로 드러난다. 기본값으로 조용히 대신하지 않고 멈출 수 있다.
  3. **닫힌 집합의 망라.** 형식을 닫힌 목록으로 두면 목록 밖의 값은 오류가 된다.
  4. **책임의 위치.** 어느 경계의 어느 선언이 어긋났는지 지목한다(2.4절의 blame).
- 이 프로젝트의 근거 [측정]:
  - 주입한 역할 실수 8종은 기존 스택에서 모두 오류 없이 틀린 답을 냈다. 기준값과의 상대 차이는 24%~385%였다.
  - rolec은 2종을 실행 전에 거부했고, 3종은 표현 자체가 불가능했고, 1종은 언어 규칙으로 사라졌고, 2종은 올바르게 실행했다.
  - 출처: `phase0/week4/results/e5_role_errors.json`.
  - 이 8종은 저자가 고른 것이고 실제 빈도는 재지 않았다(`phase0/week4/WEEK4_NOTES.md` 10절).

### 2.2 보장할 수 없는 것

**(가) 타입은 의도와 코드를 대조할 뿐, 코드와 실제 실행을 대조하지 않는다.**
- 타입 검사는 선언(의도)과 코드가 맞는지를 본다. 컴파일된 코드가 실제로 그렇게 도는지는 컴파일러, 커널, 하드웨어가 옳다는 가정 위에 있다. [판단]
- 검증된 컴파일러도 이 틈을 남긴다. [문서] (https://users.cs.utah.edu/~regehr/papers/pldi11-preprint.pdf, 초록과 3.1절)
  - Csmith 연구는 3년 동안 325건이 넘는 새 컴파일러 결함을 보고했다. 시험한 모든 컴파일러가 올바른 입력에 조용히 틀린 코드를 냈다.
  - CompCert의 검증된 중간 단계에서는 약 6 CPU-년을 들여도 틀린 코드 결함을 찾지 못했다.
  - 하지만 검증되지 않은 앞단에서 결함 6건이 나왔다. 또 PowerPC 의미 명세가 즉시값 폭 제약을 빠뜨린 결함도 나왔다. 증명도 명세가 맞다는 가정 위에 있다.
- AI 추론 서비스에서 실제로 일어났다. [문서]
  - Anthropic 사후 분석(2025-09-17)에는 세 결함이 있다(https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues).
    - XLA:TPU 컴파일러 결함: 근사 top-k에서 bf16과 fp32의 정밀도가 어긋나 "highest probability token to sometimes disappear". 증상은 앞뒤에 어떤 연산이 돌았는지, 디버그 도구를 켰는지에 따라 달라졌다.
    - TPU 서버 설정 오류: 영어 질문에 태국어·중국어 글자가 섞였다.
    - 문맥 창 라우팅 오류: 요청이 다른 문맥 길이로 설정된 서버로 갔다.
  - OpenAI(2024-02-20): "inference kernels produced incorrect results" 특정 GPU 구성에서(https://status.openai.com/incidents/ssg8fh7sfyz3).
- 두 사례는 연구자가 든 증상(다른 언어 글자가 섞임)과 같은 모양이다. [판단]
  - 원인이 컴파일러나 커널 내부에 있으면 부품 경계의 선언은 그 결함을 보지 못한다.
  - 경계 선언이 잡을 수 있는 경우는 결함이 경계를 넘는 사실의 불일치로 드러날 때뿐이다. 예를 들어 "이 연산은 fp32 누산을 요구한다"는 요구가 선언되어 있고, 소비자의 설정이 그것을 어기는 경우다.

**(나) 하드웨어 결함** [문서]
- Google은 제조 결함 때문에 조용히 틀린 계산을 하는 코어를 "mercurial"이라 부른다. 수천 대당 몇 개꼴로 관측했다(https://www.sigops.org/s/conferences/hotos/2021/papers/hotos21-s01-hochschild.pdf, HotOS '21).
- Meta는 18개월 넘게 수십만 대를 시험해 수백 개 CPU에서 조용한 데이터 손상을 찾았다(https://arxiv.org/abs/2102.11245).
- 이 층은 타입이 다루는 범위 밖이다. [판단]

**(다) 부동소수점 허용 오차**
- PyTorch 공식 문서 [문서] (https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html):
  - 부동소수점 덧셈과 곱셈은 결합법칙이 성립하지 않아서 연산 순서가 결과를 바꾼다.
  - `(A@B)[0]`과 `A[0]@B[0]`도 비트 단위로 같다고 보장하지 않는다.
  - 릴리스, 커밋, 플랫폼 사이에서도 비트 단위 동일성을 보장하지 않는다.
- 그래서 값을 비교하려면 허용 오차를 정해야 한다. `torch.testing.assert_close`의 기본 상대 허용 오차는 bf16 1.6e-2, fp16 1e-3, fp32 1.3e-6이다(https://docs.pytorch.org/docs/2.14/testing.html). [문서]
- 배치 불변성: 같은 입력을 온도 0으로 1,000번 돌렸더니 서로 다른 답이 80가지 나왔다. 배치 불변 커널을 쓰면 vLLM에서 1.6~2.1배 느렸다(https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/, 2025-09-10). [문서]
- 이 이론에 주는 뜻은 두 가지다. [판단]
  - **선언 대조의 강점.** 역할 결함 가운데 일부는 수치 잡음보다 작은 차이만 낸다.
    - 이 프로젝트가 찾은 transformers 내장 Flex 경로의 한 칸 초과 결함이 그 예다. 새 캐시에서는 sdpa와의 로짓 최대 차이가 0.06으로 bf16 잡음 수준이었고, 이전 데이터가 남은 캐시에서만 0.81로 드러났다. [측정] (`phase0/week4/WEEK4_NOTES.md` 7.2절, `phase0/week4/results/diag_hf_flex_offset.json`)
    - 값을 비교하는 사후 탐지는 이런 결함을 잡음과 구별하기 어렵다. 선언 대조는 값이 아니라 사실(위치가 값으로 넘어왔는가 참조로 넘어왔는가)을 보므로 이 한계가 없다.
  - **선언 검증의 약점.** 명제 3의 "선언을 실제 데이터로 검증"하는 단계는 허용 오차 문제를 그대로 안는다. "이 가중치는 v 예측이다"를 동작으로 판정하는 탐침은 통계적이고 틀릴 수 있다(7.4절 M7).

**(라) 결정 불가능성 때문에 선언이 필요하다** [문서]
- 함수에 대한 술어는 일반적으로 결정 불가능하다. Findler와 Felleisen은 그래서 고차 계약을 값이 실제로 쓰일 때까지 미뤄 검사한다(https://www2.ccs.neu.edu/racket/pubs/icfp2002-ff.pdf, 초록과 1절).
- 정제 타입도 검사를 결정 가능하게 하려고 논리를 제한한다. Liquid Types는 부분 타입 판정을 결정 가능한 논리의 함의 검사로 줄이고, 정제를 주어진 한정자들의 논리곱으로 제한한다(https://goto.ucsd.edu/~rjhala/liquid/liquid_types.pdf, PLDI'08, 1절).
- 결론 [판단]:
  - 코드만 보고 의미를 모두 되찾을 수는 없다. 되찾을 수 없는 사실은 누군가 선언해야 한다. 이것이 선언이 필요한 근거다.
  - 같은 이유로 선언이 틀려도 코드만으로는 알 수 없다. 이것이 (마)의 신뢰 경계로 이어진다.
  - 로컬 사례: 배치와 무관하게 같은 답이라는 성질은 프로그램의 사실이 아니라 프로그래머의 요구라서, 아무리 잘 찾는 컴파일러도 찾아낼 수 없다(`phase0/week4/WEEK4_NOTES.md` 9절 4번).

**(마) 신뢰 경계** [문서]
- 검사기가 확인하지 못하는 선언은 믿을 수밖에 없다. 대표 사례는 다음과 같다.
  - **C `restrict`.** 요구를 어기면 행동이 정의되지 않는다. 또 번역기는 restrict의 별칭 의미를 무시해도 된다(C11 초안 N1570 6.7.3.1절 4·6항, https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf).
  - **PyTorch `custom_op`의 `mutates_args`.** "This MUST be accurate, otherwise, the behavior is undefined."(https://docs.pytorch.org/docs/2.14/library.html)
    - 로컬 확인: torch 2.14는 `mutates_args=()`로 선언하고 실제로는 쓰는 연산을 오류 없이 실행한다. entail의 경계는 텐서 버전 카운터로 이것을 잡는다. [측정, 시험 코드] (`entail/DESIGN.md` 8절, `entail/tests/test_boundary.py` 164~183행)
  - **Meta spmd_types.** 사용자가 쓴 타이핑 규칙은 "completely trusted"다. 틀린 규칙을 검사로 찾을 수 없어서 `rulecheck`로 수치 시험을 한다(https://github.com/meta-pytorch/spmd_types/blob/bfa07e3a8e0ee9d25e1cf2326dc0acdb1080b11e/docs/rules.md).
  - **Rust.** Android에서 약 4%의 코드가 `unsafe{}` 블록 안에 있고, 보안 우려가 그곳에 모인다(https://blog.google/security/rust-in-android-move-fast-fix-things/, 2025-11-13).
- 증명 운반 코드는 신뢰 경계를 줄이는 방법이다. 생산자가 증명을 붙이고 소비자는 작은 검사기로 확인한다. OSDI'96 보고에서는 증명 검증이 한 번에 1~3 ms였고, 실행 중 부담은 없었다(https://www.usenix.org/legacy/publications/library/proceedings/osdi96/necula.html). [문서]
- 이 이론이 믿어야 하는 선언은 셋이다. [판단]
  1. **산출물의 선언**(config.json, safetensors 메타데이터). 틀릴 수 있다. 가상 평가에서 vLLM #51063은 유일하게 있던 선언(tie 플래그) 자체가 틀린 경우였다(`realworld/replay_vllm_sglang.md` 84행).
  2. **엔진·커널의 능력표**(`KernelCaps`). 사람이 적는다. 이 프로젝트는 성질을 붙인 실행과 뗀 실행을 비교하는 `cap_probe`로 능력표를 검증했다(`entail/DESIGN.md` 2.1절).
  3. **어댑터의 번역**("그 숫자가 이 엔진의 어디에 있는가"). 엔진이 바뀌면 낡는다(3.1절 몽키패치).

### 2.3 정적 검사와 실행 중 검사의 경계

사실을 언제 알 수 있는지로 나누면 경계가 분명해진다. [판단]

| 종류 | 예 | 알 수 있는 때 | 검사 자리 |
|---|---|---|---|
| 설정으로 정해지는 사실 | 예측 방식, 저장 형식의 종류, 위치 기준, softcap 유무, 합산 상태 | 적재 때나 추적(trace) 때 | 한 번만 검사하면 된다 |
| 값으로 정해지는 사실 | KV 유효 길이, 청크 끝, 버퍼 준비 여부 | 실행 중에만 | 매번 검사해야 한다 |

- **설정으로 정해지는 사실은 비용 없이 강제한 선례가 있다.** [문서]
  - JAX 명시적 샤딩은 샤딩을 JAX 타입에 넣고 추적 시점에 전파한다. 결과 샤딩이 모호하면 하나를 고르지 않고 오류를 내며 `out_sharding`을 요구한다(https://docs.jax.dev/en/latest/parallel.html).
  - jaxtyping은 `jax.jit` 아래에서는 추적 중에만 모양을 검사해서 실행 성능에 영향이 없다(https://docs.kidger.site/jaxtyping/api/runtime-type-checking/).
- **값으로 정해지는 사실은 실행 방식과 부딪친다.** [문서] (https://docs.pytorch.org/docs/2.14/notes/cuda.html, CUDA Graphs 절)
  - CPU와 GPU를 동기화하는 연산(`.item()` 등)은 캡처 중 금지다.
  - 캡처된 CPU 작업은 재생 때 생략된다. 따라서 그래프 안에 둔 파이썬 검사는 재생 때 돌지 않고, 장치 값을 호스트로 읽는 검사는 캡처를 깨뜨린다.
  - vLLM V1은 기본으로 균일 디코드 배치를 전체 CUDA Graph로 캡처한다(https://docs.vllm.ai/en/latest/design/cuda_graphs.html).
- **이 프로젝트의 측정도 같은 경계를 보였다.** [측정]
  - 정적 캐시에서 갱신 경계 안에 장치 비교 검사를 넣으면 47.7배 느렸고 출력이 바뀌었다(`entail/audits/cache_contract_cost.json`의 `healthy_static`).
  - 요청이 끝난 뒤 밖에서 한 번 검사하면 1.007배였고 출력이 같았다(`entail/audits/cache_contract_static.json`).
  - rolec은 호출마다 `int(last.max())`로 장치 값을 호스트로 읽는다(`phase0/week4/rolec.py`의 `주목`). 이 방식은 캡처 구간 안에 둘 수 없다. [판단, 코드 읽기]
- **파이썬 정적 타입 검사기는 텐서의 사실을 보지 못한다.** jaxtyping FAQ는 `dtype[array, shape]` 주석을 정적 검사기가 그냥 `array`로 다뤄야 한다고 적는다(https://docs.kidger.site/jaxtyping/faq/). [문서] 그래서 새 코드에서 텐서 역할을 "컴파일 단계"에서 검사하려면 mypy류가 아니라 추적 시점 검사나 전용 프런트엔드가 필요하다. [판단]

### 2.4 관련 이론이 이 이론에 주는 것

| 이론 | 핵심 사실 [문서] | 이 이론에 주는 것 [판단] |
|---|---|---|
| 정제 타입 | 값의 집합을 논리 술어로 좁힌다. 결정 가능성을 위해 논리를 제한한다. DML에서는 코드의 약 31%(줄 수로 17%)가 수동 주석이었고, Liquid Types는 배열 경계 한정자 묶음으로 이를 1% 미만으로 줄였다. 저자들은 이 주석 부담이 의존 타입의 채택을 막았다고 본다(https://goto.ucsd.edu/~rjhala/liquid/liquid_types.pdf). | `Valid(length)`, `Positions(frame)` 같은 사실은 정제 타입이다. 술어를 작고 닫힌 어휘(길이, 기준, 종류)로 두면 결정 가능성과 주석 부담이 함께 관리된다. |
| 점진적 타입(건전성과 비용 논쟁) | Takikawa 외(POPL 2016): Typed Racket 벤치마크 12개에서 부분 타입 구성의 최대 부담이 1.25~168배, 평균 0.6~68배였고, "현재 구현 기술로는" 건전한 점진적 타입이 죽었다고 결론 내렸다. 저자들은 실무가 2배 넘는 부담은 받아들이지 않으리라 봤다(https://www2.ccs.neu.edu/racket/pubs/popl16-tfgnvf.pdf, 4.2절·각주 7·7절). 반론 1: Bauman 외(OOPSLA 2017)는 Pycket JIT가 부담의 90% 이상을 없앤다고 보였다(https://2017.splashcon.org/details/splash-2017-OOPSLA/10/Sound-Gradual-Typing-Only-Mostly-Dead). 반론 2: Muehlboeck·Tate(OOPSLA 2017)는 명목 타입으로 설계하면 최악 부담이 10% 미만이고, 부담은 타입 있는 코드와 없는 코드 사이를 오가는 횟수에 비례한다고 했다. 같은 논문은 Reticulated Python의 transient 방식이 200~400%이고, 책임 추적을 더하면 흔히 두 배 넘게 는다고 적었다(https://www.cs.cornell.edu/~fabianm/papers/nomalive-oopsla17-tr.pdf, 1절·9.3~9.4절). | 비용은 "경계를 몇 번, 얼마나 깊이 검사하느냐"가 정한다. 이름표(명목) 비교는 싸고, 깊은 구조 검사나 감싸기는 비싸다. 연산마다 가로채기(2.19배)와 그릇 경계 계약(약 1.0배)이라는 이 프로젝트의 결과와 같은 방향이다(8절). |
| 계약과 책임 추적(blame) | 1차 계약은 호출 때 검사하고, 인자가 틀리면 호출자, 결과가 틀리면 함수 자신을 탓한다. 고차 계약은 감싸서 미룬다(https://www2.ccs.neu.edu/racket/pubs/icfp2002-ff.pdf, 1절). 더 타입된 쪽과 덜 타입된 쪽이 섞이면 책임은 덜 타입된 쪽에 있다(https://homepages.inf.ed.ac.uk/wadler/papers/blame/blame.pdf). | 경계 오류 문구는 "어느 생산자가 무엇을 선언했고 어느 소비자가 무엇을 요구했는가"를 담아야 쓸모가 있다. 선언이 없는 엔진 내부(덜 타입된 쪽)가 책임 위치로 지목되는 구조가 자연스럽다. 이 프로젝트의 `Invalidated` 표지(누가 무효로 만들었는지 남김)가 이 역할이다(`entail/audits/PROPAGATE.md`). |
| 증명 운반 코드 | 생산자가 증명을 붙이고 소비자는 작은 검사기로 확인한다. 검증은 한 번에 1~3 ms였다(https://www.usenix.org/legacy/publications/library/proceedings/osdi96/necula.html). | 체크포인트가 사실을 선언하고 적재기가 대조하는 구조와 같다. 저장 형식(바이트, stride, 스케일 dtype)은 결정적으로 대조할 수 있다. 그러나 "v 예측" 같은 뜻은 바이트에서 증명할 수 없어서 대조가 통계적이다(7.4절). |
| typestate(시점과 소유) | 객체에 허용되는 연산이 실행 중 상태에 따라 바뀐다(Strom·Yemini 1986, IEEE TSE 12(1), https://dblp.org/rec/journals/tse/StromY86.html). 별칭이 있으면 추적하기 어렵고, 그래서 unique·immutable 같은 권한 체계를 쓴다(Aldrich 외, https://www.cs.cmu.edu/~aldrich/papers/onward2009-state.pdf, 1절·3.3절). | TIME 사실(버퍼의 epoch, 참조로 넘긴 카운터)은 typestate 문제다. 이 프로젝트가 찾은 transformers 결함은 위치를 값이 아니라 카운터 텐서의 참조로 넘겨서, 쓰는 쪽이 올린 값을 읽는 쪽이 본 것이다(`phase0/week4/WEEK4_NOTES.md` 7.2절). 파이썬에는 별칭 통제가 없어서 이것을 정적으로 막을 수 없다. 실행 중 버전 카운터나 "호출 시점 값으로 묶는다" 같은 규칙으로만 막는다. |

### 2.5 증상과 원인의 관계

연구자가 든 증상에는 모두 배관 원인과 모델 원인이 함께 있다.

| 증상 | 배관(부품 경계) 원인의 예 | 모델·디코딩 원인의 예 |
|---|---|---|
| 다른 언어 글자가 섞임 | TPU 설정 오류(Anthropic 사후 분석, 2.2절) | DeepSeek-R1-Zero가 "poor readability, and language mixing" 문제를 겪었다(https://arxiv.org/abs/2501.12948v1 초록) |
| 같은 말 반복 | KV 캐시의 모든 층에서 토큰 하나를 떼자 모델이 질문을 되풀이했다 [측정] (`entail/audits/CACHE_CONTRACT.md`, `entail/audits/cache_contract_cost.json`의 `seeded_contract_off`) | 우도를 최대화하는 디코딩은 "bland and strangely repetitive" 글을 낸다(https://arxiv.org/abs/1904.09751 초록) |
| 맥락·논점 상실 | 문맥 창 라우팅 오류(Anthropic 사후 분석) | 긴 문맥의 중간에 있는 정보에서 성능이 크게 떨어진다(https://arxiv.org/abs/2307.03172 초록) |
| 이유 없는 성능 저하 | 사실이 지워지면 디코드가 1.3~2.5배 느렸다 [측정] (`phase0/week4/WEEK4_NOTES.md` 6절) | (해당 없음) |
| 이미지 생성 결과가 깨짐 | v 예측 모델을 eps로 돌려 기준 그림과 평균 82.64~94.78(0~255 화소값) 차이 [측정] (`issue_track/comfyui_field_test/g2_compare.json`의 `M7_vs_a_as_comfyui`) | 확인 못 함 |

- 이 프로젝트가 모은 실제 역할 계열 버그 50건의 증상 태그는 garbled 26, plausible_but_wrong 24, repetition 1, language_mixing 0이었다 [측정, 이 프로젝트 자체 분류] (`realworld/symptom_tally.json`). 연구자가 든 증상 이름은 실제 버그 보고에서 드물었다.
- 함의 [판단]:
  - 증상은 성공 기준이 될 수 없다. 이론이 막을 수 있는 것은 증상이 아니라 원인 부류(부품 경계에서 사실이 사라지거나 어긋나는 것)다.
  - 연구자가 범위를 "모델자체가 아닌 버그"로 좁혔으므로 모순은 아니다. 다만 "언어 누출을 구조적으로 막는다"는 표현은 "언어 누출을 일으키는 배관 결함 가운데 경계 사실로 드러나는 것을 막는다"로 좁혀 말해야 정확하다.

## 3. 기존 모델과 엔진을 고치지 않고 붙일 수 있는가

### 3.1 PyTorch의 수단

**(1) `__torch_function__`과 텐서 하위 클래스**
- 무엇을 보나 [문서] (https://docs.pytorch.org/docs/2.14/notes/extending.html):
  - 사용자 타입이 `__torch_function__`을 정의하면, 그 인스턴스가 `torch` 네임스페이스 함수에 들어갈 때 PyTorch가 그 구현을 부른다.
  - 1.7부터는 `torch.Tensor` 하위 클래스에 연산을 적용하면 결과도 하위 클래스로 나온다.
- torch.compile과의 관계 [문서, 설치본 소스] (`torch/utils/_python_dispatch.py` 457~557행, `torch/_dynamo/guards.py` 1134~1185행, torch 2.14.0):
  - 컴파일러가 하위 클래스를 따라가려면 `__tensor_flatten__`/`__tensor_unflatten__` 규약(`TraceableWrapperSubclass`)을 지켜야 한다.
  - 소스의 설명에 따르면 규약을 지키는 것은 필요하지만 충분하지 않다. `__torch_dispatch__` 구현이 추적 가능한 디스패처 연산으로 풀려야 하고, 별칭 의미도 지켜야 한다.
  - Dynamo는 하위 클래스의 메타데이터에 가드(`TENSOR_SUBCLASS_METADATA_MATCH`)를 건다.
  - [판단] 스텝마다 바뀌는 사실(예: 유효 길이)을 메타데이터로 들면 가드가 맞지 않아 캐시를 다시 찾거나 재컴파일할 위험이 있다. 이런 사실은 텐서로 들어야 한다.
- CUDA Graph와의 관계 [문서]: 하위 클래스의 파이썬 디스패치 코드는 CPU 작업이므로 캡처 때만 돌고 재생 때는 생략된다(https://docs.pytorch.org/docs/2.14/notes/cuda.html).
- 비용 [문서]: 아무 일도 하지 않는 `__torch_function__` 하위 클래스를 입력에 쓰자 torchvision 모델의 학습 반복이 5.3~23.3% 느려졌다. PyTorch 1.12(2022) 기준이고 이슈는 아직 열려 있다(https://github.com/pytorch/pytorch/issues/73860). 최신 버전의 값은 확인 못 함.
- 연산마다 구현이 필요하다 [설치본 소스]: torchao의 기반 클래스는 등록되지 않은 연산을 `NotImplementedError`("attempting to run unimplemented operator/function")로 거부한다(`torchao/utils.py` 678~697행, torchao 0.18.0).

**(2) `__torch_dispatch__`와 `TorchDispatchMode`**
- 무엇을 보나 [문서]: autograd 아래에서 aten 네이티브 API로 가는 호출을 모두 가로챈다. 모드를 켜면 모든 함수가 모드를 추가 인자로 받은 것처럼 동작한다(https://docs.pytorch.org/docs/2.14/notes/extending.html).
- torch.compile과의 관계 [설치본 소스, torch 2.14.0]:
  - 기반 시설용이 아닌 모드가 켜져 있으면 Dynamo가 그 프레임의 컴파일을 건너뛴다. 남기는 사유는 "non-infra torch dispatch mode present, this is not supported today in torch.compile"이다(`torch/_dynamo/convert_frame.py`, https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_dynamo/convert_frame.py).
  - 모드가 `ignore_compile_internals()`를 참으로 두면 컴파일 과정 동안 모드가 꺼진다. 그래서 컴파일로 융합된 연산은 모드에게 보이지 않는다. 기본값(거짓)이면 torch.compile이 eager 의미를 지키려고 보통 eager로 되돌아간다(`torch/utils/_python_dispatch.py` 238~266행의 설명).
  - [판단] 연산 단위 가로채기와 컴파일 성능은 함께 가질 수 없다. 둘 중 하나를 포기해야 한다.
- 비용 [측정]: 이 프로젝트의 전파 시제품은 Qwen3-4B eager 복호 64토큰에서 2.19배 느렸고, 가로챈 연산은 273,413개였다(`entail/audits/propagate_cost.json`). 8.2절에 문서 값과의 차이를 적었다.

**(3) `torch.library` 커스텀 연산**
- 컴파일러에게 불투명한 호출로 남는다. 튜토리얼은 이것을 "opaque callable with respect to torch.compile"로 설명한다(https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html). [문서]
- `mutates_args`는 필수 키워드 인자다. 그러나 정확성은 믿을 뿐이고, 틀리면 행동이 정의되지 않는다(https://docs.pytorch.org/docs/2.14/library.html). [문서]
- `torch.library.opcheck`가 스키마(변경·반환 표기), 가짜 커널, autograd 등록, 컴파일 경로를 시험한다. 시험용 도구다(같은 문서). [문서]
- [판단] 엔진이 이미 커스텀 연산으로 만든 경계에는 스키마를 읽어 붙일 수 있다(이 프로젝트의 `entail/core.py`가 그렇게 한다). 그러나 커스텀 연산이 아닌 파이썬 함수 경계는 감싸기로 붙여야 한다.

**(4) 모듈 훅** [문서] (https://github.com/pytorch/pytorch/blob/v2.14.0/docs/source/user_guide/torch_compiler/torch.compiler_nn_module.md)
- 컴파일 전에 걸고 그 뒤 바꾸지 않으면 forward·pre-forward 훅은 지원된다.
- 컴파일러가 불투명하게 부르는 'allowed modules'에 건 훅은 그래프 끊김을 일으킨다. 모델에 따라 성능이 크게 떨어질 수 있다고 문서가 적는다.
- 기본 설정 `skip_nnmodule_hook_guards=True`에서는 컴파일 뒤에 훅 사전이 바뀌어도 알아채지 못한다. state_dict 훅은 지원하지 않는다.
- 컴파일이 일어난 뒤 등록한 훅이 돌지 않는다는 보고가 있었다(https://github.com/pytorch/pytorch/issues/117758, 2024-06-10 닫힘).
- [판단] 검사기가 조용히 꺼지는 경로다. "실패는 시끄럽게"라는 원칙과 충돌한다.

**(5) 감싸기(몽키패치)** — 이 프로젝트의 경험 [측정·기록]
- 걸었던 자리:
  - transformers: `PreTrainedModel._check_and_adjust_attn_implementation`. 밑줄로 시작하는 비공개 메서드다.
  - SGLang: `ModelRunner.init_attention_backends`
  - vLLM: `KVCacheManager.allocate_slots`
  - 엔진이 자식 프로세스에서 백엔드를 정하기 때문에 `sitecustomize`로 모든 프로세스에 걸어야 했다.
  - 출처: `entail/DESIGN.md` 2.1절, `entail/audits/CACHE_CONTRACT.md`.
- 헛디딘 기록 (`entail/audits/CACHE_CONTRACT.md`):
  - `CacheLayerMixin.update`는 추상이라, 처음에는 갱신 횟수가 0으로 나왔다. 하위 클래스마다 감싸서 고쳤다.
  - SGLang 탐침이 `__slots__` 객체에 `vars()`를 불러 예외를 냈고, 스케줄러가 죽었다.
- 비용:
  - 시작 점검 한 번에 2.83 µs였다(`entail/audits/adapter_pilot.json`의 `overhead`).
  - 경계 검사는 호출당 0.12~5.0 µs였다(CPU, 16원소 텐서, `entail/audits/boundary_22_cost.json`).
- [판단] 비공개 이름에 묶이므로 엔진 버전이 바뀔 때마다 깨질 수 있다. 붙이기는 가능하지만 유지 비용이 엔진 수와 버전 수에 비례한다. 계측 코드가 예외를 내면 엔진을 죽일 수 있으므로, 계측은 절대 예외를 밖으로 내지 않아야 한다.

### 3.2 JAX의 수단

JAX에는 텐서 하위 클래스가 없다. 대신 프로그램 변환이 있다.

| 수단 | 무엇을 하나 [문서] | 제약과 비용 [문서] |
|---|---|---|
| 사용자 jaxpr 해석기 | jaxpr를 훑으며 원시 연산마다 규칙을 표에서 찾아 새 변환을 만든다(https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html) | 규칙이 없는 원시 연산은 문서 예제에서 `NotImplementedError`다. 원시 연산마다 규칙이 필요하다 |
| Quax | 사용자 변환으로 배열 같은 객체(LoRA, 양자화 배열, 단위 달린 배열)를 기존 프로그램에 그대로 넣는다. 원래 코드를 고치지 않아도 된다(https://github.com/nstarman/quax README) | 규칙이 등록되지 않은 원시 연산에서는 기본 규칙이 객체를 보통 배열로 `materialise`한 뒤 계산한다(`src/quax/_values.py`). [판단] 기본 동작에서는 사실이 조용히 떨어질 수 있다 |
| pytree 정적 필드 | `meta_fields`는 `jax.jit`에서 정적으로 다룬다. 정적이고 해시 가능하고 불변이어야 하며 JIT 캐시 키가 된다(https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_dataclass.html) | [판단] 사실을 정적 필드로 들면 사실이 바뀔 때마다 다시 추적한다. 설정 사실에 맞고, 스텝마다 바뀌는 사실에는 맞지 않다 |
| checkify | jit 안에서 쓸 수 있는 실행 중 검사다. 일반 assert는 추적 중 값이 없어서 안 된다(https://docs.jax.dev/en/latest/debugging/checkify_guide.html) | 검사를 많이 넣으면 비싸다고 문서가 적는다(모든 원시 연산에 NaN 검사를 넣는 예) |
| jaxtyping, 명시적 샤딩 | 추적 시점에 검사한다(2.3절) | 실행 비용이 없다 |
| custom_partitioning | XLA가 속을 볼 수 없는 커스텀 호출에 SPMD 규칙을 붙인다. Shardy용으로는 einsum 비슷한 `sharding_rule` 문자열을 준다(https://docs.jax.dev/en/latest/jax.experimental.custom_partitioning.html) | 불투명한 연산마다 규칙을 사람이 적어야 한다 |

- [판단] JAX에서는 "추적 시점에 한 번"이 자연스러운 검사 자리다. 설정 사실은 이 자리에서 비용 없이 강제할 수 있다. 값 사실은 checkify 같은 그래프 안 검사로만 가능하고, 비용이 든다.

### 3.3 정리

[판단] 고치지 않고 붙일 수 있는 자리와 어려운 자리는 다음과 같다.

| 자리 | 붙일 수 있나 | 근거 |
|---|---|---|
| 적재·서버 시작 경계(설정, 메타데이터, 능력표) | 가능. 비용은 한 번 µs 단위다 | 시작 점검 2.83 µs(`entail/audits/adapter_pilot.json`). JAX 추적 시점 검사, 증명 운반 코드의 한 번 검증(2.2절) |
| 호스트 쪽 파이썬 경계(스케줄러, 캐시 관리자) | 가능. 사실이 파이썬 정수로 있으면 비용이 잡히지 않는다 | 8.3절의 그릇 계약. vLLM `allocate_slots`, SGLang `prepare_for_decode` |
| 요청 경계 밖(끝난 뒤 한 번) | 가능. 대신 탐지가 늦다 | 정적 캐시 1.007배(`entail/audits/cache_contract_static.json`) |
| 컴파일·캡처 구간 안의 연산 단위 검사 | 어렵다. 가시성과 컴파일 성능 중 하나를 잃는다 | 3.1절 (2), CUDA Graph 문서, 47.7배와 출력 변화(`entail/audits/cache_contract_cost.json`) |
| 커널 안 | 이 방식으로는 불가 | 경계 선언은 커널 내부를 보지 못한다(2.2절 (가)) |

- "대부분의 모델에 붙인다"의 단위는 모델이 아니라 엔진이다. 모델은 파일 선언을 통해 들어오고, 붙이는 일은 엔진마다 한다. 규칙과 어휘는 공유하고 엔진마다 번역(어댑터)을 쓴다. 이 프로젝트에서 공유 규칙은 73줄, 엔진 하나를 붙이는 어댑터는 97~109줄이었다(`entail/audits/CACHE_CONTRACT.md` "하나의 어휘로 모았다" 절. 문서 값이며, 줄 수를 결과 파일로 다시 세지는 않았다).

## 4. 의미의 전파: 실제 제품의 사례

### 4.1 PyTorch DTensor

- **연산마다 규칙이 있다.** [설치본 소스, torch 2.14.0]
  - 샤딩 전파 규칙은 `torch/distributed/tensor/_ops/`에 있다.
  - 이 평가자가 세어 보니 이 폴더의 파이썬 파일은 모두 12,762줄이다. 파일들이 참조하는 `aten.<연산>.<오버로드>` 이름은 서로 다른 것이 454개다.
  - 등록 함수 이름의 등장 횟수는 `register_single_dim_strategy` 107, `register_op_strategy` 27, `register_prop_rule` 6, `register_sharding` 1이다. 가져오기(import)와 정의 줄도 포함한 등장 횟수라서, 등록된 연산 수와 같지 않다.
- **규칙이 없으면 멈춘다.** 규칙이 없는 연산은 "Operator ... does not have a sharding strategy registered."로 `NotImplementedError`를 낸다(`torch/distributed/tensor/_sharding_prop.py` 1046행). [설치본 소스]
- **규칙 자체가 틀릴 수 있어서 검증 도구가 있다.** `torch/distributed/tensor/_ops/strategy_validation.py`는 전체 텐서로 정답을 구하고, 여러 배치 조합을 모사해 비교한다. 그리고 "incorrect rules (DTensor claims valid but wrong)"와 빠진 규칙을 보고한다(같은 파일 1~20행). [설치본 소스]
- **실행 중 비용 때문에 지우자는 제안이 나왔다.** [문서] (https://blog.ezyang.com/2026/02/dtensor-erasure/, 2026-02-01)
  - DTensor의 eager 성능은 "famously terrible"이라고 적었다.
  - 근거로 한 논문의 측정을 인용한다. DTensor를 쓰면 학습이 종단 간 35~60% 느렸고, DTensor 연산이 실제 계산보다 7배 이상 걸렸다. 인용된 논문은 veScale(https://arxiv.org/abs/2509.07003)이다. 이 수치를 논문 본문에서 직접 확인하지는 못했다.
  - 제안은 "DTensor erasure"다. 실행은 보통 텐서로 하고, 컴파일러는 "optional type checker" 같은 곁다리 정적 분석 도구로 쓴다.
- [판단] 제품 수준의 전파도 (1) 연산마다 규칙, (2) 규칙 검증 도구, (3) 운영에서는 지우기로 귀결되었다. 이 이론이 "부품을 건너는 동안 보존"을 운영 중 상시 전파로 구현하려 하면 같은 벽을 만난다.

### 4.2 JAX, GSPMD, Shardy

- **GSPMD.** 사용자는 단일 장치용으로 프로그램을 쓰고 "give hints through a few annotations"로 텐서 분배를 알려 주면, 컴파일러가 전파해 병렬화한다(https://arxiv.org/abs/2105.04663 초록). [문서]
- **Shardy.** [문서] (https://openxla.org/shardy/propagation)
  - 연산마다 `OpShardingRule`이 있다. 어떤 차원이 어떤 축으로 나뉘면 같은 인자(factor)를 가진 다른 피연산자·결과의 차원도 같은 축으로 나눈다.
  - 데이터 흐름 연산은 `ShardableDataFlowOpInterface`로 다룬다. 커스텀 연산은 사용자가 그 메서드를 구현해야 한다.
  - 사용자 우선순위가 가장 높다. 전파는 우선순위가 더 낮은 단계에서 사용자 샤딩을 덮어쓰지 않는다.
- **custom_partitioning.** XLA가 속을 볼 수 없는 커스텀 호출에는 사람이 규칙을 적는다(3.2절). [문서]
- **명시적 모드.** 샤딩이 타입에 들어가고, 추적 시점에 전파되고, 모호하면 오류다. 아직 합산되지 않은 값(unreduced)도 타입으로 표현한다(https://docs.jax.dev/en/latest/parallel.html). [문서]
- [판단] 자동 모드의 샤딩 주석은 GSPMD 초록의 표현대로 "힌트"다. 컴파일러가 통신을 넣어 단일 장치 의미를 지키도록 설계되어 있으므로, 힌트가 나쁘면 대개 느려질 뿐 답이 틀리지는 않는다. 반면 이 이론이 다루는 사실(예측 방식, 위치 기준, 저장 형식)은 뜻을 바꾼다. 사실이 떨어지거나 틀리면 답이 바뀐다. 그래서 샤딩 전파는 전파의 "기계 장치"(연산별 규칙, 우선순위, 추적 시점 처리)가 가능하다는 근거다. 그러나 뜻을 바꾸는 사실의 안전성까지 보여 주는 근거는 명시적 모드와 unreduced 타입 쪽이다.

### 4.3 MLIR

- **양자화 타입.** `!quant.uniform` 타입은 저장 타입, 표현 타입, 스케일, 영점을 타입 안에 담는다. 양자화 값과 표현 값 사이는 `quant.qcast`/`quant.dcast` 연산으로 명시적으로 오간다(https://mlir.llvm.org/docs/Dialects/QuantDialect/). [문서]
- **희소 텐서.** 저장 형식을 텐서 타입의 인코딩 속성으로 붙인다. 문서는 이것을 "treating sparsity as a property, not a tedious implementation detail"로 설명한다(https://mlir.llvm.org/docs/Dialects/SparseTensorOps/). [문서]
- **연산별 인터페이스.** Toy 튜토리얼의 모양 추론은 연산마다 `inferShapes` 인터페이스를 구현하게 한다. 구현하지 않은 연산을 만나면 "unable to infer shape of operation without shape inference interface" 오류를 내고 패스가 실패한다(https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/). [문서]
- [판단] 닫힌 컴파일러 IR 안에서는 "뜻을 타입에 넣는다"가 이미 표준 기법이다. 이 이론의 로드맵에서 "새 코드를 위한 컴파일러" 쪽의 선례다. 다만 이 보장은 IR 안에서만 성립한다. IR 밖의 파이썬 배관(로더, 스케줄러, 캐시 관리자)에는 미치지 않는다.

### 4.4 torchao 하위 클래스 디스패치

- 기반 클래스 `TorchAOBaseTensor`는 클래스마다 독립된 디스패치 표를 두고, 기본으로 `detach`, `clone`, `alias`, `contiguous`, `copy_`, `_to_copy`를 지원한다(`torchao/utils.py` 765~780행의 설명, torchao 0.18.0). [설치본 소스]
- 이 프로젝트가 쓰는 int4 형식의 텐서 클래스 `Int4TilePackedTo4dTensor`는 여기에 `aten.linear`(와 `F.linear`), `aten.slice`, `aten.select`만 더 등록한다(`torchao/quantization/quantize_/workflows/int4/int4_tile_packed_to_4d_tensor.py` 243·244·302·365행). [설치본 소스]
- 그 밖의 연산은 `NotImplementedError`로 멈춘다(3.1절 (1)). [설치본 소스]
- [판단] 포장 형식이라는 뜻은 작고 닫힌 연산 집합을 지나는 동안만 운반되고, 집합 밖에서는 시끄럽게 멈춘다. 양자화 가중치는 주로 linear로만 쓰이므로 이 설계가 성립한다. 이 이론의 D 갈래(저장 형식을 적재부터 커널까지)가 기대는 조건과 같다.

### 4.5 Meta spmd_types

- 명시적 집합 통신에 역할 키워드를 쓴다. README의 예는 `spmd.all_reduce(x, tp, src=spmd.P, dst=spmd.R)`이고, `R * R -> R` 같은 타입 추론을 한다(https://github.com/meta-pytorch/spmd_types README). [문서]
- 검사는 `spmd.checker.typecheck()` 문맥 안에서, 가짜 프로세스 그룹으로 GPU 없이 돈다(같은 README). [문서]
- 사용자 규칙은 믿으며, 수치 시험(`rulecheck`)으로 따로 확인한다(2.2절 (마)). [문서]
- [판단] 이 이론의 설계(키워드 전용 역할 표지, 닫힌 어휘, 규칙 검증)와 가장 가까운 운영 사례다. 합산 상태 한 종류의 사실에 한정된다(`realworld/NOVELTY_CHECK.md` 2.2절).

### 4.6 이 프로젝트의 전파 시제품

- 규칙은 다섯 부류다: 그대로 옮김, 축 변경 때 무효 표시, 제자리 쓰기 때 무효 표시, 이항 연산에서 합침, 그 밖의 계산에서 멈춤(`entail/audits/PROPAGATE.md`). [기록]
- 결과 [측정] (`entail/audits/propagate_cost.json`의 `reach`):
  - 입력 토큰 ids에 붙인 사실은 `embedding`에서 멈췄다.
  - KV 캐시 텐서에 붙인 `Valid(length=5)`는 복호가 끝난 뒤 처음 붙인 텐서에 그대로 남아 있었다. 살아 있는 캐시 텐서(길이 13)에는 사실이 없었다(`tagged_length` 5, `live_length` 13, `live_cache_tensor_facts` 빈 목록, `carried` 1).
- 결론(`entail/audits/PROPAGATE.md`): 버퍼를 갈아 치우는 그릇의 사실은 텐서가 아니라 그릇의 계약에 붙여야 한다. [기록]

### 4.7 무엇이 필요했고, 무엇이 비고, 무엇이 남는가

| 사례 | 필요했던 것 | 실행 비용 | 빈틈 |
|---|---|---|---|
| DTensor | 연산별 규칙(수백 개), 규칙 검증 도구 | eager에서 큼(35~60%, 블로그가 인용한 값) | 규칙 없는 연산은 멈춤. 운영에서는 지우자는 제안 |
| JAX·Shardy | 연산별 샤딩 규칙, 불투명 호출용 사용자 규칙, 우선순위 | 추적 시점 처리라 실행 비용 없음 | 커스텀 호출은 사람이 규칙을 적어야 함 |
| MLIR | 연산별 인터페이스, 타입 인코딩 | 컴파일 시점 | IR 밖에는 미치지 않음 |
| torchao | 연산별 구현 | 작음(연산 몇 개) | 등록한 연산에서만 뜻이 운반됨 |
| spmd_types | 집합 통신·사용자 함수마다 타이핑 규칙, 수치 시험 | 검사 모드를 따로 돎 | 규칙은 믿음의 대상 |
| 이 프로젝트 전파 | 다섯 부류 규칙 | 2.19배(eager) | 계산 연산에서 멈춤, 그릇이 텐서를 갈면 사실이 낡음 |

[판단] 교훈은 다섯 가지다.
1. 전파에는 언제나 연산별 규칙이 든다. 규칙의 수는 연산 어휘의 크기를 따라간다.
2. 규칙이 없을 때의 처리는 둘로 갈린다. DTensor, torchao, MLIR은 시끄럽게 멈추고, Quax는 기본으로 뜻을 떼고 계산한다(3.2절). 이 이론은 앞의 것을 골라야 한다.
3. 실행 중 전파 객체는 운영에서 너무 비싸다(DTensor 지우기 제안, 이 프로젝트 2.19배). 운영에서는 곁다리 검사(CI, 진단)나 추적 시점 처리로 간다.
4. 버퍼를 갈아 치우는 그릇은 값에 붙인 사실을 끊는다. 그릇 경계의 계약이 필요하다.
5. 규칙 자체를 데이터로 검증해야 한다(DTensor `strategy_validation`, spmd_types `rulecheck`, 이 프로젝트의 `cap_probe`).

## 5. 실행 중 검사의 비용

### 5.1 측정·보고된 값

| 방식 | 값 | 조건 | 출처 |
|---|---|---|---|
| jaxtyping + `jax.jit` | 실행 비용 없음. 모양 검사는 추적 중에만 한다 | JAX | https://docs.kidger.site/jaxtyping/api/runtime-type-checking/ [문서] |
| jaxtyping + PyTorch·NumPy | 호출마다 beartype이나 typeguard가 검사한다. 측정 수치는 확인 못 함 | eager | 같은 문서 [문서] |
| beartype | 중첩 컨테이너를 무작위 한 갈래만 검사해서, 문서의 예에서 13.8 µs. 대신 "invites false negatives"라고 스스로 적는다 | 문서의 예제(3중 중첩 리스트) | https://beartype.readthedocs.io/en/latest/faq/ [문서] |
| typeguard | 모든 원소를 검사해서, 같은 예제에서 6,420초라고 beartype 문서가 적는다 | 같은 예제. 경쟁 도구 쪽 측정이다 | 같은 문서 [문서] |
| typeguard 운영 관행 | 파이썬을 `-O`로 실행하면 검사를 끌 수 있다고 문서가 안내한다 | | https://typeguard.readthedocs.io/en/latest/userguide.html [문서] |
| checkify | 검사를 많이 넣으면 비싸다고만 적는다. 수치는 없다 | JAX jit 안 | https://docs.jax.dev/en/latest/debugging/checkify_guide.html [문서] |
| 건전한 점진적 타입(Typed Racket) | 최대 1.25~168배, 평균 0.6~68배 | 벤치마크 12개, 모든 부분 타입 구성 | https://www2.ccs.neu.edu/racket/pubs/popl16-tfgnvf.pdf [문서] |
| 명목 점진적 타입(Nom) | 최악 10% 미만 | 위 벤치마크 일부를 옮긴 것 | https://www.cs.cornell.edu/~fabianm/papers/nomalive-oopsla17-tr.pdf [문서] |
| transient 방식(Reticulated Python) | 약 200~400%(책임 추적 없이) | Nom 논문이 인용한 값 | 같은 논문 9.3.3절 [문서] |
| PyTorch 하위 클래스(빈 `__torch_function__`) | 학습 반복 +5.3~23.3% | PyTorch 1.12, torchvision 모델 | https://github.com/pytorch/pytorch/issues/73860 [문서] |
| DTensor eager | 종단 간 35~60% 느림, DTensor 연산이 실제 계산의 7배 이상 | 블로그가 veScale 논문을 인용한 값 | https://blog.ezyang.com/2026/02/dtensor-erasure/ [문서] |
| 배치 불변 커널(검사가 아니라 보장의 비용) | 1.6~2.1배 | vLLM | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ [문서] |
| torch.compile 가드 | 호출마다 가정 6,432개를 검사해도 22.42 ms 대 22.75 ms(검사를 건너뛴 쪽)로 차이가 보이지 않음 | Qwen3-4B 디코드 한 스텝, 배치 1 | `phase0/week4/results/e1_compile_anatomy.json` [측정] |
| 연산 단위 전파(TorchDispatchMode) | 2.19배, 가로챈 연산 273,413개 | Qwen3-4B eager 64토큰 | `entail/audits/propagate_cost.json` [측정] |
| 그릇 경계 계약(동적 캐시) | 0.979배, 갱신 2,304회 모두 검사, 지적 0 | 같은 조건 | `entail/audits/cache_contract_cost.json` [측정] |
| 같은 계약을 그래프 구간 안에(정적 캐시) | 47.665배, 출력이 바뀜 | 정적 캐시, 층마다 장치 비교 | 같은 파일 `healthy_static` [측정] |
| 같은 계약을 요청 밖에서 한 번 | 1.007배, 출력 같음, 36개 층 확인 | 정적 캐시 | `entail/audits/cache_contract_static.json` [측정] |
| vLLM 페이지 캐시 계약 | 할당 96회 검사, 지적 0. 생성 시간은 끔 0.76초, 켬 0.89초(한 번씩, 1초 미만 실행이라 차이를 판정할 수 없음) | vLLM, 짧은 생성 3개 | `entail/audits/vllm_cache_on.log` 13·14행, `vllm_cache_off.log` 8행 [측정] |
| 경계 선언(호출당) | off 0.12~1.90 µs, load 0.25~2.86 µs, debug 2.01~5.00 µs(제자리 쓰기 칸에는 `mul_` 연산 자체가 들어 있다) | CPU, 16원소 텐서, 시험 코드 | `entail/audits/boundary_22_cost.json` [측정] |
| 시작 점검 | 한 번에 2.83 µs | 10,000회 평균 | `entail/audits/adapter_pilot.json` [측정] |

### 5.2 읽는 법

- **비용을 정하는 것은 검사의 위치와 횟수다.** [판단]
  - 연산마다 가로채는 방식은 2.19배(이 프로젝트), 35~60%(DTensor)다. 운영에서 상시로 켤 수준이 아니다.
  - 경계에서 이름표를 비교하는 방식은 호출당 µs 단위다. 이 프로젝트의 동적 캐시 계약은 토큰당 36회(층마다 한 번) 검사해도 잡음 안이었다(0.979배).
  - 이 결론은 점진적 타입 논쟁의 결론(부담은 경계 교차 횟수와 검사 깊이에 비례하고, 명목 검사는 싸다)과 같다(2.4절).
- **추정(측정 아님)** [판단]: 디코드 한 스텝이 7~36 ms일 때(`phase0/week4/WEEK4_NOTES.md` 6절 표), load 모드 경계(호출당 최대 2.86 µs)를 층마다 한 번씩 36회 지나면 약 0.1 ms다. 7 ms 스텝의 약 1.5%다.
  - 이 추정은 호스트 시간이 GPU 시간과 겹치지 않는다고 가정한 상한이다.
  - 또 CUDA Graph로 재생하는 구간에서는 이 파이썬 검사가 아예 돌지 않는다(2.3절). 운영 엔진에서 스텝마다 도는 검사는 결국 스케줄러 같은 호스트 쪽 경계에만 둘 수 있다.
- **eager에서 잰 수치의 한계.** 이 프로젝트의 0.979배와 2.19배는 eager 복호(초당 약 25토큰)에서 쟀다. 스텝이 길어서 µs 검사가 묻히기 쉬운 조건이다. CUDA Graph를 쓰는 운영 경로에서 호스트 쪽 경계의 비용은 재지 않았다(확인 못 함).
- **정적 대 동적.** jaxtyping과 명시적 샤딩처럼 추적 시점에 한 번 검사하면 실행 비용이 0이다. 설정으로 정해지는 사실은 모두 이 자리로 보낼 수 있다(2.3절).

## 6. 채택 가능성

### 6.1 선택 사항인 주석이 쓰이지 않은 근거

**PyTorch named tensors** [문서, GitHub]
- 2021-06-28, 유지보수자(zou3519)는 설계의 모호함을 풀 인력이 없어 적극 개발하지 않는다고 답했다(https://github.com/pytorch/pytorch/issues/60832).
- 같은 이슈의 사용자 의견:
  - 2023-02-14: 많은 기본 연산에서 오류가 나서 사실상 쓸 수 없다.
  - 2024-04-15: 이름 있는 텐서와 없는 텐서를 섞으면 오류가 나기 쉽다.
- 2026-01-30 PR #173895의 목적은 폐기된 named tensor 기능을 부담과 코드 크기 때문에 완전히 들어내는 것이었다(https://github.com/pytorch/pytorch/pull/173895). GitHub 상태는 닫힘이고 병합 표시는 없다. 이 저장소는 ghstack으로 병합하므로 상태만으로 병합 여부를 판단할 수 없다.
- 이 평가자의 확인 [측정]: 설치된 torch 2.14.0에는 `Tensor.refine_names`와 `Tensor.names`가 없다. `torch.zeros(2, 3, names=('N','C'))`는 "unexpected keyword argument 'names'"로 거부된다.
- [판단] 버려진 이유는 연산 지원의 불완전(연산별 규칙 부담), 섞어 쓰기의 어려움(점진적 경계), 설계 모호성, 부담이었다. "버그를 잡지 못해서"는 이유가 아니었다. 이 프로젝트의 선행 확인도 같은 결론이다(`realworld/NOVELTY_CHECK.md` 2.5절).

**C `restrict`**
- 표준은 이 선언을 검사하지 않는다. 어기면 정의되지 않은 행동이고, 번역기는 무시해도 된다(N1570 6.7.3.1절). [문서]
- Rust는 LLVM의 `noalias`(C의 restrict에 해당)를 참조 매개변수에 붙인다. 그런데 LLVM 결함 때문에 2018년에 이를 끄는 기본값을 넣었고(PR #54639), 그 기본값을 되돌리는 추적 이슈 #54878은 2021-03-22에 닫혔다(https://github.com/rust-lang/rust/issues/54878). [GitHub]
  - 같은 이슈 댓글(2018-10-13~14)에서 결함을 restrict를 쓴 C 코드로 줄였더니 LLVM과 GCC가 모두 잘못 컴파일했다. GCC에도 버그 87609로 보고되었다.
- C 코드에서 restrict가 실제로 얼마나 쓰이는지 잰 자료는 확인 못 함. "C가 별칭 없음 보장을 거의 주지 않아 LLVM의 그 경로가 운영에서 시험되지 않았다"는 설명은 Rust 포럼 사용자의 글에서만 찾았다(https://users.rust-lang.org/t/non-aliasing-guarantees-of-mut-t-and-rustc-optimization/34386, 2019-11-07). 1차 근거가 아니다.
- [판단] 선택형이면서 검사도 없는 선언은 그 선언을 쓰는 소비자 쪽 경로를 덜 시험된 채로 남긴다. 필수로 쓰는 사용자가 나타나자 잠복 결함이 드러났다.

**파이썬 타입 주석** [문서]
- DLS 2020: 70,000개가 넘는 저장소 가운데 주석이 있는 것은 2,678개였다. 주석을 쓴 코드도 mypy나 pytype 검사를 거의 통과하지 못했다(https://www.cs.rpi.edu/~milanova/docs/dls2020.pdf, 초록과 1절).
- FSE 2022: 9,655개 프로젝트의 1,123,393개 커밋을 분석했다. 주석은 늘고 있지만 대부분의 코드는 여전히 주석이 없었다. 커밋의 78.3%가 타입 오류가 있는 채로 들어갔다(https://software-lab.org/publications/fse2022_type_study.pdf, 초록).
- 2026-09-02 공개 연구: 파이썬 라이브러리의 91%가 타입 힌트를 한 번 이상 쓴다. 그러나 체계적으로 쓰는 저장소에서도 적용률 중앙값은 매개변수 45.8%, 반환값 35.9%였다(https://arxiv.org/abs/2609.02782).
- [판단] 선택형 주석은 "쓰긴 쓰지만 절반쯤"에서 멈춘다. 경계 검사에 쓰면 빈 곳이 곧 검사의 구멍이 된다.

**주석의 정확도** [문서]
- ASE'18 실험에서 프로그래머 71명이 물리 단위 타입 주석을 맞게 고른 비율은 51%였다. 맞는 주석 하나에 평균 136초가 걸렸다. 옳은 제안을 보여 주면 정확도가 73%로 올랐고, 틀린 제안을 보여 주면 28%로 떨어졌다(https://jpwco.com/pdf/ase18main-p15-p-bcc79e2-37685-final.pdf, 초록).
- [판단] 사용 지점마다 사람이 뜻을 적게 하면 절반은 틀린다. 선언은 사실을 가장 잘 아는 생산 지점(학습 도구, 변환 도구, 캐시 생성자)이 자동으로 적어야 한다. 자동 제안은 틀리면 오히려 해롭다.

**이 프로젝트가 본 신호** [기록]
- transformers의 선택형 엄격 적재 PR #48962가 2026-09-22에 닫혔다(`realworld/NOVELTY_CHECK.md` 2.4절).
- torch.compile의 `mark_dynamic` 힌트가 크기 1에서 경고 없이 무시되었다(`phase0/week4/WEEK4_NOTES.md` 3절).

### 6.2 필수 선언이 받아들여진 사례

- **`torch.library.custom_op`의 `mutates_args`.** 필수 키워드 인자라서 쓰지 않으면 연산을 만들 수 없다(https://docs.pytorch.org/docs/2.14/library.html). 다만 정확성은 검사하지 않는다(2.2절 (마)). [문서]
- **Rust.** [문서] (https://blog.google/security/rust-in-android-move-fast-fix-things/, 2025-11-13)
  - Android에서 메모리 안전 취약점이 처음으로 전체 취약점의 20% 아래로 내려갔다.
  - C·C++ 코드 대비 메모리 안전 취약점 밀도가 1000배 낮았다.
  - 중대형 변경의 롤백률이 C++보다 약 4배 낮았고, 코드 리뷰 시간이 약 25% 적었다.
- **GGUF.** [문서] (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
  - `general.architecture`는 필수다. 양자화된 텐서가 있으면 `general.quantization_version`도 반드시 있어야 한다.
  - 그러나 `[llm].tensor_data_layout`은 선택이고, 없으면 `reference`로 가정한다. 필수 선언과 조용한 기본값이 한 형식 안에 섞여 있다.
- **JAX 명시적 샤딩.** 모호하면 오류를 내서 선언을 강제한다(2.3절). [문서]
- **spmd_types.** Megatron-LM의 병렬화 조합에서 조용한 수치 오류 13건을 spmd_types와 퍼징 도구로 찾아 적은 이슈가 있다(https://github.com/NVIDIA/Megatron-LM/issues/7452, 2026-09-17). 본문 번호 1~13을 이 평가자가 직접 셌다. 보고자는 모든 건을 직접 검증하지는 않았고 보고서 대부분이 AI 생성이라고 밝혔다. [GitHub]

### 6.3 무엇이 필수 선언을 받아들이게 했나

[판단] 앞의 사례에서 공통점은 넷이다.
1. **한 번, 생산 지점에서 적는다.** Rust는 타입 정의와 함수 서명에, `mutates_args`는 연산 정의에 적는다. GGUF는 변환 도구가 파일에 쓴다. kohya sd-scripts는 학습 결과에 `modelspec.prediction_type`을 자동으로 쓴다(7.3절).
2. **선언 없이는 진행할 수 없다.** 컴파일 오류가 나거나 연산을 만들 수 없다.
3. **대가가 수치로 보인다.** Rust의 취약점 밀도와 롤백률이 그렇다.
4. **버려진 쪽은 반대였다.** 사용 지점마다 적어야 했고, 연산 지원이 불완전했고, 섞어 쓰기가 어려웠고, 켜면 느렸다(named tensors, DTensor eager).

이 이론에 옮기면 [판단]:
- "기존 엔진에 붙이는" 형태에서 라이브러리는 엔진에게 선언을 강제할 수 없다. 필수성은 라이브러리가 소유한 경계(적재기, 어댑터)와 파일 선언에만 걸 수 있다. 엔진 내부는 "선언 없음, 검사 못 함"으로 보고하는 것이 최선이다.
- "새 코드를 위한 컴파일러" 형태에서는 필수성을 문법(키워드 전용 인자, 닫힌 어휘)으로 강제할 수 있다. 채택의 문제는 그 코드를 누가 새로 쓰느냐로 옮겨 간다.
- 선언의 비용을 치르는 쪽(학습·변환 도구)과 이익을 얻는 쪽(추론 엔진)이 다르다. ModelSpec이 예측 방식을 선택(CAN) 항목으로 둔 것이 이 간극을 보여 준다(7.3절).

## 7. 이미지 생성 쪽: 모델의 뜻은 어디에 있는가

### 7.1 diffusers 저장소 형식: 구성 요소별 JSON에 선언된다

SDXL base 저장소를 예로 들면 다음과 같다(https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 의 파일들, 2026-09-23 조회). [문서]

| 사실 | 선언된 곳 | 값 |
|---|---|---|
| 예측 방식 | `scheduler/scheduler_config.json`의 `prediction_type` | `epsilon` |
| VAE 배율 | `vae/config.json`의 `scaling_factor` | 0.13025 |
| 구성 요소 대응(텍스트 인코더 둘, 토크나이저 둘, UNet, VAE, 스케줄러) | `model_index.json` | `text_encoder`: CLIPTextModel, `text_encoder_2`: CLIPTextModelWithProjection 등 |
| 스케줄러 설정 | `scheduler/scheduler_config.json` | `EulerDiscreteScheduler`, `timestep_spacing` 등 |

[판단] 이 형식에서는 뜻이 이미 파일에 선언되어 있다. 문제는 선언이 없는 형식(단일 파일 체크포인트)으로 들어올 때와, 소비자가 선언을 읽지 않는 경로에서 생긴다.

### 7.2 diffusers 단일 파일: 키 이름으로 추측하고 기준 저장소에서 빌린다

출처는 `src/diffusers/loaders/single_file_utils.py`다(https://github.com/huggingface/diffusers/blob/f2edf6ea305170034e086ff4895ea6dd1ce9442a/src/diffusers/loaders/single_file_utils.py, 커밋 2026-09-23). [코드]

- **모델 종류.** `infer_diffusers_model_type`이 가중치 키 이름과 텐서 모양으로 종류(`v2`, `xl_base` 등)를 정한다(593행부터).
- **설정.** `fetch_diffusers_config`가 추측한 종류를 기준 저장소에 대응시킨다(838~843행). 예를 들어 `xl_base`는 `stabilityai/stable-diffusion-xl-base-1.0`에 대응한다(164~178행). SDXL을 미세 조정한 파일은 기준 저장소의 설정을 빌려 온다.
- **예측 방식.** (1827~1835행)
  - SD2 계열(`v2`)은 `global_step == 875000`이면 `epsilon`, 아니면 `v_prediction`으로 정한다. 코드 주석이 이 방식을 "brittle global step parameter"라고 부른다.
  - 그 밖의 종류는 사용자가 주지 않으면 `epsilon`이다.
  - `prediction_type` 인자를 직접 넘기는 방식은 폐기 예정이고, 스케줄러 객체를 넘기라고 안내한다(1804~1813행).
- **VAE 배율.** LDM 형식에서 변환할 때 `latents_mean`/`latents_std`가 있으면 0.5(Playground), 원래 설정에 `scale_factor`가 있으면 그 값, 둘 다 없으면 기본 0.18215를 쓴다(1032~1075행, 390행).
- **텍스트 인코더.** 가중치 키의 접두사 목록(`LDM_CLIP_PREFIX_TO_REMOVE`)으로 CLIP 인코더 가중치를 가려낸다(396~399행 부근).
- 이 파일에는 `modelspec`이라는 문자열이 없다. safetensors 메타데이터의 선언을 이 경로에서 읽지 않는다. [이 평가자 grep, 이 파일 하나의 범위]

### 7.3 ComfyUI: 키 표지와 가중치 통계로 추측한다

출처는 `comfy/supported_models.py`와 `comfy/model_detection.py`다(https://github.com/Comfy-Org/ComfyUI/blob/6bfaacc67c2103481e5f0c84d75257cd0581d86a/comfy/supported_models.py, 커밋 2026-09-19). [코드]

- **구조.** `detect_unet_config`가 가중치 키 이름과 모양으로 구조를 정한다(`comfy/model_detection.py` 44행부터).
- **SDXL 예측 방식.** (`supported_models.py` 218~235행)
  - `edm_mean`과 `edm_std` 키가 있으면 EDM(Playground 2.5)이다.
  - `edm_vpred.sigma_max` 키가 있으면 V_PREDICTION_EDM이다.
  - `v_pred` 키가 있으면 V_PREDICTION이고, `ztsnr` 키가 더 있으면 zsnr을 켠다.
  - 그 밖에는 모두 EPS다.
- **SD2 예측 방식.** 한 가중치(`output_blocks.11.1.transformer_blocks.0.norm1.bias`)의 표준편차가 0.09보다 크면 V_PREDICTION이다. 코드 주석은 "not sure how well this will actually work"라고 적는다(`supported_models.py` 114~120행).
- **메타데이터.** 일부 새 구조에서는 파일 메타데이터의 `config`를 읽는다(`comfy/model_detection.py` 416~428행, `comfy/sd.py` 625~626행). 그러나 이 세 파일(`supported_models.py`, `model_detection.py`, `sd.py`)에는 `modelspec`이라는 문자열이 없다. 즉 SD 계열의 예측 방식은 이 경로에서 파일 선언을 읽지 않는다. [이 평가자 grep. 저장소 전체는 확인 못 함]

### 7.4 파일 선언 표준: safetensors, ModelSpec, kohya

- **safetensors.** 헤더의 `__metadata__` 키는 자유 형식의 문자열→문자열 사전이다. 값은 모두 문자열이어야 한다(https://github.com/huggingface/safetensors README 83행). [문서]
- **ModelSpec** (Stability AI, https://github.com/Stability-AI/ModelSpec, 저장소 생성 2023-07-21, 마지막 푸시 2024-06-04). [문서]
  - 키는 모두 `modelspec.`으로 시작한다. 항목을 MUST, SHOULD, CAN 세 등급으로 나눈다.
  - MUST: `sai_model_spec`, `architecture`, `implementation`, `title`, 이미지 모델의 `resolution`.
  - `architecture`는 추론 요구가 다른 모델끼리는 달라야 한다. 문서는 SDv2-512와 SDv2-768-v를 예로 든다. 그러나 "Simple finetunes of a model do not require a unique class"라고 적는다.
  - `prediction_type`(`v` 또는 `epsilon`)은 CAN, 즉 선택 항목이다.
  - 소비자는 키의 일부만 지원해도 된다고 적는다.
  - 텍스트 모델의 `data_format`은 MUST다. 이유로 형식이 "often not accurately reflected in tensor data type"이라고 적는다. 이 이론의 LAYOUT 사실과 같은 문제의식이다.
- **kohya sd-scripts.** 학습 결과에 `modelspec.prediction_type`을 자동으로 쓴다. v 매개변수화면 `v`, 아니면 `epsilon`이다(https://github.com/kohya-ss/sd-scripts/blob/34e7138b6a80c2d88f40c99fd68879c6e683f639/library/sai_model_spec.py 432~439행). [코드]
- [판단] 선언은 생산 쪽(학습 도구)에서 이미 자동으로 쓰이고 있다. 그러나 표준에서 선택 항목이고, 대표 소비자 두 곳은 해당 경로에서 읽지 않는다. 이 이론이 말하는 모양, 즉 생산자는 선언했는데 소비자에게 가는 길에서 사라지는 모양이 이미지 쪽에 그대로 있다.
  - 또 ModelSpec이 "단순 미세 조정은 고유 구조 이름이 필요 없다"고 하므로, v 예측으로 바꾼 SDXL 미세 조정본도 `architecture`는 `stable-diffusion-xl-v1-base` 그대로일 수 있다. 실제로 7.5절의 파일이 그랬다. 뜻을 바꾸는 사실이 선택 항목 하나에만 실린다.

### 7.5 이 프로젝트의 현장 측정(M7)

[측정] 출처는 `issue_track/comfyui_field_test/`의 결과 파일이다.
- 파일 AstolfoCarmix-VPredXL에는 `v_pred`와 `ztsnr` 키가 없다. 메타데이터에는 `modelspec.prediction_type: "v"`와 `modelspec.architecture: "stable-diffusion-xl-v1-base"`가 있다(`astolfo_header_check.json`).
- ComfyUI는 이 파일을 EPS로 불렀다. v를 명시한 기준 그림과 평균 절대 차이가 94.78, 87.29, 82.64였다(시드 3개, 0~255 화소값. `astolfo_m7.json`의 `a_vs_ref`).
- entail 0.3.0의 동작 탐침은 "eps처럼 동작한다(probe 0.94)"로 판정해서 놓쳤다. 켠 그림과 끈 그림이 같았다(`vpred_astolfo_plain_on.json` 48행, `astolfo_m7.json`의 `c_vs_a`).
- 파일 선언을 읽도록 바꾼 개발 사본은 "metadata modelspec.prediction_type says v-prediction"으로 해소했다. 기준 그림과 비교하면 한 시드는 화소 단위로 같았고, 나머지 둘은 0.16과 0.14 차이였다(`g2_compare.json`의 `M7_lines`, `M7_vs_b_yaml_setting`).
- 연구자의 모델 파일 44개(체크포인트 22, LoRA 22)를 읽었을 때 예측 방식을 선언한 파일은 24개, 선언끼리 어긋난 파일은 0개, 계약이 개입할 파일은 1개였다(`generality_scan.json`).
- 이 측정이 말하지 않는 것:
  - 대상은 연구자 한 사람의 모음이다. 무작위 표본이 아니다.
  - 결함이 드러난 모델은 하나다.
  - 44개 가운데 20개는 선언이 없다. 선언이 없는 파일에서는 이 방식이 동작 탐침이나 사용자 결정으로 되돌아간다.

### 7.6 이미지 쪽의 정리

[판단]
- 예측 방식, VAE 배율, 텍스트 인코더 대응, 스케줄러 설정은 세 가지 방식으로 존재한다.
  - diffusers 저장소 형식에서는 파일로 선언된다.
  - 단일 파일 체크포인트에서는 키 이름, 가중치 통계, 기준 저장소로 추측된다.
  - 표준 메타데이터(ModelSpec)는 있지만, 예측 방식은 선택 항목이고 주요 소비자 경로가 읽지 않는다.
- 이 영역은 이론이 가장 잘 맞는 곳이다.
  - 사실이 설정 사실이라 적재 때 한 번 검사하면 된다. 비용이 사실상 없다.
  - 생산 도구가 이미 자동으로 선언을 쓴다.
  - 결함의 결과가 크다(기준 그림과 평균 82~95 차이).
- 한계도 분명하다. 선언이 없는 파일에는 구조적 보장이 없다. 그 경우 할 수 있는 최선은 "모름"을 보고하고, 통계적 탐침은 교차 확인으로만 쓰는 것이다. M7이 보였듯 탐침은 틀릴 수 있다.

## 8. 이 프로젝트의 로컬 측정: 말하는 것과 말하지 않는 것

측정값은 결과 파일에서 옮겼다. "말하는 것"과 "말하지 않는 것"은 이 평가자의 판단이다.

### 8.1 역할 타입 시제품 rolec (`phase0/week4/`)

**측정값** [측정]
- 주입한 역할 실수 8종: 기존 인터페이스는 8종 모두 예외도 NaN도 없이 실행했다. 기준값과의 상대 차이는 0.235(포함 여부 착오)에서 3.847(인과 마스크 정렬 착오)까지였다(`results/e5_role_errors.json`).
- rolec 결과: 실행 전 거부 2(키·값 뒤바꿈, 받지 않는 인자), 표현 불가 3(헤드 공유 방식, 마스크 참거짓, 불리언·덧셈 마스크 혼동), 언어 규칙으로 제거 1(포함 여부), 올바르게 실행 2(차원 순서, 인과 정렬)(같은 파일).
- 올바른 호출끼리는 두 인터페이스 모두 기준값과 상대 0.23%, 0.19% 안이었다(같은 파일의 `sanity`).
- 같은 선언을 두 백엔드(Triton 손 커널, FlexAttention)로 내렸을 때 기준값과의 최대 차이가 둘 다 0.0019였다. FlexAttention 백엔드에서도 키·값 뒤바꿈은 거부되었다(`results/e6_one_spec.json`).
- 곁들인 측정: 호출마다 가드 6,432개 검사의 비용은 보이지 않았다(22.42 ms 대 22.75 ms, `results/e1_compile_anatomy.json`). 사실이 지워지면 디코드가 1.3~2.5배 느렸다(`WEEK4_NOTES.md` 6절 표).

**말하는 것**
- 이런 종류의 역할 실수는 현재 파이썬 인터페이스에서 조용하다.
- 키워드 전용 역할, 역할 타입, 이름 붙은 차원, 표 대신 사실(`Until(last)`)로 인터페이스를 바꾸면 이 실수들을 거부하거나 표현할 수 없게 만들 수 있다.
- 한 선언을 여러 백엔드로 내려도 뜻이 같게 유지될 수 있다.

**말하지 않는 것**
- 이 실수들이 실제로 얼마나 자주 일어나는지. 8종은 저자가 골랐다(`WEEK4_NOTES.md` 10절).
- rolec 검사 자체의 비용. 따로 재지 않았다. E6의 호출당 시간(0.19 ms, 1.55 ms)은 커널 실행과 블록 정보 생성이 포함된 값이다.
- 컴파일 단계 검사. rolec은 호출 시점에 파이썬으로 검사한다(`rolec.py` 머리말).
- 운영 경로와의 호환. rolec은 호출마다 `int(last.max())`로 장치 값을 읽으므로 CUDA Graph 캡처 구간에 둘 수 없다(2.3절).
- 일반성. 어텐션 한 연산, Qwen3-4B, RTX 4070 Ti에서만 쟀다.

### 8.2 연산 단위 전파 (`entail/audits/PROPAGATE.md`)

**측정값** [측정] (`entail/audits/propagate_cost.json`)
- 끔 2.553초, 디버그 5.582초로 2.19배였다. 초당 토큰은 25.1에서 11.5로 줄었다. 출력은 같았다. 가로챈 연산은 273,413개였다.
- 문서(`PROPAGATE.md`)의 표는 2.14배(5.45초, 초당 11.7토큰)이고 "다시 돌리면 2.1~2.3배"라고 적었다. 2.14배를 담은 결과 파일은 찾지 못했다. 이 평가서는 결과 파일의 2.19배를 쓴다.
- 사실의 도달 범위: 입력 ids의 사실은 `embedding`에서 멈췄다. KV 캐시 사실은 낡은 텐서에 남았다(4.6절).

**말하는 것**
- 모든 연산을 가로채는 전파는 eager에서 약 2배다. 로드맵의 "디버그 모드 2배 이하" 기준을 넘는다. 상시로 켤 방식이 아니다.
- 버퍼를 갈아 치우는 그릇은 값에 붙인 사실을 끊는다. 이것은 가설이 아니라 시제품에서 재현된 현상이다.

**말하지 않는 것**
- 가로챌 범위를 좁혔을 때의 비용. 문서가 다음 할 일로 적었고 아직 재지 않았다.
- 컴파일·CUDA Graph 경로에서의 동작. eager에서만 쟀다. torch 2.14 소스로 보면 이런 모드가 켜져 있으면 컴파일 자체가 건너뛰어진다(3.1절 (2)).
- 규칙의 정확성. 시험 8개가 고정하는 범위 밖은 모른다.

### 8.3 그릇 경계 계약 (`entail/audits/CACHE_CONTRACT.md`)

**측정값** [측정]
- transformers 동적 캐시: 끔 2.591초, 켬 2.536초로 0.979배였다. 갱신 2,304회를 모두 검사했고 지적은 0이었다. 출력은 같았다(`cache_contract_cost.json`의 `healthy`).
  - 문서의 표는 0.993배라고 적었다. 그 값을 담은 결과 파일은 찾지 못했다. 둘 다 잡음 범위로 읽힌다.
- 심은 결함(모든 층에서 토큰 하나 제거): 끄면 끝까지 돌았고, 켜면 "was 7 tokens when last written, 6 when written again"으로 멈췄다(같은 파일 `seeded_contract_off`, `seeded_contract_on`).
- transformers 정적 캐시:
  - 갱신 경계 안에서 장치로 비교하면 47.665배였고 출력이 바뀌었다(같은 파일 `healthy_static`).
  - 요청 뒤 밖에서 한 번 검사하면 1.007배였고 출력이 같았다. 심은 결함도 잡았다(`cache_contract_static.json`).
  - 문서에는 "길이를 파이썬으로 읽음 8.0배" 행도 있다. 그 값을 담은 결과 파일은 찾지 못했다.
- vLLM 페이지 캐시: 할당 96회를 검사했고 지적은 0이었다(`vllm_cache_on.log` 14행). 심은 결함은 "holds 16 KV slots for 17 tokens"로 잡았다(`vllm_cache_seeded.log` 38행).
- SGLang: 문서에 정상 실행 지적 0, 심은 결함 검출이 적혀 있다. 이 평가자는 SGLang의 원자료 로그를 열지 않았다(확인 못 함).

**말하는 것**
- 사실이 호스트 쪽 정수로 있는 그릇 경계에서는 계약 검사의 비용이 잡음 안이다.
- 모든 층에서 토큰 하나가 빠지는, 겉으로는 조용한 결함을 잡는다. 이 결함이 없으면 모델은 답하지 않고 질문을 되풀이했다.
- 컴파일·그래프 구간 안의 검사는 비용뿐 아니라 출력도 바꾼다. 그 구간은 검사 금지 구역이다.
- 세 엔진의 장부는 이름이 달라도 같은 사실이다. 공유 규칙 하나와 엔진별 번역으로 붙일 수 있었다(문서의 줄 수: 규칙 73줄, 어댑터 97~109줄).

**말하지 않는 것**
- 심지 않은 실제 결함을 잡는가. 잡은 결함은 모두 심은 것이다.
- 운영 경로의 비용. vLLM 실행은 1초 미만의 짧은 생성 한 번씩이다(끔 0.76초, 켬 0.89초, `vllm_cache_off.log` 8행, `vllm_cache_on.log` 13행). 차이를 판정할 수 없다.
- 요청 밖 검사의 탐지 시점. 결함이 난 뒤 요청이 끝나야 알린다. 그 사이의 출력은 이미 틀렸을 수 있다.

### 8.4 경계 선언 (`entail/DESIGN.md` 8절)

**측정값** [측정]
- 호출당 비용(µs, CPU, 16원소 텐서): off 0.12·0.12·1.90, load 0.25·0.60·2.86, debug 2.01·2.39·5.00(입력만·입력+결과·제자리 쓰기 순). 그냥 함수는 0.02~0.03이다(`entail/audits/boundary_22_cost.json`). 제자리 쓰기 칸에는 `mul_` 연산 자체가 들어 있다.
- 보존: 사례 01, 02, 04, 16의 생산자를 경계 선언으로 바꾸자 생산자 뒤에 손으로 다시 붙이던 호출이 1·3·2·2개(합 8개)에서 모두 0개가 되었다. 네 사례 모두 결함 판은 오류, 수정 판은 무오류였다(`entail/audits/boundary_22.json`).
- 기능 시험 9개가 통과했다고 문서가 적는다(`entail/DESIGN.md` 8절, `entail/tests/test_boundary.py`). 이 평가자는 시험을 다시 돌리지 않았다.

**말하는 것**
- 필수 역할 표지와 `returns`/`writes` 선언은 호출당 µs 단위로 돈다.
- 생산자가 결과의 뜻을 선언하면, 소비자 쪽에서 뜻을 손으로 다시 붙일 일이 없어진다(명제 3의 "보존").

**말하지 않는 것**
- 실제 엔진에서의 비용과 효과. 시험 코드에서만 확인했다(지시문이 밝힌 대로).
- GPU 텐서, 큰 텐서, 한 스텝에 지나는 경계의 수.
- 버전 카운터 대조가 torch.compile의 함수화 아래에서도 그대로 동작하는지. 재지 않았다(확인 못 함).
- 벤치마크 사례는 결함을 알고 만든 최소 모사다. 이것만으로는 넓은 방식과 깊은 방식의 우열이나 실제 효용을 가르지 못한다고 문서 스스로 적었다(`entail/DESIGN.md` 5.1절).

### 8.5 함께 볼 로컬 측정

- **시작 점검** [측정]: Gemma 2 2B·9B를 transformers sdpa로 부르면 가중치를 읽기 전에 막혔다(softcap을 읽지 않는 경로). Qwen3-4B는 세 구현 모두 통과했다. 검사 한 번에 2.83 µs였다(`entail/audits/adapter_pilot.json`).
- **RoPE 옛 이름** [측정]: vLLM에서 Llama 3.2의 `rope_scaling`을 실행 시점에 다시 넘기면 `rope_theta`가 설정에서 빠지고 경고 없이 GSM8K가 379/500에서 279/500으로 떨어졌다. 해소기를 켜면 378/500이었다(`issue_track/rope_override/SUMMARY.json`의 `LA`, `LC`, `LC_rolecheck`).
- **가상 평가** [가상 평가, 측정 아님]: 실제 버그 50건을 다섯 해법으로 다시 판정하니, 막거나 탐지한 것이 엄격 기준으로 42건, 정적 타입 없이 기존 엔진에 붙이는 조합(S2+S4)으로 37건이었다. 네 건은 시험 생성으로만 잡거나 어느 것으로도 못 잡았다(`realworld/replay_totals.json`). 판정자가 결함을 안 상태에서 한 판정이다.

### 8.6 로컬 측정 전체가 실현 가능성에 대해 말하는 것

[판단]
- **말하는 것.**
  - 설정 사실의 적재 시점 대조는 싸고(µs), 오탐 없이, 실제 결함 묶음을 막는다.
  - 호스트 쪽 그릇 경계의 계약은 비용이 잡음 안이다.
  - 연산 단위 전파는 약 2배라서 진단용이다.
  - 그래프 구간 안의 검사는 금지 구역이다.
  - 파일 선언을 읽으면 동작 추측이 놓친 결함을 고친다(M7).
- **말하지 않는 것.**
  - 모르는 실제 결함을 얼마나 찾는가. 새로 찾은 사례는 있지만 표본이 작다. 예는 RoPE 옛 이름(8.5절)과 vLLM fp8 적재다. 후자에서는 `replace_parameter` 호출 144회 모두 파라미터 클래스가 바뀌었고(`entail/audits/vllm_ledger_summary.json`), 문서는 이것을 축 사실의 소실로 해석했다(`entail/DESIGN.md` 3절).
  - CUDA Graph를 쓰는 운영 경로에서의 비용.
  - 여러 GPU, 여러 모델 계열, 여러 엔진 버전에 걸친 일반성과 어댑터 유지 비용.
  - 채택. 외부 사용자가 선언을 쓰고 켜 두는지는 아직 데이터가 없다.

## 9. 판정

확신 정도는 셋으로 적는다. **높음**은 1차 출처나 결과 파일의 근거가 여럿이고 반대 근거를 찾지 못한 경우다. **중간**은 근거가 있으나 조건이 좁거나 표본이 작은 경우다. **낮음**은 근거가 간접적이거나 데이터가 없는 경우다.

### 9.1 명제별 판정

| 명제 | 판정 | 확신 | 근거 |
|---|---|---|---|
| 1. 의미의 중립성 | **부분적으로 참.** 뜻은 이미 여러 곳에 실린다(config.json, `scheduler_config.json`, `model_index.json`, GGUF 필수 키, ModelSpec, torchao 하위 클래스, DTensor 배치 정보, MLIR 타입). 정확한 문제는 "전하지 않는다"가 아니다. 선언이 선택이고, 검사되지 않고, 소비자가 읽지 않거나 추측과 기본값으로 대신한다는 것이다 | 높음 | 4절, 6.2절, 7.1~7.4절 |
| 2. 손실 | **참.** 성능 손실(1.3~2.5배)과 조용한 오답(주입 8종, RoPE 379→279/500, M7, 토큰 하나 빠진 캐시)이 측정되었다. 외부 사건도 있다. 다만 외부 사건의 원인은 대부분 부품 경계가 아니라 컴파일러·커널·설정 안쪽이었다 | 높음(존재), 중간(비중) | 2.2절, 2.5절, 8절 |
| 3. 해법 | **조건부 참.** 설정 사실과 그릇 경계에서는 선언·강제·검증이 싸고 효과가 있다. 커널·컴파일러·하드웨어 결함, 틀린 선언, 선언이 없는 산출물은 남는다 | 중간~높음 | 9.2~9.4절 |
| 4. 형태 | **라이브러리는 경계와 적재 시점에서 가능하다.** 연산 단위 상시 전파는 성능 때문에 불가능하다. 새 코드용 컴파일러는 기술적으로 가능하지만(MLIR·JAX 선례), 채택은 별개 문제다 | 중간 | 3절, 4절, 5절, 6절 |
| 5. 진단 | **가능.** 연산 단위 전파를 CI·진단 모드에 두는 구도가 제품 쪽 흐름(DTensor 지우기 제안)과 같다 | 높음 | 4.1절, 8.2절 |

### 9.2 가능한 부분

| 부분 | 근거 | 확신 |
|---|---|---|
| **(A) 설정 사실의 적재 시점 대조.** 모델 성질과 커널 능력, 예측 방식 선언과 소비자의 선택, 저장 형식 선언과 실제 stride·스케일 dtype, 모르는 설정 키를 대조한다 | 시작 점검 2.83 µs, 정상 조합 오탐 0(`entail/audits/adapter_pilot.json`). M7에서 파일 선언을 읽어 해소(`issue_track/comfyui_field_test/g2_compare.json`). RoPE 해소기로 279→378/500(`issue_track/rope_override/SUMMARY.json`). JAX 추적 시점 검사, 증명 운반 코드의 한 번 검증(2.2~2.3절) | 높음 |
| **(B) 호스트 쪽 그릇 경계의 계약.** KV 할당량, 길이, 줄지 않음을 검사한다 | 0.979배와 갱신 2,304회 검사(`entail/audits/cache_contract_cost.json`). vLLM 96회 검사(`entail/audits/vllm_cache_on.log`). 심은 결함 검출. 세 엔진에 같은 규칙 | 비용은 높음. 실제 결함 수확은 중간(잡은 것이 모두 심은 결함) |
| **(C) 새 코드의 필수 역할 표지와 닫힌 형식 집합.** 키워드 전용 역할 인자, 역할 타입, 목록 밖이면 오류 | 주입 8종 차단(`phase0/week4/results/e5_role_errors.json`). spmd_types의 `src=`/`dst=`. `mutates_args` 필수. torchao·DTensor·MLIR의 시끄러운 실패(3~4절) | 기술적으로 높음. 채택은 9.3 (I) |
| **(D) 진단 모드의 연산 단위 전파** | 2.19배는 CI에서 받아들일 만하다. DTensor 지우기 제안이 같은 구도다(4.1절). 사실의 무효 표시로 "누가 무효로 만들었나"를 남긴다 | 높음 |

### 9.3 조건부로 가능한 부분

| 부분 | 조건 | 확신 |
|---|---|---|
| **(E) 값으로 정해지는 사실(RANGE, TIME)의 검사** | 사실이 이미 호스트 쪽 정수로 있는 엔진 구조여야 한다(vLLM·SGLang 스케줄러). 그래프 캡처 구간 밖이어야 한다. 그렇지 않으면 요청 밖의 사후 검사로 약해지고, 탐지가 늦다(8.3절) | 중간 |
| **(F) 선언을 실제 데이터로 검증** | 바이트, stride, dtype, 키로 정해지는 사실은 결정적으로 검증된다. 예측 방식이나 위치 기준 같은 뜻 사실은 동작 탐침이라 통계적이고 틀릴 수 있다(M7 탐침 0.94로 놓침). 선언을 1차 근거로 두고, 탐침은 교차 확인과 경고로만 써야 한다 | 중간 |
| **(G) 대부분의 모델에 붙임** | 단위는 모델이 아니라 엔진이다. 엔진마다 어댑터(문서 값 약 100줄)와 비공개 API 추적, 엔진 버전별 CI가 필요하다. 보장 범위는 파일 선언이 있는 모델에 한정된다(연구자 파일 44개 중 24개가 선언) | 중간~낮음(LLM 엔진 셋과 ComfyUI에서 확인, diffusers는 측정 정의 단계) |
| **(H) 부품을 건너는 보존** | 생산자의 경계 선언(`returns`/`writes`)과 그릇 계약으로 이룬다. 손 태그 8→0(`entail/audits/boundary_22.json`). 연산 단위 전파로는 운영에서 이룰 수 없다 | 중간 |
| **(I) 채택** | 선언이 생산 도구에서 자동으로 쓰여야 한다. 선언이 없으면 진행을 멈추거나 "검사 못 함"을 보고해야 한다. 대가가 수치로 보여야 한다. 선택형으로 두면 named tensors와 파이썬 주석처럼 절반쯤에서 멈출 위험이 크다(6절) | 낮음(이 이론에 대한 외부 채택 데이터 없음) |

### 9.4 불가능하거나 이론의 범위 밖인 부분

| 부분 | 근거 | 확신 |
|---|---|---|
| **(J) 컴파일러·커널·하드웨어 내부 결함을 경계 선언으로 막기** | 검증된 컴파일러도 명세 틈과 검증 밖 앞단에서 결함을 냈다(Csmith·CompCert). XLA 오컴파일과 커널 오류는 연구자가 든 증상을 그대로 냈다(Anthropic, OpenAI). 조용히 틀리는 코어가 있다(HotOS '21). 결함이 경계 사실의 불일치로 드러나는 일부만 잡힌다. 나머지는 참조 비교나 평가 점수 같은 사후 탐지의 몫이다 | 높음 |
| **(K) 증상 자체의 제거**(언어 누출, 반복, 맥락 상실) | 같은 증상에 모델 원인이 함께 있다(DeepSeek-R1, Holtzman 외, Lost in the Middle). 이론이 줄이는 것은 원인 부류다(2.5절) | 높음 |
| **(L) 생산자가 틀린 선언을 적은 경우의 완전한 검출** | 결정 불가능성과 신뢰 경계 때문이다. `restrict`와 `mutates_args`는 틀리면 정의되지 않은 행동이다. spmd_types 규칙은 믿는다. vLLM #51063은 유일한 선언이 틀렸다(2.2절 (마)). 데이터 대조로 일부만 잡는다 | 높음 |
| **(M) 모든 연산의 상시 전파를 성능 손실 없이** | 2.19배(이 프로젝트), DTensor 35~60%(블로그 인용). torch 2.14에서는 비기반 디스패치 모드가 켜지면 컴파일을 건너뛴다. CUDA Graph는 재생 때 CPU 작업을 생략한다(3.1절, 2.3절) | 높음(PyTorch 2.14 기준) |
| **(N) 파이썬 정적 타입 검사기로 텐서 사실 검사** | 모양·dtype 주석은 정적 검사기에게 그냥 배열로 취급된다(jaxtyping FAQ). 추적 시점 검사나 새 문법이 필요하다 | 높음 |
| **(O) 수치 허용 오차보다 작은 결함을 값 비교로 판정** | 한 칸 초과 결함이 새 캐시에서 0.06(bf16 잡음 수준)이었다(2.2절 (다)). 이런 결함은 선언 대조로만 가를 수 있다 | 중간~높음 |

### 9.5 세 목표를 동시에 만족하기 위한 조건

세 목표는 서로 당긴다. [판단]
- 구조적 방어는 필수 선언을 요구한다.
- 고치지 않고 붙이려면 엔진에게 선언을 강제할 수 없다.
- 성능을 지키려면 검사할 수 있는 자리가 좁아진다.

셋을 함께 만족하려면 다음 조건이 모두 필요하다.

1. **검사 자리를 셋으로 제한한다.** 적재·시작 경계, 호스트 쪽 그릇 경계, 요청 경계만 상시 검사한다. 연산 단위 전파는 CI와 진단에만 쓴다. (5절, 3.3절, 8절)
2. **컴파일·캡처 구간 안에 검사를 두지 않는다.** 장치에서 호스트로 가는 동기화도 만들지 않는다. (CUDA Graph 문서, 47.665배와 출력 변화)
3. **사실 어휘를 작고 닫힌 명목 태그로 둔다.** 술어는 길이·기준·종류 같은 단순 산술로 제한한다. 결정 가능성과 비용이 함께 관리된다. (Liquid Types, Nom, 2.4절)
4. **선언은 생산 지점에서 한 번, 도구가 자동으로 쓴다.** 사용 지점마다 사람이 적게 하지 않는다. (ASE'18의 51%, kohya의 자동 기입, Rust)
5. **선언이 없으면 조용한 기본값 대신 "검사 못 함"을 보고한다.** 뜻을 바꾸는 사실(예측 방식 등)이 비어 있으면 멈추거나 명시적 선택을 요구한다. (diffusers의 `epsilon` 기본값, ComfyUI의 EPS 기본값, GGUF의 `reference` 기본값)
6. **선언은 결정적 대조로 먼저 검증한다.** 바이트, stride, dtype, 키로 대조하고, 통계적 탐침은 교차 확인에만 쓴다. (M7)
7. **규칙·능력표·어댑터 자체를 데이터로 검증하는 장치를 함께 둔다.** (DTensor `strategy_validation`, spmd_types `rulecheck`, 이 프로젝트의 `cap_probe`)
8. **규칙은 공유 핵심에 두고, 어댑터는 "사실이 어디 있나"만 말한다.** 엔진 버전별 CI로 어댑터가 깨지는 것을 잡는다. 계측은 어떤 경우에도 예외를 밖으로 내지 않는다. (`entail/audits/CACHE_CONTRACT.md`의 SGLang 사고)
9. **범위를 밝힌다.** 커널·컴파일러·하드웨어 내부와 모델의 실력은 범위 밖이다. 참조 비교, 평가 점수 같은 사후 탐지와 함께 쓴다.
10. **"구조적"의 정도를 형태별로 나눠 주장한다.**
    - 새 코드(필수 문법): 선언된 경계 전부에 대해 구조적이다.
    - 붙이기 형태: 라이브러리가 소유한 경계와 선언된 산출물에 대해서만 구조적이다. 나머지는 "검사 못 함"을 보고한다.

성능 목표의 기준선도 적어 둔다. Takikawa 외는 실무가 2배 넘는 부담을 받아들이지 않으리라 봤다(https://www2.ccs.neu.edu/racket/pubs/popl16-tfgnvf.pdf 각주 7). 이 프로젝트의 로드맵 기준은 "디버그 모드 2배 이하"다(`entail/audits/PROPAGATE.md`). 상시 모드에서 잰 값은 경계 방식으로 0.979~1.007배였다(8.3절).
