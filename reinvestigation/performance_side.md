# 성능 쪽 주장 재조사: "의미가 전달되지 않아 헛수고와 손실이 생긴다"

- 작성: 2026-09-23, 컴파일러·성능 조사원(에이전트)
- 대상: 연구자 이론 가운데 성능 쪽 주장만. 안정성(조용한 오답) 쪽은 다루지 않는다.
- 상태: 완료 (2026-09-23)

## 요약 판정 (자세한 근거와 확신 정도는 6절)

- **(a) 주류 모델·주류 엔진: 대체로 성립하지 않는다(확신 중상).** vLLM, SGLang, FlashInfer, TensorRT-LLM, FlashAttention은 유효 길이·헤드 공유·창·softcap을 이미 손으로, 정수와 구조로 넘긴다. vLLM은 어텐션을 컴파일러가 보지 못하게 감싸고, 배치 크기를 미리 선언한다.
- **(b) 새 변형·범용 프레임워크·로컬 도구: 성립한다(확신 중).** transformers 권장 컴파일 경로에서 1.3~2.5배(로컬 실측), vLLM MLA 약 9개월·처리량 약 3배, llama.cpp Gemma 2 FA 공백 58일·최대 1.88배. 다만 새 변형 공백의 주원인은 "의미 미전달"보다 "커널 부재"이고, 의미를 받는 범용 생성기(FlexAttention, FlashInfer JIT, Ladder)는 이미 있다.
- **(c) 성능은 주된 이점이 될 수 없다. 부차적 이점이다(확신 중상).** 큰 손실은 이미 손으로 해결됐고, 남은 손실은 기존 선택적 API로 없어진다. 이론의 성능 몫은 "전문가 경로를 기본값·필수로" 만드는 데 있다.

## 0. 검증할 주장과 읽는 법

주장을 셋으로 나눠 따로 판정한다.

| 기호 | 주장 | 이 문서에서 확인하는 것 |
|---|---|---|
| C1 | 기존 GPU AI 컴파일러와 프레임워크에서는 값과 명령의 의미가 명확히 전달되지 않는다 | 의미가 인터페이스에서 지워지는 구체 지점이 1차 출처에 있는가 |
| C2 | 받는 쪽은 의미를 다시 찾거나(재발견·탐색), 추측하거나(가드·재컴파일), 보수적인 경로로 물러서며 그 결과 성능 손실이 생긴다 | 세 종류 손실 각각의 크기가 측정된 적이 있는가, 주류 스택에서도 생기는가 |
| C3 | 의미를 명시하면 그 헛수고와 손실이 사라진다 | 의미를 명시하는 기존 API·엔진이 실제로 손실을 없앴는가, 그렇다면 "새 이론"의 몫은 무엇인가 |

증거 표기:
- **[1차]** 공식 문서, 논문, 저장소 코드·릴리스 노트를 직접 열어 확인한 것
- **[로컬 실측]** 이 프로젝트의 결과 파일에서 확인한 수치(가상 평가 아님)
- **[확인 못 함]** 찾지 못했거나 직접 열어 보지 못한 것. 추측으로 채우지 않는다.
- 인용은 원문 그대로, 15단어 이하. 원문 인용은 공개 라이선스 저장소의 문서·코드 주석(PyTorch, transformers, vLLM, FlashAttention, TensorRT-LLM, diffusers 등)과 소프트웨어가 내는 문자열(오류·로그)에 한정했다. 논문, 블로그, 발표문, 이슈 댓글은 우리말로 요지를 옮겼다(MLIR 문장 하나만 예외로 인용).

## 1. 문헌: 컴파일러가 높은 수준의 의미를 잃어 최적화를 놓친다는 문제 의식과 그 크기

### 1.1 MLIR (Lattner 외, arXiv 2002.11054, 2020)

- [1차] 설계 원칙 "Maintain higher-level semantics"는 높은 수준의 의미와 구조를 분석·성능 최적화를 위해 유지해야 한다고 적는다. 인용: "Attempts to raise semantics once lowered are fragile". 한 번 낮춘 뒤 의미를 다시 끌어올리는 시도는 깨지기 쉽다는 뜻이다. 출처: https://arxiv.org/abs/2002.11054 (PDF 2절 Design Principles)
- [1차] 같은 절은 구조의 손실이 의식적으로, 실행 모델에 더는 필요 없을 때만 일어나야 한다고 하고, 이를 위해 단계적 낮춤(progressive lowering)을 원칙으로 둔다. affine 방언은 손실이 있는 낮은 수준 표현에서 affine 형태를 추론하는 일을 피하려고 설계했다고 설명한다(5.2절). 출처: 같은 PDF.
- [1차] C++ AMP, HCC, SyCL 같은 기존 흐름이 높은 수준 구성 요소를 너무 빨리 런타임 호출로 낮춘다고 비판한다. 출처: 같은 PDF.
- **크기에 대한 사실:** MLIR 논문의 평가 절은 주된 평가 기준을 여러 프로젝트에 채택되어 쓰인다는 것을 보이는 데 둔다. 정보 손실 때문에 잃은 성능을 수치로 잰 실험은 이 논문에 없다(추출한 본문 전체에서 speedup·slower 관련 측정을 찾지 못함). 출처: 같은 PDF 5절.
- 판정에의 쓰임: C1의 **문제 의식**은 컴파일러 설계의 주류 원칙이다. 그러나 MLIR은 그 손실의 **크기**를 재지 않았다. 이 이론이 "손실이 크다"를 말하려면 따로 재야 한다.

### 1.2 Halide (Ragan-Kelley 외, PLDI 2013)

- [1차] 동기: 단순 구현과 최적화 구현의 차이가 흔히 10배(자릿수 하나) 이상이다. 최적화 서브루틴 라이브러리로는 해결되지 않는데, 핵심 최적화가 단계 사이의 생산자·소비자 지역성을 위한 융합이기 때문이다. 알고리즘과 실행 전략(schedule)을 분리해 해결한다. 출처: https://people.csail.mit.edu/jrk/halide-pldi13.pdf (초록, 1절)
- [1차] 결과: 몇 시간 만에 쓴 Halide 프로그램이 전문가가 수주~수개월 손으로 튜닝한 C·CUDA보다 최대 5배 빨랐다. 출처: 같은 PDF 초록.
- [1차] 탐색 비용: 자동 튜닝은 예제마다 2시간에서 2일이 걸렸다. 한 기계에서 하루 안에 최종 성능의 15% 이내로 수렴했다. 출처: 같은 PDF 6.1절.
- 판정에의 쓰임: Halide의 문제는 "의미가 지워진다"가 아니라 "라이브러리 경계가 융합을 막는다"와 "스케줄을 사람이 짜기 어렵다"다. 의미(알고리즘)를 명시하는 쪽으로 풀었지만, 대가로 **탐색 비용**이 생겼다. 의미를 명시해도 탐색이 사라지지 않는 사례다(C3 반박 쪽).

### 1.3 TVM (Chen 외, OSDI 2018, arXiv 1802.04799)

- [1차] 동기: 기존 프레임워크는 벤더별 연산자 라이브러리에 기대고 좁은 범위의 서버 GPU만 최적화하며, 새 플랫폼 배치에는 상당한 수작업이 든다. 목표는 여러 하드웨어에 대한 성능 이식성이다. 출처: https://arxiv.org/abs/1802.04799 (초록, 1절)
- [1차] 탐색 비용을 동기로 명시한다: 구성 공간이 커서 블랙박스 자동 튜닝을 하면 탐색 비용이 크다고 적고, 그래서 학습된 비용 모델을 쓴다. 출처: 같은 PDF 1절.
- [1차] 결과: 손으로 최적화한 라이브러리를 쓰는 기존 프레임워크보다 1.2~3.8배. 연산자 융합만으로 1.2~2배. 출처: 같은 PDF 초록, 3절.
- 판정에의 쓰임: TVM의 동기는 "의미가 지워진다"보다 "하드웨어가 다양하고 라이브러리가 따라가지 못한다"다. C1을 직접 지지하지 않는다. 다만 융합 이득(1.2~2배)은 그래프 수준 정보를 연산자 경계 너머로 가져간 대가다.

### 1.4 Triton (Tillet, Kung, Cox, MAPL 2019)

- [1차] 동기: 기존 벤더 라이브러리(cuBLAS, cuDNN)를 쓸 수 없는 연산은 전문가가 직접 구현하지 않으면 장치 활용도가 낮을 위험이 있다. 벤더 라이브러리는 제한된 텐서 연산 집합만 지원한다. 출처: https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf (초록, 1절)
- [1차] 결과: cuBLAS와 대등한 행렬곱, 다른 DSL보다 최대 3배, shift-conv 같은 새 연산의 효율적 구현. 출처: 같은 PDF 1절 기여 목록.
- 판정에의 쓰임: Triton의 문제 의식은 **새 변형(새 연산)이 빠른 경로를 갖지 못한다**는 것이다. 이 이론의 (b) 쪽 주장과 가깝고, (a) 쪽 주장과는 거리가 있다.

### 1.5 별칭(aliasing) 정보와 restrict

- [1차] CUDA Programming Guide 5.4.1.4 `__restrict__`: 포인터 별칭은 코드 재배치와 공통 부분식 제거 같은 최적화를 막을 수 있고, restrict를 붙이면 컴파일러가 이를 자유롭게 할 수 있다고 설명한다. 동시에 restrict 포인터가 레지스터 압력을 늘려 점유율을 낮추면 성능이 나빠질 수 있다고 경고한다. 출처: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
- [1차] NVIDIA 기술 블로그(2014-08-07, Jeremy Appleyard): 예시 코드에서 restrict만 붙여 CPU 3.13 ms → 1.05 ms(Xeon E5-2690 v2), GPU 47.6 µs → 22.5 µs(Kepler K40). 출처: https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
  - 주의: 효과를 보이려고 고른 작은 예제다. 일반 프로그램의 평균 효과가 아니다.
- [1차] Rust는 타입(`&mut`의 유일성)으로 별칭이 없다는 사실을 알고 있지만, LLVM 오컴파일 때문에 이를 `noalias`로 넘기지 못했다. 추적 이슈 #54878(2018-10-06 작성)은 LLVM 12 대상으로 다시 켠 PR #82834가 병합된 2021-03-22에 닫혔다. 출처: https://github.com/rust-lang/rust/issues/54878. 이 PR의 성능 측정 댓글에 따르면, LLVM을 거치지 않는 벤치마크(곧 noalias로 빌드한 컴파일러 자신의 실행 속도)에서 최대 1.4%의 개선이 여럿 나왔다. 출처: https://github.com/rust-lang/rust/pull/82834 (bjorn3 댓글 2021-03-06, 2021-03-20)
- 판정에의 쓰임: 의미(별칭 없음)를 넘기면 최적화가 열린다는 것은 확립된 사실이다(C1·C3 지지). 그러나 크기는 **작은 핫 루프에서 2~3배, 큰 프로그램 평균에서 약 1%**로 갈린다. 언어가 의미를 이미 알고 있어도 백엔드가 그것을 안전하게 받아 쓰기까지 추적 이슈 기준 약 2년 반이 걸린 사례이기도 하다.

### 1.6 torch.compile의 가드, 재컴파일, 동적 shape

- [1차] 가드: Dynamo는 로컬·전역 값에 대한 가정을 가드로 표현하고, 가드가 모두 실패하면 다시 추적해 재컴파일한다. int/float는 "treated as constants and are guarded on their exact value". 기본값 `dynamic=None`은 "will only attempt dynamic shapes after the first compilation". 캐시 한도(기본 8)에 닿으면 "the function being skipped (run eagerly)". 출처: https://docs.pytorch.org/docs/2.9/_sources/compile/programming_model.recompilation.md.txt
- [1차] `torch.compile` 문서의 `dynamic` 인자: None이면 "automatically detect if dynamism has occurred"하고 재컴파일 때 더 동적인 커널을 만든다. True여도 일부 연산은 "force specialization"한다. 출처: https://github.com/pytorch/pytorch/blob/main/torch/__init__.py (`compile` docstring)
- [1차] 0/1 특수화: 크기 0과 1은 자동으로 특수화한다. 이유는 연속성·브로드캐스트 검사를 단순하게 해서 "avoids adding extra guards"이기 때문이다. 출처: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/dynamic_shapes_zero_one_specialization.html (원문 https://github.com/pytorch/pytorch/blob/main/docs/source/user_guide/torch_compiler/compile/dynamic_shapes_zero_one_specialization.md)
- [1차] `mark_unbacked` docstring: 표시한 차원은 "always be reported as not equal to zero or one"이고 값이 없는 크기로 다뤄진다. `shape_id`로 여러 텐서가 같은 크기임을 선언할 수 있다. 출처: https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/decorators.py
- [1차] 같은 파일의 `mark_dynamic` docstring은 `specialize_on`(특정 값에 특수화한 컴파일을 추가)으로 "2-8x speedups depending on the specific specialization and model architecture"를 측정했다고 적는다. 즉 **값을 모르는 일반 커널은 값을 아는 특수화 커널보다 느릴 수 있다.** 출처: 같은 파일.
- [1차] 재발견의 실제 형태: Inductor에는 matmul·softmax로 풀어 쓴 어텐션을 다시 찾아 SDPA로 바꾸는 패턴이 30개(`_sfdp_pattern_1`~`_30`) 있다. 출처: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/fuse_attention.py
- [1차] Triton도 값을 관측해 특수화한다. 정수 인자가 1이면 상수로, 16의 배수면 "D" 표지로, 포인터가 16바이트 정렬이면 "D"로 캐시 키에 넣는다. 키가 바뀌면 새로 컴파일한다. 사용자는 `do_not_specialize`로 끌 수 있다. 출처: https://github.com/triton-lang/triton/blob/main/python/src/specialize.cc (`handle_long_type` 부근), https://github.com/triton-lang/triton/blob/main/python/triton/runtime/jit.py
- 판정에의 쓰임: "추측하고, 틀리면 다시 컴파일하고, 한도를 넘으면 eager로 물러선다"는 C2의 세 행동이 torch.compile 공식 문서에 그대로 적혀 있다(C2 지지). 동시에 PyTorch는 `mark_unbacked`, `shape_id`, `specialize_on` 같은 **선언 수단을 이미 제공**한다(C3의 "명시하면 사라진다"는 이미 부분 구현되어 있다). 그리고 선언해서 동적으로 만든 커널이 특수화 커널보다 느릴 수 있다는 것도 공식 문서가 말한다(C3의 "손실이 사라진다"를 제한).

### 1.7 자동 튜닝 탐색 비용

- [1차] Halide: 예제마다 2시간~2일(1.2절).
- [1차] Ansor(OSDI 2020, arXiv 2006.06762): 한 DNN의 완전 최적화 프로그램을 만드는 데 한 기계에서 보통 몇 시간이 걸린다고 적는다. 배포 전 한 번 하는 일이라 추론에는 수용 가능하다고 본다. AutoTVM과 같은 성능에 이르는 탐색 시간을 약 10배(자릿수 하나) 줄였다고 주장한다. 출처: https://arxiv.org/abs/2006.06762 (7.4절, 8절 부근)
- [1차] `torch.compile(mode="max-autotune")`는 Triton·템플릿 행렬곱 후보를 프로파일해 고른다. 출처: https://github.com/pytorch/pytorch/blob/main/torch/__init__.py
- [로컬 실측] 이 카드에서 torch.compile 기본 모드의 콜드 컴파일 40.1 s 중 Inductor 융합 결정(`Scheduler.fused_nodes`) 12.9 s, 자동 튜너 벤치마크(`CachingAutotuner.benchmark_all_configs`) 0.80 s였다. 출처: `phase0/week4/results/e1_compile_anatomy.json`
- 판정에의 쓰임: 자동 튜닝 탐색은 비싸지만(시간~일), 그것은 **하드웨어 매핑의 선택지를 찾는 비용**이지 지워진 의미를 되찾는 비용이 아니다. 의미를 명시해도 이 탐색은 남는다(Halide가 그 예). 따라서 이 비용을 이론의 이득으로 셀 수 없다.

### 1.8 문헌 절 소결

| 주장 | 지지 근거 | 반박·제한 근거 |
|---|---|---|
| C1 의미가 전달되지 않는다 | MLIR 설계 원칙, Inductor의 어텐션 재발견 패턴 30개, Triton·Dynamo의 값 관측 특수화 | MLIR·TVM·Halide의 주 동기는 "의미 손실"보다 하드웨어 다양성·라이브러리 경계·스케줄 작성 난이도다 |
| C2 재발견·추측·후퇴로 손실 | torch.compile 문서의 가드→재컴파일→eager 후퇴, restrict 미표시 때 2~3배(예제) | 크기를 잰 1차 문헌은 드물다. MLIR은 재지 않았다. 큰 프로그램 평균의 별칭 효과는 약 1%(Rust) |
| C3 명시하면 사라진다 | restrict, noalias, mark_unbacked 같은 선언이 실제로 최적화를 연다 | 명시해도 탐색(자동 튜닝)은 남는다(Halide·Ansor). 값 모르는 동적 커널은 특수화보다 느릴 수 있다(PyTorch 문서 2~8배) |

## 2. LLM 추론의 구체 사례: 마스크 표와 의미 전달

### 2.1 커널 API가 의미를 받는 방식

- **FlashAttention** [1차]: 공개 함수에 `attn_mask` 인자가 없다. 대신 의미를 직접 받는다. `causal`, `window_size`(슬라이딩 윈도), `flash_attn_with_kvcache`의 `cache_seqlens`("keep track of the current sequence lengths"), `block_table`(페이지 KV), 가변 길이 묶음은 `flash_attn_varlen_func`로 받는다(누적 길이 `cu_seqlens_q`·`seqused_k`를 넘기는 실제 호출은 2.3절의 vLLM 코드에서 확인). GQA는 "passing in KV with fewer heads than Q"로 받는다. 출처: https://github.com/Dao-AILab/flash-attention (README)
- **PyTorch SDPA** [1차]: 백엔드를 우선순위대로 시험해 첫 번째로 조건을 통과한 것을 쓴다. flash 백엔드의 조건 `check_for_attn_mask`는 마스크가 있으면 탈락시킨다. 경고 문구: "Flash Attention does not support non-null attn_mask." 출처: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp (`can_use_flash_attention`), https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/sdp_utils_cpp.h (`check_for_attn_mask`)
- **SDPA의 `enable_gqa`** [1차]: 2.14 문서는 GQA를 "an experimental feature"로 표시하고, FlashAttention·cuDNN·math 커널에서 동작하며 메모리 효율 어텐션도 NVIDIA CUDA에서 지원한다고 적는다. 출처: https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
- **FlexAttention** [1차]: 사용자가 `mask_mod` 함수로 뜻을 적으면 `BlockMask`가 블록 희소성을 담고, 컴파일러가 커널을 만든다. 공식 블로그의 요지(의역):
  - 동기: 어텐션 변형이 기존 최적화 커널에 맞지 않으면 느린 실행과 메모리 부족을 피할 수 없다. 블로그는 이를 연구자의 "software lottery"라 부른다.
  - 의미를 사용자가 나눠 줘야 한다: `score_mod`만 받아서는 어느 부분이 마스킹인지 컴파일러가 건전하게 알아낼 방법이 없다. 그래서 `score_mod`와 `mask_mod`를 사용자가 분리해 넘긴다.
  - 재발견의 비용: `create_block_mask`는 비교적 비싼 연산이다. 블록이 완전히 비었는지 알려면 블록 안 모든 점에서 `mask_mod`를 평가해야 하기 때문이다.
  - 성능: A100에서 FlashAttention2 대비 순전파 90%, 역전파 85%. 원소마다 마스크를 적용하면 15~20% 느려진다.
  - 출처: https://pytorch.org/blog/flexattention/ (페이지 표시 갱신일 2025-05-30, PyTorch 2.5 이전 게시)
- 소결: 빠른 커널은 모두 "유효 길이·창·헤드 공유"를 **정수와 구조로** 받는다. 범용 API인 SDPA는 같은 뜻을 **참/거짓 표로만** 받을 수 있고, 표를 받는 순간 가장 빠른 백엔드가 빠진다. C1은 **범용 API 경계에서** 1차 출처로 성립한다.

### 2.2 범용 프레임워크(transformers)가 실제로 하는 일

- [1차] transformers 메인 `sdpa_attention.py`의 `use_gqa_in_sdpa`: CUDA에서는 "attention_mask is None"일 때만 `enable_gqa`를 쓰고, 아니면 `repeat_kv`로 KV를 헤드 수만큼 물리적으로 복사한다. 주석은 마스크가 있으면 "it will fall back to the math kernel"이라고 적는다. 같은 파일의 위치 편향 마스크 설명: "will usually prevent sdpa from dispatching to the most efficient kernel". 출처: https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py
- [1차] 같은 저장소 `masking_utils.py`의 `_ignore_causal_mask_sdpa`는 패딩 마스크가 전부 참인지 실행 중에 검사해(`fast_all`) 마스크를 버릴 수 있는지 **다시 찾는다**. 목적은 "allowing to dispatch to the flash attention kernel"이다. 그러나 추적 중이면 `if is_tracing(padding_mask): return False`로 이 재발견을 끈다. `is_tracing`은 Dynamo 컴파일·export, torch.jit, CUDA 스트림 캡처(CUDA Graph)를 모두 추적으로 본다. 즉 torch.compile이나 CUDA Graph로 감싸면 이 함수는 마스크를 버리지 않는다. 출처: https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py, https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py (`is_tracing`)
- [1차] transformers 문서는 속도를 위해 정적 KV 캐시와 torch.compile을 권하며 "up to a 4x speed up"이라 적는다. 동시에 배치 크기나 최대 길이가 바뀌면 "the cache is reinitialized and recompiled"이고, 재컴파일을 피하려면 `pad_to_multiple_of`로 길이를 제한하라고 한다. 출처: https://huggingface.co/docs/transformers/main/en/llm_optims
- 읽는 법: transformers가 권하는 빠른 경로(정적 캐시 + 컴파일)가 바로 "마스크 → flash 탈락 → KV 복사" 경로다. 정적 캐시는 아직 안 쓴 칸을 가려야 해서 마스크를 버릴 수 없고, 컴파일 중에는 재발견도 꺼진다. 이것은 이 프로젝트 로컬 측정(5절)의 원인을 소스 수준에서 확인해 준다.

### 2.3 주류 서빙 엔진은 이 의미를 이미 손으로 넘긴다 (이미 해결됨)

- **vLLM** [1차]
  - 어텐션 메타데이터가 `seq_lens`, `query_start_loc`, `block_table`, `slot_mapping`, `causal`을 필드로 갖고, 커널 호출에 `cu_seqlens_q`, `seqused_k`, `block_table`, `window_size`, `softcap`을 넘긴다. FA4 경로의 커널 컴파일 키에는 `qhead_per_kvhead`(헤드 공유 비율), 창 유무, softcap이 들어 있고, 모델 설정(`attn_logit_softcapping`, 슬라이딩 윈도)을 읽어 미리 컴파일할 키 목록(`get_warmup_keys`)을 만든다. 추측하지 않고 설정에서 읽는다. 출처: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py
  - torch.compile 설계 문서: 어텐션 전체를 커스텀 op으로 감싸 "Dynamo will not try to inspect any of the internal operations". 기호 shape는 "Only the input ids and position ids"뿐이고, CUDA Graph는 정해진 크기 목록(`cudagraph_capture_sizes`)으로 캡처하며, `compile_sizes`로 특정 크기 특수화도 선언한다. 출처: https://github.com/vllm-project/vllm/blob/main/docs/design/torch_compile.md
  - 같은 문서는 vLLM이 Dynamo 가드를 버리는 방식을 쓴다고 적고, 그 가드 중 다수가 "could be material"이라고 인정한다. unbacked(값 모르는 크기)의 단점으로 "missed optimization opportunities"를 든다. 자동 튜닝은 "from seconds to minutes"가 걸려 기본으로 끈다. 출처: 같은 문서.
  - 주류 엔진에도 남는 후퇴: 어텐션 백엔드는 우선순위 목록에서 설정과 호환되는 첫 백엔드를 고른다(SDPA와 같은 방식). 문서는 Blackwell에서 head_size 256용 FA4 커널이 logit softcap, attention sink 등을 지원하지 않아 그런 설정은 "transparently fall back to FA2"한다고 적는다. 뜻은 엔진이 알고 있고, 가장 빠른 커널의 지원 범위가 좁아서 생기는 후퇴다. 그 크기는 확인 못 함. 출처: https://github.com/vllm-project/vllm/blob/main/docs/design/attention_backends.md
- **SGLang** [1차]: Triton 디코드 어텐션이 `kv_indptr`, `kv_indices`, `logit_cap`을 인자로 받는다. CUDA Graph는 설정한 최대 배치 크기(`--cuda-graph-max-bs-decode`) 이하에서만 쓰며, 기본값은 작은 배치(예: 160 또는 256 미만)로 정해져 있다. 출처: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_backend.py, https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx
  - 참고: SGLang의 RadixAttention은 요청 사이의 공유 접두사를 **실행 중 자동으로 찾아** KV를 재사용하고, 논문은 최대 6.4배 처리량을 보고한다. 재발견이 싸고 이득이 큰 반례다. 출처: https://arxiv.org/abs/2312.07104
- **FlashInfer** [1차]: 가변 길이는 `indptr`, 페이지 KV는 `kv_page_indptr`·`kv_page_indices`·`kv_last_page_len`으로 받는다. 임의 마스크도 받지만 비트 압축 배열로 받는다. 변형은 JIT로 만드는 사용자 정의 템플릿으로 받는다. 논문 초록은 LLM 서빙용 컴파일러 백엔드 대비 토큰 간 지연 29~69% 감소를 보고한다. 출처: https://docs.flashinfer.ai/tutorials/kv_layout.html, https://arxiv.org/abs/2501.01005
- **TensorRT-LLM** [1차]: 패딩 없는 packed 모드에서 "a 1D tensor containing the lengths of the different sequences"를 받는다. MQA·GQA는 같은 `gpt_attention` 연산이 KV 헤드 수로 처리한다. 슬라이딩 윈도는 `max_attention_window_size`로 받는다. 출처: https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html
- 판정: 주류 엔진은 (1) 유효 길이·헤드 공유·창·softcap을 정수와 구조로 커널에 직접 넘기고, (2) 어텐션을 컴파일러가 들여다보지 못하게 감싸 재발견 자체를 없애고, (3) 배치 크기 집합을 미리 선언해 추측을 없앴다. **2.1~2.2의 손실은 주류 엔진에서는 이미 손으로 해결되어 있다.** 남은 대가는 세 가지다. 엔진마다 사람이 이 배관을 다시 짜야 하고, 가드를 버리는 선택은 안전성 쪽 위험을 남기며, 가장 빠른 커널이 지원하지 않는 변형 조합은 조용히 느린 커널로 물러난다(크기 미확인).

### 2.4 LLM 절 소결

| 주장 | 지지 근거 | 반박·제한 근거 |
|---|---|---|
| C1 | SDPA는 유효 길이를 참/거짓 표로만 받는다. transformers는 마스크가 있으면 GQA 사실을 버리고 복사한다 | FlashAttention·FlashInfer·vLLM·SGLang·TRT-LLM은 의미를 정수·구조로 받는다 |
| C2 | transformers는 마스크를 버릴 수 있는지 실행 중에 재발견하고, 컴파일 중에는 그마저 끈다. FlexAttention의 BlockMask 생성은 블록 안 모든 점을 평가한다 | 주류 엔진은 재발견·추측을 설계로 피한다. RadixAttention처럼 싸고 이득이 큰 재발견도 있다 |
| C3 | 의미를 받는 커널(FlexAttention, FlashAttention)은 빠른 경로를 쓴다 | 이미 존재하는 API다. FlexAttention은 FA2의 85~90%이고, unbacked 선언은 최적화를 잃을 수 있다(vLLM 문서) |

## 3. 새 변형 지원 지연: 빠른 경로가 생기기까지의 시간과 그동안의 경로

날짜는 모두 1차 출처(저장소 PR 병합 시각, 태그 커밋 시각, 논문 제출일, 공식 발표일)에서 읽었다.

### 3.1 사례별 연표

| 변형 | 공개 | 엔진·라이브러리의 빠른 경로 | 그동안 돈 경로 | 출처 |
|---|---|---|---|---|
| 슬라이딩 윈도(Mistral 7B) | 2023-09-27 | FlashAttention v2.3.0 태그 커밋 2023-09-27, vLLM PR #1196 병합 2023-09-28 | 공백 거의 없음. Mistral이 FlashAttention·xFormers·vLLM 관리자와 직접 협업했고, 발표문은 FlashAttention·xFormers 변경으로 2배 속도 향상을 얻었다고 적는다 | https://mistral.ai/news/announcing-mistral-7b, https://github.com/Dao-AILab/flash-attention/releases/tag/v2.3.0, https://github.com/vllm-project/vllm/pull/1196 |
| Gemma 2 어텐션 softcap | 2024-06-27 | FlashInfer 경로 vLLM PR #6051 병합 2024-07-04(7일). FlashAttention v2.6.0 2024-07-11(14일). transformers FA2 softcap PR #31887 2024-07-11. llama.cpp FA softcap PR #8542 2024-08-24(58일) | vLLM 첫 지원(PR #5908, 2024-06-27)은 모든 어텐션 커널을 고쳐야 한다는 이유로 softcap을 임시로 뺐다(PR 본문). HF 블로그는 softcap이 Flash Attention·SDPA와 호환되지 않는다고 적었다. llama.cpp는 FA 없이 일반 어텐션으로 돌았다 | https://github.com/vllm-project/vllm/pull/5908, https://github.com/vllm-project/vllm/pull/6051, https://huggingface.co/blog/gemma2, https://github.com/huggingface/transformers/pull/31887, https://github.com/ggml-org/llama.cpp/pull/8542 |
| Gemma 2 층별 교차 슬라이딩 윈도 | 2024-06-27 | vLLM PR #10584 "gemma2 full context length support" 병합 2024-11-23(약 5개월) | vLLM은 그동안 최대 길이를 4K로 잘랐다(PR #5908 본문). 8K로 강제한 사용자는 반복되는 이상 출력을 보고했다(#8580) | https://github.com/vllm-project/vllm/pull/10584, https://github.com/vllm-project/vllm/issues/6220, https://github.com/vllm-project/vllm/issues/8580 |
| MLA(DeepSeek-V2) | 2024-05-07(arXiv) | SGLang v0.3 2024-09-04(약 4개월). vLLM v0.7.1 2025-02-01(약 9개월) | vLLM PR #4650(2024-06-28 병합)은 논문의 효율적 추론 모드 구현을 할 일 목록에 남긴 채 들어갔다. v0.7.1 릴리스 노트: v0.7.0 대비 생성 처리량 약 3배, 토큰 메모리 용량 약 10배 | https://arxiv.org/abs/2405.04434, https://lmsys.org/blog/2024-09-04-sglang-v0-3/, https://github.com/vllm-project/vllm/pull/4650, https://github.com/vllm-project/vllm/releases/tag/v0.7.1 |
| AWQ 4비트 가중치 | 2023-06-01(arXiv) | vLLM 첫 지원 PR #1032 2023-09-16. 빠른 Marlin 커널 경로 PR #6612 2024-07-21(첫 지원 뒤 약 10개월) | 2023-12-02 vLLM 문서 경고: AWQ는 "under-optimized", 양자화하지 않은 모델보다 처리량이 낮다(PR #1883). 최적화 목록에 없는 양자화 방식에는 "The speed can be slower than non-quantized models" 경고를 2025-09-17까지 띄웠다(PR #25012) | https://arxiv.org/abs/2306.00978, https://github.com/vllm-project/vllm/pull/1032, https://github.com/vllm-project/vllm/pull/1883, https://github.com/vllm-project/vllm/pull/6612, https://github.com/vllm-project/vllm/pull/25012 |
| GGUF | (형식은 llama.cpp 쪽) | 현재 vLLM 문서도 "highly experimental and under-optimized"로 표시 | 확인 못 함(정량 수치 없음) | https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/gguf.md |

### 3.2 공백 기간의 느린 경로·틀린 경로 크기 (1차 출처의 수치만)

- **vLLM의 DeepSeek(MLA):** 최적화 전후 생성 처리량 약 3배, 토큰 메모리 용량 약 10배(v0.7.1 릴리스 노트). SGLang v0.3은 가중치 흡수 등 같은 계열의 최적화로 H100에서 기준 시스템 대비 3~7배 처리량을 보고했다. 그 기준 시스템의 이름은 블로그 본문에 없다(확인 못 함). 출처: 위 표.
- **llama.cpp의 Gemma 2:** 공백 기간(2024-06-28 지원 ~ 2024-08-24)에는 Gemma 2에 FlashAttention 경로를 쓸 수 없었다. 당시 FA가 기본값이었는지는 확인 못 했으므로, 아래는 FA를 켜려던 사용자가 잃은 크기로 읽어야 한다. FlashAttention(softcap 포함) 대 일반 어텐션의 프롬프트 처리(pp4096) 속도비: RTX 4090에서 마이크로배치 1일 때 1.07배, 512일 때 1.50배, 4096일 때 1.88배. RTX 3090은 1.07~1.40배. RX 6800에서는 마이크로배치 8 이상에서 오히려 0.70~0.87배로 FA가 느렸다. 출처: https://github.com/ggml-org/llama.cpp/pull/8542 (PR 본문 표)
- **vLLM의 Gemma 2:** 속도가 아니라 기능과 정확성의 공백이다. softcap을 빼고(근사), 길이를 4K로 잘랐다. 8K를 강제하면 틀린 출력이 났다.
- **vLLM의 AWQ:** 문서가 "lower throughput than unquantized version"이라 적었다. 배수는 PR 본문 그림에만 있어 텍스트로 확인 못 함.

### 3.3 이 공백의 원인은 "의미 미전달"인가

- 원인은 대부분 **커널이 없어서**다. softcap은 설정 파일에 값으로 있었고(엔진은 그 뜻을 알았다), 빠른 커널이 그 연산을 지원하지 않았다. MLA도 엔진은 모델 구조를 알았고, 잠재 공간에서 계산하는 커널과 가중치 흡수 구현이 없었다. AWQ도 형식은 알려져 있었고, 빠른 행렬곱 커널(Marlin)과 재배치가 없었다.
- 그래서 "의미를 명시하면 빠른 경로가 생긴다"는 **범용 코드 생성기가 그 의미를 받아 커널을 만들 수 있을 때만** 성립한다. 그런 도구는 이미 있다.
  - FlexAttention은 softcap을 `score_mod` 몇 줄로 표현하는 예를 공식 블로그에 싣고 있다. 출처: https://pytorch.org/blog/flexattention/
  - FlashInfer는 사용자가 바꿀 수 있는 어텐션 템플릿을 JIT로 컴파일한다. 출처: https://arxiv.org/abs/2501.01005
  - Ladder(OSDI 2024)는 사용자 정의 자료형을 일반 타입 시스템(tType)의 1급 요소로 받아 커널을 만들고, W_INT4·A_FP16에서 vLLM 대비 평균 2.3배(A100), GPU가 지원하지 않는 형식에서 최대 14.6배를 보고했다. 초록은 기존 하드웨어·소프트웨어의 새 저정밀 형식 지원이 부족하고 비효율적이라고 적는다. 출처: https://www.usenix.org/conference/osdi24/presentation/wang-lei (PDF: https://www.usenix.org/system/files/osdi24-wang-lei.pdf)
- 모델 설계까지 바꾼 사례: FlexAttention 블로그는 MosaicML이 ALiBi를 버리고 RoPE로 간 주된 이유로 커널 지원 부족을 들었다고 적는다. 빠른 경로가 없는 변형은 쓰이지 않게 된다는 뜻이다. 출처: https://pytorch.org/blog/flexattention/
- 반례: 모델 제작사가 커널 라이브러리와 미리 협업하면 공백이 0일에 가깝다(Mistral 7B). 공백은 기술 문제이면서 조직 문제다.

### 3.4 새 변형 절 소결

| 주장 | 지지 근거 | 반박·제한 근거 |
|---|---|---|
| C2 (느린·틀린 경로로 물러선다) | vLLM MLA 약 9개월·처리량 약 3배 손실, llama.cpp Gemma 2 FA 공백 58일·최대 1.88배, vLLM AWQ 약 10개월 동안 문서상 비양자화보다 느림, Gemma 2 softcap 제거·4K 절단 | 원인은 커널 부재이지 의미 미전달이 아니다. 협업이 있으면 공백이 없다(Mistral) |
| C3 (명시하면 사라진다) | 의미를 받는 범용 생성기(FlexAttention, FlashInfer JIT, Ladder)가 있으면 공백을 줄일 수 있다 | 그 생성기들은 이미 있다. 범용 생성기는 손 커널의 85~90% 수준이고(FlexAttention 블로그), MLA의 가중치 흡수 같은 대수적 재구성은 선언만으로 나오지 않는다 |

## 4. 이미지 생성 쪽: diffusers와 ComfyUI

결론 먼저: 의미가 전달되지 않아 생기는 성능 손실 사례는 **있다**. 다만 주류 추론 경로에서 확인된 크기는 1% 미만(SDXL)에서 11%(QwenImage, 사용자 보고 1건) 사이이고, 여러 곳이 이미 손으로 우회되어 있다. 큰 손실은 배치가 1보다 큰 학습과 로컬 도구의 재컴파일에서 보고되지만 정량 수치는 대부분 확인 못 했다.

### 4.1 diffusers: 불필요한 마스크와 flash 탈락

- [1차] PR #12870(2025-12-21 작성, 미병합): Qwen, Chroma, Z-Image처럼 캡션 길이가 다른 모델은 배치가 1보다 크면 마스크가 필요하다. 본문 요지: SDPA는 마스크가 없을 때만 내부 flash 알고리즘을 쓰고, 아니면 긴 시퀀스에서 특히 훨씬 느린 알고리즘으로 떨어진다. 배치를 샘플별로 나눠 어텐션을 여러 번 부르는 편이 마스크 어텐션보다 빨랐다(배치 8까지). 그러나 추론 파이프라인은 이미 CFG를 배치 1 두 번으로 불러 마스크를 피하므로 추론 쪽 실제 이득은 작다고 적는다. 수치는 그림으로만 있어 텍스트로 확인 못 함. 출처: https://github.com/huggingface/diffusers/pull/12870
- [1차] PR #12987(2026-01-27 병합) "[Qwen] avoid creating attention masks when there is no padding": 전부 참인 마스크를 None으로 바꾼다. 리뷰 댓글에서 한 사용자가 L20에서 11% 성능 향상을 보고했다. 다른 참여자는 GPU 위 마스크가 전부 참인지 확인하려면 CPU로 옮겨야 해서 컴파일 그래프가 반드시 끊긴다고 지적하고, 판단을 파이프라인(뜻을 아는 쪽)으로 옮기라고 권했다. 출처: https://github.com/huggingface/diffusers/pull/12987
- [1차] diffusers 어텐션 디스패처의 가변 길이 백엔드는 마스크에서 길이를 **다시 계산한다**: `seqlens_k = attn_mask.sum(dim=1)`, `max_seqlen_k = seqlens_k.max().item()`(호스트 동기화). 주석: "arbitrary QK masks is not supported in Flash/Sage varlen". 이 재계산의 비용은 확인 못 함. 출처: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_dispatch.py (`_prepare_for_flash_attn_or_sage_varlen_with_mask`, `_normalize_attn_mask`)
- 읽는 법: 이것은 LLM 쪽 2.2절과 같은 구조다. "패딩 없음"이라는 뜻을 아는 쪽(파이프라인)과 그 뜻이 필요한 쪽(어텐션 커널) 사이에서 뜻이 참/거짓 표로 풀리고, 받는 쪽은 표를 훑어 되찾거나 느린 경로로 간다.

### 4.2 diffusers: 해상도·LoRA 변경과 재컴파일

- [1차] PyTorch 블로그(2025-07, 페이지 갱신일 2025-07-18), H100, Flux-1-Dev:
  - DiT 컴파일로 6.7 s → 4.5 s(약 1.5배). 첫 호출은 컴파일 때문에 67.4 s. 반복 블록만 컴파일하면 콜드 9.6 s, 캐시가 있으면 2.4 s.
  - 해상도를 512에서 1024로 바꾸면 재컴파일이 일어난다. 대책은 `dynamic=True` 또는 `mark_dynamic`.
  - LoRA를 바꾸면 재추적·재컴파일이 일어나므로, 사용자가 필요한 최대 LoRA 랭크를 **미리 선언**하고 한 번만 컴파일한다(PEFT hotswap).
  - 장치→호스트 동기화는 널리 쓰이는 diffusers 파이프라인에 대부분 없다고 보고 다루지 않았다.
  - 출처: https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/
- [1차] 같은 블로그가 인용한 diffusers 측정(H100, torch 2.8 개발판, `dynamic=True`로 한 번 컴파일, 해상도 3종에서 재컴파일 없음): 1024×1024 0.153 s(컴파일 없이 0.234 s), 1536×768 0.190 s(0.260 s), 2048×2048 0.963 s(1.217 s). 동적 컴파일로도 이득 대부분이 남았다. 해상도별 정적 컴파일과의 비교는 없다. 출처: https://github.com/huggingface/diffusers/issues/11360#issuecomment-2942734765
- [1차] SDXL, A100(PyTorch 블로그 "Diffusion, Fast"): fp32 7.36 s → bf16 4.63 s → SDPA 3.31 s → 컴파일(max-autotune, channels_last) 2.54 s → QKV 결합·그래프 끊김 제거·GPU 동기화 제거 2.52 s → 동적 int8 2.43 s. 의미·배치와 관련된 손질(QKV 결합, 동기화 제거)의 몫은 0.02 s였다. 출처: https://pytorch.org/blog/accelerating-generative-ai-3/

### 4.3 ComfyUI

- [1차] `attention_pytorch`는 마스크를 SDPA에 그대로 넘기고(`attn_mask=mask`), Sage 백엔드들은 마스크가 있고 Sage가 마스크를 지원하지 않으면 PyTorch 어텐션으로 물러선다. 출처: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/modules/attention.py (`attention_pytorch`, `attention_sage`, `attention3_sage`)
- [1차] 재컴파일 보고(모두 열린 이슈, 정량 수치 없음):
  - #9972(2025-09-21): Wan 2.1에 TorchCompileModel을 붙이자 "torch._dynamo hit config.recompile_limit (64)". 마지막 가드 실패 이유는 층별 가중치 패치 객체의 문자열 키(`self.key == 'diffusion_model.blocks.6.self_attn.v.weight'`)였다. 계산과 무관한 파이썬 값이 가드로 굳어 층마다 다시 컴파일된 것이다. https://github.com/Comfy-Org/ComfyUI/issues/9972
  - #14536(2026-06-18): 같은 종류의 가드(`weight_function[0].key == ...`)로 생성 시간이 늘었다는 보고. https://github.com/Comfy-Org/ComfyUI/issues/14536
  - #6415(2025-01-09): Flux에서 해상도를 고정하고 프롬프트만 바꿔도 매번 재컴파일되어 생성이 매우 느려진다는 보고. 원인은 이슈에서 확정되지 않았다. https://github.com/Comfy-Org/ComfyUI/issues/6415
- 이 재컴파일의 크기(초 단위 멈춤, 한도 도달 뒤 eager 비율)는 확인 못 함.

### 4.4 로컬 측정(이미지)

- [로컬 실측] SDXL UNet 한 스텝(RTX 4070 Ti, fp16, 128×128 latent): eager 배치 1 129.4 ms 중 cuBLAS 68.5 ms, cuDNN 29.0 ms, FlashAttention-2 14.7 ms, ATen 요소별 14.0 ms. GPU 유휴 약 3%. 컴파일로 5~12% 개선, 요소별 14 ms 중 Inductor가 회수한 것은 약 2 ms. cuDNN이 넣던 NCHW↔NHWC 전치 커널 약 110개가 레이아웃 최적화로 사라졌다. 출처: `phase0/RESULTS.md`, `phase0/WEEK1_NOTES.md` 1절·발견 4, `phase0/week2/WEEK2_NOTES.md` 실험 C, `phase0/results/sdxl_unet.json`
- 읽는 법: 레이아웃(NCHW 대 NHWC)은 "값의 뜻" 가운데 저장 형식에 해당하고, 이것이 경계에서 맞지 않아 변환 커널이 끼었다. 그러나 이 카드의 SDXL 추론은 라이브러리 커널이 87%이고 GPU가 이미 97% 바쁘므로 의미 전달로 되찾을 여지가 작다.

### 4.5 이미지 절 소결

| 주장 | 지지 근거 | 반박·제한 근거 |
|---|---|---|
| C1·C2 | 불필요한 마스크로 flash 탈락(QwenImage, 사용자 보고 11%), 마스크에서 길이 재계산과 호스트 동기화, 가중치 키 문자열 가드로 재컴파일 한도 도달(ComfyUI) | 주류 추론 파이프라인은 배치 1 호출·None 마스크로 이미 우회. SDXL에서 의미 관련 손질의 몫은 0.02 s(2.54→2.52 s) |
| C3 | 파이프라인에서 "패딩 없음"을 알고 None을 넘기면 해결(PR #12987). LoRA 최대 랭크를 선언하면 재컴파일 없음 | 이미 사람이 손으로 하고 있다. 크기는 작거나 확인 못 함 |

## 5. 이 프로젝트의 로컬 측정과 주류 엔진 대조

모든 수치는 RTX 4070 Ti 12 GB, torch 2.14.0+cu130, Triton 3.8.0, transformers 5.17.0, Qwen3-4B에서 잰 **실측**이다. 1~3주차는 화면 출력 겸용 상태라 절대 시간에 약 10% 잡음이 있고, 교차 측정한 비율은 유효하다(`phase0/week3/WEEK3_NOTES.md` 4.1절). 4주차는 재부팅 뒤 계산 전용 상태다.

### 5.1 수치 (결과 파일에서 확인)

**(가) 컴파일된 디코드가 배치 8에서 이득을 잃는다 (1주차)**
- bf16 배치 1: eager 29.42 ms → compile+CUDA Graphs 21.84 ms. 배치 8: eager 36.72 ms, compile_default 37.26 ms, compile_graphs 36.69 ms로 이득이 0이다.
- int4 compile_graphs 배치 8(24.54 ms)에서 메모리 효율 어텐션 8.11 ms와 KV 복제 Triton 커널 6.61 ms가 스텝의 약 60%다. eager는 `flash_fwd_splitkv`를, 컴파일 경로는 `fmha_cutlassF`(메모리 효율)를 썼다.
- 출처: `phase0/RESULTS.md`, `phase0/results/llm_decode.json`, `phase0/results/llm_decode_int4.json`

**(나) 원인 분리 (2주차, 한 층 한 호출, 배치 8, 캐시 640)**
- 현재 경로(참/거짓 마스크 + KV 복제 + 메모리 효율) 495 µs, 그중 복제만 277 µs, GQA만 네이티브로 209 µs, 유효 길이·GQA를 직접 받는 Triton 커널 84 µs. flash 백엔드에 마스크를 주면 "No available kernel".
- 출처: `phase0/week2/WEEK2_NOTES.md` A1, `phase0/week2/results/decode_attn_swap.json`

**(다) 사실 하나씩 빼기 (4주차 E4, CUDA Graph, 20라운드 교차, 20라운드 전부 기본보다 빠름)**

| 설정 | 사실 없음(transformers 기본) | 헤드 공유만 | 둘 다, 범용 컴파일러(FlexAttention) | 둘 다, 손 커널 | 상한(컴파일 시점 상수) |
|---|---|---|---|---|---|
| bf16 배치 8 | 36.18 ms | 26.52 (×0.734) | 22.53 (×0.623) | 22.49 (×0.622) | 22.29 (×0.617) |
| int4 배치 1 | 9.28 ms | 8.07 (×0.868) | 7.14 (×0.769) | 7.04 (×0.757) | 7.16 (×0.769) |
| int4 배치 4 | 13.97 ms | 10.52 (×0.752) | 7.94 (×0.569) | 7.99 (×0.574) | 7.97 (×0.571) |
| int4 배치 8 | 23.03 ms | 13.36 (×0.580) | 9.25 (×0.402) | 9.39 (×0.407) | 9.07 (×0.394) |

- int4 배치 8 커널 분해: 사실 없음 = 어텐션 8.68 + KV 복사·마스크 6.50 + 행렬곱 7.03 ms. 범용 컴파일러 = 1.89 + 0.09 + 6.59 ms.
- 출처: `phase0/week4/results/e4_fact_ablation.json`, `phase0/week4/WEEK4_NOTES.md` 6절

**(라) 같은 결과의 3주차 교차 측정(재부팅 전)**: int4 배치 8 sdpa 25.11 ms → 손 커널 10.51 ms(×0.419), FlexAttention 10.42 ms(×0.415). bf16 배치 1은 21.54 → 20.25 ms(×0.937). 출처: `phase0/week3/results/ab_attention_v2.json`

**(마) 찾는 수고 자체의 크기 (4주차 E1, E3, E3b)**
- 콜드 컴파일 40.1 s. Dynamo 추적과 AOT 메타데이터 수집이 합쳐 약 8 s(약 20%), 나머지는 Inductor 코드 생성이다. 가드 6,432개를 검사할 때와 건너뛸 때의 호출 시간 22.42 / 22.75 ms(차이 없음). 출처: `phase0/week4/results/e1_compile_anatomy.json`
- 디코드 블록 정보: 선언 산술 12 µs, 컴파일된 재발견 31 µs, eager 재발견 80~86 µs(그래프 안 GPU 시간). 스텝 전체로는 그래프 안 재발견이 선언보다 0.2~1.5% 느리다. 같은 재발견을 호스트에서 매 스텝 하면 int4 배치 1이 11.24 ms로 기본(9.14 ms)보다도 느려진다. 출처: `phase0/week4/results/e3_rediscovery.json`
- 긴 문맥: 128K에서 컴파일된 재발견 60.07 ms 대 선언 1.81 ms. 36층 어텐션 대비 4.9% 대 0.1%. 32K까지는 재발견도 1~2%. 출처: `phase0/week4/results/e3b_mask_scaling_v2.json`

**(바) 추측의 비용 (4주차 E1, E2)**
- 배치 1로 컴파일한 뒤 배치 4가 오면 재컴파일 61.5 s(가드 실패 이유: 배치 차원 "expected 1, actual 4").
- 배치 1·4·8·2 순서로 들어올 때 서비스 중 멈춤: 자동 추측 111 s, `mark_dynamic` 116 s(배치 1에서 표시했는데도 배치 4에서 다시 컴파일), `dynamic=True` 171 s, `mark_unbacked` 70.3 s(한 번만), 배치 크기 집합 선언 후 사전 컴파일은 서비스 전 199.8 s·서비스 중 0 s.
- `mark_unbacked` 한 그래프의 정상 상태는 배치 1에서 22.65 ms로 자동 추측(21.78 ms)보다 약 4% 느렸다. `shape_id`까지 선언한 경우는 23.95 ms(약 10%).
- 출처: `phase0/week4/results/e2_*.json`, `phase0/week4/WEEK4_NOTES.md` 2~3절

### 5.2 주류 엔진에서도 성립하는가

| 로컬 측정 | 주류 엔진에서의 대응(1·2절 근거) | 판정 |
|---|---|---|
| (가)~(다) 마스크·KV 복사로 1.3~2.5배 느림 | vLLM·SGLang·FlashInfer·TRT-LLM은 유효 길이·헤드 공유를 정수·구조로 직접 넘긴다(2.3절). 로컬 표의 "손 커널"·"상한" 열이 이들이 이미 하는 일에 해당한다 | **주류 엔진에서는 성립하지 않는다.** 이 격차는 transformers 기본 컴파일 경로와 주류 엔진 사이의 격차다. 이 카드에서 vLLM을 직접 잰 것은 아니다(확인 못 함) |
| 이 손실은 컴파일과 정적 캐시를 켤 때 생긴다 | transformers는 정적 캐시에서 마스크를 버릴 수 없고, 추적 중에는 재발견을 끈다(2.2절). 1주차 eager(DynamicCache)는 배치 1~8 모두 `flash_fwd_splitkv`를 썼고, 컴파일(StaticCache)만 `fmha_cutlassF`로 바뀌었다(`phase0/bench_llm_decode.py` 머리 주석, `phase0/RESULTS.md`) | 범용 프레임워크에서 **빠르게 만들려고 컴파일을 켜는 순간** 생기는 손실로 한정된다 |
| (마) 찾는 수고는 작다 | vLLM은 어텐션을 불투명 op으로 감싸 재발견할 일이 없다. 가드는 버린다(2.3절) | 주류에서도 탐색 비용은 문제가 아니다. **로컬 결론과 같은 방향** |
| (바) 추측이 틀리면 1~2분 멈춤 | vLLM은 기호 shape를 입력 id·위치로 한정하고 CUDA Graph 크기를 미리 정한다. SGLang도 캡처 최대 배치를 정한다(2.3절). 로컬 "크기 집합 선언" 방식과 같다 | 주류 서빙에서는 이미 선언으로 해결. 비용은 시작 시간으로 옮겨졌다(로컬 199.8 s, vLLM 자동 튜닝은 "seconds to minutes"라 기본 끔) |
| FlexAttention(의미를 함수로 전달)이 손 커널과 2% 안 | PyTorch 블로그: A100에서 FA2의 90%(순전파)·85%(역전파) | 범용 생성기가 손 커널에 근접한다는 점은 **양쪽이 일치**. 격차 크기는 형상·하드웨어에 따라 0~15% |
| `mark_unbacked`가 약 4~10% 느림 | vLLM 문서: unbacked는 "missed optimization opportunities". PyTorch 문서: 특수화로 2~8배 | "선언하면 손실이 사라진다"에 대한 **공통된 제한**. 값을 모른다고 선언하면 특수화 이득을 잃는다 |

## 6. 판정

### (a) 주류 모델·주류 엔진에서 성능 쪽 주장이 성립하는가 → 대체로 성립하지 않는다 (확신: 중상)

근거:
1. 주류 엔진과 커널 라이브러리는 유효 길이·헤드 공유·창·softcap을 정수와 구조로 커널에 직접 넘긴다. FlashAttention에는 마스크 인자 자체가 없다(2.1, 2.3절).
2. vLLM은 어텐션을 커스텀 op으로 감싸 컴파일러가 들여다보지 않게 한다. 재발견할 일이 설계에서 없다(2.3절).
3. 추측도 선언으로 대체되어 있다. 기호 shape는 입력 id·위치뿐이고, CUDA Graph 크기는 미리 정하고, 가드는 버린다(vLLM). SGLang도 캡처 최대 배치를 정한다(2.3절).
4. 로컬 1.3~2.5배 격차는 transformers 기본 컴파일 경로와 "손 커널·상한" 사이의 격차다. 주류 엔진은 이미 후자 쪽에 있다(5.2절).
5. 찾는 수고 자체는 로컬에서도 작았다. 가드 6,432개 검사는 차이가 없었고, 그래프 안 재발견은 0.2~1.5%였다(5.1절 마).

같은 무게로 남는 반대 근거:
- 엔진마다 사람이 이 배관을 다시 짠다. 성능이 아니라 공학 비용이다.
- 비용이 시작 시간으로 옮겨 갔다. 로컬에서 크기 집합 사전 컴파일은 199.8 s였다. vLLM은 자동 튜닝이 "from seconds to minutes"라 기본으로 끈다.
- vLLM이 가드를 버리는 선택은 성능 문제가 아니라 안전성 위험이다(문서도 가드가 "could be material"이라 적음).
- 주류 엔진에서도 변형 조합이 가장 빠른 커널의 지원 범위를 벗어나면 조용히 느린 커널로 물러난다(vLLM: Blackwell FA4 → FA2). 다만 원인은 의미 미전달이 아니라 커널 지원 범위이고, 크기는 확인 못 했다(2.3절).
- MoE 라우팅, 추측 디코딩, 분산 통신 경로에서 같은 종류의 손실이 있는지는 보지 않았다.

확신 정도: 어텐션과 배치 크기에 대해서는 높다(엔진 소스와 공식 문서로 확인). 주류 스택 전체로 넓히면 중간이다. 이 카드에서 vLLM·SGLang을 같은 조건으로 재지는 않았다.

### (b) 새 변형, 범용 프레임워크, 로컬 도구에서 성립하는가 → 성립한다. 다만 원인과 해법이 이론의 그림과 다르다 (확신: 중)

성립하는 근거:
- **범용 프레임워크:** transformers가 권하는 빠른 경로(정적 캐시 + 컴파일)에서 디코드가 1.3~2.5배 느렸다(로컬 E4). 원인은 소스에서 확인된다. 마스크가 있으면 GQA 사실을 버리고 복사하고, 추적 중에는 마스크를 버릴 수 있는지 재발견하는 기능도 끈다(2.2절). 배치 추측이 틀리면 66~106 s 멈췄고, `mark_dynamic` 선언은 크기 1에서 조용히 무시됐다(로컬 E2).
- **새 변형:** vLLM은 DeepSeek MLA를 약 9개월 동안 최적화 전 경로로 돌렸고, 최적화 뒤 생성 처리량이 약 3배가 됐다. llama.cpp의 Gemma 2는 58일 동안 FlashAttention 경로를 쓸 수 없었고, FA가 들어온 뒤 FA 대 일반 어텐션의 프롬프트 처리 속도비는 RTX 4090에서 최대 1.88배(마이크로배치 512에서 1.50배)였다. AMD RX 6800에서는 반대로 FA가 느린 구간이 많았다. vLLM AWQ는 첫 지원부터 Marlin 경로까지 약 10개월 동안, 문서가 비양자화 모델보다 처리량이 낮다고 경고한 경로였다. Gemma 2는 vLLM 첫 지원에서 softcap을 빼고 길이를 4K로 잘랐다(3절).
- **로컬 도구:** diffusers QwenImage의 불필요한 마스크 제거로 11%(사용자 1명 보고). ComfyUI에서 가중치 키 문자열 가드로 재컴파일 한도에 닿는 보고가 있다. 크기는 확인 못 함(4절).

이론의 그림과 다른 점:
- 새 변형 공백의 주원인은 **커널 부재**다. softcap 값, MLA 구조, AWQ 형식은 설정과 코드에 있었다. 뜻이 전달되지 않은 것이 아니라, 그 뜻을 받을 빠른 커널이 없었다.
- 선언이 공백을 줄이려면 **범용 생성기**가 있어야 한다. 그런 생성기는 이미 있다. FlexAttention(softcap을 몇 줄로 표현), FlashInfer JIT 템플릿, Ladder(사용자 정의 자료형을 1급 타입으로, vLLM 대비 W4A16 평균 2.3배)다(3.3절).
- 범용 프레임워크의 손실은 **기존 선택적 API로 이미 없앨 수 있다.** 로컬 E4가 보인 것도, 기존 API(FlexAttention, `enable_gqa`)로 사실을 넘기면 상한과 2% 안에 든다는 것이다.
- 이미지 생성의 주류 추론 경로에서는 크기가 작다. SDXL에서 의미·동기화 관련 손질의 몫은 0.02 s였고(2.54 → 2.52 s), 파이프라인은 배치 1 호출로 마스크를 피한다(4절).

### (c) 성능이 이론의 주된 이점이 될 수 있는가 → 아니다. 부차적 이점이다 (확신: 중상)

이유:
1. 손실이 큰 곳(주류 엔진의 어텐션·배치 크기)은 이미 손으로 해결됐다. 성능을 앞세우면 "vLLM은 이미 그렇게 한다"는 반론에 답할 수 없다.
2. 남은 손실(범용 프레임워크)은 새 이론 없이 기존 선택적 API로 없어진다. 이론의 몫은 그 경로를 **기본값이자 필수로** 만드는 것이지, 새로운 속도를 만드는 것이 아니다.
3. 새 변형의 공백은 커널 작성 시간이 지배한다. 선언은 범용 생성기를 거칠 때만 도움이 되고, 그 생성기는 이미 있으며 손 커널의 85~100% 수준이다(FlexAttention 블로그 85~90%, 로컬 2% 안). MLA의 가중치 흡수 같은 대수적 재구성은 선언만으로 나오지 않는다.
4. 선언이 성능을 잃게 하는 방향도 있다. 값을 모른다고 선언하면 특수화 이득을 잃는다. 로컬에서 4~10%, PyTorch 문서는 특수화로 2~8배, vLLM 문서는 "missed optimization opportunities"라고 적는다(1.6, 5.1절).
5. 문헌은 원칙을 지지하지만 크기는 재지 않았다(MLIR 평가는 채택 사례). 별칭 정보처럼 확립된 사례도 작은 예제에서 2~3배, 큰 프로그램 평균에서는 약 1%다(1.5절).

성능이 부차적 이점으로 남는 자리(데이터가 있는 것만):
- **전문가 경로가 기본값이 된다:** 범용 프레임워크에서 1.3~2.5배(로컬), diffusers 11%(보고 1건).
- **추측 없는 컴파일:** 서비스 중 멈춤 66~106 s가 0이 된다(로컬). 다만 비용은 시작 시간으로 옮겨 간다.
- **새 변형이 빠른 경로에 닿는 시간:** 선언을 범용 생성기로 내리면 공백을 줄일 가능성이 있다. 이 이론이 그 시간을 실제로 줄였다는 측정은 없다(미측정).

주된 이점으로는 안정성 쪽이 남는다. 이 조사의 범위 밖이지만, 3절의 Gemma 2 사례(softcap 제거, 4K 절단, 8K 강제 시 반복 출력)는 새 변형의 공백이 느린 경로만큼 틀린 경로도 만든다는 것을 보여 준다.

### 6.1 이 조사에서 확인 못 한 것

- 이 카드에서 vLLM·SGLang을 로컬 E4와 같은 조건으로 잰 수치
- vLLM AWQ 대 FP16 처리량 배수(PR #6612 본문 그림에만 있음), diffusers PR #12870의 마스크 대 비마스크 수치(그림에만 있음)
- ComfyUI 재컴파일 한 번의 초 단위 크기와 한도 도달 뒤 eager 비율
- MoE 라우팅, 추측 디코딩, 분산 통신 경로의 의미 손실
- TensorRT-LLM의 새 변형 지원 날짜
- MLIR 논문 PDF는 본문을 직접 추출해 읽었고, 그 밖의 논문 인용도 추출한 본문에서 확인했다. 그림 속 수치는 읽지 않았다.
