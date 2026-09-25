# entail 공개 근거와 홍보 문안 (M10.5 초안)

- 작성: 2026-09-24. `ROADMAP.md` M10.5. 연구자 지시: 거짓 없이, 써야 할 이유와 와우 포인트를 세우고, 과장의 선을 정한다.
- 모든 문장에 근거 파일과 근거의 종류를 단다.
  - **실측:** 엔진을 실제로 돌렸다.
  - **코드:** 설치된 엔진 코드를 실행했다.
  - **문헌:** 이슈나 사후 분석에 적혀 있다.
  - **판단:** 사람이 읽고 정했다.
- 측정의 정의는 `testbed/M10_PROTOCOL.md`에 있고, 결과는 `testbed/results/m10/`에 있다.
- **이 문서는 초안이다.** 외부 게시는 연구자 허락 뒤에 한다.

## 0. 판정

1. **정직하게 세울 수 있는 와우 포인트는 둘이다.**
   - **노출:** 흔한 한 줄 설정이 인기 LLM 셋 중 하나를 조용히 틀어 버린다. entail은 설정 없이 그것을 모두 지킨다.
   - **보이지 않음:** 평가 점수로는 안 보이는 결함도 적재 때 막는다.
2. **말하면 안 되는 것이 분명해졌다.** "AI 출력 버그를 잡는 도구"라고 하면 거짓이다. 최근 실제 출력 버그를 무작위로 뽑아 재현한 8건 가운데 entail이 잡은 것은 0건이다(E3).
   - 재현한 서로 다른 결함 7개 가운데 넷은 판단상 entail이 겨냥하는 부류다. 토크나이저 어휘, token type, 커널 타일과 양자화 블록, 낡은 캐시 식별이다.
   - 그러나 넷 모두 지금의 어휘 밖이다. 나머지 셋은 부류 밖이다.
   - 이 판단은 가설을 아는 사람의 것이라 수치로 쓰지 않는다.
3. **공개 전에 고칠 것이 있다.** 1.0은 규모를 키우면 오탐이 난다.
   - 인기 모델 30개 × 엔진 셋을 돌렸다(유효 81회). entail이 실행을 망가뜨린 경우는 0이다. 해소가 필요 없던 76회는 출력이 모두 같았다.
   - 그러나 **17회(21%, 모델 30개 중 11개)에서 경보가 났다.** 원인은 다섯 가지다(4절).
   - 정적 대조에서도 인기 설정 230개 중 92개에 설정 키 경보가 뜬다.
   - 이것을 고치기 전에는 "켜 두기만 하면 된다"를 말할 수 없다(E1 L3, E2).

## 1. 써야 할 이유 (한 문단, 과장 없음)

모델 파일에는 그 모델을 어떻게 돌려야 하는지가 적혀 있다. RoPE 밑과 확장, softcap, sliding window, 채팅 템플릿, 이미지 모델의 예측 방식이 그것이다. 추론 엔진은 이 선언을 늘 읽지는 않는다. 선언이 엔진에 닿지 못하면 출력은 경고 없이 틀리고, 모델은 "갑자기 멍청해진" 것처럼 보인다. entail은 모델 파일이 이미 적어 둔 선언을 읽는다. 엔진이 실제로 고른 것과 적재·캐시·요청 경계에서 대조해, 고칠 수 있으면 결과가 나오기 전에 고치고, 못 고치면 어디서 무엇이 어긋났는지 기록한다. 정상 실행에서는 출력이 바뀌지 않고, 비용은 적재 시간의 1% 안팎이다.

## 2. 와우 포인트 (근거 붙은 문장)

| # | 문장 | 근거와 종류 | 꼭 붙일 단서 |
|---|---|---|---|
| W1 | Hugging Face에서 가장 많이 받는 LLM 300개를 GPU 없이 대조했다. vLLM에서 실행 때 `rope_scaling`을 넘기면(긴 문맥을 켜는 방법), 해당하는 180개 중 **64개(36%)**가 경고 없이 다른 RoPE 밑으로 돈다. entail을 켜면 **180개 모두** 원래 밑을 지킨다 | `E1_SUMMARY.md` L1. 코드(vLLM 0.30 자신의 `ModelConfig`). 실측 두 모델과 맞춤 | "실행 때 `rope_scaling`을 넘길 때", "vLLM 0.30, transformers 5.17", "설정 수준에서 셌고, 끝까지 잰 것은 두 모델" |
| W2 | 그 결과를 끝까지 재면, Llama-3.2-3B는 GSM8K가 379에서 **273**으로 떨어지고(entail 켬: 376~380), Qwen3-4B-Instruct-2507은 같은 YaRN 조건에서 183 대신 **175**(entail 끔)다 | 실측. `m91/SUMMARY.md` fd-rope, `E1_SUMMARY.md` L1 end to end | "밑이 얼마나 바뀌는지에 따라 크기가 다르다(50배: −28%, 5배: −4%p)" |
| W3 | 평가 점수로는 안 보인다. Gemma 2의 softcap을 버리는 백엔드는 답 500개 중 **198개**를 바꾸지만 GSM8K는 3문제 차이(p = 0.66)다. entail은 적재 때 softcap을 지키는 백엔드로 바꾼다 | 실측. `m91/SUMMARY.md` S7 | "Gemma 2, SGLang torch_native 대 triton" |
| W4 | 설정 0줄, 비용 1% 안팎. LLM 모델 폴더는 한 줄도 적지 않는다. 적재 비용은 인기 모델 30개 × 엔진 셋에서 중앙값 0.8%(최대 3.5%)다. vLLM CUDA Graph 경로에서는 잴 수 없을 만큼 작다(0.999~1.000배). 해소가 필요 없던 실행 76회는 출력이 모두 같았다 | 실측. `E2_SUMMARY.md`, `m91/SUMMARY.md` S4·S6 | "시험한 엔진·판에서". "오탐 없음"은 1.0.1 뒤 새 표본에서 확인하기 전까지 쓰지 않는다 |
| W5 | 인기 모델 25개(230개 중)가 선언한 성질(sliding window, softcap)을 SGLang의 일부 백엔드가 버린다. entail은 그 성질을 지키는 백엔드로 옮긴다 | 코드. `E1_SUMMARY.md` L2. 능력표의 근거는 flex_attention의 창·softcap과 flashinfer의 softcap이 실측이고, flashinfer의 창은 코드 읽기다 | "그 백엔드를 고르면". 기본 백엔드가 무엇인지는 모델마다 다르다. 1.0.1에서 코드 읽기 행의 해소를 고치기 전까지는 머리말에 쓰지 않는다 |
| (W6 아님) | 이미지 쪽은 와우 포인트가 아니다. Civitai 인기 단일 파일 체크포인트 중 읽을 수 있던 것의 약 90%가 예측 방식을 선언하지 않는다(전체 기간 51/55, 최근 1년 24/27). 그러나 해가 되는 v 예측 모델은 읽은 범위에서 1개뿐이었다 | 코드(entail 1.0의 머리 읽기). `E1_SUMMARY.md` I | 로그인이 필요해 읽지 못한 파일이 많다(40/100, 61/100). "선언 공백"은 말할 수 있어도 "노출이 크다"는 말할 수 없다 |

## 3. 과장의 사다리

| 단계 | 모양 | 예 | 쓸 수 있나 |
|---|---|---|---|
| 1. 사실 | 수치 + 범위 | "180개 중 64개(vLLM 0.30, 실행 때 `rope_scaling`)" | 언제나 |
| 2. 범위 안의 일반화 | 부류를 밝힌 주장 | "모델 파일의 선언이 엔진에 닿지 않아 생기는 조용한 오답을 결과 전에 막는다" | 됨. "모든 오답"이 아니라 "이 부류"라고 말하는 한 |
| 3. 비유와 구호 | 그림을 주는 말 | "추론 스택을 위한 타입 검사기", "조용한 실패를 시끄럽게" | 됨. 비유로 읽히는 자리에서. 연구자의 비유(THEORY 1절 09-23 09:12)다 |
| 4. 체감 | 경험에 대한 주장 | "안정성이 눈에 띄게 좋아진다" | **우리 목소리로는 안 됨** (아래) |
| 선 밖 | 근거와 반대되는 주장 | "AI 버그를 잡는다", "오탐이 없다", "모든 모델과 엔진", "출력이 좋아진다" | 안 됨. E3 0/8, E2 오탐, 엔진 다섯, 정상 실행에서는 출력이 같음 |

**왜 "체감"은 안 되나**
- 정상 실행에서 entail은 출력을 바꾸지 않는다. 해소가 없는 실행 18개 모두 켬과 끔의 출력이 같았다(E2).
- 이 부류의 사고를 만나지 않은 사용자는 아무것도 느끼지 못한다. 그러니 "체감된다"는 대부분의 사용자에게 거짓이 된다.

**대신 쓸 수 있는 경험의 말** (기능에 근거함)
- "설정이 엔진에 실제로 닿았는지 더는 추측하지 않는다. 실행마다 무엇을 확인했는지 남는다."
- "모델이 '갑자기 멍청해졌다'로 보이던 사고가, 어느 선언이 어디서 끊겼는지 적힌 한 줄로 바뀐다." (위치 짚기 11/11, `m91/SUMMARY.md` S8)
- 실제 사용자의 후기가 모이면 그것을 인용할 수 있다. 우리가 대신 말하지 않는다.

## 4. 공개 전에 할 일 (추천 순서)

> **2026-09-25 갱신(M11):** 1번의 다섯 원인과 잡음을 고쳐 1.0.1로 공개했고(GitHub 릴리스 `v1.0.1`, PyPI `entail-ai` 1.0.1; 연구자 지시 "공개해"), 2번의 재측정을 했다. 수치는 `testbed/results/m11/SUMMARY.md`에서 읽은 것이다.
> - 같은 인기 모델 30개 × 엔진 셋(유효 81회): 틀린 경보 **17회 → 0회**, `broken`/`refused` 결정 54 → 0, `resolved` 5 → 2(둘 다 gemma-2-2b-it의 softcap, 실측 행 근거). 망가뜨린 실행 0. 적재 비중 중앙값 0.8% → 0.7%, 최대 3.5% → 5.9%(한 실행에서 적재 경계가 한 번 75 ms 걸린 것으로, 같은 경계가 다른 두 번의 실행에서는 8 ms였다).
> - 같은 규칙으로 고른 새 표본 8개(E1 순서의 21~30번째; 유효 21회): 틀린 경보 0, `broken`/`refused` 0, 출력 동일 21/21.
> - 정적 대조(인기 설정 230개): 설정 키 `broken` 엔진마다 92 → 0(92건은 `unknown` 한 줄). 코드 읽기 행뿐인 백엔드 해소(SGLang 54회 중 27회)는 `unknown`이 됐다.
> - 시험 문제 31건: 다시 돌린 29건의 판정이 M9.1과 같다(2건은 다른 단계가 필요해 다시 돌리지 않음). E3의 DSPARK 사례는 `broken` 5줄이 `unknown` 2줄(추측 복호는 검사 못 함, 안 읽힌 설정 키)이 됐다.
> - 그래서 W4는 "인기 모델 38개 × 엔진 셋 102회에서 틀린 경보 0(시험한 엔진·판에서)"로 쓸 수 있고, W5의 flashinfer 창 행은 여전히 코드 읽기라 "어긋남을 알리되 바꾸지는 않는다"로 쓴다.

1. **1.0.1: 규모에서 드러난 오탐 다섯 가지 고치기** (`E1_SUMMARY.md` L3, `E2_SUMMARY.md`, `E3_SUMMARY.md`) — 2026-09-25 고침, 위 갱신 참조
   - **설정 키:** 어휘 밖의 읽히지 않는 키는 `unknown`으로 알린다. 어휘 키와 가까운 오타만 `broken`으로 둔다. 가늠으로는 인기 설정 경보가 92개에서 0개로 줄고 rb-15는 그대로 잡는다(`coverage_rule_whatif_vocab.json`). 규칙을 본 뒤라 새 표본에서 확인해야 한다.
   - **tie:** tie를 선언하면서 `lm_head`를 따로 담은 체크포인트(Qwen3-0.6B·1.7B, FP8·AWQ판)에서는 vLLM이 묶음을 푼다. 이때는 두 텐서를 데이터로 대조해 같으면 통과시킨다.
   - **채팅 템플릿:** 빈 줄만 다른 두 선언(`tokenizer_config.json`과 `chat_template.jinja`, Nemotron-H)을 원문 해시로 비교하지 않는다. transformers와 같은 순서로 선언 파일을 고르고, 둘이 다르면 렌더 결과로 비교한다.
   - **실측 행 없는 해소:** 능력표에서 코드 읽기로만 받쳐진 행(SGLang flashinfer의 sliding window)으로는 해소하지 않는다. 창이 문맥보다 크면 해소할 까닭도 없다(Phi-3.5).
   - **추측 복호의 KV 계약:** SGLang이 초안 토큰을 위해 미리 잡는 칸을 셈에 넣는다(DSPARK에서 33칸 예약, 17칸 기록).
   - **잡음:** 경계마다 `unknown` 줄을 묶는다(한 모델 적재에 42줄). 선언 없는 이미지 체크포인트(인기 파일의 약 90%)의 `unknown`도 한 줄 요약으로 모은다.
   - **diffusers 파이프라인:** 허브 ID로 불러도 캐시 폴더에서 선언을 읽는다.
2. **새 표본으로 E2를 다시** 재서, "인기 모델 N개에서 오탐 0"을 말할 수 있게 한다.
3. **상류 보고:** RoPE 덮어쓰기 초안이 있다(`issue_track/rope_override/UPSTREAM_DRAFT.md`). 게시는 연구자 허락이 필요하다.
4. **공개 글:** README 머리를 W1~W4로 바꾸고, 짧은 글을 쓰고, 누구나 자기 모델의 노출을 볼 수 있는 한 줄(`entail preflight`)을 둔다. 오탐을 고친 뒤에 한다.

## 5. 영어 문안 초안 (1.0.1 뒤에)

README hero:

> **Your model files say how they must be run. Your engine doesn't always listen.**
> RoPE base and scaling, soft-capping, sliding windows, chat templates, prediction types — when one of these
> declarations doesn't reach the engine, the output is wrong without a warning. entail reads what the files already
> declare, checks it where it is used, repairs it before the first token when it can, and logs exactly what broke
> when it can't. Zero config. About 1% of load time.

Short post:

> We checked the 300 most-downloaded LLMs on Hugging Face. On vLLM, passing `rope_scaling` at launch — the way you
> turn on long context — silently changes the RoPE base of 64 of the 180 it applies to. For Llama-3.2-3B that is
> GSM8K 379 → 273, with no warning. With entail on, all 180 keep their base. Some of these failures don't even show
> up in evals: dropping Gemma 2's soft-capping changed 198 of 500 answers while GSM8K moved by 3.
> entail is not a general bug finder — it guards one thing: that what a model declares is what the engine runs.

## 6. 알리는 순서와 경로 (제안, 2026-09-25)

공유용 페이지(비공개, 한국어): https://claude.ai/artifact/76uYFEutxetbhsgZYYmwLd

1. **공개 전(1.0.1):** 틀린 경보 다섯 가지를 고치고, 새 표본으로 E2를 다시 잰다. 처음 써 본 사람이 가장 많이 받는 Qwen3-0.6B에서 틀린 경보를 보면 도구를 버린다.
2. **상류 먼저:** RoPE 덮어쓰기를 vLLM·SGLang에 재현과 함께 보고한다(초안 있음, 허락 필요). 이슈에는 도구 홍보를 넣지 않는다. "상류에서 고쳐졌다"가 가장 강한 신뢰 근거가 된다.
3. **기준 문서:**
   - README 머리를 W1~W4로 바꾼다.
   - 공유 페이지의 영어판을 링크한다.
   - `entail preflight`로 자기 모델의 노출을 확인하는 세 줄을 둔다.
4. **발표 글 한 편**(영어, 한국어판): 문제 → 300개 모델 대조 → 끝까지 잰 점수 → entail이 하는 일 → 한계(E3 0/8, 오탐과 고친 이력) → 써 보기.
5. **확산 채널**
   - Hacker News(Show HN)
   - Reddit r/LocalLLaMA, r/MachineLearning
   - X·LinkedIn(180칸 격자 그림 한 장)
   - 상류 이슈가 받아들여진 뒤 vLLM·SGLang 커뮤니티
   - 한국: GeekNews, 파이토치 한국 사용자 모임, 관련 커뮤니티
6. **이어서 할 것**
   - 사용자 후기를 모은다. "체감" 문구는 그것으로만 쓴다.
   - 새 엔진 판마다 노출 표를 갱신한다.
   - 측정 방법을 워크숍 논문으로 낸다.

하지 말 것은 넷이다.
- 남의 이슈나 스레드에 도구 광고를 단다.
- 선 밖의 문구(3절)를 쓴다.
- 이미지 쪽을 앞세운다.
- 엔진을 깎아내린다. 사실만 쓴다: "이 경로에서 `rope_theta`가 빠진다".

## 7. 근거 파일

- `testbed/results/m10/E1_SUMMARY.md`: 노출
- `testbed/results/m10/E2_SUMMARY.md`: 안전성
- `testbed/results/m10/E3_SUMMARY.md`: 실제 버그
- `testbed/results/m10/E4_USE_PATTERNS.md`: 쓰임새별 묶음
- `testbed/results/m91/SUMMARY.md`, `m93/SUMMARY.md`: M9의 재측정
