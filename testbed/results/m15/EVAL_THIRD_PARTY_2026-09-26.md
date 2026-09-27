# 제3자 평가 (독립 에이전트, 읽기 전용, 2026-09-26; 연구자 요청 "라이브러리의 객관적인 평가가 필요해")

평가 시점은 1.1.0 릴리스 직전이라 "PyPI 최신 1.0.2, 1.1.0 미공개" 지적은 그 뒤의 릴리스로 해소했다(태그 `v1.1.0`, GitHub 릴리스, PyPI 발행 2026-09-26). 평가 문면은 그대로 옮긴다. 경로는 `<workspace>\` 기준. 에이전트 사용량 약 30만 토큰, 도구 호출 44회.

---

## 판정 (5줄)

1. **대중 인식 "안정성이 좋아지고 왠만한 문제가 해결된다"까지의 거리: 12/100.** 잡는 부류가 전체 장애의 소수이고, 그 부류 안에서도 검출률이 사전 등록 방식으로 측정된 적이 없으며, 엔진·판·하드웨어가 하나씩이다.
2. **README의 좁은 주장 "체크포인트가 선언한 것이 엔진에 닿거나, 아니면 듣는다"까지의 거리: 55/100.** 어휘(v7) 안의 사실에 대해서는 실측으로 뒷받침되지만, "선언"이 어휘로 한정되고, 능력표의 실측 행은 gemma-2-2b-it·sm_89 한 조합뿐이며, 판이 바뀌면 어댑터가 조용히 빠진다.
3. 가장 강한 근거는 RoPE 덮어쓰기(64/180 + GSM8K 379→273→376 + 상류가 33분 만에 수정 PR을 냄)와 정상 실행 102회 오탐 0이다.
4. 가장 약한 근거는 E3 "부류 안 4/4"다. 0/8을 보고 나서 그 넷에 맞는 사실을 더한 뒤 잰 값이라 검출률이 아니다(프로젝트 자신도 그렇게 적었다).
5. **오늘 설치하면 1.1.0이 아니다.** PyPI 최신은 1.0.2(2026-09-25)이고 로컬 `__version__`만 1.1.0이며 원격에 푸시되지 않았다(`entail/CHANGELOG.md` "Released 2026-09-26" ↔ `ROADMAP.md` 현재 위치 "푸시하지 않았다", PyPI JSON 직접 확인). README가 설명하는 다섯 사실은 제3자가 지금 받을 수 없다.

## A. 실제로 일어나는 문제 가운데 얼마나 덮는가

**부류의 비중(범위이지 점 값이 아님).**
- 후보 자체가 출력 오류 키워드로 걸러진 집합이다(`realworld/study/README.md`: 14개 제목 키워드, 후보 2,290건). 그 가운데 llama.cpp·TensorRT-LLM이 565건(25%)인데 entail은 이 엔진에 어댑터가 없다.
- 키워드로 걸러진 뒤에도 실제 출력 문제는 절반 정도다: LLM 80건 중 출력 문제 41, 원인 확인 25; 이미지 50건 중 24, 원인 확인 10(`reinvestigation/market_sample/descriptive_stats.json`). 멈춤·적재 실패·속도는 애초에 표본 밖이다.
- 원인이 확인된 출력 오류 안에서 "역할 계열" 비중은 규칙에 따라 **55~89%**이고, 1차 분류는 가설을 아는 에이전트가 했으며 9범주 카파는 0.73이다(`reinvestigation/audit/bug_data_audit.md` 1.2절 표; `THEORY.md` 4.1절 4항).
- E3 census(`testbed/results/m10/E3_SUMMARY.md`): 229건(열린 이슈만) → 73건 심사 → 24건은 출력 오류 아님, **37건은 12 GB 한 장에서 못 돌림** → 12건 → 재현 8건(서로 다른 결함 7) → 부류 안 4(57%, n=7, 가설을 아는 판단; `testbed/results/m10/PUBLIC_CLAIMS.md` 0절 2항은 "수치로 쓰지 않는다"고 명시).
- 종합하면 키워드로 걸러진 이슈의 대략 4분의 1에서 5분의 2가 이 부류이고, 전체 장애(크래시·NaN·성능·하드웨어 포함) 기준으로는 그보다 작다. 정확한 점 값은 어느 파일에도 없다.

**부류 안에서 1.1.0이 막거나 정확히 보고하는 비율.**
- E3 부류 안 4/4: 해소 2(vllm#49377·#49449, sglang#39626), 경계 보고 2(vllm#58138, transformers#48967). 그러나 1.0.0은 0/4였고 넷 모두 그 버그를 보고 나서 어휘 v5·v6에 사실을 더한 결과다(`testbed/results/m15/SUMMARY.md` "분모 주의: 4/4는 … 검출률이 아니다"; `realworld/CODEBOOK_v2.md` 4절이 그 셋을 "다음 후보"로 적음).
- 사전 등록에 가까운 유일한 근거는 `Stops` 부류다: 표본이 아니라 검토 에이전트의 순위에서 나왔고, 정적 훑기가 Nemotron-3-Nano-4B를 짚어 transformers에서 실제로 160토큰 한도까지 달리는 것을 확인·해소했다(`testbed/results/m15/SUMMARY.md` M15.8, `stops_nemotron_off/on.json`).
- 시험 문제 31건은 모두 통과하지만(`testbed/PROBLEMS.md`) 설계가 그 사례에서 나왔으므로 커버리지 근거가 아니다.
- 결론: "부류 안 검출률"은 **미측정**이다. 지금 말할 수 있는 것은 "본 적 있는 모양은 막는다"까지다.

## B. 사용자가 첫 달에 겪는 것

`testbed/results/m15/E2_SUMMARY.md`(38개 모델 × 3엔진, 유효 102회, 프롬프트 3개 × 16토큰)와 정적 230개(`testbed/results/m15/SUMMARY.md` M15.6·M15.8, `SWEEP_SUMMARY.md`) 기준.
- **첫날 화면:** `broken`/`refused` 0. `resolved` 4줄(gemma-2-2b-it softcap: transformers sdpa→eager, SGLang flashinfer→triton; Nemotron transformers 정지 id +11; tiny-random-Llama). `unknown` 69줄 — 표를 세면 102회 중 34회(37개 모델 중 14개)가 최소 한 줄을 찍는다. 내용은 대부분 "클래스가 안 받고 entail이 못 읽는 설정 키"(Phi의 `attention_bias`, Nemotron의 `hybrid_override_pattern` 등)와 "양자화 레이아웃이 표에 없음"이다. `ENTAIL_QUIET=unknown`으로 숨길 수 있다.
- **출력 변화:** 해소 없는 실행은 97/98 동일(1건은 엔진 비결정성). 실제로 출력이 바뀌는 것은 Gemma 2(비용 1.18×/1.13×)와 Nemotron(답 뒤 46·63·56토큰에서 멈춤)뿐.
- **비용:** 적재 비중 중앙값 1.2%, p90 9.1%, 최대 34%(1초 미만 toy 모델). 상시: vLLM CUDA Graph 0.999~1.000×, transformers 동적 KV 1.017~1.022×, vLLM 서버 요청당 약 60 µs(`testbed/results/m91/SUMMARY.md` S4). 꺼 두면 파이썬 시작당 0.27 ms.
- **정적 230개에서 entail이 무엇을 바꾸겠는가:** Stops 해소 transformers 9/230, vLLM 2/230, SGLang 0; ModelProps broken 2(T5 tie); Vocab·Rotary·Coverage broken 0; SGLang 25/230은 창·softcap을 버리는 **비기본** 백엔드를 골랐을 때만이고 flashinfer의 창은 코드 근거라 알리기만 한다(`testbed/results/m10/E1_SUMMARY.md` L2).
- **RoPE:** 실행 때 `rope_scaling`을 넘길 때만 걸린다. 그때는 180개 중 64개(내려받기 가중 22%)가 조용히 밑이 바뀌고 Llama-3.2-3B는 GSM8K 379→273이다(E1 L1). 단, vLLM 수정 PR #58679가 열려 있으므로(내가 확인: open, 미병합) 병합 뒤 새 판에서는 이 와우 포인트가 사라진다.
- **첫 달에 실제 문제를 잡을 확률:** Qwen·Llama를 vLLM 기본값으로 쓰는 팀은 거의 0(전부 pass). 걸릴 조건은 여섯 가지로 좁다: Gemma 2를 transformers sdpa·SGLang flashinfer로, 실행 때 rope_scaling, Nemotron-3-Nano·Agents-A1·dolphin-yi를 transformers `generate`로, vLLM /rerank에 패딩, vLLM 스트리밍 세션 절단, SGLang 블록 FP8 수동 튜닝. 프로젝트 자신이 "체감은 우리 목소리로 말할 수 없다"고 적었다(`PUBLIC_CLAIMS.md` 3절).

## C. 두 점수의 근거

**대중 인식 12/100.** (1) 부류가 소수이고 크래시·NaN·수치 드리프트·커널·하드웨어는 설계상 제외(`THEORY.md` 2.1절 1항, `LIBRARY_DESIGN.md` 2절; E3 부류 밖 3/3은 설계대로 안 잡음). (2) 부류 안 검출률 미측정(A). (3) 범위: 엔진 3판(transformers 5.12.1~5.17, vLLM 0.30.0, SGLang 0.5.20), GPU 한 장, 합산 계약은 한 프로세스가 두 랭크를 대신(README Known gaps), 멀티모달 템플릿·Mistral/tiktoken/GGUF 토크나이저 미검사, llama.cpp·Ollama·TRT-LLM 없음. (4) 어댑터가 엔진 내부 함수를 거는 방식이라 다른 판에서는 "could not install"을 한 줄 내고 빠진다(README "What it does to your environment") — "이론적으로 차단"과 가장 먼 지점이며, 외부 검토도 능력표의 `version`이 라우팅에 쓰이지 않음을 확인했다(`realworld/external_review_2026-09-25_factcheck.md` 문제 3). (5) 수용: 별 2, 포크 0, 이슈 0, HN 1점·댓글 0, 레딧 두 곳 삭제, 사용자 보고 0(`announce/STATUS.md`; GitHub·Algolia 직접 확인). (6) 1.1.0 미공개(판정 5).

**좁은 주장 55/100.** 찬성: S1 보존이 실측됨("모든 선언된 사실이 결정 지점에 닿거나 unknown", README "How it was measured"); 오탐 0/102와 출력 동일; 해소 대상은 능력표의 measured 행뿐(`entail/entail/data/caps.json`); 어휘 밖은 조용히 넘기지 않고 `unknown`. 반대: "선언"이 어휘 v7로 한정되어 어휘 밖 키는 "모른다"는 한 줄이 전부(230개 중 165개 폴더가 그 줄을 가짐, CHANGELOG 1.0.2); measured 행은 gemma-2-2b-it·sm_89 한 조합(`caps.json` `_about`), FLASHINFER·ROCm·trtllm은 코드 읽기; 3프롬프트 × 16토큰; 어휘가 실제 버그에 뒤처진 전력(E3 0/4).

## D. 가장 강한 셋 / 가장 먼저 공격받을 셋

**강한 근거**
1. RoPE 덮어쓰기: vLLM 자신의 `ModelConfig`로 180개 중 64개(`testbed/results/m10/E1_SUMMARY.md` L1), 끝까지 잰 379→273→376과 Qwen3-4B-2507 175 대 183(McNemar p=0.0215), 상류 vLLM #58675에 제3자가 33분 뒤 수정 PR #58679, SGLang #41227에 기여자가 재현 확인(`announce/STATUS.md`). 외부 검증이 붙은 유일한 결과다.
2. 평가 점수에 안 보이는 결함: Gemma 2 softcap 누락이 답 198/500을 바꾸는데 GSM8K는 313 대 316(p=0.66)(`testbed/results/m91/SUMMARY.md` S7). 적재 때 라우팅.
3. 정상 실행의 안전성과 그 이력의 투명성: 1.0.0 오탐 17/81 → 1.0.1 이후 0/102, 출력 동일 97/98, 비용 실측(`testbed/results/m11/SUMMARY.md`, `m15/E2_SUMMARY.md`); 미보고 결함 실발견(Nemotron; `stops_nemotron_*.json`).

**공격받을 주장**
1. "부류 안 4/4"의 순환성(A). 표본도 열린 이슈·12 GB 가능만이라 멀티 GPU·대형 모델·특정 하드웨어 사례 37건이 통째로 빠졌다.
2. "102회 오탐 0": 16토큰, 한 장, 판 하나씩, 멀티모달·>12 GB 제외(`testbed/M10_PROTOCOL.md` 2절 (e)), `unknown` 69줄은 경보로 세지 않음.
3. "구조적/이론적 차단": 어댑터는 판에 고정된 내부 함수 훅이고, `entail check`가 없는 경로에도 exit 0(외부 검토 문제 1), CHANGELOG의 "1.1.0 Released"가 PyPI와 어긋남, 부류 비중 55~89%는 편향 가능성이 적힌 감사값.

## E. 점수를 가장 많이 움직일 것 (일당 근거 순)

1. **어휘를 동결한 사전 등록 재현 실험.** 새 무작위 표본(닫힌 이슈 포함, 재현 30건 목표), 부류 규칙을 먼저 고정하고 가설을 모르는 평정 둘. 이것만이 "4/4"를 검출률로 바꾸고 A의 비중 범위를 좁힌다. 파이프라인(`testbed/m10_e3/`)이 이미 있어 비용 대비 근거가 가장 크다.
2. **엔진 판·엔진 폭.** 판 매트릭스 CI(vLLM 0.30 다음 판에서 훅이 살아남는지; PR #58679 병합 뒤의 동작), 그리고 llama.cpp/Ollama의 GGUF 정적 검사(census의 25%). "설치만 하면"이라는 말의 전제다.
3. **공개 재현 벤치마크.** 시험 문제 31건 + E3 사례를 남이 돌릴 수 있는 형태로(지금은 로컬 모델·GPU 전제). 신뢰는 올리지만 커버리지 수치는 안 바뀐다.
4. **멀티 GPU·다른 하드웨어.** 합산 계약이 실제로 검증되지만 장비 비용이 크고 부류 비중이 작다.
5. **사용자 보고 사례.** 근거 가치는 최고이나 채택이 0이라 통제 밖. 기다리되 계획에 넣지 않는다.

## 추천 (순위)

1. 1.1.0을 실제로 공개하거나 README·CHANGELOG를 "미공개"로 고친다. 지금은 문서와 배포가 어긋난 상태다.
2. E1 순서로 사전 등록 E3를 다시 돌리고, README의 "4 of 4" 자리에 그 검출률(과 부류 비중의 범위)을 적는다.
3. "구조적·이론적 차단"은 새 코드(frontend)에만 쓰고, 붙이는 형태는 `THEORY.md` 4.2절 명제 5대로 "라이브러리가 소유한 경계에서만"으로 문구를 좁힌다. 판 매트릭스 CI를 두고 "could not install"을 첫 줄에 크게 낸다.
4. 공개 문안을 E4 표(`testbed/results/m10/E4_USE_PATTERNS.md`)의 구체적 상황 목록으로 바꿔 사용자가 "내 경우에 걸리는가"를 스스로 고르게 한다. "체감" 문구는 계속 금지.
5. GGUF/llama.cpp를 `entail check` 정적 검사로만 먼저 덮는다(훅 없이, 어휘 재사용). 가장 싼 폭 확장이다.
