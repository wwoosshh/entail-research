# AI 모델 실행 소프트웨어로 인한 품질 저하 사례 조사 (2024~2026)

- 작성일: 2026-09-23
- 짝 파일: `market_incidents.json` (사례별 객체 28개)
- 모은 기준: 같은 모델인데 실행 소프트웨어나 설정 때문에 결과가 나빠진 경우만 모았다. 실행 소프트웨어에는 추론 엔진, 서빙 업체, 로컬 실행 도구, 에이전트 하네스, 이미지 생성 도구가 들어간다. 모델 자체의 실력 문제는 뺐다.
- 인용: 원문 그대로 두고 15단어 이하로 잘랐다. 원인이 적혀 있지 않으면 '원인 미기재'로 두었다.

---

## 1. 요약

### 사례 수

- 모두 28건이다. LLM 20건(L01~L20), 이미지 생성 8건(I01~I08).
- 연도별(시작일 또는 첫 보고일 기준)로는 2024년 10건, 2025년 13건, 2026년 5건이다.
- 셈 규칙은 출처가 밝힌 원인 하나에 1건이다.
  - L08~L10은 Anthropic 사후 분석 한 편(2025-09-17)에서, L14~L16은 다른 한 편(2026-04-23)에서 나왔다.
  - I01은 같은 증상이 SD3, Flux, Z-Image에서 되풀이된 것을 1건으로 묶었다.
- 출처 종류
  - 25건은 1차 출처로만 확인했다.
  - L07과 L19는 2차 출처(Simon Willison 블로그, TechCrunch)에 기댄다. L06의 Meta 발언도 2차(VentureBeat)다.
  - X와 Reddit 원문은 접속이 막혀 열지 못했다.

### 출처가 말한 범위 안의 공통점

1. **같은 모델에서 결과가 달라진다고 출처가 직접 적었다.**
   - OpenRouter는 "the same model weights (with the same quantization) should yield the same results"라고 쓴 뒤 "differences emerge"라고 적었다.
   - Moonshot AI는 제3자 API와 공식 API 사이에서 "a stark contrast"를 봤다고 했다.
   - Artificial Analysis는 "Providers parse and format tool calls differently"라고 적었다.
2. **대부분 사용자 쪽에서 먼저 드러났다.** 발견 경로가 적힌 26건 가운데 23건이 사용자 보고, 피드백, 사용자의 교차 비교에서 시작했다고 출처에 적혀 있다.
   - 측정 기관의 벤치마크로 드러난 것은 L07이다.
   - 업체가 먼저 관찰한 것은 L10(Haiku 3.5)이다.
   - I08은 모델 제작자의 사용 경고다.
   - L01과 L19는 발견 경로가 적혀 있지 않다.
3. **업체 사후 분석 세 편 모두 처음에는 내부 평가나 내부 사용으로 재현하지 못했다고 적었다.**
   - Anthropic 2025: "The evaluations we ran simply didn't capture the degradation users were reporting"
   - Anthropic 2026: "neither our internal usage nor evals initially reproduced the issues identified"
   - OpenAI Codex: "Despite no immediate evidence from our own usage or top-line product metrics"
4. **원인이 적힌 정도**
   - 25건은 업체, 유지보수자, 병합된 수정 커밋이나 PR이 원인을 적었다. 단, L07에서 AWS의 원인은 적혀 있지 않다.
   - 3건(L18, I05, I07)은 이슈 참여자의 분석이나 추정만 있고, 유지보수자가 원인을 확정한 기록이 없다.
5. **크기 수치**
   - 같은 조건에서 전후나 기준 대비 품질 수치를 준 사례는 7건이다(L06, L07, L11, L13, L16, L17, L18).
   - 영향받은 비율만 준 사례는 2건이다(L08, L12).
   - 이미지 8건에는 품질 수치가 없다.
6. **경고 표시**
   - 경고가 없었다거나 조용히 나빠졌다고 출처가 적은 사례는 4건이다.
     - L05: 사용자 화면에 알림이 없었다.
     - L17: "silently restores"
     - I02: "No missing keys reported in the console."
     - I07: "silently corrupted"
   - 콘솔 경고가 있었다고 적은 사례는 2건이다.
     - I01: 'lora key not loaded'
     - I05: 'invalid value encountered in cast'
   - 나머지 22건은 경고가 있었는지 적혀 있지 않다.

### 출처가 말한 범위 안의 차이점

1. **출처 형태**
   - LLM 쪽에는 업체 사후 분석 4편이 있다(OpenAI 상태 페이지, Anthropic 2편, OpenAI Codex 보고서).
   - 모델 제작사의 측정 2건(Moonshot AI, MiniMax)과 제3자 측정(Artificial Analysis)도 있다.
   - 이미지 쪽 8건은 모두 GitHub 이슈, 커밋, PR, 모델 설명이다. 이미지 쪽의 업체 사후 분석이나 체계적 측정은 찾지 못했다.
2. **보고에서 수정까지 걸린 기간** (출처 날짜로 계산)
   - 4일 이내: L01, L02, L03(soft-capping), L06, L16, L19(하루 중 일부), I01, I03, I06
   - 1~3주: L04, L07(AA 갱신까지 8일), L09, L10, L15, L20, I02
   - 1개월 이상
     - L08: 수정 배포까지 30일
     - L14: 34일
     - L12: 보고서까지 46일
     - L17, L18: 73일
   - 이슈가 열려 있거나 우회책만 있는 사례: L05, L17 후속(#53912), I04(릴리스 미반영), I05, I07
3. **수정 방식**
   - 엔진이나 도구 코드 수정이 가장 많다.
   - 기본값이나 설정을 바꾼 사례: L05, L07(Azure 엔진 갱신), L14, L16
   - 모델 제작사가 권고를 내거나 API 필드를 더한 사례: L11, L13
   - 사용자에게 설정을 안내한 사례: I04(dev 브랜치), I08(CLIP skip)

### 측정 수치가 큰 사례

지표가 서로 달라 직접 비교할 수는 없다.

| 사례 | 지표 | 수치 |
|---|---|---|
| L18 vLLM v0.11.0 → v0.12.0, Qwen3-VL-2B | RefCOCO | 86% → 19% |
| L17 vLLM prefix caching + MTP, Qwen3.6 | 도구 호출 시나리오 | 약 90% → 약 50% (보고자별) |
| L11 Kimi K2 업체들 | tool_call_f1 | Nebius 50.60% (공식 평균 84%) |
| L13 MiniMax-M2 사고 기록 폐기 | Tau² | 87 → 64 |
| L07 gpt-oss-120b Azure | AIME25 | 80.0% (다수 업체 93.3%) |

---

## 2. 사례 표

'보고→수정'은 출처에 적힌 날짜로 계산했다.

| ID | 날짜 | 누가 | 사용자가 본 것 | 발견 | 보고→수정 | 크기 | 출처 |
|---|---|---|---|---|---|---|---|
| L01 | 2024-02-20 | OpenAI ChatGPT | 뜻 없는 단어 나열 | 미기재 | 약 1일 | 미기재 | 1차 |
| L02 | 2024-04-25 | llama.cpp (Llama 3 GGUF) | 쉬운 산수를 같은 방식으로 틀림 | 사용자 교차 비교 | 4일 | 집계 없음 | 1차 |
| L03 | 2024-06-28 | transformers, llama.cpp (Gemma 2) | 환각, 오탈자, 쉬운 문제 오답 | 사용자 교차 비교 | 0~2일, 9B는 16일 | eq-bench 49.16 (수정 전) | 1차 |
| L04 | 2024-06-20 | llama.cpp CUDA (Qwen2) | "GGGG...", 무작위 토큰 | 사용자 보고 | 21일 | 미기재 | 1차 |
| L05 | 2024-09-30 | Ollama | 긴 입력이 잘려 답이 이상함 | 사용자 이슈 | 이슈 열림 | 로그 예 10983→2048 토큰 | 1차 |
| L06 | 2025-04-05 | Llama 4 서비스, vLLM 등 | 서비스별 품질 차이 | 사용자 보고, 엔진 기여자 | 4일 | MMLU Pro 68.58→71.53 | 1차 + 2차 |
| L07 | 2025-08-12 | gpt-oss-120b 업체들 | 업체별 점수 차이 | 제3자 벤치마크 | 8일 (AA 갱신 기준) | AIME25 93.3 vs 80.0 | 2차 |
| L08 | 2025-08-05 | Anthropic (라우팅) | 품질 저하 | 사용자 보고 | 30일 (롤아웃 42~44일) | Sonnet 4 요청 최대 16% | 1차 |
| L09 | 2025-08-25 | Anthropic (TPU 설정) | 태국어·중국어 문자, 문법 오류 | 사용자 보고 | 8일 | 비율 미기재 | 1차 |
| L10 | 2025-08-25 | Anthropic (XLA:TPU) | 토큰 선택 오류 | 업체 관찰, 사용자 보고 | 10~18일 | 비율 미기재 | 1차 |
| L11 | 2025-09-09 | Kimi K2 업체와 엔진 | 도구 호출 오류 | 사용자 피드백, 제작사 측정 | 권고 | schema 100% vs 71.96% | 1차 |
| L12 | 2025-10-31 | OpenAI Codex | 성능 저하, 문장 중 언어 전환 | 사용자 보고 | 보고서까지 46일 | 세션 0.25% 미만 | 1차 |
| L13 | 2025-11-03 | OpenAI 호환 통합 (MiniMax-M2) | 에이전트 품질 저하 | 사용자 피드백, 제작사 측정 | 필드 추가 | Tau² 87 vs 64 | 1차 |
| L14 | 2026-03-04 | Anthropic Claude Code (추론 수준 기본값) | 응답 품질 저하 | 사용자 보고 | 34일 | 미기재 | 1차 |
| L15 | 2026-03-26 | Anthropic Claude Code (사고 기록 삭제 버그) | 잊어버림, 반복 | 사용자 보고 | 15일 | 미기재 | 1차 |
| L16 | 2026-04-16 | Anthropic Claude Code (시스템 프롬프트) | 코딩 품질 저하 | 사용자 보고, 업체 ablation | 4일 | 평가 3% 하락 | 1차 |
| L17 | 2026-05-25 | vLLM prefix caching + MTP | 정확도 급락 | 사용자 설정 조합 비교 | 73일, 후속 이슈 열림 | 약 94→75%, 약 90→50% | 1차 |
| L18 | 2025-11-27 | vLLM 0.11.1~0.12.0 (Qwen3-VL) | 좌표·박스 오류 | 사용자 버전 비교 | 73일 | RefCOCO 86→19% | 1차 |
| L19 | 2025-08-07 | OpenAI GPT-5 라우터 | GPT-5가 멍청해 보임 | 미기재 | 하루 중 일부 | 미기재 | 2차 |
| L20 | 2025-03-12 | LM Studio MLX, mlx-lm/vlm (Gemma 3) | `<pad>`만 출력 | 사용자 보고 | 6~12일 | 미기재 | 1차 |
| I01 | 2024-06-13 | ComfyUI LoRA 로더 | LoRA 효과 없음 | 사용자 보고 | 0~3일 (3회) | 미기재 | 1차 |
| I02 | 2024-08-09 | ComfyUI lowvram | LoRA 효과 약화 | 사용자 비교 | 7일 | 미기재 | 1차 |
| I03 | 2024-08-22 | ComfyUI ops.py | 두 번째 그림부터 노이즈 | 사용자 보고 | 0일 | 미기재 | 1차 |
| I04 | 2024-12-14 | A1111 v1.10.1 | v-pred 그림 과포화 | 사용자 보고 | 릴리스 미반영 | 미기재 | 1차 |
| I05 | 2025-07-28 | SageAttention + ComfyUI | 검은 출력 | 사용자 보고 | 이슈 열림 | 미기재 | 1차 |
| I06 | 2025-06-16 | diffusers Chroma | 품질 저하 | 사용자 대조 | 2일 | 미기재 | 1차 |
| I07 | 2026-03-08 | diffusers MPS 로더 | 검은 그림 | 사용자 보고 | 원인 분석 166일, 열림 | 손상 값 약 1e37 | 1차 |
| I08 | 2024-01-08 | Pony V6 XL을 쓰는 도구 설정 | 품질 낮은 덩어리 | 제작자 경고 | 해당 없음 | 미기재 | 1차 |

---

## 3. 사례별 상세

### LLM

#### L01. OpenAI ChatGPT 무의미 출력 (2024-02-20)
- **사용자가 본 것:** "produced word sequences that made no sense"
- **발견:** 원문에 적혀 있지 않다. 상태 페이지 기준 2024-02-20에 발생해 2024-02-21에 해결됐다.
- **원인 (OpenAI 상태 페이지):** "inference kernels produced incorrect results when used in certain GPU configurations". 사용자 경험 최적화가 들어가면서 버그가 생겼다고 적었다.
- **수정:** 수정을 배포하고 해결을 확인했다. 세부 내용은 없다.
- **크기:** 수치 미기재
- **출처:** https://status.openai.com/incidents/ssg8fh7sfyz3 (1차)

#### L02. llama.cpp의 Llama 3 사전 토큰화 오류 (2024-04-25)
- **사용자가 본 것:** "What is 3333+777?"을 meta.ai와 Groq는 맞혔다. llama.cpp로 돌린 Llama 3 GGUF(8B fp16, 8B Q8_0, 70B Q4_0)는 모두 틀렸다. "All of the llama.cpp instances got the problem wrong in exactly the same way."
- **발견:** Reddit 언급을 본 사용자(coder543)가 여러 실행 환경을 교차 비교해 보고했다. 이슈는 2024-04-25에 열렸고 수정 PR #6920은 2024-04-29에 병합됐다(4일).
- **원인 (PR #6920):** "the pre-tokenization splits the input string in the wrong way". 그 결과로 "This leads to poor generation quality"라고 적었다.
- **수정:** BPE 사전 토큰화 종류를 GGUF 헤더에 기록한다. 변환할 때 토큰 해시로 확인하고, 모르는 토큰화면 스크립트를 갱신하라고 요구한다. 수정 전에 만든 GGUF는 기본 사전 토큰화로 되돌아가므로 다시 변환해야 한다.
- **크기:** 집계 수치 없음. 단일 프롬프트로 재현했다.
- **출처:** https://github.com/ggml-org/llama.cpp/issues/6914 , https://github.com/ggml-org/llama.cpp/pull/6920 (1차)

#### L03. Gemma 2 soft-capping 누락: transformers, llama.cpp (2024-06-28)
- **사용자가 본 것**
  - Hugging Face 토론 제목: "Hallucinations, misspellings etc. Something seems broken?"
  - llama.cpp Q8_0 27B는 AI Studio가 맞히는 문제를 "still gets the answer wrong to even simple problems".
- **발견:** 사용자들이 Google AI Studio, NVIDIA NIM 결과와 비교했고 eq-bench로 측정했다. llama.cpp 지원 PR은 2024-06-27에 생성됐고 첫 보고는 06-28에 나왔다.
- **원인과 수정**
  - transformers PR #31698 "Gemma capping is a must for big models": 06-28 병합
  - llama.cpp PR #8197 "This PR adds the missing attention layer and final logit soft-capping.": 06-30 병합. GGUF를 다시 만들어야 한다.
  - llama.cpp PR #8227 sliding window mask: 07-01 병합
  - llama.cpp PR #8444/#8473 "Gemma 9b should use 256 and not 224": 07-14 병합
  - Hugging Face 직원: "float16 should not be used for this model"
- **크기:** 수정 전 eq-bench v2 49.16. 수정 후 수치는 없다.
- **출처:** https://huggingface.co/google/gemma-2-27b-it/discussions/10 , https://github.com/ggml-org/llama.cpp/issues/8183 , https://github.com/ggml-org/llama.cpp/pull/8197 , https://github.com/huggingface/transformers/pull/31698 , https://github.com/ggml-org/llama.cpp/pull/8444 (1차)

#### L04. llama.cpp CUDA에서 Qwen2 "GGGG" 출력 (2024-06-20)
- **사용자가 본 것:** Qwen 공식 문서에 "Previously, Qwen2 models generate nonsense like GGGG... with llama.cpp on GPUs."라고 적혀 있다. issue #8025는 Qwen2-72B가 숫자, 기호, 여러 언어 조각을 무작위로 낸다고 보고했다.
- **발견:** 사용자 보고. issue #8025는 2024-06-20에 열렸고 PR #8412는 2024-07-11(b3370)에 병합됐다(21일).
- **원인 (PR #8412 제목):** "use F32 precision in Qwen2 attention and no FA". 본문: "few reports of these models generating "GGGG" when FA is disabled"
- **수정:** Qwen2 attention을 F32로 계산한다. 그 전에는 Qwen 문서가 `-fa`와 전체 GPU 오프로드를 우회책으로 안내했다. 문서에는 "Both should be no longer necessary after b3370"이라고 적혀 있다.
- **크기:** 수치 미기재
- **출처:** https://qwen.readthedocs.io/en/v2.0/run_locally/llama.cpp.html (1차, 모델 제작사 문서), https://github.com/ggml-org/llama.cpp/pull/8412 , https://github.com/ggml-org/llama.cpp/issues/8025 (1차)

#### L05. Ollama 기본 컨텍스트 초과 입력의 조용한 잘림 (2024-09-30)
- **사용자가 본 것:** "The user can only see some weird behaviour from the LLM's answers." (issue #7043 댓글, 2025-01-07)
- **발견:** 사용자 요청 이슈가 2024-09-30에 열렸고 2026-09-23 기준으로도 열려 있다.
- **원인 (서버 로그):** `msg="truncating input prompt" limit=2048 prompt=10983 keep=5 new=2048`. 댓글에 따르면 이 경고는 `ollama serve` 로그에만 남고 `ollama run` 화면에는 전달되지 않는다.
- **수정과 변경**
  - v0.5.13(2025-02-27): `OLLAMA_CONTEXT_LENGTH` 환경 변수 추가
  - v0.6.7(2025-04-26): "Increased default context window to 4096 tokens"
  - v0.9.3(2025-06-25): 학습 길이로 컨텍스트 제한
  - v0.10.0(2025-07-18): `ollama ps`에 컨텍스트 길이 표시
  - v0.15.5(2026-02-03): VRAM에 따라 기본값 4k/32k/256k
  - 별도 이슈 #17427(2026-07-27): num_ctx를 설정해도 넘치는 입력은 num_ctx/2+2 토큰으로 잘린다는 보고였다. 유지보수자가 두 단계 잘라내기 동작이라고 설명하고 08-12에 닫았다.
- **크기:** 로그 예시에서 10983토큰이 2048토큰으로 줄었다.
- **출처:** https://github.com/ollama/ollama/issues/7043 , https://github.com/ollama/ollama/releases/tag/v0.6.7 , https://docs.ollama.com/context-length , https://github.com/ollama/ollama/issues/17427 (1차)

#### L06. Llama 4 공개 직후 엔진 구현 오류 (2025-04-05 공개)
- **사용자가 본 것:** Meta의 Ahmad Al-Dahle은 "reports of mixed quality across different services"라고 말했다(VentureBeat 인용, X 게시물 2025-04-07 UTC).
- **발견:** vLLM 기여자(luccafong)가 QK norm 문제를 찾아 MMLU Pro로 측정했다. Unsloth는 QK norm epsilon 문제를 찾았다. 공개 4일 뒤 vLLM에서 수정됐다.
- **원인**
  - vLLM PR #16311: "QKNorm should not be across head, we need to do qknorm per head"
  - Unsloth: "this means using 1e-05 and not 1e-06" (QK norm epsilon)
- **수정**
  - vLLM PR #16311: 2025-04-09 병합
  - transformers PR #37418 "use `rms_norm_eps` for the L2Norm for Llama4": 04-10 병합
  - llama.cpp PR #12889(Scout RoPE 설정 변경 대응): 04-11 병합
- **크기:** vLLM MMLU Pro 0.6858 → 0.7153 ("score goes up from 0.6858 to 0.7153")
- **출처:** https://github.com/vllm-project/vllm/pull/16311 , https://github.com/huggingface/transformers/pull/37418 , https://github.com/ggml-org/llama.cpp/pull/12889 (1차), https://unsloth.ai/docs/basics/dynamic-3.0-ggufs (1차, Unsloth 자체 설명), https://venturebeat.com/ai/meta-defends-llama-4-release-against-reports-of-mixed-quality-blames-bugs (2차)

#### L07. gpt-oss-120b 업체별 점수 차이 (2025-08-12)
- **사용자가 본 것:** 같은 gpt-oss-120b(reasoning effort high)인데 업체마다 AIME25 점수가 달랐다. Simon Willison 글 제목은 "Open weight LLMs exhibit inconsistent performance across providers"다.
- **발견:** Artificial Analysis가 업체별 정확도 벤치마크를 돌렸다(GPQA Diamond 16회, AIME25 32회, IFBench 8회).
  - AA 게시는 2025-08-12 UTC다(X 게시물 ID로 계산).
  - 모델 공개는 2025-08-05, 측정 데이터는 08-11이다(OpenRouter 블로그).
- **점수 (AIME25x32, Simon Willison 2025-08-15 글)**

  | 업체 | 점수 |
  |---|---|
  | Cerebras, Nebius Base, Fireworks, Deepinfra, Novita, Together.ai, "vLLM 0.1.0"(원문 표기) | 93.3% |
  | Parasail | 90.0% |
  | Groq | 86.7% |
  | Amazon | 83.3% |
  | Azure | 80.0% |
  | CompactifAI (고압축 모델) | 36.7% |

- **원인 (Microsoft Azure의 Lucas Pickup, Simon Willison 인용):** "Old vLLM commits that didn't respect reasoning_effort, so all requests defaulted to medium." AWS에 대해서는 "No news yet on what went wrong"이라고 적혀 있다. 원인 미기재다.
- **수정:** Azure는 모든 서빙 인스턴스를 수정했다. 2025-08-20 갱신에서 Groq와 Azure는 93.3%였고, Google Vertex가 83.3%로 새로 올라왔다.
- **출처:** https://simonwillison.net/2025/Aug/15/inconsistent-performance/ (2차). 1차 출처인 https://x.com/ArtificialAnlys/status/1955102409044398415 는 X가 막혀 열지 못했다.

#### L08. Anthropic: 짧은 요청이 1M 컨텍스트 서버로 잘못 라우팅됨 (2025-08-05)
- **사용자가 본 것:** Claude Code 사용자의 약 30%가 "had at least one message routed to the wrong server type"였다.
- **발견:** 사용자 보고가 쌓여 8월 말에 조사를 시작했다. 08-29의 부하분산 변경으로 영향이 커졌다.
- **원인:** "misrouted to servers configured for the upcoming 1M token context window". 08-29 변경에 대해서는 "a routine load balancing change unintentionally increased"라고 적었다.
- **수정:** 라우팅 로직을 고쳐 09-04에 배포했다. 롤아웃은 1P와 Vertex가 09-16, Bedrock이 09-18에 끝났다.
- **크기**
  - 처음에는 Sonnet 4 요청의 0.8%였다.
  - "At the worst impacted hour on August 31, 16% of Sonnet 4 requests were affected."
  - Bedrock은 최대 0.18%, Vertex는 0.0004% 미만이었다.
- **출처:** https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues (1차)

#### L09. Anthropic: TPU 서버 설정 오류로 출력 손상 (2025-08-25)
- **사용자가 본 것:** "producing Thai or Chinese characters in response to English prompts". 코드에는 명백한 문법 오류가 섞였다. 영어 답 중간에 "สวัสดี"가 나온 예를 들었다.
- **발견:** 사용자 보고 후 업체가 조사했다. Opus 4.1과 Opus 4는 08-25~28에, Sonnet 4는 08-25~09-02에 영향을 받았다.
- **원인:** "we deployed a misconfiguration to the Claude API TPU servers". 런타임 성능 최적화가 드물어야 할 토큰에 높은 확률을 줬다.
- **수정:** 09-02에 롤백했다. 배포 과정에 예상 밖 문자 출력 검출 테스트를 더했다.
- **크기:** 비율 미기재
- **출처:** 위와 같은 사후 분석 (1차)

#### L10. Anthropic: 근사 top-k의 XLA:TPU 컴파일 오류 (2025-08-25)
- **사용자가 본 것:** 토큰 선택이 틀려 품질이 떨어졌다. 근사 top-k가 "sometimes returned completely wrong results, but only for certain batch sizes and model configurations".
- **발견:** Anthropic이 Haiku 3.5에서 먼저 관찰해 09-04에 롤백했다. 이후 Opus 3 사용자 보고를 받아 09-12에 롤백했다. Sonnet 4는 재현하지 못했지만 함께 롤백했다.
- **원인:** "inadvertently triggered a latent bug in the XLA:TPU compiler". 혼합 정밀도 연산과, 이전 우회책이 가리고 있던 근사 top-k 문제가 겹쳤다고 적었다.
- **수정:** 근사 top-k를 정확 top-k로 바꾸고 일부 연산을 fp32로 통일했다. XLA:TPU 팀과 컴파일러 버그를 고쳤다.
- **크기:** 비율 미기재. 제3자 플랫폼은 영향이 없었다.
- **출처:** 위와 같은 사후 분석 (1차)

#### L11. Kimi K2 제3자 업체와 엔진의 도구 호출 차이 (2025-09-09 저장소 생성)
- **사용자가 본 것:** Moonshot AI는 "we have received numerous feedback on the precision of Kimi K2 in toolcall"이라고 적었다. K2 Thinking 뒤에는 벤치마크 점수가 이상하다는 피드백이 이어졌다고 했다.
- **발견:** 모델 제작사가 4,000개 요청에 대한 각 업체 응답을 공식 API와 비교했다.
  - tool_call_f1은 공식 API 대비 도구 호출을 트리거하는 일치도다.
  - schema_accuracy는 트리거된 호출 중 JSON 스키마를 통과한 비율이다.
- **원인**
  - K2VV README: "Some vendors may not meet the requirements due to using incorrect versions."
  - 이 밖에 잘못된 도구 호출 ID 형식과 guided encoding 부재를 들었다.
  - Kimi Vendor Verifier 글: "a significant portion of these cases stemmed from the misuse of Decoding parameters"
  - 이어서 "we observed a stark contrast between third-party API and official API"라고 적었다.
- **수정:** 권장 엔진 버전을 제시했다(K2-0905는 vLLM v0.11.0, SGLang v0.5.3rc0 이상). 도구 호출 ID를 `functions.func_name:idx`로 바꾸고 guided encoding을 쓰라고 권했다. 공식 API의 Thinking 모드에는 Temperature=1.0, TopP=0.95를 강제했다.
- **크기 (2025-11-15 시험)**
  - K2-0905 schema_accuracy: 공식 100%, vLLM 76.00%, SGLang 73.13%, Volc 72.86%, Baseten 72.49%, AtlasCloud 72.44%, Together 71.96%
  - K2-0905 tool_call_f1: Groq 69.52%, Nebius 50.60%. 공식 API를 여러 번 돌린 평균은 84%, 허용 기준은 80%다.
  - K2-Thinking schema_accuracy: vLLM 87.22%, Chutes 83.05%
  - K2-Thinking tool_call_f1: Novita 72.22%, Chutes 68.10% (기준 73%)
- **출처:** https://github.com/MoonshotAI/K2-Vendor-Verifier , https://www.kimi.ai/blog/kimi-vendor-verifier (1차)

#### L12. OpenAI Codex: 구형 하드웨어와 제약 샘플링 버그 (2025-10-31 보고서)
- **사용자가 본 것:** 2025-09-15 GPT-5-Codex 출시 무렵부터 성능이 떨어졌다는 공개 보고가 늘었다. "the model would switch languages mid sentence as part of the final answer". 보고서에는 한국어로 바뀐 예가 실려 있다.
- **발견:** 업체가 조사했다. 하드웨어 종류별로 평가를 돌렸고, 요청 특성(클러스터, 하드웨어, CLI 버전 등)으로 이용 유지를 예측하는 모델을 만들었다. 불만이 늘기 시작한 지 46일 뒤에 보고서가 나왔다.
- **원인**
  - 하드웨어: "slight performance issues with some of our older hardware"
  - 제약 샘플링: "A subtle bug in the implementation that led the token sequence to become out-of-distribution"
- **수정:** 해당 하드웨어를 fleet에서 뺐다. 제약 샘플링 수정은 "next couple of days"에 배포하겠다고 했고, 부하분산도 개선하기로 했다.
- **크기:** 언어 전환은 "Less than 0.25% of overall sessions". 하드웨어 차이의 크기는 적혀 있지 않다.
- **출처:** https://docs.google.com/document/d/1fDJc1e0itJdh0MXMFJtkRiBcxGEFtye6Xc6Ui7eMX4o/mobilebasic (1차, "Ghosts in the Codex Machine")

#### L13. MiniMax-M2: OpenAI 호환 통합이 이전 사고를 버림 (2025-11-03)
- **사용자가 본 것:** 다회차 에이전트 작업의 품질이 떨어졌다. MiniMax는 "interleaved thinking is sometimes not applied correctly in practice"라고 적었다.
- **발견:** 모델 제작사가 사용자 피드백을 분석하고, 이전 사고를 유지할 때와 버릴 때를 벤치마크로 비교했다.
- **원인:** "the widely-used OpenAI Chat Completion API does not support passing reasoning content back"
- **수정:** OpenAI 호환 API에 `reasoning_details` 필드를 도입했다. OpenRouter, Ollama, Droid, Vercel, Cline과 협력한다.
- **크기 (유지 vs 폐기)**

  | 벤치마크 | 유지 | 폐기 |
  |---|---|---|
  | Tau² | 87 | 64 |
  | BrowseComp | 44.0 | 31.4 |
  | GAIA | 75.7 | 67.9 |
  | xBench | 72.0 | 66.0 |
  | SWE-Bench Verified | 69.4 | 67.2 |

- **출처:** https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2 (1차)

#### L14~L16. Anthropic Claude Code 하네스 변경 3건 (2026-03~04, 사후 분석 2026-04-23)
세 건의 공통 사항은 다음과 같다.
- **사용자가 본 것:** "reports that Claude's responses have worsened for some users". 영향은 Claude Code, Claude Agent SDK, Claude Cowork에 있었고, 사후 분석은 "The API was not impacted."라고 적었다.
- **발견:** 사용자 보고로 3월 초에 조사를 시작했다. 세 변경이 서로 다른 트래픽에 다른 일정으로 적용돼 전체적으로는 들쭉날쭉한 저하로 보였다고 적었다.
- **출처:** https://www.anthropic.com/engineering/april-23-postmortem (1차)

**L14. 기본 reasoning effort 변경 (2026-03-04, Sonnet 4.6·Opus 4.6)**
- 기본값을 high에서 medium으로 바꿨다. 사후 분석은 "This was the wrong tradeoff."라고 평했다.
- 04-07에 원복했다(34일). 크기 수치는 없다.

**L15. 유휴 세션 사고 기록 삭제 버그 (2026-03-26, Sonnet 4.6·Opus 4.6)**
- 사용자는 Claude가 잊어버리고 같은 말을 되풀이한다고 느꼈다("made Claude seem forgetful and repetitive").
- 원인은 한 번만 지워야 할 이전 사고 기록이 "keep happening every turn for the rest of the session instead of just once"였다.
- 04-10에 수정했다(v2.1.101, 15일). 원인을 찾는 데 대해 "it took us over a week to discover and confirm the root cause"라고 적었다.

**L16. 장황함을 줄이는 시스템 프롬프트 (2026-04-16, Sonnet 4.6·Opus 4.6·Opus 4.7)**
- "we added a system prompt instruction to reduce verbosity"
- 결과는 "In combination with other prompt changes, it hurt coding quality"였다.
- 업체가 시스템 프롬프트를 줄 단위로 빼 보는 ablation에서 발견했다. 한 평가에서 "a 3% drop for both Opus 4.6 and 4.7"이 나왔다.
- 04-20에 원복했다(v2.1.116, 4일).

#### L17. vLLM: prefix caching과 MTP 추측 디코딩을 함께 쓸 때 정확도 저하 (2026-05-25)
- **사용자가 본 것:** 두 기능을 함께 켰을 때만 정확도가 떨어졌다. 한 사용자는 응답이 "as if it didn't see the prompt"였다고 적었다.
- **발견:** 보고자가 네 가지 설정 조합(둘 다 끔, 하나씩, 둘 다 켬)으로 같은 사내 분류 데이터셋 2,600건을 비교했다. 보고 후 PR #51113이 병합되기까지 73일이 걸렸다.
- **원인 (수정 PR #51113):** "Every request resuming from that hash silently restores a truncated state." 하이브리드 Mamba/GDN 모델에서 mamba align 모드의 prefill 청크가 블록 경계에 맞지 않았다.
- **수정:** PR #51113이 2026-08-06에 병합되고 이슈가 닫혔다. 그러나 후속 이슈 #53912(2026-08-26)는 v0.28.0에서도 손상이 있다고 보고했고 열려 있다.
- **크기**
  - 분류 정확도: 약 94% → 약 75% (보고자)
  - 도구 호출 시나리오: 약 90% → 약 50% (다른 사용자)
  - 34개 프롬프트 채점: 0.778 → 0.741
  - 재현하지 못했다는 보고도 있다(-0.67%).
- **출처:** https://github.com/vllm-project/vllm/issues/43559 , https://github.com/vllm-project/vllm/pull/51113 , https://github.com/vllm-project/vllm/issues/53912 (1차)

#### L18. vLLM 0.11.1~0.12.0: Qwen3-VL 위치 지정 붕괴 (2025-11-27)
- **사용자가 본 것:** "The model produces incorrect bounding box coordinates and poor spatial reasoning results."
- **발견:** 사용자가 vLLM 버전 사이를 비교했다(0.11.0 대비).
  - #29595(Hopper)는 2025-11-27에 열려 2026-02-08에 닫혔다(73일).
  - #36117(A100)은 2026-03-05에 열려 03-07에 닫혔다.
- **원인:** 유지보수자가 확정한 기록은 없다.
  - #29595 기여자: "the issue may be in torch 2.9.0 handling of triton 3.5.0" (추정)
  - #36228 보고자 분석: v0.12.0에서 Conv3d 구현이 바뀌었고, PyTorch 2.9.0에서 cuBLAS GEMM이 선택됐으며, 기본 `mp` 실행기와 겹쳤다.
- **수정:** 우회책은 triton 3.4.0으로 내리거나 `--enforce-eager`를 쓰는 것이다(#29595). A100 경로는 `--distributed-executor-backend ray`로 우회했다. 유지보수자(ywang96)는 "resolved since we released 0.17.0"이라고 적었다.
- **크기 (Qwen3-VL-2B, 4×A100, v0.11.0 → v0.12.0)**
  - PointBench: 57.6% → 약 13%
  - RefCOCO_g_test: 86% → 19%
  - PixmoPointsEval: 50% → 15%
- **출처:** https://github.com/vllm-project/vllm/issues/29595 , https://github.com/vllm-project/vllm/issues/36117 , https://github.com/vllm-project/vllm/issues/36228 (1차)

#### L19. OpenAI GPT-5 실시간 라우터 장애 (2025-08-07)
- **사용자가 본 것:** Sam Altman은 "the result was GPT-5 seemed way dumber"라고 말했다. GPT-5는 프롬프트마다 어떤 모델이 답할지 정하는 실시간 라우터를 쓴다.
- **발견:** 원문에 적혀 있지 않다. 내부 장애(sev)로 불렀다.
- **원인:** "the autoswitcher was out of commission for a chunk of the day"
- **수정:** 라우터의 판단 경계를 조정하고, 어느 모델이 답하는지 보여 주겠다고 했다.
- **크기:** 수치 미기재
- **출처:** https://techcrunch.com/2025/08/08/sam-altman-addresses-bumpy-gpt-5-rollout-bringing-4o-back-and-the-chart-crime/ (2차, 2025-08-08 기사)

#### L20. LM Studio MLX 런타임에서 Gemma 3가 `<pad>`만 출력 (2025-03-12)
- **사용자가 본 것:** "Using Gemma 3 on MLX only generates <pad><pad><pad> as the output."
- **발견:** 사용자 보고(댓글 60개). 보고 6일 뒤인 03-18에 mlx-vlm v0.1.18로 수정됐고 03-24에 이슈가 닫혔다.
- **원인:** Blaizzy가 인용한 설명은 "The dynamic range is too large for fp16."이다. MLX의 awni는 "there was a bug with the window attention."이라고 했다.
- **수정:** bfloat16으로 다시 변환한 모델을 올렸다. mlx-vlm v0.1.18과 mlx-lm이 수정됐고 LM Studio MLX 런타임도 갱신됐다.
- **크기:** 수치 미기재
- **출처:** https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/513 (1차)

### 이미지 생성

#### I01. ComfyUI: 다른 형식의 LoRA 키가 적용되지 않음 (2024-06-13부터 반복)
- **사용자가 본 것:** 그림은 나오지만 LoRA 효과가 없었다.
  - Z-Image(#11158): "images are generated but the effect is not seen"
  - ModelScope LoRA(#11763): "the LoRA has no effect on the output image"
- **발견:** 사용자 보고. 콘솔에 'lora key not loaded' 경고가 찍혔다.
  - SD3: 2024-06-13 보고, 같은 날 커밋 ac151ac "Support SD3 diffusers lora."
  - Flux(SimpleTuner): 2024-08-04 보고, 08-05 커밋 78e133d "Support simple diffusers Flux loras."
  - Z-Image(ModelScope): 2026-01-09 보고, 01-12 PR #11805
- **원인:** "Loras trained with Diffusers' scripts can't be loaded with stock Lora Loader" (#3701). 학습 도구마다 다른 키 형식을 로더가 모르는 문제였다.
- **수정:** 형식별 키 매핑을 추가했다. Musubi Tuner LoRA는 ComfyUI 형식으로 변환해야 한다(#11487).
- **크기:** 수치 미기재
- **출처:** https://github.com/Comfy-Org/ComfyUI/issues/3701 , https://github.com/Comfy-Org/ComfyUI/issues/4202 , https://github.com/Comfy-Org/ComfyUI/issues/11158 , https://github.com/Comfy-Org/ComfyUI/issues/11763 , https://github.com/Comfy-Org/ComfyUI/issues/11487 (1차)

#### I02. ComfyUI lowvram 모드에서 Flux LoRA가 일부만 적용됨 (2024-08-09)
- **사용자가 본 것:** "LoRA is getting loaded only partially and producing an image with diminished effect"
  - 다른 사용자: "Seems like a strength of 1 in ComfyUI is more like 0."
  - 같은 LoRA를 Forge에서 쓰면 기대대로 나왔다.
- **경고:** "No missing keys reported in the console." 다른 댓글은 "with no error / warnings output"이라고 적었다.
- **발견:** 8GB GPU 사용자가 LoRA 강도 0, 원인 커밋 전후, 다른 도구와 결과를 비교했다.
- **원인과 수정:** 원인 커밋은 08f92d5 "Partial model shift support."(2024-08-08)다. 수정 커밋은 83f3431 "Fix potential lowvram issue."(2024-08-16)로, 보고 7일 뒤다.
- **크기:** 수치 미기재
- **출처:** https://github.com/Comfy-Org/ComfyUI/issues/4282 (1차)

#### I03. ComfyUI: Flux + LoRA에서 두 번째 그림부터 노이즈 (2024-08-22)
- **사용자가 본 것:** "only the first render is good". 이후 그림은 노이즈였다.
- **발견:** 사용자 보고. 다른 사용자가 원인 커밋을 지목했다.
- **원인과 수정:** 원인 커밋은 538cb06 "Make cast_to a nop if weight is already good."(08-20)다. 수정 커밋은 c7ee4b3 "Try to fix some lora issues."(08-22)로, 보고 당일이다.
- **크기:** 수치 미기재
- **출처:** https://github.com/Comfy-Org/ComfyUI/issues/4549 (1차)

#### I04. AUTOMATIC1111 릴리스 v1.10.1의 v-prediction 미지원 (2024-12-14)
- **사용자가 본 것:** NoobAI-XL V-Pred로 만든 그림이 과포화됐다. "my pictures is just deep fried"
- **발견:** 사용자 보고. 같은 날 협업자가 "you are on v1.10.1 ... use dev branch"라고 답했다.
- **원인:** 릴리스에는 v-pred 지원이 없다.
  - dev 브랜치 PR #16567 "Support and automatically detect SDXL V-prediction models"는 2024-10-19에 병합됐다. state_dict에 `v_pred` 키가 있으면 v-pred로 본다(ComfyUI의 감지 방식을 따름).
  - 모델 카드는 "THIS MODEL WORKS DIFFERENT FROM EPS MODELS!"라고 경고한다. diffusers로 쓸 때는 `prediction_type="v_prediction"`을 손으로 지정하라고 안내한다.
- **수정:** dev 브랜치를 쓰라고 안내했다. 2026-09-23 조회 기준 dev는 v1.10.1보다 96커밋 앞서 있고 새 릴리스는 없다. dev에서도 v-pred 모델이 검은 그림을 낸다는 보고가 있다(#17067, 2025-07, 열림).
- **크기:** 수치 미기재
- **출처:** https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16721 , https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/16567 , https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/17067 , https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0 (1차)

#### I05. SageAttention + ComfyUI에서 검은 출력: Wan 2.1/2.2, Qwen-Image (2025-07-28)
- **사용자가 본 것:** 영상과 그림이 완전히 검게 나왔다. "Sage Attention with WAN FP8 model (or FP8 quantization) causes black output." (#221 제목 일부)
- **경고:** 콘솔에 RuntimeWarning "invalid value encountered in cast"가 찍혔다(댓글).
- **발견:** 2025-07부터 2026-07까지 사용자 보고가 이어졌다. ComfyUI #9077과 SageAttention #221은 열려 있다.
- **원인:** 확정 기록은 없다. SageAttention 유지보수자는 다른 모델(Open-Sora) 이슈 #93에서 "fp16 has a limited range and may encounter overflow error as the accumulator."라고 적었다.
- **수정:** 공식 수정은 없다. 사용자들이 쓴 우회책은 다음과 같다.
  - SageAttention 2.0.1로 내리기(#9184)
  - fp32 누산 커널 강제
  - `--fast` 옵션 제거
- **크기:** 수치 미기재
- **출처:** https://github.com/Comfy-Org/ComfyUI/issues/9077 , https://github.com/Comfy-Org/ComfyUI/issues/9184 , https://github.com/thu-ml/SageAttention/issues/221 , https://github.com/thu-ml/SageAttention/issues/93 (1차)

#### I06. diffusers ChromaPipeline의 attention mask 누락 (2025-06-16)
- **사용자가 본 것:** 출력 품질이 참조 구현보다 나빴다. 보고자는 "The masks are computed ... and then thrown away and not returned."라고 적었다.
- **발견:** 사용자가 참조 구현 코드와 대조했다. 파이프라인은 2025-06-14에 병합됐고 보고는 06-16에 나왔다.
- **원인 (수정 PR #11725):** "we neglected passing the modified attention mask to the transformer model, leading to quality issues"
- **수정:** PR #11725가 06-18에 병합됐다(2일). v0.34.0 릴리스(06-24) 전이라 main 브랜치 사용자만 영향을 받았다.
- **크기:** 수치 미기재
- **출처:** https://github.com/huggingface/diffusers/issues/11724 , https://github.com/huggingface/diffusers/pull/11725 (1차)

#### I07. diffusers: Apple MPS로 곧바로 올릴 때 가중치가 조용히 손상됨 (2026-03-08)
- **사용자가 본 것:** GLM-Image가 완전히 검은 그림을 냈다. 보고자는 "some model parameters become silently corrupted"라고 적었다.
- **발견:** 사용자 보고. 다른 기여자가 2026-08-21에 원인을 분석했다(보고 후 166일). 이슈는 열려 있다.
- **원인 (기여자 분석):** "a non-blocking CPU→MPS copy can read source storage that was already released". 이 분석은 PyTorch #189690을 인용했다. 기여자는 GLM에 한정되지 않는다고 적었다.
- **수정:** 우회책은 CPU에 먼저 올린 뒤 MPS로 옮기는 것이다.
- **크기:** 손상된 값이 약 1e37이었다.
- **출처:** https://github.com/huggingface/diffusers/issues/13227 (1차)

#### I08. Pony Diffusion V6 XL의 CLIP skip 요구 (2024-01-08)
- **성격:** 사건 기록이 아니라 모델 제작자의 사용 경고다.
- **사용자가 본 것 (제작자 설명):** "otherwise you will be getting low quality blobs"
- **원인 (필요한 설정):** "Make sure you load this model with clip skip 2 (or -2 in some software)"
- **수정:** 사용자가 설정을 바꿔야 한다.
- **크기:** 수치 미기재
- **출처:** https://civitai.com/models/257749/pony-diffusion-v6-xl (1차). 버전 생성일 2024-01-08은 https://civitai.com/api/v1/models/257749 에서 확인했다.

---

## 4. 업계 대응 (출처가 말한 대로)

### 모델 제작사의 검증 도구

- **OpenAI: gpt-oss 구현 검증** (호환성 테스트 커밋 2025-08-11)
  - 도구 호출과 API 형태를 보는 호환성 테스트, 그리고 AIME·GPQA·Healthbench 평가 도구를 공개했다.
  - 호환성 테스트는 "This largely acts as a smoke test"라고 스스로 적었다.
  - 성공 기준: "0 invalid requests and over 90% on both pass@k and pass^k"
  - harmony 형식이 틀리면 "cascading generation issues"가 난다고 경고했다.
  - 출처: https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations , https://github.com/openai/gpt-oss/tree/main/compatibility-test
- **Moonshot AI: K2 Vendor Verifier → Kimi Vendor Verifier**
  - K2VV 저장소는 2025-09-09에 생겼다. 업체를 주기적으로 평가하고 명단에 없는 업체는 참여를 신청할 수 있다.
  - 이후 Kimi Vendor Verifier는 여섯 가지를 본다: API 파라미터 사전 검증, OCRBench, MMMU Pro Vision, AIME2025, K2VV ToolCall, SWE-Bench.
  - AIME2025는 "Catches KV cache bugs and quantization degradation that short benchmarks hide."
  - 공식 API의 Thinking 모드에는 Temperature=1.0, TopP=0.95를 강제했다.
  - 상류 수정: "We embed with vLLM/SGLang/KTransformers communities to fix root causes"
  - 출처: https://github.com/MoonshotAI/K2-Vendor-Verifier , https://www.kimi.ai/blog/kimi-vendor-verifier
- **MiniMax** (2025-11-03): `reasoning_details` 필드를 도입했다. OpenRouter, Ollama, Droid, Vercel, Cline과 협력한다.

### 라우터와 측정 기관

- **OpenRouter: Exacto** (2025-10-21)
  - 도구 호출 정확도가 높은 업체만 묶은 `:exacto` 엔드포인트를 만들었다.
  - "In aggregate, we have measured the accuracy of billions of LLM tool calls."
  - 출처: https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/
- **OpenRouter: Auto Exacto** (2026-03-12)
  - 약 5분마다 처리량, 도구 호출 원격 측정, 벤치마크 점수로 업체를 다시 평가한다. 도구 요청에는 기본으로 켜진다.
  - 원인 진술: "The actual culprit, more often than not: tool-call parsers."
  - 이어서 "that's typically an inference engine issue, not a provider cutting corners"라고 적었다.
  - 출처: https://openrouter.ai/blog/announcements/auto-exacto/
- **Artificial Analysis**
  - 업체별 정확도 벤치마크(2025-08, gpt-oss-120b)에 이어 Endpoint Accuracy Index(2026-08-04)를 냈다.
  - 방식: "self-host the official weights at the lab's recommended precision" 한 참조 배포와 각 엔드포인트를 비교한다.
  - 출처: https://artificialanalysis.ai/articles/endpoint-accuracy-index

### 서빙 업체의 약속

- **Anthropic 2025-09-17**
  - 운영 평가: "we will run them continuously on true production systems"
  - 더 민감한 평가를 만들고, 개인정보를 지키며 커뮤니티 피드백을 디버깅하는 도구를 만든다.
  - 배포 과정에 예상 밖 문자 검출 테스트를 더했다.
- **Anthropic 2026-04-23**
  - 내부 직원이 Claude Code 공개 빌드를 그대로 쓴다.
  - 시스템 프롬프트를 바꿀 때마다 모델별 평가 묶음을 돌리고 줄 단위 ablation을 계속한다.
  - 지능과 맞바꿀 수 있는 변경에는 "soak periods, a broader eval suite, and gradual rollouts"를 둔다.
- **OpenAI Codex 2025-10-31**
  - `/feedback`을 클러스터와 하드웨어에 연결했다.
  - "Moved all internal usage within OpenAI to use the same setup as our external users"
  - 기능 플래그 60개 이상을 없앴고 80개를 더 없애는 중이다.
  - 클러스터와 하드웨어 조합별 평가를 추가했다.

### 추론 엔진과 로컬 도구

- **결정적(배치 불변) 추론**
  - Thinking Machines(2025-09-10): 배치 불변 커널을 공개했다.
  - SGLang(2025-09-22): 결정적 모드를 추가했다. 평균 부하는 34.35%다.
  - vLLM `VLLM_BATCH_INVARIANT=1`: 문서에 "may impact performance compared to the default non-deterministic mode"라고 적혀 있다.
  - 출처: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ , https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/ , https://docs.vllm.ai/en/latest/features/batch_invariance/
- **llama.cpp** (2024-04-29, PR #6920): 변환할 때 사전 토큰화 해시를 확인하고, 모르는 경우 갱신을 요구한다. `test-tokenizer-0`로 검증한다.
- **Ollama**
  - v0.6.7: 기본 4096
  - v0.9.3: "limit context length to what the model was trained against"
  - v0.10.0: `ollama ps`에 컨텍스트 길이 표시
  - v0.15.5: VRAM 기준 기본값

### 이미지 도구와 표준화

- **v-pred 표지 감지**
  - A1111 PR #16567(2024-10-19, dev)과 ComfyUI는 state_dict에 `v_pred` 키가 있는지로 판단한다.
  - ComfyUI #12579(2026-02-22, 열림)는 메타데이터를 읽어 달라고 요청했다: "model metadata has been explictly set with prediction_type="v"" (원문 철자 그대로). 우회책은 Model Sampling Discrete 노드로 덮어쓰는 것이다.
- **Stability AI ModelSpec** (규격 v1.0.1, 저장소 첫 커밋 2023-07-21, 마지막 커밋 2024-06-04)
  - safetensors 헤더 메타데이터 규격이다. 목적은 "an inference engine can determine how to load it correctly"다.
  - `prediction_type`(v/epsilon), `encoder_layer`(clip skip), `unet_dtype`, `vae_dtype`은 모두 CAN(선택) 등급이다.
  - 출처: https://github.com/Stability-AI/ModelSpec

### 사용 안내

- **Google Gemini 3 문서:** 기본값 유지를 권한다. temperature를 1.0 아래로 내리면 "looping or degraded performance"가 날 수 있다고 적었다. 출처: https://ai.google.dev/gemini-api/docs/troubleshooting

### 학계 도구

- **Model Equality Testing** (2024-10): API가 참조 가중치와 같은 분포를 내는지 MMD 기반 통계 검정으로 확인한다.
- **Cai et al.** (2025-04): "software-only methods are fundamentally unreliable"이라고 보고, 신뢰 실행 환경(TEE)을 제안했다.
- **IRIS** (2026-07): 게이트웨이에서 모델 대체와 라우팅 희석을 감사한다.
- **Ekka** (2026-06): 조용한 오류를 자동으로 진단한다. "Ekka also diagnoses 4 new silent errors from serving frameworks". 개발자들이 모두 확인했다.

---

## 5. 체계적 측정

| # | 측정 (공개일) | 대상 | 결과 (출처 수치) | 출처 |
|---|---|---|---|---|
| M1 | Artificial Analysis Endpoint Accuracy Index (2026-08-04) | GLM-5.2, gpt-oss-120b, DeepSeek V4 Pro 서버리스 엔드포인트 | 아래 참조 | https://artificialanalysis.ai/articles/endpoint-accuracy-index , https://artificialanalysis.ai/models/gpt-oss-120b/providers |
| M2 | Artificial Analysis 업체별 정확도 (2025-08) | gpt-oss-120b | L07 참조. AIME25x32 93.3%~80.0% (압축 모델 36.7%) | Simon Willison 글 (2차) |
| M3 | OpenRouter 도구 호출 원격 측정 (2025-10-21, 2026-03-12) | Kimi K2, DeepSeek, GLM, gpt-oss, Qwen3 Coder 등 | 아래 참조 | OpenRouter 블로그 2편 |
| M4 | K2 Vendor Verifier (2025-11-15 시험) | Kimi K2-0905, K2-Thinking의 업체와 엔진 | L11 참조. 요청 4,000개 | GitHub README |
| M5 | Model Equality Testing, Gao·Liang·Guestrin (arXiv 2024-10-26) | 2024년 여름 상용 API, Llama 4종, 엔드포인트 31개 | "11 out of 31 endpoints serve different distributions than reference weights released by Meta" | https://arxiv.org/abs/2410.20247 |
| M6 | IRIS, Zhang 외 (arXiv 2026-07-23) | OpenRouter에서 한 공개 가중치 모델을 업체별로 고정 | "flags 14/15 provider pairs as distinguishable" | https://arxiv.org/html/2607.20860v1 |
| M7 | Give Me FP32 or Give Me Death?, Yuan 외 (arXiv 2025-06-11) | DeepSeek-R1-Distill-Qwen-7B, bf16, greedy | "up to 9% variation in accuracy and 9,000 tokens difference in response length" | https://arxiv.org/abs/2506.09501 |
| M8 | Defeating Nondeterminism in LLM Inference, Thinking Machines (2025-09-10) | Qwen3-235B-A22B-Instruct-2507, temperature 0, 1000회 | 아래 참조 | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ |
| M9 | SGLang 결정적 추론 (2025-09-22) | 50회 샘플링, FlashInfer·FA3·Triton | 아래 참조 | https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/ |
| M10 | A First Look at Bugs in LLM Inference Engines, Liu 외 (arXiv 2025-06-11, 2026-01-09 개정, TOSEM) | llama.cpp, vLLM, DeepSpeed, MLC-LLM, TensorRT-LLM의 버그 929건 | 아래 참조 | https://arxiv.org/abs/2506.09713 |

측정별 세부 결과는 다음과 같다.

**M1. Artificial Analysis Endpoint Accuracy Index**
- 방식: BFCL-500(3회), HLE-250(10회), AA-LCR-25(10회)를 같은 비중으로 합쳐, 참조 배포를 100으로 둔 값이다.
- gpt-oss-120b 엔드포인트 (참조 SGLang 100)

  | 엔드포인트 | 값 |
  |---|---|
  | Amazon | 100.77 |
  | SambaNova | 98.15 |
  | Parasail | 97.9 |
  | CoreWeave | 97.63 |
  | DeepInfra | 97.01 |
  | Scaleway | 96.61 |
  | Nebius (Base) | 96.59 |
  | Azure | 92.9 |
  | Novita | 90.84 |
  | Cerebras | 87.28 |
  | Groq | 86.36 |
  | DeepInfra (Turbo) | 84.4 |
  | Together AI | 76.71 |
  | Google Vertex | 72.2 |
  | Cloudflare | 69.83 |

- gpt-oss-120b 도구 호출: "some endpoints score 22% on BFCL-500 against 37% for the reference"
- GLM-5.2: 가장 제한이 심한 엔드포인트는 "half the reference or less on HLE-250"이었다. 페이지 값으로는 Scaleway 74.83, DeepInfra (FP4) 73.04가 가장 낮다.
- DeepSeek V4 Pro: "Majority of the endpoints are at reference parity"
- 출력 토큰: 가장 낮은 엔드포인트는 "roughly half the reference's output tokens"를 냈다.

**M3. OpenRouter 도구 호출 원격 측정**
- 측정 규모: "billions of LLM tool calls". 판정 기준은 세 가지다. 반환된 도구 호출이 올바른 JSON인지, 도구 이름이 입력에 있었는지, 스키마가 맞는지를 본다.
- Auto Exacto 적용 후 도구 호출 오류율

  | 모델 | 변화 |
  |---|---|
  | GLM-5, GLM-4.7 | 88%, 80% 감소 (약 8% → 약 1%) |
  | gpt-oss-120b | 5.6% → 3.5% |
  | DeepSeek V3.2 | 16% 감소, TauBench 69% → 74% |

**M8. Thinking Machines 비결정성 측정**
- 1000회 생성 중 서로 다른 완성문이 80개였다. 103번째 토큰에서 처음 갈렸다.
- 배치 불변 커널을 쓰면 1000개가 모두 같았다.
- 원인: "the load (and thus batch-size) nondeterministically varies"
- 시간: vLLM 기본 26초, 결정적 55초, 개선된 attention 커널 42초

**M9. SGLang 결정적 추론**
- 일반 모드에서는 같은 요청이 여러 출력을 냈다. 단일 시험에서 3~4개, prefix 시험에서 최대 10~18개였다.
- 결정적 모드에서는 모두 1개였다.

**M10. 추론 엔진 버그 연구**

| 증상 | 비율 |
|---|---|
| Crash | 65% |
| Unexpected Output | 13% (추론·서빙 단계에서는 "nearly 20%") |
| Feature Failure | 11% |
| Abnormal Performance | 6% |
| System Hang | 4% |
| Silent Error | 1% |

- 논문 표현: "over 35% of bugs manifest as non-crash symptoms"
- 모델 변환 단계의 17%는 잘못된 모델 파일을 만들었다.

**이미지 생성 쪽:** 같은 모델을 도구마다 돌려 품질 차이의 빈도나 크기를 잰 체계적 측정은 이번 조사에서 찾지 못했다. diffusers에 "The different quality between ComfyUI and Diffusers ?"라는 토론(#9265, #9271)은 있지만 측정은 아니다.

---

## 6. 조사 방법과 한계

- 모든 사례는 해당 페이지를 직접 열어 확인했다.
  - GitHub는 `gh issue view`, `gh pr view`, `gh api -X GET`로 읽기만 했다.
  - 웹 페이지는 원문 텍스트에서 인용문과 수치를 다시 찾아 대조했다.
- 열지 못한 원문
  - X 게시물: Artificial Analysis, Ahmad Al-Dahle, Lucas Beyer
  - Reddit GPT-5 AMA
  - Unsloth 블로그(Cloudflare 차단). Unsloth 문서 페이지로 대신 확인했다.
  - 이 때문에 L07과 L19는 2차 출처에 기대고, L06의 Meta 발언도 2차 출처다.
- 제외한 후보
  - 원인이 실행 소프트웨어로 확인되지 않은 것: DeepSeek V3.1의 '极' 토큰 현상
  - 모델 자체 행동인 것
  - 유지보수자가 저하가 아니라고 반박한 것: ComfyUI 0.3.51의 Qwen-Image-Edit 출력 변화(#9509)
- 날짜, 기간, 비율은 출처에 적힌 값과, 그 날짜 사이를 센 값만 썼다.
