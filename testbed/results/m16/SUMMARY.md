# M16 결과: 어휘를 동결한 사전 등록 재현 실험 (2026-09-26)

규약은 `testbed/M16_PROTOCOL.md`(결과 전에 적음)이고, 이 문서를 쓴 뒤에도 규약은 바꾸지 않았다(6절 변경 기록 없음). 동결 = entail **1.1.0**(`63e0949`, 어휘 v7). 실험 중 어휘·별칭·규칙·능력표·어댑터를 바꾸지 않았고(작업 사본 `git status` 깨끗, HEAD `63e0949`), 실험 중에 본 어휘 후보는 `DEFERRED.md`에 두었다.

## 1. 결론 (수치는 모두 결과 파일에서)

| 항목 | 값 | 출처 |
|---|---|---|
| 심사 | 150건 심사, 통과 17 (not_output 63, cannot_run 67, no_repro_info 3) | `screening.json` |
| 평정 대상(규칙 1 통과) | 87건 | `rating_packet.json` |
| 평정 일치 | 7범주 카파 **0.860**(관찰 일치 0.897), K1 대 나머지 카파 **0.947**(0.977); 갈린 9건은 셋째 눈가림 평정자가 정함 | `ratings.json` |
| 부류(K1) 비중 | 합의 **29/87 = 33%**(평정자 A 27/87 = 31%, B 29/87 = 33%); K7(알 수 없음) 32/87; K7을 뺀 55건 기준 29/55 = 53% | `ratings.json` |
| 재현 | 통과 17건 가운데 **15건 재현**, 2건 미재현 | `results.json`, `cases/` |
| **검출률** | **0 / 7** (재현된 합의 K1 7건 가운데 resolved·reported 0). 평정자별 분모: A 6, B 7 → 0/6, 0/7 | `results.json` |
| 부류 밖 오탐(`false`) | **0** (재현된 부류 밖 8건과 미재현 2건의 실행 어디에도 `broken`·`refused` 없음) | `cases/*_on.record.jsonl` |

한 줄로: **1.1.0의 어휘는 처음 보는 실제 버그 가운데 부류 안의 것을 하나도 잡지 못했고, 부류 밖의 것을 하나도 잘못 잡지 않았다.** "부류 안 4/4"(버그를 본 뒤 어휘를 더해 잰 값)는 검출률이 아니었고, 이 실험이 그 자리를 채운다. README의 그 문장은 이 수치로 바꾼다.

## 2. 조사와 심사 (M16.1, M16.2)

- 조사(`m16_census.py`): 네 저장소(vLLM, SGLang, transformers, diffusers), 2026-03-26~09-26, 열린 것과 닫힌 것, 제목 키워드 14개, 563건 → M10 E3가 심사한 73건을 뺀 **490건**. 순서는 시드 20260926.
- 심사(`m16_screen.py`, `screening.json`): 순서대로 150건, 건너뛰지 않음. 판정과 이유를 모두 적었다.
  - `pass` 17, `not_output` 63(멈춤·크래시·적재 실패·성능·문서·기능 요청), `cannot_run` 67(대형 모델, 다중 GPU, Blackwell·AMD·XPU·NPU·MPS 전용), `no_repro_info` 3.
  - **경계 판정 5건**(이유에 `borderline` 표시): transformers#45698(잘못된 커스텀 모듈 적재), vllm#46585(sm90 커널 단위 시험), transformers#47752(generation_config 우선순위), vllm#46988(프롬프트 토큰 id 불일치), diffusers#14451(잘못된 파이프라인 클래스). 증거가 모델 출력이 아니라 적재된 클래스·설정값·프롬프트 토큰이어서 규칙 1의 엄격한 읽기로 `not_output`으로 뺐다. 다섯 모두 "선언의 우선순위·치환" 성격이라 평정하면 K1일 가능성이 크다. 넣었다면 평정 대상은 92건이고, 재현 가능한 것은 셋(45698, 47752, 14451; CPU)이라 검출률 분모가 늘 수 있었다. 이 선택은 결과를 보기 전에 규칙대로 내렸고, 되돌리지 않는다.
  - 대체 모델은 쓰지 않았다(규칙 2는 보고된 모델 기준). 모델 없는 재현(파서·커널·스케줄러·토크나이저 수준)은 보고가 그 경로를 제시한 경우에만 통과시켰다.

## 3. 눈가림 평정 (M16.3)

- 자료(`m16_rating_packet.py` → `rating_packet.md/json`): 규칙 1을 통과한 87건의 제목·상태·라벨·본문 요약(환경 덤프 제거, 2,500자)·연결된 PR 제목과 설명(700자). 심사 이유와 `expected_boundary`는 넣지 않았다. 순서는 별도 시드 20260927.
- 평정자: 세션과 독립인 에이전트 둘(A: Opus, B: Sonnet), 코드북 `testbed/m16/RATING_CODEBOOK.md`(중립 범주 K1~K7)만 받음. 갈린 9건은 셋째 평정자(C: Opus, 새 문맥)가 정했다. 결과 `ratings_A.txt`, `ratings_B.txt`, `ratings_C.txt`, 합산 `ratings.json`(`m16_rate_score.py`).
- 분포(합의): K1 29, K2 5, K3 6, K4 9, K5 4, K6 2, K7 32.
- 한계: 평정자는 이 프로젝트의 세션 안에서 띄운 에이전트라 프로젝트 안내문이 문맥에 있었을 수 있다. 자료와 코드북에는 가설이 드러나는 말이 없고, 평정 지시는 "자료의 글만으로 판단하고 다른 파일을 열지 말라"였다. 두 평정자가 모델이 다르고 일치도가 높은 것(0.86/0.95)이 편향의 상한을 좁히지만, 완전한 외부 평정은 남은 일이다.

## 4. 재현과 판정 (M16.4)

실행 관례는 M10 E3와 같다: `testbed/m10_e3/run_case.sh`로 entail 끄고 한 번, 켜고 한 번(`ENTAIL=load`, 기본 정책), 결과는 `results/m16/cases/<case>_{off,on}.json`과 `_on.record.jsonl`. 환경: `gpu`(transformers 5.17.0, diffusers 0.40.0), `vllm`(0.30.0), `sglang`(0.5.20; 재현 스크립트가 요구한 `peft`를 더함), 그리고 규칙 4의 보고된 판 환경 셋(최대 6): `tf5121`(transformers 5.12.1, CPU), `vllm0190`(vLLM 0.19.0), `vllm0220`(vLLM 0.22.0; GLM-OCR 적재를 위해 transformers만 5.17.0으로 올림).

| # | 이슈 | 사례 | 판·환경 | 재현 | 합의 | entail 켠 실행의 판정 | 판정 |
|---|---|---|---|---|---|---|---|
| 1 | vllm#56655 프롬프트 임베딩 마스크가 접두 캐시 키에 없음 | `vl56655` | 0.30.0 | 재현(B after A: 32토큰 재사용, A의 출력) | K1 | pass 9(적재 경계만) | **missed** |
| 6 | transformers#47885 Whisper 특징 행렬 전체 NaN, 무음 전사 '0' | `tf47885` | 5.17.0 | 재현 | K2 | unknown 3(config Coverage) | missed(부류 밖) |
| 14 | vllm#49316 kimi_k2 스트리밍이 스키마 형 변환을 건너뜀 | `vl49316` | 0.30.0 | 재현(4/4: "3" 대 3) | K4 | pass 2 | missed(부류 밖) |
| 19 | vllm#48231 Gemma 4 이미지 요청 fp16 넘침 | `vl48231` | 0.30.0, dtype=float16 | 재현(토큰 1023 " our" 반복; bf16 대조는 정상) | K2 | unknown 3, pass 8 | missed(부류 밖) |
| 40 | sglang#38573 W4AFP8 MoE 그룹 64 대 커널 chunk 128 | `sg38573` | 0.5.20 | **못 돌림**(스킴이 capability 90 요구, 이 카드 89) | K1 | — | not_reproduced |
| 42 | vllm#43728 thinking 대 enable_thinking | `vl43728_0220` | **0.22.0**(0.30.0은 두 이름을 다 읽음, 0.19.0은 kwargs 이전) | 재현(content null) | K1 | pass 1 | **missed** |
| 44 | sglang#40835 use_rslora 무시 | `sg40835` | 0.5.20 | 재현(PEFT 16.0 대 SGLang 2.0) | K1 | pass 2(transformers 쪽만) | **missed** |
| 54 | vllm#49412 스트리밍 공백 불일치 | `vl49412` | 0.30.0 | 재현(3/3) | K4 | pass 2 | missed(부류 밖) |
| 64 | transformers#46032 Mamba2 캐시+청크 | `tf46032` | **5.12.1** | 재현(오차 2.9) | K4 | pass 1 | missed(부류 밖) |
| 67 | transformers#46612 빔 서치가 cache_params를 재정렬 안 함 | `tf46612` | **5.12.1** | 재현(캐시 있는 빔 ≠ 없는 빔) | K1(A K4, B K1, C K1) | unknown 1, pass 3 | **missed** |
| 76 | vllm#48895 Marlin MoE topk 가중치 행 어긋남 | `vl48895` | 0.30.0, sm89 | 재현(119/128행 틀림; 외부 곱은 0/128) | K1 | 기록 없음 | **missed** |
| 81 | transformers#48293 Switch 라우터 반환·용량 | `tf48293` | 5.17.0 | 재현 | K2 | unknown 1, pass 2 | missed(부류 밖) |
| 92 | vllm#39468 Gemma 4 파서 구분자 누출 | `vl39468`(+`_0190`) | 0.30.0, **0.19.0** | 미재현(보고의 두 문장과 수정 PR의 회귀 문자열이 두 판 모두 깨끗이 파싱됨) | K4 | — | not_reproduced |
| 102 | transformers#46710 토크나이저 클래스 치환 | `tf46710` | **5.12.1**(토크나이저 수준) | 재현(LlamaTokenizer, id 불일치) | K4 | pass 2(**경계에서 Vocab pass**) | missed(부류 밖) |
| 105 | diffusers#13411 단조 아닌 sigma 일정 | `df13411` | 0.40.0 | 재현(출력·입력 코사인 -0.17, 대조 +0.96; 보고의 1.45배는 안 나옴, 0.875) | K6(A K6, B K4, C K6) | 기록 없음 | missed(부류 밖) |
| 115 | sglang#21843 GDN a/b 비연속 strides | `sg21843` | 0.5.20(고쳐지지 않음) | 재현(연속 대비 g 4.7, beta 0.78 차이) | K1 | 기록 없음 | **missed** |
| 125 | vllm#42016 GLM-OCR mrope 회전 방식 | `vl42016_0220` | **0.22.0**(0.30.0은 #49906으로 고쳐짐) | 재현(두 시험 이미지에 'STOP'; HF 참조와 일치 0토큰) | K1 | unknown 14(로컬 폴더 없음), pass 6 | **missed** |

재현 기준을 보고와 다르게 둔 곳(모두 스크립트 머리말과 여기 적음): df13411은 놈 비율 대신 부호(보고의 1.45배 계산이 denoised=0을 가정), vl48231은 토큰 0 대신 "한 토큰 반복"(같은 퇴화 형태, bf16 대조 정상), vl42016은 보고의 시험 이미지 두 장을 더함. 모두 보고 자체의 증거 쪽으로 좁힌 조정이다.

## 5. 왜 하나도 못 잡았나 (기록에서 읽은 것)

- **판정 자체가 없는 자리**: 재현된 부류 안 7건 가운데 3건(vl48895, sg21843, 그리고 부류 밖 df13411)은 기록 파일이 아예 없다. 모델을 적재하지 않고 커널·스케줄러만 부르는 경로에는 어댑터가 붙지 않는다. sg40835(LoRA 설정), vl43728(요청 kwargs), vl56655(접두 캐시 키), tf46612(빔 재정렬), vl42016(mrope 커널)에서는 적재 경계의 판정(Coverage·Vocab·Rotary·Stops·ModelProps·Layout)만 있고 결함이 생긴 경계에는 없다. 1.1.0의 경계는 적재·컨테이너·요청 셋이지만, 실제로 판정이 붙는 자리는 대부분 적재다.
- **경계에서 통과**: tf46710은 토크나이저 클래스가 치환됐는데 `load:transformers.tokenizer`의 Vocab이 pass였다. Vocab 사실은 크기를 비교하지 클래스 선언을 보지 않는다(부류 밖 K4로 평정됐지만 `DEFERRED.md` 4번에 남김).
- **어휘 밖**: 7건의 사실은 모두 v7 어휘에 없다 — 접두 캐시 키의 구성 요소, 어댑터의 스케일 규칙, 커널 입력의 strides, 요청 설정의 이름 대응, 빔-캐시 행 대응, 회전 짝짓기 방식, 라우팅 가중치 행 대응. 후보는 `DEFERRED.md`에 있다.
- **entail 자신의 결함**(규약 1절의 예외 후보): vLLM 0.22.0에서 `vllm_scoring` 어댑터 설치가 순환 import로 실패했다(기록되고 실행은 계속됨; 결과에 영향 없음). 허브 id로 적재한 모델(GLM-OCR)에서는 Vocab·Stops 검사가 "로컬 폴더 없음"으로 13~14줄 cannot-check를 냈다(M11.6은 diffusers만 다뤘다). 둘 다 이번 실험에서는 고치지 않았다.

## 6. 범위와 한계

- 분모가 7이라 0/7의 구간은 넓다(단측 95% 상한 약 35%). "부류 비중 33%"도 K7 32건(보고만으로 알 수 없음)을 분모에 둔 값이며, 판정 가능한 55건 기준으로는 53%다.
- 심사는 세션이 했다(가설을 앎). 규칙은 기계적이고 이유를 모두 적었으며, 경계 판정 5건을 2절에 드러냈다.
- 미재현 2건(sg38573 하드웨어, vl39468 파서 수준)은 검출률 분모에 넣지 않았다.
- 보고된 판 환경은 셋(transformers 5.12.1, vLLM 0.19.0, 0.22.0)을 만들었고, 1.1.0의 어댑터가 옛 판의 코드에 붙는지는 각 실행의 기록으로만 확인했다(붙은 것: 토크나이저·config 경계; 못 붙은 것: 0.22.0의 scoring 어댑터).
- 12 GB 한 장이라 통과 17건 가운데 다중 GPU·대형 모델은 애초에 심사에서 빠졌다(cannot_run 67).

## 7. 파일

- 규약 `testbed/M16_PROTOCOL.md`; 코드북 `testbed/m16/RATING_CODEBOOK.md`.
- 도구 `testbed/m16_census.py`, `m16_screen.py`, `m16_rating_packet.py`, `m16_rate_score.py`, `m16_results.py`.
- 결과 `testbed/results/m16/`: `census.json`, `order.json`, `screening.json`, `rating_packet.{json,md}`, `rating_disagreements.md`, `ratings_{A,B,C}.txt`, `ratings.json`, `verdicts.json`, `results.json`, `cases/`(17 사례 + 보고된 판 재실행 `*_0190`, `*_0220`, bf16 대조 `vl48231bf16`), `DEFERRED.md`.
- 사례 스크립트 `testbed/m16/cases/*.py`(`_parser_util.py`는 파서 사례 공용).
- 내려받은 것: Qwen3-0.6B(rev c1899de), gemma-4-12B-it-qat-w4a16-ct(9.6 GB), zai-org/GLM-OCR, openai/whisper-tiny, google/switch-base-8, state-spaces/mamba-130m-hf, DeepSeek-R1-Distill-Llama-8B 토크나이저, gemma-4-E2B-it 토크나이저, vLLM 시험 이미지 두 장. 가상 환경 `~/venvs/tf5121`, `vllm0190`, `vllm0220`.

## 8. 다음 (연구자가 정함)

1. `DEFERRED.md`의 후보 10개 가운데 무엇을 어휘에 넣을지. 이번 실험의 수치는 그것을 넣기 **전**의 값이다. 넣은 뒤에는 새 표본으로 다시 재야 검출률이다.
2. 판정이 붙는 자리를 적재 밖으로 넓힐지(커널 호출·요청 파서·어댑터 설정 파일). 이번 7건 가운데 5건은 어휘가 있어도 그 자리에 어댑터가 없으면 잡히지 않는다.
3. README의 "4 of 4" 문장은 이 수치로 바꿨다(1.1.0 문서 커밋, 푸시는 허락 뒤).
