# M10 E4: 쓰임새별 묶음 (새 측정 없음)

- 작성: 2026-09-24. `testbed/M10_PROTOCOL.md` 4절의 정의를 따른다.
- 이미 잰 결과를 사람들이 엔진을 쓰는 방식별로 다시 묶었다.
- 수치는 모두 적힌 결과 파일에서 옮겼다. 새로 잰 것은 없다.
- 사례의 종류는 셋이다. 섞지 않는다.
  - **실제:** 엔진이나 모델에 실제로 있는 동작이다.
  - **모의:** 시장 사례의 원인을 그대로 만든 것이다.
  - **심음:** 연구 도구로 결함을 넣은 것이다.
- 장치는 RTX 4070 Ti 한 장이다. 엔진은 transformers 5.17, vLLM 0.30, SGLang 0.5.20, diffusers 0.40, ComfyUI 0.34.1이다.

## 1. OpenAI 호환 서버 (vLLM, SGLang)

| 사고 | 종류 | entail 없이 | entail 켬 | 근거 파일 |
|---|---|---|---|---|
| 긴 문맥을 켜려고 실행 때 `rope_scaling`을 넘김(`--hf-overrides`, `--json-model-override-args`). Llama-3.2-3B | 실제 | GSM8K 379 → **273**/500(vLLM), 161 → **106**/200(SGLang). 경고 없음 | 376~380/500(vLLM), 161/200(SGLang) | `testbed/results/m91/SUMMARY.md` fd-rope, `m93/SUMMARY.md`, `entail/README.md` |
| 같은 원인이 걸리는 인기 모델의 수 | 코드 | 적용되는 180개 중 **64개**가 다른 RoPE 밑으로 돔 | 180개 모두 원래 밑 | `results/m10/E1_SUMMARY.md` L1 |
| 백엔드가 모델이 선언한 softcap을 버림. Gemma 2, SGLang | 실제 | 답 500개 중 198개가 바뀜. GSM8K로는 안 보임(313 대 316, p = 0.66) | 적재 때 softcap을 지키는 백엔드(triton)로 바꿈. 비용 1.13배(그 백엔드의 값) | `m91/SUMMARY.md` S7, `entail/README.md` |
| 모델이 선언한 것과 다른 채팅 템플릿으로 렌더 | 모의 | 조용히 다른 프롬프트로 생성 | 기록(`broken`), 응답 200. 엄격 정책에서는 생성 전에 거부(vLLM 400, SGLang 500) | `m91/SUMMARY.md` S1, `m93/SUMMARY.md` |
| 모델이 "유지"라고 선언한 이전 사고 기록을 버림(시장 사례 L13) | 모의 | 조용히 버림 | 기록, 엄격이면 400 | `m91/SUMMARY.md` S2 mk-L13 |
| 요청 필드를 아무도 읽지 않음(`reasoning_effort`, 시장 사례 L07) | 모의 | 조용히 무시 | 기록, 엄격이면 400 | `m91/SUMMARY.md` S2 mk-L07 |
| 도구 호출 파서가 모델의 형식을 읽지 못함(시장 사례 L11) | 모의 | 구조화된 도구 호출 0개(본문에 글자로 남음) | 해소: 구조화된 호출 1개 | `m91/SUMMARY.md` S7 mk-L11 |
| 기본 문맥 길이로 긴 입력을 자름(시장 사례 L05) | 모의 | 10,983토큰 중 2,048토큰만 읽고 틀린 답 | 모델에 여유가 있으면 해소, 없으면 기록 | `m91/SUMMARY.md` S7 mk-L05 |
| 적재 중 가중치 행이 한 칸 밀림 | 심음 | GSM8K 468 → **3**/500. 서버는 정상으로 뜸 | 기록, 엄격이면 서버가 뜨기 전에 멈춤 | `m91/SUMMARY.md` S7 fd-shift |
| 재배치된 가중치(roll, transpose, stride) | 심음 | 조용히 틀림(roll) | 기록·엄격 거부. stride는 데이터로 해소 | `m91/SUMMARY.md` S2 fd-repack |
| KV 캐시가 토큰을 잃음 | 심음 | 틀린 출력 | 기록, 엄격 거부 | `m91/SUMMARY.md` S2 fd-kv |

비용은 셋이다.
- 적재 시간의 0.3~2.6%
- vLLM CUDA Graph 경로 0.999~1.000배
- 요청당 약 60 µs(렌더링 62~197 µs 옆에서 잰 값)

근거는 `m91/SUMMARY.md` S4다.

## 2. transformers 스크립트

| 사고 | 종류 | entail 없이 | entail 켬 | 근거 파일 |
|---|---|---|---|---|
| 기본 어텐션(sdpa)이 Gemma 2의 softcap을 버림 | 실제 | 조용히 다른 계산 | eager로 바꿈. 토큰이 기준과 같음. 비용 1.18배 | `entail/README.md` |
| `from_pretrained` 키워드나 속성으로 준 RoPE 값(transformers 4의 이름) | 실제 | 설정에서 빠짐 | `config.json` 경로와 같은 값(4계열 × 2경로 × 3값) | `entail/README.md` |
| 설정 키의 오타(`rope_scale`) | 심음 | 조용히 무시 | 기록, 엄격 거부 | `m91/SUMMARY.md` S2 rb-15 |
| 선언과 다른 채팅 템플릿을 `apply_chat_template`에 넘김 | 모의 | 조용히 다른 프롬프트 | 기록, 엄격이면 렌더 전에 거부 | `m93/SUMMARY.md`, `entail/tests/test_transformers_template.py` |

비용은 셋이다.
- 적재 시간의 0.7~2.6%
- 동적 KV 캐시 계약 1.017~1.022배(기준 1.02배의 경계)
- 정적 캐시 1.005배

근거는 `m91/SUMMARY.md` S4와 `m93/SUMMARY.md`다.

## 3. 이미지 생성 (diffusers, ComfyUI)

| 사고 | 종류 | entail 없이 | entail 켬 | 근거 파일 |
|---|---|---|---|---|
| v 예측 체크포인트를 eps로 샘플링(시장 사례 I04) | 실제 | 작성자 기준 그림과 55~95/255 차이. 실행은 "성공" | diffusers는 픽셀까지 같음, ComfyUI는 같거나 0.14~0.16/255 | `m91/SUMMARY.md` S7 fd-m7·mk-I04, `entail/README.md` |
| 모델에 닿지 않는 LoRA(시장 사례 I01) | 실제 | LoRA 없는 그림과 픽셀까지 같음 | 닿는 모듈 수를 기록, 엄격이면 샘플링 전에 멈춤 | `m91/SUMMARY.md` S7 mk-I01, `entail/README.md` |
| 단일 파일 VAE가 다른 모델의 배율을 받음 | 실제 | 15~16/255 차이 | 선언 파일의 배율로 픽셀까지 같음 | `m91/SUMMARY.md` S7 fd-vae |
| ComfyUI: 샘플링 노드의 스케줄이 워크플로가 끝난 뒤에도 남음(#16490) | 실제 | 다음 실행이 다른 그림(55.8/255), 이어서 검은 그림 | 새 세션과 픽셀까지 같음(3/3) | `entail/README.md` |

## 4. 직접 짠 코드 (코드 경계, 디버그 모드)

| 사고 | 종류 | entail 켬 | 근거 파일 |
|---|---|---|---|
| 재현 벤치마크 16건: 배치, 스트라이드, 양자화, 위치, 시점, 합산(한 프로세스가 두 랭크를 대신함) | 실제 버그의 재현 | 해소 또는 거부. 수정 판은 통과 | `m91/SUMMARY.md` S2 rb-01~17 |
| 선언 부담 | - | 코드 경계 서명 31줄, 손 태그 0. LLM 모델 폴더는 0줄 | `m91/SUMMARY.md` S5·S6 |

## 5. 이 묶음이 말하지 않는 것

- 일반 프로젝트가 이 사고를 얼마나 자주 만나는지는 말하지 않는다. 그것은 E1(노출도)이 잰다.
- 무작위로 뽑은 실제 버그를 몇 건 막는지도 말하지 않는다. 그것은 E3가 잰다.
- 위의 표는 사고가 났을 때의 차이다.
