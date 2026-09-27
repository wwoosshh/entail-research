# 후보 어휘: 회전 쌍의 짝짓기 (`Rotary.pairing`) — 근거 메모 (2026-09-26, M15.7 탐색)

M15.7 정적 훑기가 어휘 밖으로 남긴 RoPE 이름 가운데 `rope_interleave`(GLM-5.2 계열 4개, `rotary_sweep.json`·`SUMMARY.md` M15.7)의 뜻은
"회전이 머리 차원을 **반으로 갈라**(NeoX 식) 짝짓는가, **이웃끼리 끼워**(GPT-J 식) 짝짓는가"다. 엔진은 이를 대개 config가 아니라
**아키텍처 관례**로 정한다. 이 메모는 그 뜻이 실제로 부품 경계에서 사라진 사례가 있는지를 읽기 전용으로 확인한 것이다(GitHub 조회만; 게시 없음).

## 공개 이슈 (vLLM, `gh` 조회 2026-09-26)

| 이슈 | 날짜 | 상태 | 무엇이 어디서 사라졌나 |
|---|---|---|---|
| vllm#49290 | 2026-07-21 | open(댓글은 #49906로 해결됐다고 함) | MRoPE Triton 커널이 NeoX 식 회전을 **고정**해, GPT-J 식 모델(GLM-OCR)에서 틀린 결과. 모델 코드는 `is_neox_style=False`를 선언했지만 커널 경로가 읽지 않았다. |
| vllm#53063 | 2026-08-20 | open(#51655로 고침) | DFlash 드래프트 모델이 타깃의 RoPE 짝짓기(`is_neox_style`)를 **물려받지 않음**. 고침은 로더가 타깃에서 읽어 `hf_config.is_neox_style`로 찍는 것. |
| vllm#53063 후속 댓글 (2026-08-29) | | | 고친 뒤에도 문제: 어떤 공개 DFlash2 체크포인트는 타깃과 **반대 관례**로 증류됐는데, 관례가 선언이 아니라 추론이라 config로 바로잡을 길이 없다는 보고(A/B 수치 첨부). |

## 설치된 vLLM 0.30.0에서 확인한 것 (`~/venvs/vllm`, 읽기만)

- `model_executor/layers/rotary_embedding/mrope.py`: Triton 커널이 `is_neox_style`과 `mrope_interleaved`를 constexpr로 받는다(#49290의 고침이 들어 있음).
- `model_executor/models/` 가운데 **20개 파일**이 `is_neox_style=False`(GPT-J 식)를 코드 상수로 선언한다: glm4, glm, deepseek_v2, gpt_j, commandr, ernie45, bailing_moe*, exaone_moe, openpangu, kimi_k25_vit, mllama4 등.
- config에서 이 뜻을 읽는 곳은 드물다: `transformers_utils/configs/bailing_moe_v3_vl.py`(`rope_interleave`), `glm5_next.py`(`indexer_rope_interleave`). DFlash(`models/qwen3_dflash.py:305`)는 `config.is_neox_style`을 읽되 없으면 True로 **가정**한다.

## 어휘로 넣을 때의 판단 재료

- **선언 출처:** config 키(`rope_interleave`, `rope_interleaved`, `is_neox_style` 찍기)가 있으면 파일이 선언한다. 없으면(대부분) 관례뿐이라 entail이 지어낼 수 없다. 사용자 선언(`declare_on`, M6 방식)이 채널이 될 수 있고, 이는 #53063 후속 댓글이 말한 "config로 바로잡을 길이 없다"를 정확히 메운다.
- **소비자:** vLLM `RotaryEmbedding.is_neox_style`(객체가 든 값), MRoPE 커널의 constexpr, DFlash 로더의 찍기. SGLang·llama.cpp(GGUF는 아키텍처 표 `llm_arch_rope_type`)는 미확인.
- **정상 실행 오탐 위험:** 선언이 없는 모델에서는 판정하지 않아야 한다(unknown조차 소음이 될 수 있어 "선언 있을 때만 비교"). 선언이 있는 모델(GLM-5.2, Bailing)에서 객체 값과 비교하면 된다.
- **자로 잰 값:** 실제 보고 2건(하나는 커널, 하나는 드래프트 모델)과 후속 보고 1건이 있고, 둘 다 "선언은 있는데(코드 상수) 소비자가 안 읽음" 또는 "선언 자체가 없어 추론"이라 THEORY의 부류 안이다. 다만 현 vLLM 0.30.0에서는 둘 다 고쳐져 있어 E3식 재현은 옛 판(0.29 이전)이 필요하다.

판단은 검토 에이전트와 연구자에게 넘긴다. 이 메모는 근거 파일이며 구현이 아니다.
