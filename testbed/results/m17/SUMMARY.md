# M17 결과 (2026-09-26 시작; 자는 ROADMAP M17에 결과 전에 적음)

## M17.1 선언-소비 대응, LoRA 설정 (`adapter_config.json`)

**무엇:** PEFT 어댑터의 설정 파일을 선언 파일로 읽는다. 키 41개(PEFT 0.21.0 `LoraConfig`)를 종류(가중치가 나름 / 추론에 뜻이 있음(중립값 명시) / 학습 전용)로 적고, 소비자마다 읽는 키를 코드 줄 근거로 적었다(`entail/data/adapter_config_keys.json`): PEFT(transformers·diffusers)는 전부, vLLM 0.30.0은 `r, lora_alpha, target_modules, use_rslora` + 큰 소리로 거부하는 셋(`use_dora, modules_to_save, bias`), SGLang 0.5.20은 `r, lora_alpha, target_modules, use_dora`. 규칙은 하나(`entail/adapter_config_contract.py`): 중립값이 아닌 값으로 선언됐는데 이 소비자가 읽지 않는 추론 키는 `broken`, 소비자가 담을 수 있으면 `resolved`(담는 값은 핵심이 계산: rsLoRA는 `lora_alpha / sqrt(r)`), 거부하는 키는 note와 pass, 모르는 키는 값이 비어 있지 않을 때 한 번 `unknown`. 어댑터: `sglang_lora`(`LoRAAdapter.__init__`, scaling 대입), `vllm_lora`(`PEFTHelper.from_local_dir`). `entail check <어댑터 폴더>`가 엔진마다 판정.

**표를 적은 순서(눈가림 여부):** `use_rslora` 행은 sglang#40835를 안 뒤에 적었으므로 회고다. 나머지 행(`rank_pattern`, `alpha_pattern`, `lora_bias`, `bias`, `modules_to_save`, `fan_in_fan_out`, `target_parameters`, `trainable_token_indices`, `alora_invocation_tokens`, QALoRA·BD-LoRA·변형들)은 PEFT 스키마와 엔진 코드에서 적었고 어떤 사례도 보지 않았다.

**회고(검출률 아님):** sglang#40835의 스크립트(`testbed/m16/cases/sg40835.py`)를 새 어댑터로 다시 돌림(`testbed/results/m17/retro/sg40835_{off,on}.json`):

| | PEFT scaling | SGLang scaling, entail 끔 | SGLang scaling, entail 켬 |
|---|---|---|---|
| use_rslora=true (r=64, alpha=128) | 16.0 | 2.0 | **16.0** (`resolved`, `apply_use_rslora` 16.0) |
| use_rslora=false | 2.0 | 2.0 | 2.0 (`pass`) |

**실제 엔진 적재(오탐 자):** PEFT 0.21.0으로 Qwen2.5-3B-Instruct에 어댑터 둘을 만들었다(`testbed/m17/make_lora.py`; r=16, alpha=32, q_proj·v_proj, B=0이라 출력은 바뀌지 않음; `use_rslora`만 다름). `testbed/m17/lora_vllm.py`, `lora_sglang.py`로 entail 끄고·켜고 적재해 한 프롬프트를 생성(`results/m17/retro/lora_*`):

| 실행 | 출력(끔=켬) | entail 판정 |
|---|---|---|
| vLLM 0.30.0, default | 같음 | pass |
| vLLM 0.30.0, rslora | 같음 | pass (vLLM은 use_rslora를 읽음) |
| SGLang 0.5.20 Engine, default | 같음 | pass |
| SGLang 0.5.20 Engine, rslora | 같음(B=0) | **resolved**: scaling 8.0 (= 32/sqrt(16)), 스케줄러 프로세스에서 |

`broken`·`refused` 0. 비용: 경계당 첫 판정 27.5 ms(SGLang, 표 적재 포함)·71.8 ms(vLLM), 그 뒤 8.5·10.8 ms.

**측정이 드러낸 오탐과 수정:** 첫 실행에서 어댑터 넷 모두가 `broken`이었다 — PEFT 0.21.0은 `use_bdlora: null`(과 `arrow_config: null` 등)을 모든 파일에 쓰는데, 표의 중립값이 `false`라 `null ≠ false`를 "선언됨"으로 봤다(검토 에이전트가 미리 경고한 "있으면 쏘는" 오탐의 한 형태). 수정: `null`은 키를 막론하고 아무것도 선언하지 않는다(`_neutral`). 단위 시험의 기본값 묶음을 PEFT가 실제로 쓴 파일로 바꾸고, 모든 키를 `null`로 둔 경우를 회귀 시험에 넣었다.

**시험:** 47 파일 423 검사(`bash tests/run_all.sh`, `~/venvs/ci`). 새 시험 `tests/test_adapter_config.py` 8건; `test_adapter_rules`의 v2 목록과 허용 import에 `sglang_lora`, `vllm_lora`, `adapter_config_contract`.

**남은 것(M17.5로):** rsLoRA 해소의 출력 수준 확인(B≠0인 어댑터로 SGLang 출력 대 PEFT 참조), `entail check`의 어댑터 폴더 정적 판정을 인기 LoRA 폴더들로 훑기(오탐 자).

## M17.1b 선언-소비 대응, 요청 설정의 이름 (`chat_template_kwargs`와 추론 파서)

**무엇:** 한 설정이 여러 이름으로 오간다는 것(`thinking`, `enable_thinking`, `thinking_mode`)을 데이터로 두고(`entail/data/request_settings.json`의 group), vLLM 판별로 추론 파서마다 읽는 이름을 코드 줄 근거로 적었다(0.30.0: kimi_k2·kimi_k3·deepseek_v3/v4/v41·glm47_moe·ling3는 두 이름, qwen3·gemma4·nemotron_v3는 `enable_thinking`, minimax_m3는 `thinking_mode`; 0.22.0 행은 회고용: kimi_k2가 `thinking`만). 규칙(`request_contract.setting_names`): 요청이 준 이름을 **템플릿이 읽었고** 파서는 같은 설정을 다른 이름으로 읽으면 `broken`, 어댑터가 그 값을 파서가 읽는 이름으로 넘기면 `resolved`; 템플릿이 읽지 않은 이름(기존 설정 규칙이 말함)이나 그 설정을 읽지 않는 파서에서는 판정하지 않는다. 자리: 서버가 요청마다 만드는 파서 객체 — `vllm_serve`가 `ParserManager.get_parser`가 돌려주는 클래스를 감싸 생성 인자의 `chat_template_kwargs`에서 결정하고 같은 dict에 값을 쓴다.

**회고(규칙 수준, 검출률 아님):** vllm#43728 — 0.22.0 행(kimi_k2는 `thinking`)에 `{enable_thinking: false}`와 Kimi 템플릿의 변수(`enable_thinking`)를 주면 `resolved`(`thinking=False`가 파서에 넘어감); 0.30.0 행(두 이름)에서는 `pass`. 옛 판의 어댑터는 만들지 않았다(어댑터는 시험한 판에 붙는다).

**실제 서버(오탐 자):** M5.3의 요청 경계 하니스(`testbed/m53_serve.py healthy on`, vLLM 0.30.0 + Qwen3-4B + `--reasoning-parser qwen3`, 요청 10종: 생각 켬·끔·effort·도구·심은 결함 넷)를 entail 켜고 다시 돌림(`testbed/results/m17/serve/m53/healthy_on.*`). `request:vllm.parser_settings`: 검사 14, `pass` 14, 건너뜀 4(생각 설정이 없는 요청), `broken` 0. 기존 판정(심은 결함 넷의 `broken`)은 전과 같다.

**시험:** `tests/test_request_settings.py` 5건(표, 규칙, 회고, 무판정 조건, 어댑터 감싸기). 전체 48 파일 428 검사.

## M17.2 컨테이너 키의 완전성 (vLLM 접두 캐시 해시, transformers 빔 재정렬)

**무엇:** 같은 선언-소비 표(`entail/data/cache_key_fields.json`)로 두 소비자를 적었다. (1) vLLM 0.30.0의 블록 해시: 요청의 모델 입력 필드 여섯(`prompt_token_ids, prompt_embeds, prompt_is_token_ids, mm_features, lora_request, cache_salt`; `v1/request.py` 줄 근거)이 선언, 해시 함수가 읽는 다섯(`kv_cache_utils.py generate_block_hash_extra_keys` L600-636 + `hash_block_tokens`)이 소비 — `prompt_is_token_ids`만 빠져 있다(수정 PR #56656 미병합). (2) transformers의 빔 재정렬: 모델 forward의 캐시 인자 이름이 선언, 재정렬이 만지는 이름이 소비(5.17.0: `ALL_CACHE_NAMES` 다섯; 5.12.1 행: `past_key_values`만). 규칙(`entail/cache_key_contract.py`, `cache_key_incomplete`): 선언된 필드가 키에 없으면 `broken`, 어댑터가 키를 늘릴 수 있으면 `resolved`. 어댑터: `vllm_cache_key`(`Request.__init__`; 해소 = `generate_block_hash_extra_keys`를 한 번 감싸 블록마다 마스크의 digest를 extra key로 더하고 그 요청의 해시를 다시 만듦), `transformers_beam`(`GenerationMixin._beam_search`; 해소 없음, 보고만).

**회고(검출률 아님):**

| 사례 | 판 | entail 끔 | entail 켬 |
|---|---|---|---|
| vllm#56655 (`vl56655`) | 0.30.0 | B after A: 32토큰 재사용, A의 출력 `[753]*16` | `resolved`(`extend_key_prompt_is_token_ids`); B after A: 재사용 0, B 자신의 출력 `[9,0]*8`; A after A는 여전히 32토큰 적중(같은 요청의 반복은 살아 있음) |
| transformers#46612 (`tf46612`) | 5.12.1 | 캐시 있는 빔 ≠ 없는 빔 | `broken`(reported) at `request:transformers.generate.beam_reorder`: "cache_params ... the key does not cover it"; 출력은 그대로(해소 없음) |
| 같은 사례 | 5.17.0 (`tf46612_517`) | — | `pass`(재정렬이 모든 이름을 만짐) |

**오탐 자(실제 vLLM):** `vl49449`(스트리밍 세션 + 접두 캐시 + cache_salt, SmolLM2): 새 경계 `pass` 2, M14의 `identity_recompute` 해소 1은 전과 같음; `lora_vllm_default`(LoRA + 접두 캐시): `pass` 2. `broken` 0.

**시험:** `tests/test_cache_key.py` 4건. 전체 49 파일 432 검사.

## M17.3 커널 호출 계약 (Triton, 엔진 공통)

**무엇:** 규칙 (b)를 판정 의미대로 핵심에 뒀다(`entail/kernel_launch_contract.py`, `kernel_stride_assumed`): 텐서 인자의 크기>1인 안쪽 차원의 stride가 1이 아닐 때, 커널 서명에 `stride`가 든 이름의 인자가 하나도 없으면 `broken`(커널이 알 길이 없음), 있는데 그 값이 정수 인자 어디에도 없으면 `unknown`("비교 못 함", 한 번), 정수 인자에 그 stride가 있으면 `pass`; stride 0(expand 뷰)과 크기 1 차원은 대상 밖. 자리는 엔진 공통 훅 하나(`entail/adapters/triton_launch.py`, `triton.runtime.jit.JITFunction.run`; Triton 3.7·3.8): 인자 이름은 커널 자신의 서명(`JITFunction.params`)에서 오므로 커널을 몰라도 호출 인자를 이름에 묶는다. 경계는 `kernel:<엔진>.<커널 이름>`(엔진은 커널의 모듈 이름에서). 워밍업(autotune) 호출은 보지 않고, (커널, 텐서 배치) 조합마다 한 번, 커널마다 처음 8개 배치까지만 본다(아래 비용). 원칙 6(그래프 캡처 중 검사 안 함)과의 관계: 캡처 때의 호출은 그 배치의 첫 호출이라 재생 전에 한 번 판정되고, 재생에는 파이썬이 없다.

**회고(검출률 아님):** sglang#21843의 텐서(`testbed/m16/cases/sg21843.py`; SGLang 0.5.20의 `fused_gdn_gating_kernel`에 stride (32, 2)인 a·b를 넘김)를 entail 켜고 다시 돌림(`testbed/results/m17/retro/sg21843_{off,on}.json`): 재현됨(비연속 대 연속 출력의 최대 차 g 4.74, beta 0.78), entail은 미리 적은 정의대로 **`unknown`** 둘 — "a is strided in its innermost dimension (dim 1, stride 2); fused_gdn_gating_kernel takes stride arguments (stride_a, stride_b) but none of its integer arguments equals 2: whether it knows cannot be told from the launch". 커널이 행 stride만 받으므로 `broken`이 아니라 `unknown`이다(판정 의미대로; unknown_only).

**오탐 자(실제 엔진):** 오늘의 vLLM 0.30.0 실행(S4 하니스 셋, 짝짓기 확인, 회고들; `entail_logs/record-2026-09-26.jsonl`)에서 Triton 커널 경계 19개(`_gumbel_sample_kernel`, `_apply_write_kernel`, `_compute_slot_mappings_kernel`, …), 판정 전부 `pass`, `broken`·`unknown`·`refused` 0. SGLang 0.5.20의 LoRA 실행(`retro/lora_sglang_default_on.record.jsonl`) 커널 5개 `pass` 17. 즉 정상 실행에서 이 규칙이 말한 것은 없다.

**비용(S4, `testbed/m55_graph.py`: vLLM 0.30.0 CUDA Graph 경로, Qwen3-4B, 64→128토큰, 8라운드 on/off/control 회전, 중앙값 on/off):**

| 실행 | B=1 | B=8 | B=32 | control/off (B=1/8/32) | 파일 |
|---|---|---|---|---|---|
| 모든 어댑터, 첫 측정(커널당 상한 없음) | | | 1.048 | | 상한을 넣은 뒤 다시 재서 덮임(`triton_launch.py` 머리말에 남김) |
| 모든 어댑터, 커널당 8배치 상한 | 1.0150 | 1.0098 | 1.0616 | 1.0025 / 0.9945 / 0.9908 | `results/m17/m55/graph.json` |
| `triton_launch` 뺌(`ENTAIL_SKIP`) | 1.0039 | 1.0178 | 1.0447 | 0.9998 / 0.9994 / 0.9951 | `m55_skip_triton/m55/graph.json` |
| `vllm_cache_key` 뺌 | 1.0009 | 1.0017 | 1.0036 | 0.9973 / 0.9990 / 1.0036 | `m55_skip_cachekey/m55/graph.json` |
| 모든 어댑터, 기록 파일을 열어 둔 뒤 | **1.0054** | **1.0086** | **0.9994** | 0.9944 / 1.0033 / 1.0032 | `m55_v2/m55/graph.json` |

세 배치 모두 2% 안이고 B=32는 잡음 안이다(control과 같은 크기). 출력은 켜고 끄고 같다(`same_tokens_on_off` true, 세 배치).

**원인(귀속):** Triton 훅이 아니라 요청마다 도는 `vllm_cache_key`였고, 그 안에서도 판정이 아니라 **기록 줄 쓰기**였다. `load.safely`는 경계마다 timing 한 줄을 기록 파일에 쓰는데, 줄마다 파일을 열고 닫았다. 이 프로젝트는 WSL에서 `/mnt/c`(9P) 위에 있어 한 번에 4.6 ms다(`scratchpad/bench_record.py`: 9P 4.633 ms, ext4 0.005 ms; 판정 경로 자체는 0.021 ms, 기록 없는 `safely` 0.008 ms). 요청 32개 × 4.6 ms ≈ 147 ms가 2.5초 라운드의 6%다. 기록의 timing 줄 1,998개 가운데 판정 시간 중앙값은 0.075 ms였다(p99 4.2 ms).

**수정(`entail/record.py`):** 프로세스마다 기록·로그 파일을 열어 두고 줄마다 쓰고 flush한다(9P에서 0.176 ms, ext4 0.004 ms; `bench_append.py`). fork한 자식은 물려받은 핸들을 버리고 자기 것을 append 모드로 연다(같은 파일의 끝에 붙음). 파일은 프로세스당 8개까지(날짜가 바뀌면 새 파일), `atexit`에 닫는다. 시험 `tests/test_logs.py`에 한 건 추가(파일당 핸들 하나, 줄마다 바로 보임, 닫으면 다시 열림). Triton 훅의 커널당 상한(8배치)은 그대로 둔다: 커널 수백 개 × 배치가 늘어나는 서버에서 stride 읽기가 정상 상태에 남지 않게 하는 상한이다.

**시험:** `tests/test_kernel_launch.py` 4건.

## M17.4 `Rotary.pairing` (어휘 v8)

**무엇:** 회전 임베딩이 회전하는 차원을 짝짓는 방식(`split`: i와 i+d/2, rotate_half; `interleaved`: 2i와 2i+1, GPT-J)을 `Rotary` 사실의 필드 `pairing`으로 더했다(`entail/facts.py` v8, `ADDED_IN[("Rotary","pairing")] = 8`; 판별 `READABLE_VERSIONS` 1~8). 선언의 출처는 셋이고 이 순서다(`entail/rotary_pairing_contract.declared`): (1) `Rotary` 사실이 `pairing`을 나름(`readers.read_hf_dict`가 config 키 `rope_interleave`·`rope_interleaved`·`is_neox_style`을 읽음; `aliases.json`), (2) config.json 또는 `text_config`의 그 키, (3) 아키텍처 표(`entail/data/rotary_pairing.json`: transformers 5.17.0 참조 구현의 파일·줄 근거; glm4·glm·glm4v·glm_ocr·ernie4_5·cohere·cohere2·gptj = interleaved, llama·qwen2·qwen3·gemma3·mllama·deepseek_v3 = split). 표에 없는 아키텍처에 키도 없으면 선언이 없어 판정하지 않는다(entail이 관례를 지어내지 않음). 규칙 둘: `rotary_pairing_mismatch`(만들어진 층이 선언과 다르게 짝지음; 어댑터가 층의 관례를 놓을 수 있으면 `resolved`, 아니면 `broken`), `rotary_pairing_ignored`(엔진의 커널 경로가 층과 무관하게 split로 짝지음; 표의 `kernels` 행, 판별로: vLLM의 Triton MRoPE 커널은 0.27.0 전까지(#49906) — interleaved를 선언한 MRoPE 모델에서 `broken`, 놓을 것이 없음). 어댑터 `vllm_pairing`(`process_weights_after_loading` 훅; 층의 `is_neox_style`을 읽고 놓음; `is_neox_style=True`가 split).

**회고(규칙 수준, 검출률 아님):** vllm#42016(GLM-OCR, vLLM 0.22.0의 MRoPE 커널): 표의 커널 행으로 0.22.0에서 **`broken`**(`rotary_pairing_ignored`), 0.30.0에서는 `pass`(`tests/test_rotary_pairing.py`). 옛 판의 어댑터는 만들지 않았다.

**실제 엔진(vLLM 0.30.0, `testbed/m17/pairing_vllm.py`, `results/m17/pairing/`):** 첫 실행이 오탐이자 해로운 해소였다 — GLM-OCR(`zai-org/GLM-OCR`)에서 "26 layer(s) pair split"로 `resolved`가 나서 26개 모듈을 interleaved로 바꿨다(`run1_glm_ocr_on.json`, `run1_glm_ocr_record.jsonl`). 그 26개는 언어 모델이 아니라 **비전 타워**의 모듈이었다(vLLM `glm_ocr.py` L287 `is_neox_style=True`와 24개 블록의 `ApplyRotaryEmb`; 참조 구현도 비전 타워는 split이다 — 표의 근거 줄 자체가 "L335 rotate_half (split) is the vision tower's"라고 적고 있었다). 언어 모델의 MRoPE 모듈 둘은 선언대로 interleaved였다. 수정: 표의 아키텍처 행은 텍스트 모델의 관례이므로 어댑터는 **언어 모델의 모듈만** 읽고 놓는다(vLLM `SupportsMultiModal.get_language_model`; 없으면 모델 전체). 비전 타워 등 다른 부분은 대조하지 않는다(빈틈으로 적음). 단위 시험에 이 모양을 넣었다(언어 모델 둘 interleaved + 비전 셋 split → `pass`, 비전은 손대지 않음; 언어 모델이 split이면 그 둘만 놓음). 두 번째 실행(언어 모델만, `testbed/m17/pairing_run2.sh`):

| 실행 | 층의 관례(모델 전체, 실행 뒤) | `load:vllm.rotary_pairing` | 출력(8토큰, greedy) | 경계 비용 |
|---|---|---|---|---|
| GLM-OCR, entail 끔 | split 26, interleaved 2 | — | " the city of Paris. Paris is the" | — |
| GLM-OCR, entail 켬 | split 26, interleaved 2 (그대로) | **pass** "2 rotary layer(s) pair interleaved, as declared" | 같음 | 2.6 ms |
| Qwen3-0.6B, entail 켬 | split 2 | **pass** "2 rotary layer(s) pair split, as declared" | " Paris. The capital of Italy is Rome" | 2.6·3.3 ms |
| Qwen3-4B, S4 하니스(`m55_v2`) | — | pass (checks 1, 두 규칙 passed) | on = off | — |

`broken`·`refused` 0. 첫 실행의 파일은 `run1_*`로 남겼다.

**시험:** `tests/test_rotary_pairing.py` 5건(선언의 출처 순서, 커널 행의 판별, vllm#42016 회고와 층 대조, 어댑터, 멀티모달의 언어 모델만). 전체 **51 파일 442 검사**(`bash tests/run_all.sh`, `~/venvs/ci`). `tests/test_load.py`의 기대 하나를 바꿨다: SmolLM2의 `rope_interleaved`는 v7까지 "읽지 않는 키"(`unknown`)였고 v8부터는 `Rotary.pairing`의 별칭이라 짝짓기 자리에서 대조된다(`pass`, "compared where a consumer of the fact reads it").

**검토 반영(2026-09-26, 읽기 전용 검토 에이전트, M17.3·M17.4 한 번; 지적 22건 가운데 코드 9건·문서 6건 반영):**
- **[결함, 높음] DeepSeek-V3 행이 틀렸다.** 표는 `split`이었으나 transformers 5.17의 `DeepseekV3Config.rope_interleave` 기본값은 `True`이고(`configuration_deepseek_v3.py` L114) 모델링은 그때 interleave를 적용한다(L464); vLLM도 interleaved로 만든다. 오늘까지 살아남은 것은 `hf_config.to_dict()`가 클래스 기본값을 실어 키 경로로 판정됐기 때문이고, `--trust-remote-code`로 저장소의 `configuration_deepseek.py`(그 키 없음)를 쓰면 표로 떨어져 언어 모델 전체를 split로 뒤집었을 것이다(GLM-OCR 1차 실행과 같은 모양, 주력 모델에서). 행을 `interleaved`로 고치고 근거를 적었다. 어떤 실행에서도 나기 전에 검토가 잡았다.
- **[위험, 중간] stride 이름 휴리스틱.** `"stride" in 이름`만 보면 `q_s0`(vLLM sparse indexer), `sxm`(mxfp8), `a_s0`·`sq_d`(SGLang)처럼 다른 이름으로 stride를 받는 커널에 거짓 `broken`이 난다(검토가 AST로 vLLM 690·SGLang 715 커널을 훑어 찾음; 오늘 실행에서 안 난 것은 그 호출 자리가 지금은 연속화하기 때문). 규칙을 미리 적은 의미 안에서 좁혔다: 정수 값 인자가 안쪽 stride와 같으면 이름과 무관하게 `pass`(told), stride류 이름(`stride`, `_s0`, `s`+글자 1~2, `s_`, `ld`)이 있으면 `unknown`, 둘 다 없을 때만 `broken`. 정수는 커널의 값 인자(constexpr·launch option 제외)에서만 센다(BLOCK=2나 num_warps=2가 "told"로 새지 않게).
- **[위험, 중간] 배치 상한이 모양으로 소진됨.** 메모 키에 shape가 들어 있어 디코드 모양 8개면 커널이 더는 보이지 않았다(`_apply_write_kernel`이 짧은 GLM-OCR 실행에서 이미 8). 키를 shape 없이(텐서마다 rank와 안쪽 stride) 바꾸고 상한은 strided 패턴만 센다: 연속 텐서의 모양 20개는 패턴 하나, 새 모양의 strided 텐서는 여전히 보인다.
- **[위험] 훅 서명·워밍업 문구.** `run(self, *args, **kwargs)`로 통과시켜 Triton 서명이 바뀌어도 엔진을 깨지 않게 했다. "워밍업은 보지 않는다"는 compile-only `warmup=True`만 뜻하고 autotuner의 벤치마크 호출은 보통 호출로 본다(문구 수정).
- **[위험, 중간] 선언 하나 대 모듈 여럿.** DSA 모델은 indexer가 자기 키(`indexer_rope_interleave`)로 짝지으므로 `indexer` 모듈은 대조에서 빼고, 남은 모듈이 두 관례를 다 들면 `unknown`으로 알리고 아무것도 놓지 않는다(뒤집으면 자기 키로 짝짓는 것까지 옮김).
- **[위험] 언어 모델을 못 찾을 때.** `get_language_model`이 실패하면 모델 전체로 넓히지 않고(1차 실행의 실패 모양) `unknown`("언어 모델을 찾지 못함")으로 알린다.
- **[위험, 중간] 커널 행의 조건.** 0.22~0.26의 MRoPE 커널 경로는 CustomOp이 켜질 때(enforce_eager, ROCm, `+rotary_embedding`)만 돌고, 기본 컴파일 경로의 `forward_native`는 층을 따른다. 어댑터가 MRoPE 모듈의 실제 디스패치(`_forward_method`)를 읽어 `forward_native`가 아닐 때만 커널 행을 적용한다. M16 사례(vl42016)는 `enforce_eager=True`였으므로 `broken`은 그대로 맞고, 이슈 #42016은 MI300(ROCm) 보고다.
- **[문서] 판정 문장.** Coverage 자리표 대신 `Rotary(pairing=…)` 사실 둘을 `agrees()`로 비교해 판정 줄이 두 관례를 그대로 보인다("declared Rotary(pairing=interleaved) …; uses Rotary(pairing=split)"). "N layer(s)"는 인스턴스 수라 "module(s)"로 바꿨다(vLLM은 `get_rope`로 인스턴스를 공유하므로 GLM-OCR의 층 16개가 인스턴스 2개다).
- **[정직] M17.4의 판정 의미는 구현 전에 적지 않았다.** ROADMAP M17의 사전 의미는 (a)(b)(c)에 관한 것이고 짝짓기 규칙 둘은 구현 뒤 체크 항목에 처음 나온다. 이 문서 첫 줄의 "자는 결과 전에 적음"은 (a)(b)(c)와 오탐·비용·검출률 자에만 해당한다.
- **[문서] `rope_interleaved`의 Coverage.** v8에서 짝짓기 키가 어휘 키가 되자 transformers·SGLang에서도 "다른 경계에서 대조됨"(pass)이 됐는데 그 엔진들에는 짝짓기 소비자가 없다. 엔진별 소비자 표(`load.COMPARED_BY`)로 vLLM에서만 pass, 나머지는 v7처럼 unread(unknown)로 되돌렸다(정적 훑기의 transformers·SGLang Coverage pass는 98로 돌아간다).
- **[문서] 기록 파일.** 쓰기+flush에 잠금, 실패한 파일은 닫고 다시 열지 않음(줄마다 4.6 ms가 조용히 돌아오지 않게).
- 반영하지 않은 것: `set_pairing`이 `get_rope`가 공유하는 인스턴스를 바꾸는 것(vLLM 자신의 모델들도 같은 방식으로 바꿈; 새 인스턴스로 갈아 끼우는 방식은 어댑터를 두껍게 함 — 빈틈으로 적음).

**남은 것(M17.5로, 짝짓기):** 멀티모달 모델의 비전 타워는 대조하지 않는다(참조 구현마다 다르고 표가 없음; README 빈틈에 적음). 정적 검사(`entail check`)는 키로 선언된 `pairing`만 `Rotary` 사실에 싣고 아키텍처 표는 어댑터만 본다. transformers·SGLang에는 짝짓기 어댑터가 없다(각 참조 구현이 곧 소비자인 transformers는 대조할 것이 없고, SGLang은 `is_neox_style` 상당을 아직 읽지 않았다).

## M17.5 측정: 오탐, 자리 덮임, 비용, 동결 (진행 중)

자는 ROADMAP M17에 결과 전에 적은 그대로다. 수치는 결과 파일에서만 옮긴다.

| 자 | 정의 | 결과 | 파일 |
|---|---|---|---|
| 정상 실행 오탐(E2) | 인기 모델 38개 × transformers·vLLM·SGLang, entail 켬, `broken`·`refused` 0이어야 통과; 출력은 M10/M11의 entail 끈 실행과 비교 | **1차(커밋 `99a4841`): 통과.** 110회 가운데 유효 102(제외 8은 1.1.0 때와 같은 환경 실패: transformers의 FP8·compressed-tensors 커널 없음, vLLM의 작은 시험 모델 엔진 초기화 실패), entail이 깨뜨린 실행 0, `broken`·`refused` 0, 해소 4(1.1.0 최종과 같음: softcap 2, add_stops 2), 해소 없는 실행의 출력 같음 97/98, 적재 비중 중앙값 **1.3%** · p90 6.5% · 최대 41.9%. `unknown` 79(1.1.0 최종 75) — 늘어난 4줄은 모두 Nemotron-3-Nano-4B의 causal_conv1d 커널(SGLang `_causal_conv1d_fwd_kernel` 2, vLLM `_causal_conv1d_update_kernel` 2): 채널 우선 배치(x (9728, 11) stride (1, 17504))의 stride를 커널이 `stride_x_token: tl.constexpr`로 받는데, 검토 반영이 constexpr를 정수 인자에서 모두 빼서 "told"를 놓쳤다. **수정:** stride류 이름의 constexpr는 센다(`triton_launch.told_params`). **2차(수정 커밋 `ce79b19`): 통과.** 유효 102, `broken`·`refused` 0, 해소 4, 출력 같음 97/98, `unknown` 69(유효 실행 기준; 기록 전체 75) — 1.1.0 최종과 경계별로 **모두 같음**(새 경계 다섯 어느 것도 정상 실행에서 말한 것이 없음), 적재 비중 중앙값 **1.4%** · p90 8.2% · 최대 41.2%. sg21843 회고도 같음(`unknown` 2) | `testbed/results/m17/e2_99a4841/`(1차), `testbed/results/m17/e2/`(2차), `testbed/m17_e2_breakdown.py` |
| 정적 오탐(E1) | 인기 폴더 230개, 엔진 셋 정적 검사(`entail check` 경로), `broken` 0(T5 tie 2건은 1.0.1부터 알려진 것) | **통과.** 커밋 `b352cc5`, 400 s. `broken`은 ModelProps의 T5 tie 2건(vlt5-base-keywords, xflux_text_encoders)뿐, 세 엔진 같음. 1.1.0 최종과 다른 것은 Coverage뿐: pass 98 → **102**, unknown 123 → 119 — SmolLM2 폴더 넷의 `rope_interleaved`(=false, split)가 v8부터 `Rotary.pairing`의 별칭이라 짝짓기 자리에서 대조됨. Rotary pass 196/unknown 6, Vocab 214/16, Stops resolved transformers 9·vLLM 2·SGLang 0은 그대로. **검토 반영 커밋(`99a4841`)에서 다시:** 같되 Coverage가 엔진별로 갈림 — vLLM pass 102/unknown 119, transformers·SGLang 98/123(짝짓기 키는 vLLM에서만 대조되므로 그 엔진들에서는 v7처럼 unread), `broken`은 T5 tie 2건뿐, 397 s. 정적 검사는 Triton 훅을 거치지 않으므로 `ce79b19`에도 그대로 적용된다 | `testbed/results/m10/e1_llm/static_1.2.0-v8.json`, `static_1.2.0-v8b.json` |
| 부류 밖 오탐(M16) | 재현된 부류 밖 8건을 v8 커밋에서 다시 돌려 `broken`·`refused` 0 | **통과.** 8건(tf47885, tf48293, df13411, tf46032, tf46710, vl49316, vl49412, vl48231) 모두 `broken`·`refused` 0; pass 아닌 판정은 `unknown`뿐(config의 읽지 않는 키, vl48231의 양자화 경계 1) — M16과 같음 | `testbed/results/m17/m16_rerun/summary.json` |
| 경계 판정 3건 회고 | 심사에서 경계 판정으로 뺀 transformers#45698(커스텀 모듈 적재 우선순위), #47752(generation_config 우선순위), diffusers#14451(파이프라인 클래스) — 재현 스크립트가 없어 규칙 수준: M17의 어느 규칙도 이 셋을 읽지 않는다(선언 파일 표·커널 stride·컨테이너 키·짝짓기 모두 무관). 판정 없음 = 오탐 없음, 검출도 없음 | 규칙 수준 | `results/m16/screening.json` |
| 자리 덮임 | 재현된 부류 안 7건 가운데 기대 경계에 판정이 하나라도 있는 비율(M16: 2/7) | **5/7** (sg40835, sg21843, tf46612, vl56655, vl42016_0220; 없음: vl48895(규칙 밖), vl43728_0220(어댑터 미설치, 아래)) | `m16_rerun/summary.json` |
| 회고(검출률 아님) | 부류 안 7건의 판정(resolved / reported / unknown_only / adapter_not_installed / missed), 보고된 판의 환경에서 끝까지 돌린 것 | **resolved 2**(sg40835: SGLang scaling 2.0 → 16.0; vl56655: B after A 재사용 0) · **reported 2**(tf46612: transformers 5.12.1 빔 경계 `broken`; **vl42016: vLLM 0.22.0 실제 실행에서 `broken at load:vllm.rotary_pairing`** — "vllm.mrope_triton (vllm 0.22.0) pairs split-wise regardless of the layer", 출력은 그대로 쓰레기(해소 없음), 실행은 이어짐) · **unknown_only 1**(sg21843) · **adapter_not_installed 1**(vl43728_0220: `vllm_serve`는 0.30의 `ParserManager`를 감싸는데 0.22.0은 `ReasoningParserManager`이고 사례가 파서를 프로세스 안에서 만들어 자리에 닿지 않음; 규칙 수준으로는 resolved) · **missed 1**(vl48895, 어느 규칙에도 들지 않음) | 같은 파일, `m16_rerun/cases/*_on.record.jsonl` |
| 비용 S4 | vLLM CUDA Graph 경로 on/off 중앙값, B=1/8/32 ≤ 1.02 | 기록 수정 뒤(`b352cc5`) **1.0054 / 1.0086 / 0.9994**; 검토 반영 커밋(`99a4841`, shape 없는 메모 키) **1.0076 / 1.0029 / 1.0061**, control 1.0041 / 0.9971 / 1.0005, 출력 같음 | `m55_v2/m55/graph.json`, `m55_v3/m55/graph.json` |
| 검토가 바꾼 판정의 재확인(`99a4841`) | 검토 반영이 회고·실제 엔진 판정을 바꾸지 않았는지 | sg21843 `unknown` 2(같음); vl42016 vLLM 0.22.0 `broken`(같음; 판정 줄이 이제 `Rotary(pairing='interleaved')` 대 커널 경로를 보임, `enforce_eager`라 MRoPE 모듈이 `forward_cuda`로 디스패치); GLM-OCR vLLM 0.30 Vocab·Stops·짝짓기 `pass`(같음) | `m16_rerun_v8b/cases/`, `pairing/glm_ocr_v8b_*` |
| 비용 적재 비중 | E2 실행의 entail 시간 / 적재 시간, 중앙값·p90 | (측정 중) | `e2/E2_SUMMARY.md` |
| 사전 점검 (1) 허브 id | 허브 id로 적재한 모델의 Vocab·Stops가 "로컬 폴더 없음"으로 판정을 못 내는 것 | **원인과 수정.** 캐시가 없어서가 아니었다: huggingface_hub 1.32/1.33의 `snapshot_download(local_files_only=True)`가 엔진이 가져오지 않은 파일(`.gitattributes`, `.eval_results/*`)이 빠진 스냅샷을 `IncompleteSnapshotError`로 거부한다(GLM-OCR 탐침 `scratchpad/hub_cache_probe.py`; `try_to_load_from_cache(config.json)`는 그 스냅샷 폴더를 돌려줌). `load.local_folder`가 거부되면 캐시된 config.json의 폴더로 되짚는다(내려받기 없음). 단위 시험 1건(가짜 캐시 폴더). **실제 엔진 확인:** vLLM 0.30.0에 `zai-org/GLM-OCR`을 허브 id로 적재(커밋 `95b4377`, `results/m17/pairing/glm_ocr_hubid_*`): `load:transformers.tokenizer` Vocab **pass**, `load:vllm.input_processor` Stops **pass**, 짝짓기 pass — "no local folder to read" 줄이 사라짐 | `entail/load.py local_folder`, `tests/test_load.py` |
| 사전 점검 (2) 옛 판 어댑터 | 옛 판 환경에서 어댑터 설치 실패를 기록하고 따로 셈 | **기록.** vLLM 0.22.0: `could not install entail.adapters.vllm_scoring`(partially initialized module; M16과 같음), `vllm_serve`의 자리(`ParserManager`) 없음 → vl43728_0220은 adapter_not_installed로 셈. 그 밖의 어댑터(짝짓기, 캐시 키, Triton, LoRA)는 0.22.0에 설치됨(vl42016_0220의 `broken`이 그 증거) | `m16_rerun/cases/entail_logs/entail-2026-09-26.log` |

**M17.5 판정: 통과.** 자 셋(정상 실행 오탐 0, 정적 오탐 0, 부류 밖 오탐 0)과 비용(S4 ≤ 2%, 적재 비중 중앙값 1.4%)이 통과했고, 자리 덮임은 2/7 → 5/7이다. 어휘 v8은 커밋 **`ce79b19`**(2026-09-26)에 동결한다(`testbed/M16_PROTOCOL.md` 7절에 적음). 동결 뒤의 커밋은 문서만 바꾼다. 검토 에이전트의 지적 22건 가운데 코드 반영 10건(DeepSeek-V3 행 포함)과 그 반영이 낸 회귀 1건(constexpr stride)까지 이 커밋에 들어 있다. 다음은 M17.6(2차 사전 등록 재현, 151번부터)이다.

## M17.6 2차 사전 등록 재현 (어휘 v8 = `ce79b19` 동결; 규약 `testbed/M16_PROTOCOL.md` 7절; 끝, 2026-09-26: 검출 0/8)

**심사(M16 규약 3절 그대로, 순서 151~300, 건너뜀 없음, 세션이 함):** 150건 심사에서 멈춤(통과 30건에 못 미침). `pass` **20**, `not_output` 64, `cannot_run` 63, `cannot_install_version` 1, `no_repro_info` 2 (`testbed/results/m17/replay2/screening.json`; 결정 파일 `decisions_*.json`). 심사 뒤 고친 판정 둘을 기록에 남겼다: sglang#28180(재현 절차는 모델을 열어 두지만 환경 절의 모델이 Qwen3.6-27B)과 transformers#48256(스크립트의 모델이 Nemotron-Nano-9B-v2)은 규칙 2로 `cannot_run`. 경계 판정(borderline)으로 뺀 것은 1차의 판례(증거가 모델 출력이 아닌 것: 토큰 id·자리표 인덱스·클래스 선택·코드 비교)를 따랐고 이유에 적었다: vllm#52924, #46817(토큰 id), #39532(config 클래스), transformers#46664(num_frames의 뜻), #48310(past_key_values 이름), diffusers#13546(ERNIE-Image RoPE 짝짓기 질문), transformers#45910(DeepSeek V4 RoPE theta 질문). 이 가운데 짝짓기·이름·뜻 세 건은 부류 안으로 보이는 선언 질문이라 `results/m16/DEFERRED.md`에 적을 후보다.

**보고된 판 환경(규칙 4, 최대 6개):** `tf540`(transformers 5.4.0), `tf580`(5.8.0), `tf5102`(5.10.2), `vllm0240`(vLLM 0.24.0 + transformers 5.12.1), `vllm0230`(0.23.0; transformers 4.57.6가 딸려 옴), `vllm0111`(0.11.1). 기존의 `vllm0190`(0.19.0)도 쓴다. 여섯을 다 써서 transformers 5.6.0(Zamba2, #47475)은 설치하지 않았다.

**눈가림 평정 자료:** 규칙 1 통과 86건(`not_output`이 아닌 것)의 `rating_packet.{json,md}`(시드 20260928, 1차와 다른 순서, 심사 메모 없음). 평정자는 규약대로 가능하면 세션 밖에서 연구자가 돌린다. 세션 안 에이전트로 돌리려면 연구자 허락이 필요하다(에이전트 정책). 평정 전까지 검출률의 분모는 없다.

**재현(사례 스크립트 `testbed/m17/replay2/cases/`, 실행 `testbed/m10_e3/run_case.sh`, 결과 `results/m17/replay2/cases/`):** 진행 중. 판정은 `results/m17/replay2/verdicts.json`에 사례마다 적는다(resolved / reported / unknown_only / missed / not_reproduced; 정의는 1차와 같고 **회고가 아닌 검출**이다: 어휘·어댑터는 심사 시작 전에 동결됐다).

**재현 결과(2026-09-26, 평정 전; `results/m17/replay2/verdicts.json`, `M16_OUT=testbed/results/m17/replay2 python testbed/m16_results.py`):**

| 사례 | 판·환경 | 재현 | entail 켠 실행 | 판정 |
|---|---|---|---|---|
| vllm#58532 INT8 MoE 채널 스케일 | 0.30.0 | 재현(라우팅된 expert의 스케일 ×4에 출력 불변) | pass 1 | missed |
| vllm#48217 Gemma4 파서 스트리밍 | **0.24.0** | 재현(비스트리밍 content, 스트리밍 reasoning만) | 판정 없음 | missed |
| diffusers#13425 rescale_noise_cfg NaN | 0.40.0 | 재현(NaN 16384) | 판정 없음 | missed |
| vllm#38643 Qwen3.5-4B NVFP4 쓰레기 | 0.30.0 정상 → **0.19.0**(보고 nightly에 가장 가까운 판; 체크포인트는 보고에 이름이 없어 공개 내보내기 AxionML 사용) | 재현(다국어 토큰 죽) | unknown 4(config 키, NvFp4 재포장 서명 없음) | missed |
| vllm#56578 softcap NaN | 0.30.0 | 재현(NaN·inf) | pass 1 | missed |
| sglang#36938 prefill logprobs 어긋남 | 0.5.20 | 재현(수정 PR의 CPU 시험 그대로) | 판정 없음 | missed |
| vllm#52576 블록 FP8 K 타일 | 0.30.0 | 재현(BLOCK_SIZE_K 256/그룹 128: 상대 오차 0.75) | pass 1(Triton 호출) | missed |
| vllm#47986 DeepSeek V4 도구 파서 | 0.30.0 | 재현(tool_b가 tool_a의 스키마로 풀림) | 판정 없음 | missed |
| transformers#45356 Kimi-K2.5 토크나이저 | 5.17.0 정상 → **5.4.0** | 재현(163607 → "", `</think>` → 163604) | **unknown 2**: 토크나이저 경계에서 "엔진의 토크나이저는 163584 기본 토큰, 모델 어휘는 163840: 모델의 토크나이저가 아니고 누구의 것인지 말하는 것이 없음" | unknown_only |
| vllm#57740 `<|image_pad|>` 바인딩 | 0.30.0 | 재현(토큰 수준, 보고와 같은 위치; 답은 안 뒤집힘) | unknown 2(config 키) | missed |
| transformers#46489 deepseek-coder 토크나이저 | 5.17.0 정상 → **5.10.2** | 재현(LlamaTokenizer, 4/4 문자열 id 다름) | pass 2 | missed |
| transformers#47475 Zamba2 인과 위반 | 5.17.0 | **미재현**(위치 400 앞 차이 0.0); 5.6.0은 환경 상한으로 미설치 | — | not_reproduced |
| vllm#48831 리랭커 긴 입력 | **0.24.0** | **미재현**(8,674토큰: 오프라인 0.825, 서버 0.830·0.827·0.827) | unknown 2 | not_reproduced |
| transformers#45812 Granite 토크나이저 | 5.17.0 정상 → **5.8.0** | 재현(GPT2Tokenizer, 2/5 다름) | pass 2 | missed |
| vllm#40466 enable_thinking=False 스트리밍 | 0.30.0, **0.11.1** | **미재현**(둘 다 content로 옴) | pass | not_reproduced |
| vllm#41207 DeepSeek-OCR "Ġ" | 0.30.0(Triton 3.7이 원격 커널 거부), **0.19.0+tf 5.6.2**(원격 코드가 transformers 5에 없는 클래스 import) | **실행 불가** | — | not_reproduced |
| vllm#42047 Gemma4 도구 파서 실수 | **0.19.0** | 재현(108.2 → 108.02) | pass 2 | missed |
| sglang#25055 logprobs가 reasoning 포함 | 0.5.20 | 재현(155토큰, `<think>`부터) | pass 15 | missed |
| vllm#50427 FlexAttention int32 | torch 2.13 | 재현(-8.0) | 판정 없음(Inductor 커널) | missed |
| vllm#45734 hidden-states 커넥터 NaN | **0.23.0**(+tf 5.12.1) | **미재현**(저장된 hidden states 205×2560에 NaN 없음; CPU 참조와 코사인 중앙값 0.97) | 모든 어댑터 켜면 **`vllm_serve` 감싸기가 0.23.0의 `is_harmony` 키워드를 못 받아 API 서버가 죽음(entail이 실행을 깨뜨림, DEFERRED 21)**; `ENTAIL_SKIP=vllm_serve`로 다시: unknown 3 + `container:vllm.allocate_slots` **broken 3**(KV 그룹 0·1·2, "512 slots for 205 tokens"; 기대 경계 아님, 분석은 DEFERRED 22) | not_reproduced |

**집계(재현 끝, 평정 전; `M16_OUT=testbed/results/m17/replay2 python testbed/m16_results.py`):** 통과 20 → 재현 **15**, 미재현 5(Zamba2·리랭커·0.11.1 스트리밍·DeepSeek-OCR 실행 불가·hidden-states). 재현 15건의 entail 판정: **missed 14, unknown_only 1, resolved·reported 0 — 검출 0.** `broken`은 2차 전체에서 한 실행(vl45734, vLLM 0.23.0)의 KV 할당 경계에서만 났고(3줄, 기대 경계 아님; 재현되지 않은 사례라 검출도 오탐 계산의 대상도 아니며 2차 뒤에 원인을 가린다), `refused`는 없다. **entail이 깨뜨린 실행 1**(같은 사례: 0.23.0에서 `vllm_serve` 감싸기의 서명 불일치 — 원칙 12 위반, 2차 뒤 수정). 자리 덮임(기대 경계에 판정 하나라도): 1/15(Kimi 토크나이저). 검출률의 분모(재현 ∩ 합의 K1)는 평정 뒤에 정해지며, 분자는 0이다(3의 법칙 상한 3/n).

**미리 보는 뜻(평정과 무관한 사실):** 재현된 15건 가운데 entail의 경계가 있는 자리는 토크나이저 셋(Kimi·deepseek-coder·Granite)뿐이고, 거기서도 Vocab 규칙은 크기를 비교하지 어떤 id를 내는지는 비교하지 않는다(Kimi만 크기 어긋남을 알아채 `unknown`). 나머지 12건은 파서의 상태 기계(3), 커널 안의 산술·스케일 배치(4), 스케줄러·응답 조립(2), 멀티모달 자리표 바인딩(1), 컴파일된 FlexAttention(1), 선형 어텐션 커널의 입력 배치(1)로, v8에 자리가 없다. M17.5의 회고 5/7과 달리 **검출은 0**이다. 이것이 사전 등록의 답이다.

**2차 뒤 수정(재현이 끝난 뒤, 기록된 결과는 `ce79b19` 그대로):** DEFERRED 21(`vllm_serve`의 `get_parser` 감싸기가 0.23.0의 키워드를 못 받아 서버가 죽음)은 커밋 `6b28513`에서 고쳤다 — 인자를 이름으로 묶고 나머지는 그대로 넘긴다(검토 지적 7과 같은 모양의 결함; 시험 51 파일 445 검사). DEFERRED 22(0.23.0의 하이브리드 모델에서 KV 할당 `broken` 3줄)는 탐침 둘로 가렸다: 같은 모델·판에서 추측 디코딩 없이 생성하면 나지 않고(`probe_alloc_0230.json`), **0.30.0의 ngram 추측 디코딩(Qwen3-0.6B)에서는 난다**(`probe_alloc_spec.json`: "holds 48 KV slots for 32 tokens, unit 16") — 엔진이 lookahead 슬롯을 미리 잡고 거부한 초안의 블록을 두는 정상 동작이라 **오탐 부류**다. 규칙을 "토큰보다 적게 잡으면 broken, 많이 잡는 것은 손실이 아님"으로 좁혔다(`kv_contract._rules`; E2에는 추측 디코딩이 없어 1.0~1.1의 오탐 자에 걸리지 않았다 — 오탐 자를 추측 디코딩까지 넓혀야 한다). 좁힌 뒤 같은 탐침: `probe_alloc_spec_after.json`.

**평정(2026-09-26, 연구자 허락 "에이전트 진행해" 뒤 세션 안 에이전트; 규약 7절은 세션 밖을 선호하므로 그 사실을 여기 적는다):** 눈가림 평정자 둘(A: Opus, B: Sonnet)이 코드북과 `rating_packet.md`(86건)만 받아 `ratings_A.txt`·`ratings_B.txt`를 썼다. 일치도(`m16_rate_score.py`): 7범주 카파 **0.718**(관찰 일치 0.791), K1 대 나머지 카파 **0.679**(0.860) — 1차(0.860/0.947)보다 낮다. B는 K1을 23건, K4(경로·상태 기계)를 20건으로 봐서 A(K1 31, K4 10)보다 경로 쪽으로 기울었다. 갈린 18건은 셋째 눈가림 평정자(C: Opus, 새 문맥)가 `rating_disagreements.md`만 보고 정했다(`ratings_C.txt`): 16건은 A와 같았고 B와 같은 것은 없었으며, 2건은 셋째 범주였다(vllm#48898 K3/K5→K7, sglang#26745 K6/K4→K1; 합의 `split`, 둘 다 재현 사례가 아니다). 합의: K1 **31/86(36%)**, K7 31, K4 10, K2 5, K5 5, K3 2, split 2(`ratings.json`).

**최종 집계(`M16_OUT=testbed/results/m17/replay2 python testbed/m16_results.py` → `results.json`):**

| 자 | 값 | 파일 |
|---|---|---|
| 심사 | 151~300번 150건 → 통과 20 | `screening.json` |
| 재현 | 15 / 20 (미재현 5) | `verdicts.json`, `cases/` |
| 분모(재현 ∩ 합의 K1) | **8**: vllm#58532, #38643, #47986, #57740, transformers#45356, #46489, #45812, sglang#25055 | `results.json` |
| **검출** | **0 / 8**(resolved 0, reported 0; 평정자별 A 0/8, B 0/4). 3의 법칙 상한 3/8 = 0.375 | `results.json` |
| 자리 덮임(기대 경계에 판정 하나라도) | 1 / 8(transformers#45356: 토크나이저 경계의 `unknown`) | `verdicts.json` |
| 부류 밖 오탐(재현 7건의 broken·refused) | 0 | `results.json` |
| entail이 깨뜨린 실행 | 1(vllm#45734, 미재현; `6b28513`에서 고침) | DEFERRED 21 |
| 부류 비중 | 합의 K1 31/86(A 31, B 23) | `ratings.json` |

**뜻:** 1차(0/7)와 같이 **0**이다. 부류 안 8건 가운데 셋은 같은 자리(토크나이저 경계)에 있고 entail의 경계가 거기 있다: 규칙이 크기만 비교해서 놓쳤다(하나는 `unknown`). 나머지 다섯은 v8에 자리가 없다(커널이 받는 스케일 배치, FLA 커널의 입력 배치, 도구 파서의 슬롯, 멀티모달 자리표의 출처, 응답의 logprobs 범위). 어휘 후보는 `results/m16/DEFERRED.md` 11~20행(23행에 평정 결과). 다음 방향(예: 토크나이저 id 수준 규칙; 오탐 자를 추측 디코딩·다중 KV 그룹·PD 분리로 넓히기)은 연구자가 정한다. README(EN/KO)의 "다음 검출률은 2차에서" 문장을 이 결과로 바꿔 1.2.0에 실었다.
