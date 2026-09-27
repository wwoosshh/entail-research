# M15 어휘 v6: 실제 버그 표본의 부류 안 결함을 사실로 — 측정

- 시작: 2026-09-25. 자: E3의 서로 다른 재현 결함 7개 가운데 부류 안 4개를 설치만으로 차단하거나 그 경계에서 정확히 보고하는가. 환경: WSL, vLLM 0.30.0(`~/venvs/vllm`), SGLang 0.5.20(`~/venvs/sglang`), transformers 5.17(`~/venvs/gpu`), RTX 4070 Ti. 단위 시험은 CPU(`~/venvs/ci`).
- 판정의 뜻은 E3 규약(`testbed/M10_PROTOCOL.md` 3.3)을 따른다. **검출** = 해소해 출력이 기대대로 나오거나, 결함이 생긴 경계와 사실에서 `broken`/`refused`를 낸다.

## M15.1 `TokenType` — vllm#58138 (cross-encoder 패딩의 토큰 종류)

**결함.** vLLM 0.30의 점수 경로(`_apply_post_tokenization_to_token_type_ids`)는 `padding="max_length"`의 패딩 자리에 토크나이저가 선언한 `pad_token_type_id`(0)가 아니라 마지막 실제 토큰의 종류(문서, 1)를 준다. /rerank 점수가 바뀐다(첫 문서 8.155 → 7.167).

**사실.** `TokenType(role="pad", type_id)`(MAPPING, 어휘 v6). 선언 = 토크나이저의 `pad_token_type_id`(config, declared). 엔진 = 패딩이 받은 종류(engine, verified). 규칙은 핵심 `request_contract.pad_type`, 어댑터는 `adapters/vllm_scoring.py`(함수를 감싸 패딩 구간만 읽음).

**소비자가 담을 수 없는 수리.** 첫 구현은 패딩 자리에 선언값 0을 써 넣었고(`resolved` 4건), 함수 검사는 transformers와 같아졌지만 서버의 패딩 요청이 **400**으로 죽었다. 원인: vLLM은 토큰 종류를 "0 다음 1"의 첫 1 위치 하나로 압축한다(`compress_token_type_ids`). 문서 뒤의 패딩에 0을 주면 압축이 실패한다. 즉 이 소비자는 선언된 패딩 종류를 **담을 수 없다**. 이를 능력표 행으로 적었다(`caps.json`: `vllm.scoring.io_processor` × `TokenType.type_id`, honours false, evidence measured). 해소는 능력표가 담을 수 있다고 할 때만 제안된다(`pad_type(repairable=None → 표 조회)`).

**측정** (`e3_58138_on.json`, `e3_58138_stop.json`, 기록 `e3_58138_logs/`):

| 정책 | 함수 검사 | 서버 /rerank 패딩 없음 | 서버 /rerank 패딩 | entail 판정 |
|---|---|---|---|---|
| off (E3, 09-24) | vLLM ≠ transformers | 8.155 / 1.658 / -11.264 | 7.167 / 1.221 / -11.203 (틀림) | 없음 (놓침) |
| on, 기본(보고하고 계속) | 같음(vLLM ≠ transformers) | 같음 | 같음(틀린 점수, 요청은 진행) | `broken` 4건, 능력표 note 포함 |
| on, `ENTAIL_ON_BROKEN=stop` | (함수 검사가 먼저 멈춤) | 8.155 / 1.658 / -11.265 (정상) | **점수 전에 거부** (HTTP 500, entail 문구) | `refused` |

**판정.** E3 규약으로 **검출**(경계와 사실에서 `broken`/`refused`). 해소는 이 소비자에서 불가능하며 그 이유가 능력표에 측정 근거로 적혀 있다. 멈춤 정책에서는 틀린 점수가 나가기 전에 막힌다. 기본 정책에서는 보고만 하고 틀린 점수가 나간다(연구자 결정 M5.4).

**단위 시험.** `tests/test_request_contract.py`(+4), `tests/test_vllm_scoring.py`(6), `test_caps.py`(행 39, measured 24). `run_all.sh`: 40개 파일, 354 checks.

**배운 것.** "수리"는 소비자가 담을 수 있을 때만 수리다. 담을 수 없는 수리를 적용하면 틀린 출력이 요청 실패로 바뀐다(더 나쁨). 어댑터가 아니라 능력표가 그 사실을 들고, 핵심이 표를 보고 해소를 제안한다.

## M15.2 `KernelConfig` — sglang#39626 (블록 FP8 커널의 K 타일 대 양자화 블록)

**결함.** SGLang 0.5.20의 블록 FP8 Triton matmul은 설정의 `BLOCK_SIZE_K`가 양자화 블록 `block_k`보다 크면 스케일 포인터가 블록 경계를 못 맞춰(`n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K` = 0) 틀린 값을 낸다. 엔진의 sanitiser는 타일이 작을 때만 올려 잡고, 큰 타일은 거르지 않는다. 결함은 수동·외부 튜닝 설정 경로에서 난다(배포된 튜닝 설정 119 파일·1,887 항목을 훑어 `BLOCK_SIZE_K > block_k` 0건, 나누어떨어지지 않는 작은 타일 0건, `BLOCK_SIZE_N > block_n` 27건은 열마다 스케일을 찾아 무해).

**사실.** `KernelConfig(tile_k, tile_n)`(LAYOUT, 어휘 v6). 선언 = 양자화 블록(허용되는 가장 굵은 타일). 엔진 = 고를 설정의 타일. 규칙 `tile_over_block`(핵심 `tile_contract`): 타일이 블록의 약수여야 한다. 해소 `clamp_tile_k`: 설정의 타일을 블록으로 묶는다(엔진의 기본 설정과 같아 정의상 옳음). 어댑터 `adapters/sglang_fp8_tile.py`는 matmul을 감싸 설정 맵을 맵 정체당 한 번만 검사한다(상시 경로는 집합 조회).

**측정** (`e3_39626_on.json`, 기록 `e3_39626_logs/`):

| | BLOCK_SIZE_K=32 (블록 32) | BLOCK_SIZE_K=64 (블록 32, 수동 설정) | entail |
|---|---|---|---|
| off (E3, 09-24) | 288 (맞음) | **64 (틀림)** | 없음 (놓침) |
| on | 288 | **288 (맞음)** | `resolved` 1건(`clamp_tile_k` → 32) |

**판정.** **해소.** 틀린 값이 나오기 전에 설정이 고쳐졌다. E3 #62 놓침 → 해소.

**단위 시험.** `tests/test_tile_contract.py`(5), `tests/test_sglang_fp8_tile.py`(4). `run_all.sh`: 42개 파일, 363 checks.

## M15.3 `Vocab` — transformers#48967 (폴더에 두 어휘, 엔진이 모델의 것이 아닌 토크나이저를 만듦)

**결함.** 모델 폴더(VARabic/Sentence-ALDi, 리비전 d8ce98a)가 `vocab.txt`(100,000줄, 모델의 것: config `vocab_size` 100,000)와 다른 모델의 `tokenizer.json`(WordPiece 32,000)을 함께 들고 있었다. transformers 5는 `tokenizer.json`을 먼저 취해 32,000 어휘의 토크나이저를 만들고, 같은 문장의 id가 [2, 2413, 9200, 3]에서 [2, 27966, 42, 1, 3]으로 바뀌어 회귀 점수가 0.0006에서 0.716이 됐다. 저장소는 2026-09-22에 `tokenizer.json`을 지워 고쳤다.

**사실.** `Vocab(size, added)`(MAPPING, 어휘 v6). 핵심 `vocab_contract`가 폴더를 읽는다: `tokenizer.json`(model.vocab), `vocab.txt`(줄), `vocab.json`, sentencepiece 모델(protobuf 1번 필드 조각 수, 라이브러리 없이), 그리고 모델의 어휘(safetensors 임베딩 행, 없으면 config `vocab_size`). 어댑터 `transformers_tokenizer`는 `PreTrainedTokenizerBase.from_pretrained`(transformers 5.17의 토크나이저 클래스가 상속)를 감싸 엔진이 든 토크나이저의 기본 어휘와 최대 id+1만 넘긴다. **범위(검토 M15.3에서 정정):** vLLM과 SGLang이 AutoTokenizer로 만드는 토크나이저는 이 훅을 지나지만, vLLM auto 모드의 Mistral 토크나이저(`tekken.json`·`tokenizer.model.v*`), transformers의 `MistralCommonBackend`, SGLang의 tiktoken(.json)·GGUF 경로, `VLLM_USE_FASTOKENS=1`이 나중에 덮은 경우는 지나지 않는다(무음). 정적 검사는 Mistral 파일이 있으면 그 사실을 note로 적고, 실행 시 우회 경로의 "검사 못 함" 보고는 M15.6에서 훅 발화 수를 세며 넣는다. 허브 id는 로컬 캐시 폴더로 푼다(`load.local_folder`, 내려받지 않음). 규칙 둘, 임계값 없음: `vocab_out_of_range`(토크나이저 최대 id ≥ 임베딩 행), `vocab_not_the_models`(폴더의 출처가 갈리면 모델 어휘와 같은 쪽이 모델의 것; 엔진이 다른 쪽을 들면). 출처에 없는 크기는 `unknown`(어디서 왔는지 말할 수 없음), `broken` 아님. 수리는 없다(모델의 것인 출처를 판정문에 이름해 사용자가 그 파일로 만들 수 있게).

**측정** (`e3_48967_on.json`, `e3_48967_stop.json`, 기록 `e3_48967_logs/`; 정적은 `entail check`):

| 실행 | 결과 | entail |
|---|---|---|
| off (E3, 09-24) | 틀린 id, 점수 0.716 | 없음 (놓침) |
| on, 기본 | 틀린 id 그대로(요청 진행) | `broken` 1건: "vocab.txt is the model's tokenizer and the engine built the other one" |
| on, `ENTAIL_ON_BROKEN=stop` | **토크나이저 적재에서 거부, id 하나도 안 나옴** (exit 1; 결과 파일은 쓰이기 전에 멈추므로 없음, 표준 오류에 `RoleError` 문구) | `refused` |
| `entail check` 옛 스냅숏 | exit 1 | `broken` (같은 문구) |
| `entail check` 고친 리비전 | exit 0 | `pass` (vocab.txt 100,000 = 엔진 100,000) |
| `entail check` Qwen3-4B (정상) | exit 0 | `pass` (tokenizer.json 151,643, added 26, 행 151,936) |

**판정.** **검출**(경계와 사실에서 `broken`/`refused`). 정적 검사는 CI 게이트로 쓸 수 있다(exit 1). 정상 모델에서 임베딩 패딩(Qwen)은 통과한다. E3 #73 놓침 → 검출.

**함께 잡은 것.** 자동 설치 목록(`sitecustomize.TARGETS`)에 `transformers.tokenization_utils_base` 키가 이미 있어(템플릿 어댑터) 새 줄이 dict 안에서 덮여 어댑터가 설치되지 않았다. 한 항목으로 합치고, 키 중복과 진입점 존재를 검사하는 `tests/test_autoinstall_targets.py`를 더했다.

**단위 시험.** `tests/test_vocab_contract.py`(9), `tests/test_transformers_tokenizer.py`(4), `tests/test_autoinstall_targets.py`(2).

## M15.7 (첫 발견) 배포된 SGLang fused-MoE 설정의 K 타일이 양자화 블록을 넘음 — 미보고

**출처.** M15.2 검토 에이전트가 SGLang 0.5.20의 배포 튜닝 설정 87파일·1,538항목을 훑어 찾았다: `srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`의 M=64·128·256·512 항목이 `BLOCK_SIZE_K=256`(블록 128). fused-MoE 설정 조회(`try_get_optimal_moe_config`)에는 dense 쪽과 달리 살균이 전혀 없다.

**기전(코드).** 블록 FP8 fused-MoE 커널(`kernels/ops/moe/fused_moe_triton_kernels.py`)은 K 타일마다 스케일 하나를 `offs_ks = k_start // group_k`로 고른다. 타일 256이 블록 128 둘에 걸치면 뒤 절반이 앞 블록의 스케일을 받는다.

**재현(커널 수준, 이 카드, `testbed/m15/moe_tile.py`).** 설정은 엔진의 조회 함수를 `override_config`로 통과시켜(entail이 감싸는 자리) 상위 matmul(`invoke_fused_moe_kernel`)에 넘겼다. A = 1, 가중치 = 1, K 블록 둘의 스케일 1과 3 → 정답 512, 타일 하나에 스케일 하나면 256.

| | 타일 128 | 타일 256 (배포 H100 설정의 값) |
|---|---|---|
| off | 512 (맞음) | **256 (틀림)** |
| on | 512 | **512** — `resolved at load:sglang.fused_moe_kernel_config`(타일을 128로 묶음) |

**판정.** 배포 설정 자체가 이 커널에 대해 틀렸다. H100에서 E=512·N=256·fp8 블록 [128,128]인 MoE를 Triton fused-MoE 경로로 돌리면 M=64~512 배치에서 조용히 틀린 값이 나온다(H100이 없어 끝까지는 재지 못함; 기전은 sm_89에서 재현). 설치만으로 entail이 타일을 묶어 막는다. 상류 보고 후보(연구자 허락 뒤). `moe_tile_off.json`, `moe_tile_on.json`.

### M15.3 검토 반영 (2026-09-25)

- **오탐 하나(S3):** `~/models/gemma-3-1b-it`가 `vocab_out_of_range`로 `broken`이었다. added 토큰 `<image_soft_token>`(id 262,144)이 텍스트 전용 1B의 행 262,144 밖인데, 텍스트로는 나오지 않는 id다. 규칙을 좁혔다: **기본 어휘(`vocab_size`)가 행보다 클 때만 `broken`**, added 토큰이 행 밖이면 pass에 note. 시험 `test_an_added_token_past_the_rows_is_noted_not_broken`.
- 검토가 "시험 없음"으로 짚은 앞선 변경 둘에 시험을 두었다: 임베딩 접미사가 위치 임베딩을 집지 않게 config `vocab_size` 이상의 행만 택함(`test_a_position_embedding_under_a_similar_name_does_not_stand_in_for_the_rows`); 두 출처가 모두 모델 어휘와 다르면 `unknown`(`test_two_sources_neither_equal_to_the_model_are_unknown_not_judged`).
- 실폴더 42개(검토 에이전트, CPU)에서 위 오탐 하나 말고는 `broken` 없음: Llama-3.2, Qwen2/2.5/3 ×20, Gemma-2, Phi-3.5/Phi-4-mini, TinyLlama, ms-marco-MiniLM, tiny-GptOss 모두 pass. sentencepiece 조각 수 걷기는 실파일 5개에서 transformers `vocab_size`와 같음.

## M15.4 RoPE 어휘 v6 — E2의 `unknown`을 없앤다 (S1)

**빈틈.** E2/M11에서 Phi-3.5-mini·Phi-4-mini(longrope의 `long_factor`/`short_factor`), gpt-oss(yarn의 `beta_fast`/`beta_slow`/`truncate`)가 `load:*.config.rope_parameters`에서 "RoPE keys … are not in vocabulary"로 `unknown`이었고, Gemma 3(층별 두 기저)은 "RoPE set per layer type"으로 사실이 아예 안 나왔다. 어휘 밖이면 대조하지 않으므로 엔진이 그 값을 잃어도 보이지 않는다.

**어휘 v6의 `Rotary` 선택 필드(ADDED_IN 6):** `beta_fast`, `beta_slow`, `attention_factor`, `mscale`, `mscale_all_dim`, `truncate`(yarn); `long_factor_sha256`, `short_factor_sha256`, `factor_terms`(longrope: 48~64개 실수 목록은 JSON의 SHA-256과 항 수로 비교, 기록에는 요약만); `local_theta`(Gemma 3의 `rope_local_base_freq`, 또는 `rope_parameters.sliding_attention.rope_theta`). 별칭 표(`aliases.json`)와 읽기 함수(`readers._rotary`)가 같은 자리에서 선언 쪽과 엔진 쪽을 모두 읽으므로, 엔진이 이 값 하나라도 잃으면 `broken`/`refused`가 된다(`tests/test_load.py`: beta_fast를 잃은 held config가 refused).

**정적 확인(`entail check`, 이 카드 없이):** Phi-3.5-mini·Phi-4-mini는 `Rotary(rope_type='longrope', theta=10000, factor_terms=…)`로, tiny-GptOss는 `Rotary(rope_type='yarn', theta=150000, factor=32, original_max_position=4096, beta_fast=32.0, beta_slow=1.0, …)`로 읽힌다(어휘 밖 문제 없음). 실행 시 `unknown`이 사라졌는지는 M15.6의 E2 재측정에서 센다.

**시험.** `tests/test_sources.py`(+1: 다이제스트·yarn·Gemma 3 옛 표기·v5 거부), `tests/test_load.py`(Gemma 3 모양이 비교됨; 잃은 beta_fast가 refused; 어휘 밖 키는 여전히 unknown). `run_all.sh`: 45개 파일, 385 checks.

## M15.5 SGLang radix 캐시에 같은 모양의 낡음이 있는가 — 코드 확인, 미룸

SGLang 0.5.20의 radix 캐시는 열쇠가 해시 사슬이 아니라 **토큰 id 그 자체**다(`srt/mem_cache/radix_cache.py` `RadixKey.token_ids`; 조회는 요청의 현재 토큰으로 매번 맞춘다). vLLM의 결함은 "토큰과 따로 저장된 정체(블록 해시)가 토큰이 잘린 뒤에도 남는 것"인데, SGLang에는 그런 별도 정체가 없다. 세션 이어가기(`managers/session_controller.py`)는 이력 토큰으로 새 `Req`를 만들어 다시 맞추고, 되돌림(retraction)은 `prefix_indices`·`last_node`를 다시 잡는다. 같은 모양의 결함을 코드에서 찾지 못했으므로 `Identity`의 SGLang 소비자는 **미룬다**(검토 M15.1의 권고대로: 없으면 미룸). 재현된 사례가 오면 다시 본다.

## M15.6 재측정 (2026-09-26 완료)

### E1 정적 (230 config, 어휘 v6, `testbed/results/m10/e1_llm/static_1.1.0-v6.json`)

- 123 s(1.0.2: 62.5 s; 토크나이저 검사가 폴더마다 AutoTokenizer를 시도). 이 실행 시점의 E1 폴더에는 토크나이저 파일이 아직 없었다(정적 훑기 M15.7이 그 뒤에 채운다).
- 엔진 셋 모두 같은 분포: ModelProps pass 202 / broken 2 / unknown 23(1.0.2와 같음; broken 2는 T5의 tie), Coverage pass 56 / unknown 165(같음), Layout unknown 41(같음), **Vocab pass 137 / unknown 77, broken 0**. 어휘 v6가 정적 검사에 새 `broken`을 만들지 않았다.

### E2 정상 실행 (인기 모델 38개 × 엔진 셋, 최종 코드; `testbed/results/m15/e2_final/`, `E2_SUMMARY.md`)

- 실행: `testbed/m15_e2_run.sh`(`E2_DIR=testbed/results/m15/e2_final`), 커밋 `a719aad`의 트리(그 뒤의 `82644c1`은 NeoX 별칭·GPT-J 값 가드·Laguna reader 수정이라 이 38개 모델에는 닿지 않는다). 대조는 M10·M11의 entail off 실행. 규약은 `testbed/M10_PROTOCOL.md` 2절.
- 짝이 있는 실행 110회, **유효 102회**(제외 8: transformers가 돌리지 못하는 양자화 모델 4(AWQ·FP8·w4a16), 엔진 쪽 실패 4 — off도 같이 실패).
- **entail이 실행을 깨뜨린 것 0, broken·refused 0, 틀린 경보 0.** resolved 2(gemma-2-2b-it의 softcap 백엔드, transformers·SGLang; 실측 행이 뒷받침). 해소 없는 실행의 출력 일치 **99/100**(다른 1은 tiny-GptOss transformers: M11에서 off-대-off로도 다른 엔진 비결정성).
- **unknown 76줄**(같은 모델·코드가 섞인 앞선 실행 `e2`의 364줄, M11(1.0.1)의 69줄): Coverage 64(클래스가 안 받고 entail이 못 읽는 키 — Phi의 `attention_bias`·`mlp_bias`, Nemotron의 `hybrid_override_pattern`, SmolLM2의 `rope_interleaved` 등; 이제 프로세스당 한 번), Layout 17(vLLM·SGLang의 quant 경로, 1.0.2와 같음), ModelProps 1. Qwen의 `rope_scaling: null`이 만들던 240줄은 pass가 됐다.
- **비용:** 적재 시간 대비 라이브러리 시간 **중앙값 1.16%**, p90 7.8%, 최대 37%(toy 모델 tiny-Qwen3: 적재 0.77 s에 고정 비용 280 ms; gemma-3-1b-it transformers 30%는 tokenizer.json 33 MB + sentencepiece를 첫 실행에서 세는 값이며 캐시 뒤에는 사라짐). 실행당 라이브러리 시간 중앙값: transformers 144 ms(모델마다 첫 실행이라 어휘 출처 캐시가 비어 있음), vLLM 142 ms(프로세스 2), SGLang 82 ms(프로세스 3).
- Vocab 판정: transformers 38·vLLM 73·SGLang 114(프로세스당 한 번; 앞선 실행은 SGLang 217), 전부 pass.
- 섞인 코드로 돈 앞선 실행(`e2`, `E2_MIXED_SUMMARY.md`)은 23:44~00:23 사이의 커밋 다섯 개에 걸쳐 있어 수치를 쓰지 않는다. v6가 만든 오탐(SmolLM2 `rope_interleaved` → `mrope_interleaved` 오타 판정)이 거기서 안 보인 것은 SmolLM2가 mrope 커밋 전에 먼저 돌았기 때문이고, `e2_recost`(3모델 × 3엔진)에서 세 엔진 모두 broken으로 드러나 고쳤다(`LIBRARY_DESIGN.md` 11절 2026-09-26 둘째 행).

### 비용 재측정 (`testbed/results/m15/e2_recost/`, `E2_RECOST_SUMMARY.md`)

- 어휘 출처 캐시·tokenizer.json 조건부 세기·`added_tokens_decoder` 뒤, Qwen3-4B·Llama-3.2-3B·SmolLM2-135M × 3엔진 9회: 라이브러리 시간 Qwen3-4B 264/872/999 ms → **32/185/94 ms**(transformers/vLLM/SGLang), Llama 219/693/1799 → 37/156/158, SmolLM2 130/325/264 → 42/199/98. 적재 비중 중앙값 1.28%, 최대 5.9%(SmolLM2 transformers, 적재 0.7 s). transformers의 첫(콜드) 실행은 Qwen3-4B 199 ms(tokenizer.json + vocab.json 세기 한 번), Llama 39 ms(출처가 tokenizer.json뿐이라 세지 않음).
- 이 9회에서 SmolLM2 세 엔진이 Coverage broken 11건을 냈고, 그것이 v6의 새 오탐이었다(위). 고친 뒤의 `e2_final`에서 0.

### E1 정적, 최종 코드 (230 config, `testbed/results/m10/e1_llm/static_1.1.0-final.json`)

- 최종(M15.6 검토 반영 뒤, 커밋 `a98e66c`, 369 s; 토크나이저 파일이 훑기로 채워져 230개 폴더 모두에서 토크나이저를 만든다). 엔진 셋 같은 분포: ModelProps pass 202 / broken 2(T5의 tie, 1.0.1부터 알려진 것) / unknown 23, **Coverage pass 98 / unknown 123 / broken 0**(v6 첫 실행 56/165: 어휘가 다른 경계에서 비교하는 키만 남은 폴더가 pass로; 도구 표지 키 셋은 note), Layout unknown 41, **Vocab pass 214 / unknown 16 / broken 0**(unknown은 remote-code 클래스 등 토크나이저를 못 만든 폴더), **Rotary pass 196 / unknown 6 / broken 0**(검토 지적으로 `entail check`가 이제 클래스가 만든 config 객체의 RoPE를 파일과 대조한다; unknown 6 = DeepSeek-V4 계열 4의 층 종류별 분할 `compress/main`(어휘 밖, "비교 못 함"으로 말함) + `attn_factor` 2(읽는 엔진 없음)). 검토 에이전트가 따로 돌린 CPU 대조(300폴더, `rotary_held`)도 pass 196 / broken 0이었다.

### 자로 잰 판정 (M15 전체)

- **차단:** E3 부류 안 4/4(해소 2, 경계에서 정확히 보고 2). 1.0.0은 0/4였다.
- **정상 실행:** 38개 × 3엔진 유효 102회에서 틀린 경보 0, 출력 동일 99/100, 정적 230개에서 broken 0(T5 tie 2 제외). v6가 만든 오탐 둘(SmolLM2 오타 판정, Laguna reader 죽음)은 이 단계 안에서 잡아 고쳤고 회귀 시험을 두었다.
- **비용:** 적재 비중 중앙값 1.2%(E2 final), 재측정 1.3%. 상시 경로(디코드)에는 이 단계가 손대지 않았다(M14의 Identity 훅은 스트리밍 세션 갱신에서만 돈다).
- **분모 주의:** 4/4는 census 229건에서 씨앗 순서로 뽑아 재현된 표본이며 검출률이 아니다. 정적 훑기의 "어긋남 0"은 인기 모델의 폴더가 대개 출처 하나라 비교 상대가 없다는 뜻이지 검사가 무의미하다는 뜻은 아니다(출처 둘인 폴더 1개는 unknown으로 정직하게 남았다).

## M15.8 `Stops` — 생성이 끝나는 id (어휘 v7; 검토 M15.6이 자로 1순위로 꼽은 후보)

- **부류:** 같은 뜻(생성이 끝나는 토큰 id)이 세 파일에 선언되고 엔진마다 읽는 부분집합이 다르다. transformers 5.17은 generation_config.json만(`modeling_utils.py` from_pretrained L4320 → `generation/utils.py` adjust_generation_fn L514; 파일이 없을 때만 config의 id로), vLLM 0.30은 토크나이저 eos + generation_config(`v1/engine/input_processor.py` L55·373), SGLang 0.5.20은 config ∪ generation_config(`srt/configs/model_config.py` L1903)에 **더해 스케줄러가 토크나이저 eos도 맞춘다**(`schedule_batch.py` check_finished L1719-1722; `skip_tokenizer_init`이면 제외 — M15.8 검토가 잡은 정정: 첫 판은 SGLang이 토크나이저를 안 읽는다고 적었다). 어느 파일이 선언한 끝을 엔진이 안 보면 답이 끝난 뒤에도 생성이 이어진다. 실제 사례: Llama 3(2024-04) 무한 생성.
- **규칙(핵심 `stops_contract.py`):** 출처들의 합집합이 뜻이다. 소비자의 정지 집합이 합집합을 덮으면 pass(더 멈춰도 됨), 빠뜨리면 `stop_dropped` — 해소는 빠진 id를 소비자 집합에 더하는 것(세 엔진 모두 담음: `generation_config.eos_token_id` 리스트, `generation_config_fields` dict, `hf_eos_token_id` set). 선언된 id가 토크나이저 밖이면 `stop_id_out_of_range`(broken). 특수 토큰이 아닌 eos는 note. 정적 검사는 엔진이 읽는 파일을 데이터(`data/stops_sources.json`)로 들고 그 집합을 만든다.
- **재현(`testbed/m15/stops_replay.py`, `results/m15/stops_replay_off.json`·`_on.json`):** Llama-3.2-3B-Instruct의 사본(가중치 심볼릭 링크, generation_config.json의 eos만 128001로 좁힘; config.json은 128001·128009를 선언). transformers 5.17, 채팅 템플릿, greedy, max_new_tokens 160, 프롬프트 3개.
  - **off:** 세 답 모두 **160 토큰까지 달렸다**(답의 끝 `<|eot_id|>` 128009는 7·17·36번째에 나왔지만 엔진의 집합에 없어 멈추지 않음). 적재 뒤 엔진이 든 eos: 128001.
  - **on:** 적재에서 `[entail] resolved at load:transformers.generation_config` — config.json이 선언한 128009를 더함(`held_eos_after_load` [128001, 128009]). 세 답이 **8·18·37 토큰에서 멈췄다**(끝 id 바로 뒤). 1.0.0~1.1.0.dev(M15.7까지)는 이 부류를 읽지 않았다.
- **E1 정적, 첫 실행(230개, `testbed/results/m10/e1_llm/static_1.1.0-stops-first.json`, 커밋 `fcfc090`):** Stops 판정 transformers pass 214 / broken 20, vLLM 220 / 8, SGLang 218 / 16. broken은 세 갈래다(측정 정의대로 수를 본 뒤 verdict를 정한다).
  - (가) **퇴화 토크나이저 오탐 — 고친다.** 토크나이저 파일이 없는 폴더(GGUF 저장소 unsloth/Qwen3-4B-GGUF, 드래프트 모델 MiniCPM5-2B-DSpark·Kimi-K3-DSpark·Qwen3.8-27B-DFlash2/DSpark)에서 transformers가 vocab 1짜리 Qwen2Tokenizer를 만들고(M15.3에서 본 그 모양), 정적 검사가 그 크기(1)와 eos(0)를 믿어 "id가 토크나이저 밖"·"0을 빠뜨림"이 났다. 폴더에 토크나이저 출처가 없으면 만든 토크나이저를 쓰지 않는다(Vocab이 이미 unknown으로 두는 것과 같은 판단).
  - (나) **토크나이저의 eos가 엔진 집합 밖 — 실제 부류.** config·generation_config는 `</s>`(2)만 말하고 채팅 템플릿과 토크나이저 eos_token은 `<|im_end|>`(11)인 nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16(generation_config는 `_from_model_config`로 자동 생성된 것), 같은 모양의 dolphin-2.9.1-yi(7 vs 2)·Ornith-1.5-9B(248046 vs 248044)·tiny-random 둘. transformers(generation_config만)와 SGLang(config ∪ generation_config)은 이 끝을 못 본다. 인기 230개 중 토크나이저 eos id가 transformers의 집합 밖인 폴더 7. 실행 시 어댑터가 이 선언을 아직 못 읽는다(transformers·SGLang 어댑터는 파일의 id만 읽음) — tokenizer_config.json의 `eos_token`을 `added_tokens_decoder`로 id로 바꿔 읽는 reader를 더한다. Nemotron은 실제 폴더 그대로 transformers에서 재현한다(아래).
  - (다) **config.json의 eos가 generation_config에 없음 — 규칙대로 해소.** saiga_llama3(128001 vs 128009), sarvam-30b(1 vs 26), Ornith, dolphin-yi. 기본 모델의 끝(`<|end_of_text|>`류)을 소비자 집합에 더하는 해소는 안전하다(그 id에서 멈추는 것은 언제나 옳다). 정적 검사에서는 "entail이 더할 것"으로 resolved로 말한다(어텐션 경로 전환과 같은 뜻).
- **E2 첫 실행(38개 × 3엔진, 커밋 `fcfc090`, `testbed/results/m15/e2_stops/`, `E2_SUMMARY.md`; 앞 단계의 실행은 `E2_FINAL_SUMMARY.md`로 이름을 바꿈):** 유효 102회, entail이 깨뜨린 실행 0, broken·refused 0, resolved 3(gemma-2 softcap 둘 + **`add_stops` 하나**: hmellor/tiny-random-LlamaForCausalLM의 vLLM — generation_config.json에 eos가 없고 config.json이 1을 선언, vLLM은 토크나이저 eos 2만 들어 1을 더함; 규칙대로), 틀린 경보 0(규약 2.3을 고침: 핵심 규칙의 해소는 능력표 행이 아니라 재현 실측이 근거 — `M10_PROTOCOL.md` 6절 2026-09-26), unknown 70(그중 1은 **entail 자신의 오류**: 같은 모델의 transformers에서 generation_config에 eos가 없어 소비자 집합이 비었는데 빈 `Stops`를 사실이 거부해 "entail failed here"로 기록됨 — 실행은 그대로 이어짐(`load.safely`); 빈 집합을 값으로 허용하고 회귀 시험을 둠), 출력 동일 98/99(다른 1은 앞 실행과 같은 tiny-GptOss 비결정성), 적재 비중 중앙값 1.3%, 최대 37.8%(toy 모델). 실행당 라이브러리 시간 중앙값 transformers 136 / vLLM 158 / SGLang 121 ms(SGLang은 앞 실행 82 ms에서 올랐다: 프로세스 셋이 각각 폴더의 파일 사실을 읽음 — 어휘 출처 캐시 같은 캐시가 없음).
- **미개척 발견 — nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16, 파일 그대로(`testbed/results/m15/stops_nemotron_off.json`·`_on.json`, `stops_nemotron_on.record.jsonl`):** 정적 훑기(나)가 짚은 폴더를 transformers 5.17에서 실제로 돌렸다(채팅 템플릿, greedy, max_new_tokens 160, 프롬프트 3). config.json과 `_from_model_config`로 자동 생성된 generation_config.json은 `</s>`(2)만 끝이라 하고, tokenizer_config.json의 `eos_token`과 채팅 템플릿은 `<|im_end|>`(11)이다.
  - **off:** 세 답 모두 **160토큰 한도까지 달렸다.** 답은 45·62·55번째의 `<|im_end|>`에서 끝났지만 엔진의 집합 {2}에 없어 멈추지 않았고, 그 뒤로 `<|im_end|>`를 계속 냈다(본문에 `<|im_end|> <|im_end|> <|im_e…`).
  - **on:** 적재에서 `resolved` — tokenizer_config.json이 선언한 11을 더해 집합 [2, 11]. 세 답이 **46·63·56토큰에서 멈췄다**(끝 id 바로 뒤). 출력 본문은 off와 같은 토큰까지 같다.
  - 판정: 인기 모델(E1 상위 300 안)이 배포된 파일 그대로 transformers에서 답이 끝난 뒤에도 생성을 잇는다. 뜻(끝 = `<|im_end|>`)은 토크나이저와 템플릿에 선언돼 있고 generation_config.json이 그것을 잃었다. vLLM은 토크나이저 eos를 1차 eos로 들어 멈추고, SGLang도 스케줄러가 토크나이저 eos를 맞춰 멈춘다(검토가 코드로 확인; 첫 판의 "SGLang도 같은 증상"은 틀렸다). 증상은 transformers `generate()`에서 난다. 상류(모델 저장소의 generation_config.json) 보고 초안은 연구자 허락 뒤 게시. 이 발견은 검토 에이전트의 순위(특수 토큰 id) → 정적 훑기 → 실행 확인의 순서로 나왔고, 보고된 이슈에서 출발하지 않았다.
- **E1 정적, 보정 뒤(230개, `static_1.1.0-stops-second.json`; 토크나이저 선언 reader + 퇴화 토크나이저 제외 + 정적 해소 표기):** Stops **broken 0**. transformers pass 220 / resolved 8(Nemotron-3-Nano-4B 11, dolphin-yi 7, Ornith 248046, Agents-A1-4B 248046 — 토크나이저가 선언한 `<|im_end|>`류; saiga 128001, sarvam 1 — config의 기본 끝; tiny-random-Llama 둘 2), vLLM pass 226 / resolved 2(saiga, sarvam: config의 끝만), SGLang pass 224 / resolved 4(Nemotron, Agents-A1, tiny 둘). 첫 실행의 broken 20/8/16은 (가) 퇴화 토크나이저 오탐이 사라지고 (나)(다)가 "entail이 더할 것"(resolved)으로 바뀐 것이다. InternScience/Agents-A1-4B는 generation_config.json이 없고 config.json의 `text_config.eos_token_id` 248044를 transformers가 기본 집합으로 들며 토크나이저 eos는 `<|im_end|>` 248046이라, transformers에서 같은 증상이 예상된다(미실측; SGLang은 토크나이저 eos를 맞추므로 아님).
- **최종(커밋 `bd176b8`) E1 정적(`static_1.1.0-stops.json`):** Stops broken 0; transformers pass 221 / resolved 8, vLLM 227 / 2, SGLang 225 / 4(둘째 실행과 같은 폴더들). 다른 사실은 M15.6 최종과 같다(Coverage·Vocab·Rotary broken 0).
- **최종 E2(`testbed/results/m15/e2_final2/`, `E2_SUMMARY.md`; 첫 Stops 실행은 `E2_STOPS_SUMMARY.md`, M15.6 실행은 `E2_FINAL_SUMMARY.md`):** 유효 102회, entail이 깨뜨린 실행 0, **broken·refused 0, 틀린 경보 0**, resolved 9줄 = softcap 2(gemma-2, 실측 행) + `add_stops` 7줄 = **고유 5**(모델 × 엔진; SGLang은 프로세스 둘이 같은 결정을 기록): 실측된 실제 결함 1(Nemotron/transformers 11), toy 체크포인트 2(tiny-random-Llama transformers 1·2, vLLM 1 — 그중 1은 토크나이저의 `<s>`(bos)라 끝이 아님), **SGLang 2는 아무것도 바꾸지 않는 해소**(스케줄러가 토크나이저 eos를 이미 맞춤; 검토 지적). 검토 뒤 고침: SGLang 어댑터는 토크나이저의 선언된 끝을 든 것으로 세고, bos로 선언된 id는 끝에 더하지 않으며, 규약 2.3의 근거는 (해소, 엔진)별로 센다 — 다시 돈 결과는 아래. 해소 없는 실행의 출력 동일 94/95(다른 1은 tiny-GptOss 비결정성). unknown 69(M15.6 76; entail 자신의 오류 0). 적재 비중 중앙값 **1.4%**, p90 9.4%(M15.6 7.8%; 원인은 tokenizer_config.json 한 파일이 아니라 어댑터(config·loader·attention·stops)마다 `sources.read_all`이 폴더 전체 — safetensors 헤더 포함 — 를 다시 읽는 구조: 검토 지적. 폴더 스탬프 메모로 고침, 아래 재측정), 최대 36%(toy 모델). 실행당 라이브러리 시간 중앙값 transformers 155 / vLLM 166 / SGLang 118 ms.
- **검토 반영 뒤 최종 E2(커밋 `b0a7257`, `testbed/results/m15/e2_final3/`, `E2_SUMMARY.md`; 앞선 실행들은 `E2_FINAL2_SUMMARY.md`·`E2_STOPS_SUMMARY.md`·`E2_FINAL_SUMMARY.md`):** 유효 102회, entail이 깨뜨린 실행 0, **broken·refused 0, 틀린 경보 0**(규칙 해소의 근거를 (해소, 엔진)별로 센 뒤에도), resolved 4 = softcap 2(gemma-2, 실측 행) + **`add_stops` 2, 둘 다 transformers**(Nemotron-3-Nano-4B 11 — 실측된 실제 결함; hmellor/tiny-random-Llama 2 — 토크나이저가 선언한 `</s>`; config의 eos 1은 토크나이저의 `<s>`라 더하지 않고 note). vLLM·SGLang의 add_stops는 0(둘 다 토크나이저 eos를 이미 맞춤). 해소 없는 실행의 출력 동일 97/98(다른 1은 tiny-GptOss 비결정성). unknown 69. 적재 비중 **중앙값 1.2%, p90 9.1%, 최대 34%**(toy 모델); 실행당 라이브러리 시간 중앙값 transformers 129 / vLLM 146 / SGLang 109 ms(폴더 메모로 앞 실행 155/166/118에서 내려감).
  - 이 실행 뒤 규칙을 한 번 더 좁혔다(커밋 `b0a7257` 다음): bos 제외가 "bos이면서 eos이기도 한 id"(GPT-2·OPT·Pythia·Qwen base의 `<|endoftext|>`)까지 지워 42/230 폴더의 판정을 없앴던 것을, "어느 출처가 bos라 하고 eos라 하지 않는 id"로 좁혔다. 이 변경은 E2의 판정을 바꾸지 않고(그 모델들에 pass 기록이 더해질 뿐) 아래 부록 실행으로 확인했다.
- **최종 E1 정적(커밋 `5905138`, `static_1.1.0-stops.json`; 앞선 실행들은 `-first`(보정 전), `-second`, `-third`(bd176b8), `-fourth`(b0a7257: bos 규칙이 넓어 42개 폴더의 판정이 사라졌던 실행)):** Stops **broken 0**; transformers pass 219 / resolved 9(Nemotron 11, dolphin-yi 7, Ornith 248046, Agents-A1-4B 248046, tiny-random-Llama 둘 2, tiny-random-Gemma2 1 — 토크나이저가 선언한 끝; saiga 128001, sarvam 1 — config의 기본 끝), vLLM pass 226 / resolved 2(saiga, sarvam), **SGLang pass 228 / resolved 0**(스케줄러가 토크나이저 eos를 맞추므로 더할 것이 없다). 판정 없음 2(GGUF 저장소 등 파일에 끝이 없는 폴더).
- **부록(`e2_final3_addendum/`, 커밋 `5905138`):** bos가 곧 eos인 base 모델(Qwen3-4B-Base)을 세 엔진에서 다시 돌려 Stops pass가 기록됨을 확인했다(transformers 1, vLLM 1, SGLang 2 — 프로세스 둘). b0a7257의 E2 판정은 바뀌지 않는다.
- **자로 잰 판정(M15.8):** 부류 안 결함 1건 추가 차단(재현 해소) + 배포된 인기 모델 1건에서 미보고 결함을 찾아 해소(Nemotron/transformers; Agents-A1-4B는 같은 모양, 미실측). 정상 실행 102회 틀린 경보 0(해소 2는 파일의 선언과 실측에 근거). 비용은 중앙값 1.2%로 제약 안이며 p90(9.1%)은 적재 1초 미만의 toy 모델 집단에서 나온다. 검토 에이전트가 잡은 오류(SGLang 표)와 그 반영은 위에 적었다.






### M15.4 검토 반영 (2026-09-26)

- **빈틈 A(잡힘 없이 통과하던 값):** Phi의 `original_max_position_embeddings`는 config 최상위 키라 읽히지 않았고, 엔진 객체의 값은 "클래스 기본값"으로 오기록되어 비교에서 빠졌다(엔진이 잃어도 PASS). 이제 scaling dict에 없으면 최상위에서 읽는다. `tests/test_load.py`: 잃은 held config가 refused.
- **빈틈 B:** `partial_rotary_factor`(Phi-4-mini 0.75; transformers 5는 rope_parameters 안에도 둔다)를 어휘에 더했고, 엔진 쪽 reader 문제(어휘 밖 키)도 경계에서 말한다(전에는 버려져 무음).
- **Gemma 3:** 국소 층의 스케일링(`local_factor`, 선언은 None)을 담아, 전역 factor를 국소 층에 적용한 엔진 객체가 broken이 된다. 한계: 진짜 결정 지점은 모델 코드(층 종류별 rotary 생성)라, config 경계는 "설정이 변환에서 깨진" 경우만 본다. 층별 생성 지점 어댑터는 사례가 오면.
- **MoE down 설정:** 엔진이 호출마다 새 dict를 만들므로 정체 메모는 매번 판정하고 무한히 쥔다. 내용으로 메모(한 번 기록, 복사본마다 다시 묶음, 상한 4,096). 재확인: 커널 수준 타일 256 → 128, 512(맞음), 기록은 resolved 1줄·timing 2줄.
- RoPE 다이제스트 비교의 오탐 여부는 검토 에이전트가 7모델×3엔진의 실제 config 객체로 CPU 실측해 **모두 일치**(거짓 refused 없음).

## M15.7 정적 훑기 결과 (`testbed/results/m15/sweep_static.json`, `SWEEP_SUMMARY.md`, `rotary_sweep.json`)

- 대상: E1의 230개 폴더(config 있는 인기 모델). 토크나이저 파일 3,048 MB를 내려받고 임베딩 행은 safetensors 헤더만 range 요청으로 읽었다(201개; 29개는 헤더에서 못 찾아 config `vocab_size`가 대신 섬).
- **Vocab: pass 219, unknown 11, broken 0.** unknown은 토크나이저를 못 만든 9(GGUF 저장소, remote-code 클래스), 토크나이저 파일 없는 폴더 1, 두 어휘가 모두 모델 어휘와 다른 폴더 1(EleutherAI/gpt-neox-20b: tokenizer.json 50,254·vocab.json 50,277·행 50,432 — added 토큰 차이, 무해). **인기 모델 230개에서 미보고 어휘 어긋남은 0.** 이것이 데이터다: 이 부류는 인기 모델에는 드물고, #48967 같은 폴더 편집 실수에서 난다.
- **RoPE:** 첫 훑기에서 어휘 밖 키가 25개 모델(Qwen-VL 계열 mrope: `mrope_section`·`mrope_interleaved`·`partial_rotary_factor`)에서 났다. `mrope_section`·`mrope_interleaved` 필드, `proportional`(Gemma 4 계열 5개), `attn_factor` 별칭, dict 안의 `partial_rotary_factor`를 어휘에 더한 뒤 재집계: 어휘 밖 문제 34 → **0**. 검토 반영(2026-09-26): `mrope`는 종류가 아니다(transformers 5가 객체에서 `default`로 바꾸므로 종류로 두면 파일과 객체가 늘 어긋난다; 절이 사실이다). GPT-NeoX 계열(Pythia 7개)의 제 이름 `rotary_emb_base`·`rotary_pct`를 별칭으로 더해(transformers 5.17은 이를 `rope_theta`·`partial_rotary_factor`로 든다) 현재 코드로 다시 훑음: **Rotary 사실 208**(230 중; 나머지 22는 RoPE 없는 모델·GPT-J의 `rotary_dim`·DeepSeek MLA의 `qk_rope_head_dim` 등 어휘 밖 이름), 종류 default 167, yarn 23, llama3 12, proportional 5, longrope 2, linear 1, 어휘 밖 문제 **3**: poolside/Laguna-S-2.1의 국소 층 `partial_rotary_factor` 1.0 ≠ 전역 0.5(v6가 못 담는 모양이라 보고; 첫 수정은 이 모양에서 reader가 죽어 사실을 통째로 잃었고 시험을 두어 고쳤다) 1, `attn_factor` 2(DeepSeek-R1-0528-Qwen3-8B 계열; M15.6 검토: transformers 5.17도 vLLM 0.30도 읽지 않는 이름을 별칭으로 넣어 pass시키고 있었다 — 뺐고, 이제 "비교 안 됨"으로 말한다). `rotary_sweep.json`
- Mistral 토크나이저 파일이 있는 모델 1(unsloth/mistral-7b-v0.3-bnb-4bit): 검사 밖(note).
- 판정: 훑기의 결과는 "찾은 결함 목록"이 아니라 **선언이 소비자에 안 닿던 곳의 목록**이며, 그 목록(mrope 25, proportional 5, partial 26, NeoX 이름 7)은 어휘에 넣어 닫았다. 남은 어휘 밖은 3(Laguna의 국소 partial 1, 읽는 엔진이 없는 `attn_factor` 2)이며, GPT-J `rotary_dim`(2)·DeepSeek `qk_rope_head_dim`(15)·SmolLM3 `no_rope_layers`(3)·GLM-5.2 `rope_interleave`(4)는 이름은 RoPE지만 v6 필드가 아니라 Coverage의 unknown(안 읽힌 키)으로 남는다. 이 가운데 `rope_interleave`(회전 쌍을 끼워 넣는가 반으로 가르는가)는 엔진이 관례로 정하는 뜻이라 후보 어휘로 검토 에이전트에 넘긴다.
- **미개척 탐색(2026-09-26, 읽기 전용):** 그 짝짓기 뜻이 실제로 경계에서 사라진 공개 사례를 찾았다. vllm#49290(MRoPE Triton 커널이 NeoX 식을 고정해 GPT-J 식 모델 GLM-OCR이 틀림; 모델 코드의 `is_neox_style=False` 선언을 커널 경로가 안 읽음), vllm#53063(DFlash 드래프트가 타깃의 짝짓기를 안 물려받음)과 그 후속 댓글(관례가 선언이 아니라 추론이라 config로 바로잡을 길이 없음). 설치된 vLLM 0.30.0에는 둘 다 고쳐져 있고, 모델 파일 20개가 GPT-J 식을 코드 상수로 든다. 근거와 판단 재료는 `PAIRING_CANDIDATE.md`. 구현은 검토·연구자 판단 뒤.

### E3 재측정 (실제 버그 8건, 재현된 것 전부; 규약 `M10_PROTOCOL.md` 3.3)

| # | 이슈 | 부류 | 1.0.0 (E3, 09-24) | 1.1.0 (M14+M15) | 근거 |
|---|---|---|---|---|---|
| 24, 37 | vllm#49377·#49449 낡은 블록 해시 (같은 결함) | 안 (정체·시점) | 놓침 | **해소**: 후보 첫 토큰 `,`→`
`, 캐시 16→0 = 참조 | `r4/e2e_off.json`, `e2e_on.json` |
| 62 | sglang#39626 K 타일 > 양자화 블록 | 안 (형식·입도) | 놓침 | **해소**: 64→288 | `m15/e3_39626_on.json` |
| 68 | vllm#58138 패딩 토큰 종류 | 안 (형식·입도) | 놓침 | **검출**: 기본 `broken` 4건, 멈춤은 점수 전 거부 | `m15/e3_58138_on.json`, `e3_58138_stop.json` |
| 73 | transformers#48967 두 어휘 | 안 (형식·입도) | 놓침 | **검출**: 기본 `broken`, 멈춤은 첫 id 전 거부, `entail check` exit 1 | `m15/e3_48967_on.json` |
| 22 | sglang#37606 CUDA 그래프 약한 참조 | 밖 (엔진 내부 메모리 수명) | 놓침 | 놓침(설계대로), 판정 0 | `m15/e3/sg37606_on.json` |
| 32 | sglang#35564 파서 스트리밍 | 밖 (파서 논리) | 놓침 | 놓침(설계대로), 판정 0 | `m15/e3/sg35564_on.json` |
| 53 | diffusers#14569 스케줄러 NaN | 밖 (부품 내부 산술) | 놓침 | 놓침(설계대로); 비-pass 1줄은 기존 `unknown`(허브 id 미캐시), `broken` 0 | `m15/e3/df14569_on.record.jsonl` |

- **부류 안 4/4** 차단(2 해소) 또는 정확히 보고(2, 멈춤 정책에서 차단). **부류 밖 3/3**은 설계대로 잡지 않으며 틀린 경보도 없다. 1.0.0의 0/8에서 왔다.
- **M15.8 추가(표본 밖, 2026-09-26):** 정지 id 부류(Llama 3 2024-04의 모양)를 transformers 5.17에서 재현해 **해소**했다(`stops_replay_off.json`·`_on.json`: off는 세 답이 160토큰 한도까지, on은 8·18·37토큰에서 멈춤). 이 부류는 E3 표본에 없었고 정적 훑기가 아닌 검토 에이전트의 순위에서 왔다.
- 분모에 주의: 8건은 229건 census에서 씨앗 순서로 12건을 골라 재현된 것이다. 이 수치는 "이 표본의 부류 안 결함"에 대한 것이지 전체 장애의 검출률이 아니다.
