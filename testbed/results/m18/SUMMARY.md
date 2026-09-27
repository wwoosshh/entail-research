# M18 참조 대조: 선언이 코드 속에만 있는 자리 (2026-09-26~)

연구자 지시 "진행해봐"(2026-09-26). 설계는 `LIBRARY_DESIGN.md` 11절 M18 행, 단계는 `ROADMAP.md` M18. 자(결과 전에 적음)는 로드맵에 있다: 회고는 검출률이 아니고, 검출률은 어휘 동결 뒤 3차 사전 등록 재현에서만 잰다.

## M18.1 토크나이저 참조 대조 (끝, 2026-09-26)

**규칙(`entail/tokenizer_contract.py`, 어휘 v9 `Tokenization`):** 폴더가 선언한 토크나이저를 실제로 돌린다 — tokenizer.json을 `tokenizers` 라이브러리로(파일의 padding·truncation은 지우고, BPE dropout이 있으면 비교하지 않음; sentencepiece 모델은 특수 토큰을 나눠 인코딩하는 참조를 재기 전에는 unknown). 엔진이 만든 토크나이저와 고정 탐침 10개(문장, 들여쓴 코드, 앞뒤 공백, 숫자·문장부호, 여러 문자, 이모지, 탭·CRLF, 마크업, 긴 반복)를 같이 인코딩해 id가 정확히 같아야 한다(`tokenizer_ids`). tokenizer_config.json의 `added_tokens_decoder`(없으면 tokenizer.json의 `added_tokens`, added_tokens.json)가 선언한 추가 토큰 전부를 엔진 토크나이저에서 찾아 같아야 한다(`added_token_id`). 어긋남은 `broken`(보고, 실행 계속; 고침 없음 — 결정은 만든 클래스와 처음 다른 탐침을 적는다). 돌릴 수 없는 선언(파일 없음, 라이브러리 없음, tiktoken.model은 pre-tokenization 패턴이 없어 추가 토큰만 비교)은 폴더마다 한 번 `unknown`. 폴더 자신의 플래그(`legacy`, `add_prefix_space`)가 파이프라인을 정확히 재현하는 자리의 차이는 선언끼리 어긋난 것(`sources_disagree`, unknown), 사용자가 준 빌드 인자가 설명하는 차이는 `user_choice`(unknown). 참조의 탐침 id는 `entail_logs/tokenizer_ids.json`(시작 폴더마다)에 파일 스탬프·라이브러리 판·참조 판으로 두어 첫 프로세스만 참조를 만든다.

**회고(`testbed/results/m18/retro/cases/`, `run_case.sh`로 off/on; 보고된 판의 venv):**

| 사례 | 판 | 재현 | entail 켠 실행 | 판정 |
|---|---|---|---|---|
| transformers#45356 Kimi-K2.5 `</think>` | 5.4.0 (tf540) | 재현 | **broken** `added_token_id`: `<|start_header_id|>`를 163590으로 선언, 엔진은 163589; 추가 토큰 23개 중 18개 다름(+ 탐침은 tiktoken이라 unknown) | reported |
| transformers#46489 deepseek-coder | 5.10.2 (tf5102) | 재현(LlamaTokenizer) | **broken** `tokenizer_ids`: 탐침 9/10 다름 | reported |
| transformers#45812 Granite | 5.8.0 (tf580) | 재현(GPT2Tokenizer) | **broken** `tokenizer_ids`: 4/10 | reported |
| transformers#46710 R1-Distill-Llama | 5.12.1 (tf5121) | 재현(LlamaTokenizer) | **broken** `tokenizer_ids`: 9/10 | reported |
| 같은 넷 | 5.17.0 (gpu) | 미재현 | pass 3(탐침 10/10, 추가 토큰 22·96·256 일치), Kimi는 unknown(추가 토큰 23/23 일치, 탐침은 비교 못 함) | — |

2차 재현에서 놓친 부류 안 8건 가운데 셋(그리고 1차의 tf46710)이 이 규칙에 잡힌다. **회고이지 검출률이 아니다.**

**오탐과 비용(`testbed/results/m18/e2_tokenizers/`, `testbed/m18_tokenizers.py`; E2 38 모델의 폴더, transformers 5.17.0, 세 엔진 모두 이 AutoTokenizer로 만든다; 검토 반영 뒤의 최종값):**

| 자 | 값 |
|---|---|
| pass | 36 |
| broken | 2 — TinyLlama-1.1B-Chat과 hmellor/tiny-random-Llama: 그 집합의 유일한 Llama-2 시절 토크나이저(38개 폴더에 서로 다른 토크나이저는 11개, 23개는 Qwen2의 것) |
| unknown | 0 |
| 첫 프로세스 추가 비용(참조 만들기 포함) | 중앙값 +241 ms, p90 +443, 최대 +894(33 MB tokenizer.json) (`summary_run1_before_disagree_rule.json`) — 맨 빌드 중앙값 189 ms 위에 |
| 기계 캐시 뒤 추가 비용 | 중앙값 +2 ms, p90 +36, 최대 +152 (`summary.json`) — 폴더마다 off/on 한 쌍이라 수 ms는 분해능 아래(검토 지적 11); M18.6에서 어댑터 하나만 끄고 짝 비교로 다시 잰다 |

**정적 폴더 300개(`testbed/results/m18/static/tokenizers.json`, `testbed/m18_static_tokenizers.py`; M10 E1 말뭉치, `sites.check_static`의 토크나이저 검사, transformers 5.17.0, 원격 코드 없이):** pass **201**, broken **9**, unknown **5**, 비교할 토크나이저 없음 85(GGUF 저장소 등). broken 9는 두 모양이다. (a) Llama-2 시절 legacy 내보내기 여섯(TinyLlama, CodeLlama-7b-hf, MiniCPM-SALA-AWQ, EuroLLM-22B-Instruct, 시험 폴더 둘): 공백으로 시작하는 탐침에서 파일과 다름(아래 발견 1). (b) 바이트 수준 BPE tokenizer.json 위에 `LlamaTokenizerFast`를 선언한 셋 — **deepseek-ai/DeepSeek-R1-0528-Qwen3-8B**, deepseek-ai/deepseek-coder-7b-instruct-v1.5, lmstudio-community의 MLX 내보내기: 5.17.0이 `LlamaTokenizer`를 만들어 파일의 Split 정규식 pre-tokenizer를 버리고 Metaspace를 달아, 탐침 10개 중 9개가 다르고 `"How are you doing?"`이 `How are y oud o ing ?`로 나뉘어 `Howareyoudoing?`으로 복원된다(`static/llamafast_check.json`; transformers#46710의 부류가 이 폴더들에서는 5.17.0에 살아 있음. 상류 검색 결과 없음. 초안 `issue_track/llama_legacy/ISSUE_transformers_llamatokenizerfast_bytelevel.md`, 게시는 연구자 결정). unknown 5: dolphin-2.9.1-yi-1.5-34b(`add_prefix_space=true`·`legacy=true` 대 접두 없는 파일: 선언끼리 어긋남, 플래그의 파이프라인이 엔진의 id를 재현), OPT 둘·macbert·tiny-gpt2(vocab.json/vocab.txt만: 돌릴 수 있는 선언 없음).

**발견 1(엔진의 실제 차이; 검토 지적 3으로 고친 서술; `results/m18/llama_legacy/legacy_flag_check.json`):** transformers 5의 `LlamaTokenizer`는 tokenizer.json을 그대로 쓰지 않고 vocab·merges로 다시 만들며 `Metaspace(prepend_scheme=always|first)`를 단다(`tokenization_llama.py`). **플래그는 따른다**: `legacy=true`면 always, false면 first이고, `"</s>Hello there"`는 always에서 파일과 같은 `[2, 15043, 727]`, first에서 `[2, 10994, 727]`. 플래그와 무관한 차이는 공백으로 시작하는 텍스트다: 파일(과 sentencepiece의 add_dummy_prefix)은 `'▁'`를 겹쳐 `[1678 '▁▁▁', ...]`, Metaspace는 always든 first든 `[259 '▁▁', ...]`. 그래서 Llama-2 시절 폴더는 legacy 값과 무관하게 그 탐침에서 `broken`이다(정적 말뭉치 여섯, E2 둘). TinyLlama(`legacy=false`)는 특수 토큰 뒤 구간에서도 파일과 다르고(`</s>` 뒤 `'▁'` 29871이 사라짐), 이것은 플래그가 설명하므로 결정에 "1 further differ only after a declared token"으로 적힌다. 효과(`llama_legacy/tinyllama_effect.json`, TinyLlama-1.1B-Chat bf16, 탐욕 48토큰): 채팅 프롬프트 8개 모두 id가 하나(29871) 다르고 **탐욕 출력이 7/8 다르다**(대개 표현 차이; 실질적으로 다른 하나는 파일 쪽 id가 틀린 설명을 냄). 4.x의 fast 토크나이저는 파일을 그대로 썼으므로 4.x 사용자의 id는 파일 쪽이었다; 어느 쪽이 학습 때의 id인지 폴더는 말하지 않는다. 초안 `issue_track/llama_legacy/ISSUE_transformers_llama_legacy.md`(고쳐 씀), 게시는 연구자 결정.

**검토 반영(읽기 전용 검토 에이전트 1회, 지적 15건; `scratchpad/review_m181_report.md`; `LIBRARY_DESIGN.md` 11절 "M18.1 검토 반영" 행):** 반영 11건 — 선언끼리 어긋남은 플래그의 파이프라인이 재현하는 자리만 `sources_disagree`(TinyLlama는 unknown → broken), "5.17이 legacy를 무시" 서술 정정, sentencepiece 참조는 재기 전에는 unknown, padding·truncation·dropout, 추가 토큰 전부 비교와 예외 처리, 캐시 키(라이브러리 판·참조 판)와 환경 의존 실패 미캐시, 엔진 토크나이저의 padding 되돌리기, 콘솔 인코딩 오류 대비, 사용자 인자 `user_choice`, 정적 검사의 원격 코드·선언 없는 폴더·개별 try, subfolder, README·CHANGELOG 문구(회고 표시, Kimi unknown, 추가 토큰 서술, 집합 설명, 두 비용). 미룬 것 4건은 `results/m16/DEFERRED.md` 24행.

**설계 결정(`LIBRARY_DESIGN.md` 11절 M18.1 행 둘):** 선언이 데이터로 있는 자리는 선언과, 코드 속에만 있는 자리는 참조를 돌려 행동으로 비교한다. 폴더의 선언끼리 어긋나면 플래그가 설명하는 자리만 `unknown`, 나머지는 `broken`. 참조의 실행 비용은 기계 캐시로 낸다.

**시험:** `tests/test_tokenizer_contract.py` 13건(일치, 탐침 어긋남, 추가 토큰 어긋남·미지·예외, 선언 없음, tiktoken, 플래그가 설명하는 어긋남과 아닌 것, 사용자 인자, padding·truncation·dropout, 캐시와 키, 정지 정책). 전체 54 파일 481 검사.

**남은 것:** sentencepiece-only 폴더는 특수 토큰을 나눠 인코딩하는 참조를 재기 전에는 unknown이다. tiktoken 폴더의 텍스트 탐침은 비교하지 못한다(pre-tokenization 패턴이 원격 코드에만 있음). vLLM의 `tokenizer_mode=mistral`(tekken.json)과 transformers의 `MistralCommonBackend`는 이 훅을 지나지 않는다. 미룬 검토 항목은 DEFERRED 24행.

## M18.2 커널 참조 대조: vLLM CustomOp의 커널 대 정의 (끝, 2026-09-26)

**규칙(`entail/kernel_reference_contract.py`, 어휘 v9 `KernelReference`; 어댑터 `adapters/vllm_kernel_reference.py`):** vLLM의 CustomOp는 정의(`forward_native`)와 커널(`forward_cuda`)을 함께 들고 다닌다. 모델이 다 만들어진 뒤(`process_weights_after_loading`) 커널 경로로 디스패치된 모듈마다 `_forward_method`를 감싸고, 클래스·입력 패턴마다 프로세스에 한 번, 실제 입력의 복제본을 토큰 차원 64행으로 잘라 커널과 정의를 돌려 비교한다(정의는 입력 dtype으로 한 번, float32로 한 번). 규칙 `kernel_reference_mismatch`: max|커널−정의| > FACTOR(8)×바닥(dtype 정의와 float32 정의의 차이) + ATOL_ULPS(4)×출력 dtype ulp×scale이면 `broken`. 결정 뒤 원래 메서드를 되돌린다(정상 경로 비용 0). 더미 입력(모두 0)이나 정의가 정확한 입력(위치 0의 rotary: 항등)은 아무것도 말하지 않으므로 다음 호출을 기다린다(최대 256회: 프로파일 실행이 모든 층에 0과 위치 0을 주고 rotary 인스턴스는 층 사이에 공유된다). 토큰 차원은 가장 큰 텐서의 첫 차원이고, 모든 텐서가 그 차원을 공유해야 자른다(MRoPE 위치 [3, n]은 마지막 차원). 공유하지 않는 연산(어텐션의 cu_seqlens)·forward를 덮어쓴 연산(mamba 믹서)·CUDA 그래프 캡처 중의 호출은 비교하지 않는다(앞의 것은 `unknown` 한 번).

**회고(`results/m18/retro/cases/vl42016_*`):** vllm#42016(GLM-OCR, vLLM 0.22.0의 Triton MRoPE 커널이 짝짓기를 무시)에서 `MRotaryEmbedding`의 33번째 호출(프로파일 실행 뒤 첫 실제 입력)에서 **broken**: max|커널−정의| 10.5, 바닥 0.0309, scale 10.9, 허용 0.588 — 값 크기만큼 어긋난다. 같은 사례를 0.30.0(커널 수정 뒤)에서 돌리면 pass(0.00343, 바닥 0.00343). M17.4의 표(아키텍처별 짝짓기, 판별 조건) 없이 잡힌다. 부류 밖 두 건(vllm#56578 softcap NaN, #50427 FlexAttention int32)은 CustomOp가 아니라 닿지 않는다.

**정상 실행(`results/m18/e2_kernels/`, `testbed/m18_kernel_summary.py`; E2 8개 모델, vLLM 0.30.0 eager; 처음 측정 — 검토 2회차 뒤의 재측정과 그 뜻은 아래 "M18.2·M18.3 검토 2회차 반영" 절이 대신한다: 이 32건의 대부분은 정의를 정의와 비교한 것이었다):** 결정 32건 모두 pass(RMSNorm·GemmaRMSNorm·SiluAndMul·GeluAndMul·RotaryEmbedding·Llama3RotaryEmbedding·Mixer2RMSNormGated·UnquantizedFusedMoEMethod). 감싼 모듈 수: Qwen3-0.6B 143, Llama-3.2-3B 87, GLM-OCR 257. diff/바닥 비는 중앙값·p90·최대 모두 **1.00**(최대 절댓값 오차가 둘 다 같은 최대 크기 원소의 bf16 반올림에 지배되기 때문: 요소별 연산에서는 비가 정보를 주지 않는다), diff/scale 중앙값 0.0022, 최대 0.0068(bf16 ulp 두 개 안). 그래서 FACTOR 8·ATOL_ULPS 4는 정상 실행에서 넉넉하고, 짝짓기 버그는 diff/scale ≈ 1로 어떤 자에도 걸린다. 값을 고정한다(설계 11절 M18.2 행 (3)). `unknown` 두 종류: `ApplyRotaryEmb`·`MMEncoderAttention`(비전 타워: 인수가 토큰 차원을 공유하지 않음).

**비용:** 클래스·패턴마다 한 번(64행) + 더미 호출의 재시도(작은 슬라이스). 8개 모델의 적재 시간은 entail 없는 실행과 같은 범위(12.9~23 s, 이전 측정과 같음); 정상 경로는 감싸기를 되돌려 0. 정밀 측정은 M18.6에서 한다.

**남은 것(M18.2b):** 정의가 없는 연산 — FusedMoE의 양자화 경로(vllm#58532 INT8 스케일 배치, #48895 Marlin 행 대응), FLA 커널(#38643) — 은 entail이 torch 참조를 들고 다녀야 잡힌다. 연구자 결정 사항이다(참조를 라이브러리가 소유하면 "엔진의 정의"가 아니라 "entail의 정의"가 된다).

## M18.3 파서 참조 대조: 스트리밍 대 전체 텍스트, 도구 호출 대 스키마 (끝, 2026-09-26)

**규칙(`entail/parse_contract.py`, 어휘 v9 `Parse`; 어댑터 `adapters/vllm_parse.py`):** vLLM 0.30의 채팅 파서는 같은 텍스트를 두 길로 푼다 — 스트리밍은 `parse_delta`(토큰마다, 마지막에 `finished=True`), 비스트리밍은 `parse`(전체). 어댑터는 서버가 요청마다 파서를 만드는 클래스(`ParserManager.get_parser`가 돌려주는 것; serve 어댑터의 감싸기와 합성)를 감싸, 인스턴스가 만들어진 인자를 기억하고 `parse_delta`가 내는 델타(content·reasoning·tool_calls의 이름과 인자 조각)를 모은다. 스트림이 끝나면 같은 클래스의 새 인스턴스로 전체 텍스트를 `parse`하고 둘을 정확히 비교한다(인자는 JSON 값으로): `stream_differs_from_full`. 도구 호출의 인자는 요청이 선언한 도구의 `parameters.properties`에 맞아야 한다(`additionalProperties`가 허용이 아닐 때 선언에 없는 키, 선언에 없는 도구 이름): `tool_args_outside_schema` — 스트리밍 끝과 `parse` 둘 다에서. 요청이 reasoning을 안 받으면(`include_reasoning=False`) reasoning은 비교하지 않는다. 고침 없음(클라이언트는 이미 스트림을 받았다).

**회고(규칙 수준: 엔진의 실제 파서 클래스 + entail의 실제 어댑터, HTTP 서버는 아님; `testbed/m18/parse_retro.py`, `results/m18/parse_retro/summary.json`, vLLM 0.30.0):**

| 사례 | 파서 | 판정 |
|---|---|---|
| vllm#49316 스트리밍이 형 강제를 건너뜀(4 텍스트) | kimi_k2 | **broken** 4/4: `'{"count": "3"}'` 대 `'{"count": 3}'` 등 |
| vllm#49412 도구 호출 주변 공백(3 텍스트) | qwen3 | **broken** 2/3: `' done.'` 대 `''`, `' mid '` 대 `''`(content 자체가 사라짐); `'Sure! '` 대 `'Sure!'`는 양끝 공백뿐이라 pass에 메모(M18.5에서 정함: 실제 서버의 정상 도구 호출 스트림마다 나는 차이) |
| vllm#47986 tool_b가 tool_a의 스키마로 풀림(전체 텍스트) | deepseek_v4 | **broken**: "tool call 1 (tool_b) lacks its required ['arguments'] (it carries ['city']); ... carries ['city'], which its declared parameters (arguments) do not have and the tool forbids" — 검토 2회차 뒤 tool_b의 선언을 정확히 해서(`arguments` 객체 하나, required, additionalProperties false) 사례가 가른다: 파서 자신의 풀기 함수에 바른 슬롯을 준 대조(고친 조회가 낼 결과 `{"arguments": {"city": "Tokyo"}}`)는 같은 규칙에서 **pass** |
| 정상 텍스트 7개(kimi_k2 형 맞는 호출·평문, qwen3 호출 하나·둘·평문, deepseek_v4 호출 하나, 고친 슬롯 조회 대조) | — | 결정 없음 7(조용한 pass는 세기만 하고 기록하지 않는다: 검토 2회차 결정); broken 0 |

기대한 8건 가운데 7건이 broken, 1건(양끝 공백)은 pass+메모, 정상 7건에 오탐 0(`parse_retro/summary.json`, 검토 2회차 반영 코드로 다시 잰 최종값). vllm#48217(0.24.0 Gemma4, 스트리밍이 답을 reasoning으로)·#42047(0.19 Gemma4ToolParser 부동소수)은 0.30에 없는 옛 API(`extract_*_streaming`)라 이 어댑터의 자리가 아니다(단위 시험의 모사로만 덮음). 평정자 합의로는 #47986만 부류 안(K1)이고 나머지는 K4(경로·상태 기계)지만 자리는 같다.

**남은 것:** 실제 서버(스트리밍 요청)에서의 오탐과 비용은 M18.5·M18.6에서 잰다. qwen3 `after_tool`에서 전체 텍스트 경로가 도구 호출 뒤의 content를 아예 버리는 것(`''`)은 보고(#49412)의 "공백" 설명보다 큰 차이다 — 기록만 한다.

## M18.2·M18.3 검토 2회차 반영 (2026-09-26)

읽기 전용 검토 에이전트(보고 `scratchpad/review_m182_m183_report.md`, 15건: 반드시 7·권고 6·메모 2)의 지적을 반영하고 같은 자로 다시 쟀다. 결정은 `LIBRARY_DESIGN.md` 11절 "M18.2·M18.3 검토 반영" 행에, 미룬 것은 `results/m16/DEFERRED.md` 28~30에 있다.

**커널 참조(`kernel_reference_contract.py`, `adapters/vllm_kernel_reference.py`), 바뀐 것:** 값마다·출력 텐서마다 자기 dtype으로 판정(비유한 값은 그 자체로 어긋남; NaN이 `max`에 삼켜지던 결함을 고침), 정의 실행 뒤 연산의 텐서를 되돌림(rotary의 캐시 dtype 변환), 엔진 상태를 든 연산·랭크 둘 이상·torch.compile 아래·모듈에서 닿지 않는 연산·자를 수 없는 큰 인자는 한 번 unknown, 정지 정책은 한 번 서고 비켜섬, 자름은 커널이 인자를 손대기 전에, 키에 모듈 경로·인스턴스 스칼라 속성 추가, TRIES 64, `_dummy_run` 표시로 엔진의 프로파일·워밍업 호출은 세지도 비교하지도 않음.

**측정 중 드러난 것 둘.** (1) vLLM 0.30은 GPU 모델 러너가 둘이다(`vllm.v1.worker.gpu_model_runner`와 새 `vllm.v1.worker.gpu.model_runner`; 워커가 하나를 고르고 이 환경은 새 쪽을 썼다). 첫 반영은 옛 러너만 훅해 프로파일 실행이 실제 호출로 세어졌고, 결정이 "call 29"(층 수만큼 지난 뒤)에 나거나 RMSNorm 넷이 "같은 행이 64번" unknown이 됐다(`e2_kernels/probe/qwen3_0.6b*.record.jsonl`: `dummy_hook: false`, 커널 경계의 timing 줄 349). 세 러너 모듈(`gpu_model_runner`, `gpu.model_runner`, `mm_encoder_model_runner`)의 `_dummy_run`을 모두 표시한 뒤 모든 결정이 실제 입력의 **call 1**에 나고 timing 줄은 프로세스당 7~11이다. (2) **eager 모드의 vLLM 0.30에서 RMSNorm의 "커널 경로"는 정의 자체다.** `RMSNorm.forward_cuda`는 배치 불변 모드가 아니면 `return self.forward_native(x, residual)`이고(융합 rms_norm 커널은 torch.compile의 융합 패스에서만 닿는다), GemmaRMSNorm도 같으며, Nemotron의 Mixer2RMSNormGated는 n_groups≠1이라 정의로 빠진다. 처음 E2가 "32건 pass"라고 한 것의 대부분은 정의를 정의와 비교한 것이었다(검토 지적 11b는 둘만 짚었는데 실제로는 더 많다). 커널을 돌리는 동안 정의가 실제로 불리는지 감시해(인스턴스 속성으로 잠시 가로챔) 결정마다 "the dispatched method calls the definition itself on this input"이라고 적는다.

**정상 실행(`results/m18/e2_kernels/`, 같은 8개 모델, vLLM 0.30.0 `enforce_eager`, 최종 코드):** 결정 35건 = **pass 34, unknown 1**(Qwen3-4B-FP8: 선형 커널 도우미가 든 `quant_fp8` 144 인스턴스는 모듈에서 닿지 않아 한 번 unknown). broken 0. 결정은 모두 실제 입력의 call 1(Nemotron은 28행: 프롬프트가 28토큰). **pass 34 가운데 독립된 커널을 비교한 것은 13건**(RotaryEmbedding 5, Llama3RotaryEmbedding 1, SiluAndMul 5, GeluAndMul 1, ReLUSquaredActivation 1)이고 **21건은 정의를 정의와 비교한 것**(RMSNorm 17, GemmaRMSNorm 2, Mixer2RMSNormGated 1, UnquantizedFusedMoEMethod 1: 위 (2)). 비트까지 같은 28건 = 정의 경로 21 + 활성 커널 7(vLLM의 활성 커널은 PyTorch 구현의 반올림 순서를 그대로 따른다); rotary 6건은 비트까지 같지 않다. 값별 허용치 대비 최악 비(margin)는 34건에서 **중앙값 0.062, 최대 0.095**, 독립 커널 13건만 보면 0.059~0.081 — 지금의 FACTOR 8·ATOL_ULPS 4는 정상 실행에 약 10배 여유가 있고, 이 값은 M18.6의 E2(38 모델)에서 데이터로 정한다. 회고 vl42016(0.22.0의 Triton MRoPE 커널: 독립 커널)은 코드가 바뀐 뒤 다시 돌리지 않았다(값 크기만큼의 차이라 값별 규칙에서도 broken이다; M18.6에서 재확인). **뜻:** eager 모드의 vLLM 0.30에서 이 규칙이 실제로 감시하는 독립 커널은 rotary와 활성 함수뿐이고, 정규화·양자화의 융합 커널은 torch.compile 아래에 있어 지금은 비교하지 않는다(DEFERRED 28).

**파서 참조(`parse_contract.py`, `adapters/vllm_parse.py`), 바뀐 것:** 델타는 기록 없이 모으고 스트림 끝에 한 번 `safely`로 판정, 조용한 pass는 세기만(메모 있는 pass는 기록), 끝나지 않은 출력(토큰 한도, 열린 reasoning, 강제 도구의 인자가 JSON이 못 됨)의 차이는 unknown, 폴백이 reasoning을 content로 다시 보낸 것은 메모, 스키마는 `required` 누락과 `additionalProperties: false`의 추가 키만 broken(열린 도구의 추가 키는 pass의 메모), 키가 모델 텍스트에 있는지로 출처를 말함, 참조 파서 생성 실패는 클래스마다 한 번 unknown. **회고 재측정(`parse_retro/summary.json`):** 기대 8건 중 7 broken + 1 pass·메모(양끝 공백), 정상 7건(고친 슬롯 조회 대조 포함) broken 0. vl47986은 tool_b의 선언을 정확히 해서(`arguments` 객체 하나, required, additionalProperties false) 결함(`{"city": ...}`: required 누락 + 금지된 키)과 고친 조회(`{"arguments": {...}}`: pass)를 가른다 — 처음 사례는 어느 쪽이든 broken이라 가르지 못했다(지적 13c).

**시험:** 55 파일 497 검사.

## M18.4 자리표 출처와 logprobs 범위 (끝, 2026-09-26)

**규칙 1(`entail/placeholder_contract.py`, 어휘 v9 `Placeholder`; 어댑터 `adapters/vllm_multimodal.py`):** vLLM의 멀티모달 프로세서가 자리표를 펼치고 항목마다 묶은 구간을 정한 뒤(`_maybe_apply_prompt_updates`), 모델 설정이 선언한 마크업(`vision_start_token_id` 뒤에 `image_token_id`/`video_token_id`; Qwen-VL 계열)에 대해 구간 바로 앞 토큰이 선언된 시작 토큰인지 본다. 아니면 그 구간은 사용자 글에서 온 것이다(`placeholder_outside_markup`, broken). 마크업을 선언하지 않는 모델은 판정하지 않는다. 엔진의 합성 프롬프트(메모리 프로파일링: 0번 토큰부터 자리표가 잇달아, 템플릿 없음)는 구간 앞에 아무것도 없거나 다른 자리표가 있으므로 판정하지 않는다(첫 실행에서 프로파일 입력 5줄이 broken으로 나와 고침).

**규칙 2(`parse_contract.check_logprobs`; `adapters/sglang_serve.py`의 `_build_chat_response` 훅):** 채팅 응답의 logprob 토큰을 이어 붙인 것이 메시지의 content(양끝 공백 제외)여야 한다. 서버가 갈라낸 reasoning 구간(과 `<think>` 마커)까지 덮으면 `logprobs_cover_other_text`(broken).

**회고(`results/m18/retro/cases/vl57740_*`, `sg25055_*`):**

| 사례 | 판·환경 | entail 켠 실행 | 판정 |
|---|---|---|---|
| vllm#57740 사용자 글 속 `<\|image_pad\|>` | vLLM 0.30.0, Qwen2.5-VL-3B | 공격 순서: **broken** "the image placeholder bound at tokens 20..275 is preceded by id 220, not by the declared start of the markup (id 151652)"; 대조 순서(15..270, `<\|vision_start\|>` 뒤): pass | reported |
| sglang#25055 logprobs가 reasoning을 덮음 | SGLang 0.5.20, Qwen3-0.6B, qwen3 파서, `separate_reasoning` | **broken** "the 155 logprob tokens cover the reasoning span (545 characters and its markers) as well as the content (12 characters)" | reported |

둘 다 규칙을 그 버그에서 썼으므로 회고다.

## M18.5 오탐 자 넓히기 (2026-09-26)

| 자 | 설정 | 결과 |
|---|---|---|
| 실제 서버, 모든 어댑터 | vLLM 0.30.0 `serve` Qwen3-0.6B, `--reasoning-parser qwen3 --tool-call-parser hermes --enable-auto-tool-choice`, eager; 요청 9종 × (스트리밍, 비스트리밍) = 18(평문, thinking on/off, 도구 호출 둘, tool_choice=none, 길이 잘림, n=2, include_reasoning=false) | 오류 0. 결정: Parse pass 10·broken 0(양끝 공백 메모를 정하기 전에는 broken 2: hermes 파서가 도구 호출 스트림마다 `'\n\n'`을 흘리고 전체는 `''`), KernelReference pass 5, Tokenization pass 2, 나머지 pass. `results/m18/e2_server/` |
| 추측 디코딩 | vLLM 0.30.0 ngram(초안 3, Qwen3-0.6B), 모든 어댑터 | broken 0, refused 0(`allocate_slots` 결정 없음; M17.6 뒤 `kv_needed`를 좁힌 결과가 유지됨). KernelReference pass 5. `results/m18/e2_spec/` |
| 다중 KV 그룹(하이브리드) | Nemotron-3-Nano-4B(Mamba+어텐션), vLLM 0.30.0 eager, M18.2 8개 모델 실행 | broken 0, refused 0(unknown은 Coverage·Layout의 알려진 것). `results/m18/e2_kernels/engines/` |
| PD 분리 | — | 한 장으로는 재지 않았다(기록만). |

**M18.6에서 할 것:** E2 102회(38 모델 × 3 엔진)를 모든 어댑터로 다시 돌려 오탐 0과 적재 비중을 확인하고, 어휘 v9를 동결한 뒤 3차 사전 등록 재현(순서 301번부터)을 한다.

## M18.6 3차 사전 등록 재현 (2026-09-27, 연구자 지시 "사전 등록 재현으로 새 능력 재봐"; 평정 전)

- **동결:** entail `07fceac`(M18.1~M18.5 + M19 L3; 규약 `testbed/M16_PROTOCOL.md` 6절 2026-09-27 행과 8절). M18.6 E2의 오탐 6회(Tokenization, Llama-2 시절 폴더 둘)는 고치지 않은 채 동결했다. 심사·재현 중 entail은 바꾸지 않았다(이 절의 수정은 재현 스크립트와 결과 도구뿐).
- **심사:** 순서 301~450번 150건(규약 3절 그대로, 건너뛰지 않음) → `pass` 10, `cannot_run` 83, `not_output` 54, `no_repro_info` 3. 판정과 이유는 `replay3/screening.json`(묶음별 `decisions_*.json`, 보여 준 본문 `show_*.txt`). 한도를 넘는 모델은 원리가 작은 모델에서 보여도 보고된 모델로 판정했다(1·2차의 선례). 커널이 이 GPU에서 빌드되지 않는 것 하나(sglang#35257, JIT 실패)는 시험해 확인했다.
- **재현:** 사례마다 entail 끄고 한 번, 켜고 한 번(`testbed/m10_e3/run_case.sh`, 기본 정책; 스크립트 `testbed/m18/replay3/cases/`, 결과 `replay3/cases/`). 보고된 판 환경은 새로 하나(vLLM 0.21.0) 만들었다(한도 6).

| 사례 | 재현 | entail 켬(기대 자리) | 판정 |
|---|---|---|---|
| vllm#58406 mm_processor_kwargs 병합 | 0.30.0, 보고의 스니펫: 비디오 fps None, size 바뀜 | 결정 없음(설정 수준 스니펫, 자리 없음) | missed |
| vllm#57353 kimi_k3 비스트리밍 대 스트리밍 | 0.30.0, 서버와 같은 경로(ParserManager): 같은 글이 스트리밍은 reasoning, 전체 파싱은 content | **제자리(`request:vllm.parser`) unknown**: 차이를 봤으나 "추론 블록 안에서 끝난 출력"이라 탓하지 않는 규칙(M18.3 결정 7) | unknown_only |
| diffusers#14213 조상 스케줄러 NaN | 0.40.0, 보고의 스니펫: NaN 192 | 결정 없음 | missed |
| transformers#48051 convert_to_rgb | 5.17.0 재현 안 됨(고쳐짐), 5.12.1 재현(모양 (2,2,1,1)) | 결정 없음 | missed |
| transformers#47328 Omni DiT 로터리 | 5.12.1, 라이브러리 자신의 모듈로: 스프레드 9.9(5.17.0: 4.5e-6) | 결정 없음(짝짓기 규칙은 vLLM 쪽에만) | missed |
| transformers#49066 Qwen2Tokenizer 사전 토큰화 | 5.17.0, 보고 스크립트: 힌디어 글 id 6 대 10 | **제자리(토크나이저) pass**: 탐침 10개가 모두 같게 인코딩됨. 탐침에 결합 부호(`\p{M}`)가 없다 | missed |
| transformers#45491 EmbeddingGemma NaN | — 모델이 게이트(라이선스 수락 필요, 401) | — | not_reproduced |
| vllm#49250 거절된 KV 적재 뒤 재계산 | 0.30.0, 보고의 커넥터·모델: 기준 " Rome." 대 동기 " capital of Spain is Madrid." | 설정 키·토크나이저·템플릿 unknown, 커널 호출 pass; 재계산 경로에는 자리 없음 | missed |
| vllm#43962 Triton 어텐션 오답 | 0.30.0·0.21.0 모두 재현 안 됨(두 백엔드 모두 "512") | — | not_reproduced |
| vllm#43602 Qwen3-VL deepstack(컴파일) | 0.22.0, 고친 0.30.0을 대조로: 이미지 첫 토큰 분포 차 0.75~3.74 nats(0.30.0은 0.13~0.25), 설명 하나가 틀림 | KernelReference unknown 26(컴파일 아래 커스텀 연산), deepstack 입력 자리에는 없음 | missed(재현은 사후 대조, 아래) |

- **평정 전 머리 수치:** 재현 8(엄격하게는 7), **검출(resolved·reported) 0/8**. unknown_only 1, missed 7, 재현 안 됨 2. 자리 덮임(기대 경계에 판정이 하나라도 있음) **2/8**(1차 2/7, 2차 1/8). entail 켬 실행 전체에서 broken·refused **0**(오탐 0). entail이 깨뜨린 실행 0(Qwen3-VL 첫 시도의 엔진 실패는 스크립트의 메모리 설정 때문이고 끔·켬 모두 같았다). 검출률의 분모(재현 ∩ 합의 K1)는 눈가림 평정 뒤에 정해진다. 검출이 0이므로 어떤 분모에서도 점은 0이고, 3의 법칙 상한은 3/n(n ≤ 8)이다.
- **새 능력이 닿은 곳:** 이번 표본에서 M18.3 파서 참조는 한 건(vl57353)에 닿아 unknown을 냈고, M18.1 토크나이저 참조는 한 건(tf49066)에 닿아 pass를 냈다. M18.2 커널 참조는 컴파일 모드라 비교하지 않았다(unknown). M19 L3의 정의(vLLM `fused_experts`·블록 FP8 행렬곱, SGLang GDN 게이트)를 부르는 사례와 자리표 규칙(M18.4)에 닿는 사례는 없었다.
- **사후 판단을 밝힘(vl43602):** 사례 스크립트에 미리 적은 기준은 "이미지 답만 다르고 글만 있는 답은 같다"였다. 글만 있는 프롬프트 하나가 두 판 모두에서 경로 사이에 갈려(근소한 동률) 이 기준은 충족하지 않았다. 고친 판(0.30.0)의 같은 측정을 본 뒤 재현으로 셌다. 이 대조가 없으면 not_reproduced이고, 어느 쪽이든 검출 수는 0이다.
- **실험 중 본 것:** 채택하지 않고 `results/m16/DEFERRED.md` 31~35행에 적었다(토크나이저 탐침에 결합 부호, 파서의 미완결 규칙이 분류 뒤집힘까지 덮음, KV 커넥터 재계산, 처리기 kwargs 병합, transformers 쪽 짝짓기). 3차 사례로 규칙을 고치면 회고가 되므로 3차 검출률에 넣지 않는다.
- **남은 것(연구자 허락):** 눈가림 평정. 자료 `replay3/rating_packet.md`(96건, 시드 20260929; 규칙 1을 통과한 모든 이슈)와 지시문은 준비됐다. 평정자 둘(+갈린 것은 셋째)은 세션 밖이나, 허락이 있으면 세션 안 에이전트로 돌린다.
