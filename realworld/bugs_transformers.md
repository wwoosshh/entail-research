# huggingface/transformers 추론 출력 정확성 버그 파일럿

실행일: 2026-09-22. 검색 API 호출은 12:34–12:36 UTC에 했다. GitHub에는 GET 요청만 보냈다. 댓글, 반응, 라벨은 남기지 않았다.

## 1. 후보 수집

호출 형식: `gh api -X GET search/issues -f q='<q>' -f per_page=100`. 호출 사이에 10초를 쉬었다. 모든 키워드의 total_count가 100 이하라서 2쪽은 받을 필요가 없었다. 모든 응답에서 `incomplete_results`는 false였다.

| KW | q (그대로) | total_count |
|---|---|---|
| wrong | `repo:huggingface/transformers is:issue in:title wrong created:2025-01-01..2026-09-22` | 55 |
| incorrect | `repo:huggingface/transformers is:issue in:title incorrect created:2025-01-01..2026-09-22` | 74 |
| garbage | `repo:huggingface/transformers is:issue in:title garbage created:2025-01-01..2026-09-22` | 4 |
| gibberish | `repo:huggingface/transformers is:issue in:title gibberish created:2025-01-01..2026-09-22` | 1 |
| nonsense | `repo:huggingface/transformers is:issue in:title nonsense created:2025-01-01..2026-09-22` | 0 |
| accuracy | `repo:huggingface/transformers is:issue in:title accuracy created:2025-01-01..2026-09-22` | 7 |
| mismatch | `repo:huggingface/transformers is:issue in:title mismatch created:2025-01-01..2026-09-22` | 45 |
| "different output" | `repo:huggingface/transformers is:issue in:title "different output" created:2025-01-01..2026-09-22` | 11 |
| corrupted | `repo:huggingface/transformers is:issue in:title corrupted created:2025-01-01..2026-09-22` | 3 |
| degraded | `repo:huggingface/transformers is:issue in:title degraded created:2025-01-01..2026-09-22` | 0 |

- 키워드별 합계 200. 이슈 번호로 합친 합집합은 **196개**다.
- 합집합의 상태: closed/completed 188, closed/not_planned 4, open 4.

## 2. 필터

판정 방법:
- 196개 전부를 제목으로 먼저 나눴다. 제목만으로 통과가 확실한 것 52개, 모호한 것 61개, 제외가 확실한 것 83개였다.
- 모호한 61개는 본문 앞부분을 읽고 판정했다. 선정 구간(아래) 안에 드는 것은 본문, 댓글, 연결된 PR까지 모두 읽었다. 61개 중 23개가 통과했다. 제목만으로 통과시킨 52개 중 선정 구간 밖의 것은 본문을 읽지 않았으므로 75는 대략적인 수다. 선정된 25개는 모두 본문, 댓글, 타임라인, 연결 PR을 읽고 분류했다.
- 통과 기준: 사전학습 모델의 `generate()`나 forward 결과(로짓, 생성 토큰, 예측, 은닉 상태)가 틀리거나, 깨지거나, 조건에 따라 달라진다는 보고. 해당 체크포인트의 토크나이저가 토큰 ID나 디코딩 문자열을 틀리게 만드는 경우도 넣었다(N3 범주가 이 경우를 위해 있다).
- 제외 기준: 예외로 멈추기만 하는 경우, 로딩 실패, 학습 전용(손실, 기울기, 가중치 초기화), 문서, 오류 메시지, 성능, 설치, 모델 출력에 영향이 보고되지 않은 전처리 유틸리티 반환값.

결과:
- **필터 통과 75개** (closed/completed 제한 전). open 4, closed/not_planned 1, **closed/completed 70**.
- 선정: closed/completed 70개를 closed_at 내림차순으로 정렬해 앞 **25개**를 골랐다. 닫힌 날짜 범위는 2026-02-25 ~ 2026-09-21이다.
- 선정 구간 안에서 읽은 뒤 제외한 11개: #48835 (예외로 멈춤), #48051 (전처리 유틸리티 반환 모양, 모델 출력 없음), #48057 (예외, 보고자 본인 실수), #47458 (처음부터 학습할 때의 가중치 초기화), #46439 (RuntimeError), #45698 (사용자 정의 코드 클래스 선택 문제, 장난감 출력만 있음), #46682 (ValueError), #46728 (프로세서 도우미 함수의 개수, 모델 출력 보고 없음), #45357 (저장 키 이름, 다시 불러오기 실패), #45072 (dtype 불일치 예외), #43866 (커뮤니티 체크포인트 로딩 실패).
- stale 봇(github-actions)이 닫은 이슈는 25개 중 5개다. 그중 #47030, #47246은 고친 변경을 찾았으므로 원인대로 분류했다. 나머지 3개(#47405, #44945, #43377)는 고친 변경이 없어 N4로 두었다.

## 3. 결과 표

| issue # | title (shortened) | closed date | category | silent | evidence phrase | source URL |
|---|---|---|---|---|---|---|
| 48293 | SwitchTransformers router: cumsum over wrong dim; router_logits equal probs | 2026-09-21 | R3 | yes | "`cumsum(dim=-2)` accumulates over the singleton dimension — a no-op" | https://github.com/huggingface/transformers/pull/48421 |
| 47030 | FineGrainedFP8 DeepGEMM silently wrong on SM100 (UE8M0 scale rounding) | 2026-09-01 | R3 | yes | "float32 scales would be ceil-rounded to UE8M0 without requantizing" | https://github.com/huggingface/transformers/pull/47623 |
| 47405 | Converted checkpoint loaded via wrong Auto class generates fluent garbage | 2026-08-26 | N4 | no | "we already show a table of mismatched/missing weights" | https://github.com/huggingface/transformers/issues/47405#issuecomment-5021113183 |
| 47246 | Nemotron-H Mamba2 slow path: wrong inter-chunk recurrence | 2026-08-19 | R3 | yes | "sums over the target-chunk axis while states is broadcast over it" | https://github.com/huggingface/transformers/pull/47250 |
| 47752 | Pipeline ignores values set on model.generation_config | 2026-08-18 | R1 | yes | "`defaults_only=True` silently ignored the user's `model.generation_config` settings" | https://github.com/huggingface/transformers/pull/47953 |
| 47328 | Qwen2.5-Omni Token2Wav DiT: half-split cos/sin with interleaved rotate | 2026-07-23 | R3 | yes | "A half-split cos/sin combined with an interleaved rotate is not a valid rotation" | https://github.com/huggingface/transformers/issues/47328 |
| 47475 | Zamba2 torch_forward causality violation | 2026-07-22 | R3 | yes | "inter-chunk state recurrence reduces over the wrong axis" | https://github.com/huggingface/transformers/issues/47475 |
| 46489 | DeepSeek-Coder v1 tokenizer strips whitespace in v5 | 2026-06-28 | N3 | yes | "LlamaTokenizer is SentencePiece-based and cannot consume a ByteLevel-BPE tokenizer.json correctly" | https://github.com/huggingface/transformers/issues/46489 |
| 46612 | Beam search cache reorder skipped for Mamba, XLNet, RWKV, Reformer | 2026-06-24 | R4 | yes | "beam hypotheses may continue using stale or incorrectly aligned cached states" | https://github.com/huggingface/transformers/issues/46612 |
| 46032 | Mamba2Mixer: use_cache with seq_len > 1 gives wrong results | 2026-06-23 | R2 | yes | "dt[:, 0, :] silently dropped all tokens after index 0" | https://github.com/huggingface/transformers/pull/46084 |
| 46710 | DeepSeek-R1-Distill-Llama-8B output regression 4.55 → 5.9 | 2026-06-23 | N3 | yes | "causing the tokenizer to fall back to LlamaTokenizer by default" | https://github.com/huggingface/transformers/issues/46710#issuecomment-4727054833 |
| 45920 | AutoTokenizer wrong token IDs for OLMo2, HyperClovaX, Yi, others (v5) | 2026-06-19 | N3 | yes | "Update tokenizer mappings to use TokenizersBackend for additional models" | https://github.com/huggingface/transformers/pull/46091 |
| 45812 | AutoTokenizer wrong token IDs for all Granite models (v5) | 2026-05-18 | N3 | yes | "AutoTokenizer routes Granite to GPT2Tokenizer, whose __init__ hardcodes ByteLevel(use_regex=True)" | https://github.com/huggingface/transformers/issues/45812 |
| 45910 | DeepSeek-V4 RoPE theta mismatch, main vs compressed KV | 2026-05-12 | R1 | unclear | "the final attention seems to mix KV entries encoded with different RoPE bases" | https://github.com/huggingface/transformers/issues/45910 |
| 44945 | Molmo2 wrong output with manual multi-GPU layer split | 2026-05-01 | N4 | yes | "it seems to operate as if the checkpoints were not loaded correctly" | https://github.com/huggingface/transformers/issues/44945 |
| 45381 | Qwen2.5-VL video vision_position_ids wrong (5.3+) | 2026-04-14 | R2 | yes | "All video frames share the same position_temporal and the position_height is also wrong" | https://github.com/huggingface/transformers/issues/45381 |
| 45356 | Kimi-K2.5 tokenizer: `</think>` decodes to empty (5.4+) | 2026-04-13 | N3 | no | "causing every token after ID 163588 to be assigned wrong IDs" | https://github.com/huggingface/transformers/pull/45359 |
| 45242 | Gemma 4: use_cache=False gives garbage logits | 2026-04-09 | R1 | yes | "kv states should ALWAYS be shared, even during training or inference without Cache" | https://github.com/huggingface/transformers/pull/45312 |
| 44155 | AudioFlamingo3 batched inference leaks embeddings across items | 2026-03-25 | R2 | yes | "The processor and model compute audio embedding counts differently" | https://github.com/huggingface/transformers/issues/44155 |
| 44671 | CamemBERT wrong masked-LM predictions in v5 | 2026-03-23 | R1 | no | "`CamembertConfig` was missing `tie_word_embeddings: bool = True`" | https://github.com/huggingface/transformers/pull/44931 |
| 44779 | DeepSeek-R1 tokenizer drops spaces in v5 | 2026-03-18 | N3 | yes | "fix for having incorrect tokenizer class on the hub" | https://github.com/huggingface/transformers/pull/44801 |
| 44448 | pegasus-cnn_dailymail incoherent output in v5 | 2026-03-18 | N3 | yes | "we just were missing a convert from_spm for this model" | https://github.com/huggingface/transformers/issues/44448#issuecomment-4031864865 |
| 43377 | Mimi encoder: batched differs from single (padding_mask unused) | 2026-03-15 | N4 | yes | "`MimiModel._encode_frame` does not contain appropriate logic to handle `padding_mask`" | https://github.com/huggingface/transformers/issues/43377 |
| 43697 | RTDetrV2 different outputs in v5.0 with identical inputs | 2026-03-05 | R3 | yes | "_tied_weights_keys mapping direction, which prevents v4-style model.decoder.* head weights from loading correctly" | https://github.com/huggingface/transformers/issues/43697#issuecomment-4003544801 |
| 43975 | deepseek-coder-6.7b-instruct detokenizes wrongly in v5 | 2026-02-25 | N4 | yes | "on a clean environment with transformers 5.2 issue is gone" | https://github.com/huggingface/transformers/issues/43975#issuecomment-3957497132 |

## 4. 집계

| 범주 | 개수 | 이슈 |
|---|---|---|
| R1 의미 누락 | 4 | #47752, #45910, #45242, #44671 |
| R2 범위·위치 | 3 | #46032, #45381, #44155 |
| R3 배치·순서·매핑 | 6 | #48293, #47030, #47246, #47328, #47475, #43697 |
| R4 낡은 상태 | 1 | #46612 |
| N1 수치·커널 산술 | 0 | |
| N2 플랫폼·툴체인 | 0 | |
| N3 토크나이저·템플릿·샘플링 | 7 | #46489, #46710, #45920, #45812, #45356, #44779, #44448 |
| N4 버그 아님·고친 변경 없음 | 4 | #47405, #44945, #43377, #43975 |
| N5 원인 불명 | 0 | |

- **R1–R4 비중: 14 / 21 = 66.7%.** 분모는 25개에서 N4 4개를 뺀 수다. N5는 0개다.
- 민감도: #47246과 #47475를 규칙 그대로 N4로 두면(이슈에 연결된 수정이 없음) 12 / 19 = 63.2%다.
- 토크나이저(N3) 7개를 빼면, 분류된 나머지 14개가 모두 R1–R4였다. N1과 N2는 0개였다.
- silent: yes 21 / 25 (84%), no 3 (#47405, #45356, #44671), unclear 1 (#45910). R1–R4 14개만 보면 yes 12, no 1, unclear 1이다.

## 5. 개별 판정 메모

- #47030: stale 봇이 닫았다. 그러나 메인테이너가 main에서 고쳤다고 답했다. PR #47623은 SM100에서 float32 스케일 체크포인트를 DeepGEMM으로 보내지 않게 바꿨다. 반올림 문제(N1)로 볼 수도 있다. 커널이 요구하는 UE8M0 스케일 형식과 체크포인트의 fp32 스케일이 어긋난 것이 원인이라 R3로 두었다.
- #47246, #47475: 같은 코드다. Nemotron-H가 Zamba2의 `torch_forward`를 물려받는다. 두 이슈 모두 병합된 수정이 연결되어 있지 않다. 파일 이력에서 커밋 a79421ace4 (PR #47452, 2026-07-24 병합, 리팩터)가 문제의 세 줄을 Mamba2의 올바른 식으로 바꾼 것을 확인했다. #47475는 같은 날 "Code agent slop" 라벨과 함께 댓글 없이 닫혔다. #47246의 근거 문구는 병합되지 않은 제안 PR #47250에서 가져왔다. 축을 잘못 골라 합한 계산이라 N1로 볼 여지도 있다. 텐서 축의 역할을 혼동한 것이라 R3로 두었다.
- #48293: `keepdim=True`로 모양이 바뀌어 `dim=-2`가 크기 1인 축을 가리켰다. 그래서 용량 초과 토큰이 버려지지 않았다. 같은 PR이 `router_logits` 자리에 확률이 들어가던 문제도 고쳤다.
- #46612: 캐시가 `past_key_values`가 아닌 이름으로 저장된 모델에서 재정렬 단계가 건너뛰어졌다. 그래서 순서가 맞지 않는 옛 캐시를 다음 단계가 읽었다. 이름을 무시한 점(R1)보다 상태를 제때 갱신하지 않은 점(R4)을 주 원인으로 봤다.
- #46032: 캐시가 있으면 한 토큰 디코드라고 가정했다. 그래서 두 번째 토큰부터 버렸다(R2). 캐시 상태를 청크 스캔의 초기 상태로 넘기지 않은 문제(R1)도 함께 있었다.
- #45910: 코드를 읽고 올린 보고라 실행 결과가 없다. 그래서 silent를 unclear로 두었다. 수정(PR #45892)은 CSA/HCA 층의 주 어텐션도 그 층의 compress RoPE(yarn 적용)를 쓰게 바꿨다.
- #44671: 수정 PR 설명에 따르면 로드 보고서에 `lm_head.decoder.weight: MISSING`이 찍혔다. 그래서 silent를 no로 두었다.
- #45356: 잘못된 조언을 담은 `fix_mistral_regex` 경고가 떴다. 그래서 silent를 no로 두었다.
- #47405: MISSING/UNEXPECTED 키 경고가 있었다. 메인테이너는 이미 누락 가중치 표를 보여 주므로 오류를 내는 편이 낫지 않다고 답했다. 오류를 내는 PR #47408은 병합되지 않았고 stale 봇이 닫았다. 사용자가 클래스를 잘못 고른 경우이기도 하다.
- #43377: stale 봇이 닫았다. PR #43378은 아직 열려 있다. 현재 main의 `_encode_frame`에도 padding_mask 지원을 미룬 TODO가 그대로 있다.
- #43975: 메인테이너가 재현하지 못했고 보고자가 새 환경에서는 사라졌다며 닫았다. 넉 달 뒤 #46489가 같은 모델 계열에서 같은 증상을 다시 보고했고 PR #46091로 고쳐졌다. 이 이슈 자체에는 원인과 수정이 없어 N4로 두었다.
- #43697: 원인은 협업자가 Codex 분석을 옮긴 댓글에서 나왔다. PR #41549의 diff가 rt_detr_v2의 decoder head 묶음 항목을 지웠고, 보고자는 v5.1.0에서 해결됐다고 확인했다.
- #44155: 전용 수정 PR은 없다. 담당자는 PR #43538(Music Flamingo 추가) 리뷰 댓글에서 원인을 찾아 고쳤다고 답했다. 그 댓글에 따르면 모델의 길이 계산이 평균 풀링은 반영했지만 다운샘플링은 반영하지 않았다.

## 6. 관찰

25개 중 13개는 v5 계열로 올린 뒤 생긴 회귀였다: 토크나이저 8개(N3 7개는 체크포인트와 맞지 않는 토크나이저 클래스나 변환을 쓴 경우, 1개는 재현되지 않은 #43975), 가중치 묶음 2개(CamemBERT, RT-DETR v2), RoPE 배치·MoE 라우터 축·Qwen2.5-VL 영상 위치를 바꾼 v5 변경 3개. 모델 계열로는 DeepSeek 체크포인트가 6개(그중 5개가 토크나이저)로 가장 많았고, Mamba2 계열 경로가 4개(Nemotron-H, Zamba2, Mamba2 청크 프리필, Mamba 빔 서치 캐시)였으며 그중 2개는 CUDA 커널 경로는 맞고 순수 PyTorch 경로만 틀렸다. 어텐션 관련 R 사례 3개는 SDPA, FlashAttention, flex 커널 자체가 아니라 RoPE 배치·기저(Qwen2.5-Omni DiT, DeepSeek-V4)와 Gemma 4의 층 간 KV 공유에서 나왔다.
