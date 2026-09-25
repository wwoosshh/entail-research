# vLLM 출력 오류 버그 파일럿 조사

실행일: 2026-09-22. 대상: `vllm-project/vllm` 이슈. GitHub에는 `gh api` GET 요청만 보냈다. 댓글, 반응, 라벨 같은 쓰기 작업은 하지 않았다.

질문: 추론 중 잘못된 출력을 낸 실제 버그 가운데 역할 계열 결함(R1–R4)이 원인인 것은 얼마나 되는가. 역할 계열 결함은 각 구성 요소가 제 몫을 올바르게 계산하지만, 값에 관한 사실(역할, 범위, 배치, 유효 시점)이 구성 요소 사이에서 사라지거나 어긋나거나 잘못된 시점에 읽히는 경우다.

## 1. 방법

### 1.1 검색

명령 형식: `gh api -X GET search/issues -f q='<쿼리>' -f per_page=100`. total_count가 100을 넘으면 `-f page=2`를 한 번 더 호출했다. 검색 호출 사이에는 10초를 기다렸다. 403과 429 응답은 없었다. 모든 응답의 `incomplete_results`는 false였다.

정확한 쿼리 10개:

```text
repo:vllm-project/vllm is:issue in:title wrong created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title incorrect created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title garbage created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title gibberish created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title nonsense created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title accuracy created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title mismatch created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title "different output" created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title corrupted created:2025-01-01..2026-09-22
repo:vllm-project/vllm is:issue in:title degraded created:2025-01-01..2026-09-22
```

| KW | total_count | 받은 쪽 |
|---|---:|---|
| wrong | 81 | 1 |
| incorrect | 109 | 1, 2 |
| garbage | 47 | 1 |
| gibberish | 35 | 1 |
| nonsense | 2 | 1 |
| accuracy | 117 | 1, 2 |
| mismatch | 118 | 1, 2 |
| "different output" | 18 | 1 |
| corrupted | 27 | 1 |
| degraded | 12 | 1 |
| 합 | 566 | |

200건을 넘는 키워드가 없어서 2쪽까지로 전부 받았다.

### 1.2 합집합

이슈 번호로 합쳐서 555건이 되었다(검색 결과 566건 중 11건 중복). PR은 섞이지 않았다. 상태별로는 closed·completed 264건, closed·not_planned 177건, closed·duplicate 1건, open 113건(reopened 1건 포함)이다.

### 1.3 출력 오류 필터 (제목 기준, 대략)

555건의 제목을 모두 읽고 세 갈래로 나눴다.

- 통과(Y): 제목이 생성 결과(텍스트, 토큰, logprob, 점수, 임베딩)나 평가 정확도가 틀리다, 깨졌다(garbage, gibberish, corrupted, incoherent), 떨어졌다(accuracy drop, degraded, regression), 실행마다 다르다고 명시한 경우.
- 탈락(N): 크래시와 shape·dtype mismatch 오류, 오류·경고 메시지 문구, 메트릭과 로그 값, 문서, 설치·빌드·버전, 성능만의 문제, 기능 요청과 RFC, 로드 실패. tool-call·reasoning 파서가 올바른 토큰을 다른 응답 필드로 보내는 API 계층 문제도 탈락으로 했다. 모델이 낸 토큰 자체는 맞기 때문이다.
- 모호(B): CI 정확도 테스트 실패, 플레이스홀더·프롬프트 구성·채팅 형식 불일치, 출력 영향이 제목에 없는 수식·인덱스 버그처럼 제목만으로 판단할 수 없는 경우.

결과: Y 280건, B 77건, N 198건. 필터 통과 수는 280건으로 보고하고, 모호한 77건은 따로 적는다.

| 판정 | 합 | closed·completed | closed·not_planned | open |
|---|---:|---:|---:|---:|
| Y | 280 | 125 | 93 | 62 |
| B | 77 | 51 | 18 | 8 |

필터를 통과하고 closed·completed인 이슈는 125건이다(모호 51건 별도).

### 1.4 25건 선정

closed·completed인 Y와 B 176건을 `closed_at` 내림차순으로 놓고 위에서부터 본문을 읽었다. 필요하면 댓글도 읽었다. 본문 단계의 기준은 이렇다. 이슈나 댓글이 출력 수준의 증상(생성 텍스트나 토큰, logits나 logprob, 점수나 임베딩, 평가 정확도)을 보고하면 통과다. 내부 값(인덱스, 수식, 프롬프트 토큰 id, 단위 테스트 허용 오차)만 보고하거나 크래시나 HTTP 오류만 있으면 제외했다.

31건을 읽어 25건이 통과했다. 제목이 Y인 26건 중 25건, B인 5건 중 0건이 통과했다. 25번째 이슈는 2026-06-23에 닫혔다.

본문 단계에서 제외한 6건:

| issue | 제목(줄임) | closed | 제외 이유 |
|---|---|---|---|
| [#56949](https://github.com/vllm-project/vllm/issues/56949) | Dense DP weight transfer selects wrong IPC payload | 2026-09-17 | 결과가 HTTP 500 오류다. 잘못된 출력은 없다. |
| [#52924](https://github.com/vllm-project/vllm/issues/52924) | `find_mm_placeholders` may return incorrect position | 2026-08-20 | 인덱스 버그 보고만 있다. 보고자가 댓글에서 확인된 end-to-end 실패는 아직 없다고 했다. |
| [#50825](https://github.com/vllm-project/vllm/issues/50825) | `expanded_block_table_buffer` width mismatch (DSA indexer, DCP) | 2026-08-03 | 크래시(RuntimeError)다. |
| [#46988](https://github.com/vllm-project/vllm/issues/46988) | Gemma4 video prompt expansion / timestamps mismatch | 2026-06-30 | 프롬프트 토큰 불일치만 보고했다. 출력 증상이 없다. |
| [#41236](https://github.com/vllm-project/vllm/issues/41236) | Dynamic NTK RoPE scaling is wrong | 2026-06-29 | 코드를 읽고 찾은 수식 버그다. 출력 차이를 보고한 사람이 없고, 한 관리자는 MTEB 결과가 기준과 같다고 했다. |
| [#46585](https://github.com/vllm-project/vllm/issues/46585) | `test_flashinfer_cutlass_mxfp4_fused_moe` accuracy mismatch (H20) | 2026-06-27 | 낡은 단위 테스트였다. 보고자가 실제 추론에서는 정확도 문제가 없다고 확인했다. |

같은 기간에 닫힌 N 판정 이슈 중 제목이 애매한 5건(#51971, #41037, #49804, #44796, #48217)도 본문을 읽었다. 모두 크래시, 단언 실패, 버전 검사, 파서 필드 문제여서 N을 유지했다.

### 1.5 원인 분류 절차

- 읽은 자료: 이슈 본문, 모든 댓글, 타임라인(cross-referenced PR, closed와 referenced 이벤트), 수정 PR의 제목과 본문. #43781은 diff까지 읽었다.
- 수정 PR: 이슈가 닫힌 시각에 병합된 PR, 또는 댓글에서 수정으로 지목된 PR.
- N4: 관리자나 보고자가 사용자 오류, 설정 문제, 정상 동작이라고 한 경우. 수정 PR을 지목하지 않은 채 재현 안 됨, 이미 고쳐짐, stale로 닫힌 경우도 N4로 했다. 수정 PR이 출력 오류가 아니었다고 설명한 경우(¶)도 여기에 넣었다.
- N5: 수정은 있지만 글만으로는 기전을 알 수 없는 경우. 이번 표본에서는 0건이다.
- 한 이슈에서 결함 두 개가 함께 고쳐졌으면, 보고된 조용한 오답을 설명하는 쪽을 1차 원인으로 잡았다.
- 표시
  - †: 기전은 PR 본문에 명시되어 있다. 하지만 그 PR은 병합되지 않았고, 이슈는 우회책이나 "지금은 된다"는 확인으로 닫혔다.
  - ‡: 댓글에서 수정으로 지목된 PR 하나가 R1 결함과 R3 결함을 함께 고쳤다. 어느 쪽이 이 이슈의 원인인지는 글로 구분되지 않는다.
  - §: 더 깊은 원인이 의존성이나 컴파일러에 있을 수 있다.
  - ¶: 기전은 역할 계열이지만, 수정 PR은 정확도 저하가 틀린 계산이 아니라 느린 스텝 때문이라고 설명했다.
- silent: 서빙 스택(vLLM)이 오류나 경고를 냈는지로 판단했다. 평가 도구의 알림(lm_eval의 "null content")과 테스트 스크립트의 비교 단언은 세지 않았다. 본문에 적혀 있지 않으면 unclear로 했다.

## 2. 결과 (closed_at 내림차순 25건)

| issue # | title (shortened) | closed date | category | silent | evidence phrase | source URL |
|---|---|---|---|---|---|---|
| [#52644](https://github.com/vllm-project/vllm/issues/52644) | DeepSeek-V4 accuracy drop, MRV2 + FULL_DECODE_ONLY (ROCm MI350/355) | 2026-09-12 | N4 | yes | "This issue is no longer observed in the last few nightlies." | https://github.com/vllm-project/vllm/issues/52644#issuecomment-5643150040 |
| [#48058](https://github.com/vllm-project/vllm/issues/48058) | XPU FP8 W8A8 (dynamic) garbage on Arc Pro B70 | 2026-09-08 | N2 | yes | "I verified that vllm-xpu-kernels=0.1.11(with vllm=0.23.1rc1.dev1162+g2bd895762.xpu) can fix this issue." | https://github.com/vllm-project/vllm/issues/48058#issuecomment-5054708375 |
| [#52276](https://github.com/vllm-project/vllm/issues/52276) | DeepSeek-V4 NIXL receive failure returns corrupted reasoning | 2026-08-21 | R1 † | unclear | "The failed request is then reported as finished receiving without any failure signal" | https://github.com/vllm-project/vllm/pull/52232 |
| [#51063](https://github.com/vllm-project/vllm/issues/51063) | Mistral3 VLM ties lm_head from top-level config, drops real lm_head | 2026-08-20 | R1 | yes | "a checkpoint with a real `lm_head` silently discards it and generates gibberish" | https://github.com/vllm-project/vllm/pull/51665 |
| [#47087](https://github.com/vllm-project/vllm/issues/47087) | MTP spec decode degenerates into garbage loops (Qwen3.6 hybrid) | 2026-08-19 | R2 | yes | "one request's chunk ends mid-block, leaving its running-state slot holding e.g. `state@364`" | https://github.com/vllm-project/vllm/pull/51113 |
| [#50681](https://github.com/vllm-project/vllm/issues/50681) | Qwen3.6-35B-A3B corrupted with EP + sequence parallelism | 2026-08-14 | R3 | yes | "The Qwen3Next model path infers the hidden-state layout from the first tensor dimension" | https://github.com/vllm-project/vllm/pull/50685 |
| [#29341](https://github.com/vllm-project/vllm/issues/29341) | sleep level 2 causes gibberish | 2026-08-13 | N4 | yes | "I believe that in this case it was user error." | https://github.com/vllm-project/vllm/issues/29341#issuecomment-5277858850 |
| [#51326](https://github.com/vllm-project/vllm/issues/51326) | DeepSeek-V4-Flash corrupted on H100 TP8+EP (0.26.0) | 2026-08-08 | N4 | yes | "The issue seems fixed and the model output looks normal." | https://github.com/vllm-project/vllm/issues/51326#issuecomment-5223379557 |
| [#51094](https://github.com/vllm-project/vllm/issues/51094) | OffloadingConnector wrong output at exact chunk boundary (mamba all mode) | 2026-08-07 | R2 | yes | "Mamba then resumed from a state that had already consumed the last prompt token" | https://github.com/vllm-project/vllm/pull/51100 |
| [#43559](https://github.com/vllm-project/vllm/issues/43559) | ~20% accuracy drop with prefix caching + MTP (Qwen3.6-35B-A3B) | 2026-08-06 | R2 | yes | "Every request resuming from that hash silently restores a truncated state." | https://github.com/vllm-project/vllm/pull/51113 |
| [#41472](https://github.com/vllm-project/vllm/issues/41472) | ROCM_ATTN incorrect output for LFM2 (hybrid) | 2026-08-06 | R2 † | yes | "the kernel read from the wrong slots and silently corrupted attention outputs" | https://github.com/vllm-project/vllm/pull/42420 |
| [#42182](https://github.com/vllm-project/vllm/issues/42182) | Qwen3.5-27B P/D GSM8K collapse with async scheduling | 2026-08-06 | R4 | yes | "the delayed zeroing kernel can erase the received attention KV" | https://github.com/vllm-project/vllm/pull/48481 |
| [#41207](https://github.com/vllm-project/vllm/issues/41207) | DeepSeek-OCR multimodal output broken after Transformers upgrade | 2026-07-31 | R1 § | yes | "Fixes DeepSeek-OCR for Transformers v4 which was resetting `model_type` to `deepseek_vl_v2`" | https://github.com/vllm-project/vllm/pull/41460 |
| [#50435](https://github.com/vllm-project/vllm/issues/50435) | GLM-5.2-FP8 P/D GSM8K about 5% lower on nightly | 2026-07-30 | N4 | yes | "The earlier 89.31% result was likely caused by a deployment config mismatch" | https://github.com/vllm-project/vllm/issues/50435#issuecomment-5132654322 |
| [#46347](https://github.com/vllm-project/vllm/issues/46347) | Qwen3.5 accuracy drop when VLLM_CPU_KVCACHE_SPACE is set (CPU) | 2026-07-29 | N4 | yes | "Fix already in place" | https://github.com/vllm-project/vllm/issues/46347#issuecomment-5115435195 |
| [#49692](https://github.com/vllm-project/vllm/issues/49692) | EPD correctness test: different output for multi-image prompts | 2026-07-28 | N4 | yes | "it's actually normal to have slight deviation like that" | https://github.com/vllm-project/vllm/issues/49692#issuecomment-5100442279 |
| [#48611](https://github.com/vllm-project/vllm/issues/48611) | FlashMLA sparse dense-MHA split: OOB write, corrupted fp8_ds_mla gather | 2026-07-27 | R3 | yes | "does not understand the `fp8_ds_mla` cache layout — silently wrong K/V without DCP" | https://github.com/vllm-project/vllm/issues/48611 |
| [#40373](https://github.com/vllm-project/vllm/issues/40373) | DeepSeek-R1 NVFP4 P/D accuracy drop with flash_infer_one_sided prefill | 2026-07-21 | N4 | yes | "automatically marked as stale because it has not had any activity within 90 days" | https://github.com/vllm-project/vllm/issues/40373#issuecomment-5029450660 |
| [#48831](https://github.com/vllm-project/vllm/issues/48831) | Qwen3-Reranker wrong scores on >8K-token inputs | 2026-07-17 | R4 § | yes | "the pooled hidden states reading a reused/aliased buffer whose producing kernel has not completed" | https://github.com/vllm-project/vllm/pull/48901 |
| [#48324](https://github.com/vllm-project/vllm/issues/48324) | FlashInfer allreduce+RMSNorm+quant fusion corrupts output (FP32 norm weights) | 2026-07-12 | R3 | yes | "could match graphs where the activation and RMSNorm weight have different dtypes" | https://github.com/vllm-project/vllm/pull/48330 |
| [#47239](https://github.com/vllm-project/vllm/issues/47239) | GLM-5.2 on MRV2: low aa_lcr accuracy and TPOT spikes (B300) | 2026-07-09 | N4 ¶ | yes | "(via eval-client timeouts/truncation) the low long-context scores" | https://github.com/vllm-project/vllm/pull/47381 |
| [#47300](https://github.com/vllm-project/vllm/issues/47300) | Gemma4 gibberish for long inputs with images on SM90 FA4 | 2026-07-05 | R1 | yes | "The mask implemented `causal OR mm_prefix` instead of `(causal AND sliding_window) OR mm_prefix`" | https://github.com/vllm-project/vllm/pull/47332 |
| [#40018](https://github.com/vllm-project/vllm/issues/40018) | ROCM_AITER_MLA_SPARSE prefill garbage above ~20K tokens (GLM-5.1-FP8) | 2026-06-26 | R1 ‡ | yes | "relies on the RoPE implementation mutating the input arguments in-place" | https://github.com/vllm-project/vllm/pull/43781 |
| [#43602](https://github.com/vllm-project/vllm/issues/43602) | Qwen3-VL-2B Geo3K accuracy below SGLang | 2026-06-24 | R4 | yes | "this can make the decoder graph specialize to the no-deepstack path" | https://github.com/vllm-project/vllm/pull/43617 |
| [#42007](https://github.com/vllm-project/vllm/issues/42007) | FP8 MoE models corrupted when serving LoRA adapters | 2026-06-23 | R3 | yes | "the MoE LoRA kernel receives hidden states that have already been quantized to `torch.float8_e4m3fn`" | https://github.com/vllm-project/vllm/issues/42007 |

증거 문구는 모두 출처 원문과 글자 그대로 대조했다. 굵은 글씨 표시(`**`)와 줄바꿈만 없앴다.

### 이슈별 메모 (수정 경로와 기전)

- #52644: 수정 PR이 없다. 우회 PR #52646은 병합되지 않았다. 이후 #57229는 이 문제가 main에서 더는 해당하지 않는다며 #55095의 우회 기본값을 되돌렸다.
- #48058: 가중치 stride 배치를 고치는 PR #48108(R3 가설)은 병합되지 않았다. 기여자가 vllm-xpu-kernels 0.1.11에서 해결됨을 확인했고, 관리자가 우선 닫았다. 커널 패키지 안에서 무엇이 바뀌었는지는 글에 없다.
- #52276: 관리자가 제안한 설정(`enforce_handshake_compat: false`)을 쓰자 재현이 사라져 보고자가 닫았다. 기전은 관리자가 쓴 열린 PR #52232에 있다. HMA 수신이 실패해도 그 실패 신호가 스케줄러까지 전달되지 않아, 디코드가 무효한 KV로 계속 진행한다.
- #51063: #51665로 고쳤다. 체크포인트를 보지 않고 설정값만으로 가중치 공유 여부를 정해서, 실제로 학습된 lm_head를 버렸다.
- #47087, #43559: #51113으로 고쳤다. 47087은 보고자가 "Resolved by"로 확인했고, 43559는 PR 병합과 함께 닫혔다. mamba align 모드에서 청크가 블록 중간에서 끝나면, 다른 위치의 상태가 블록 경계 상태의 해시로 캐시되었다.
- #50681: #50685로 고쳤다. 은닉 상태가 시퀀스 병렬로 쪼개졌는지를 텐서의 첫 차원으로 추론했는데, TP2 단일 토큰 디코드에서는 전체 입력과 샤드 모두 한 행이라 구별되지 않았다.
- #29341: 관리자는 사용자 오류로 판단했다. level 2 깨우기 뒤에는 `reload_weights`를 불러야 한다. 부르지 않고 생성해도 경고가 없었다.
- #51326: nightly에서 사라졌다고 해서 닫혔다. 이후 0.27.1에서 다시 나왔고, 관리자는 deepgemm과 관련 있을 것으로 추정했다.
- #51094: #51100으로 고쳤다. 청크 경계 N에서 N−1 토큰까지만 조회하면서 N 시점의 Mamba 상태를 복원해, 마지막 토큰이 두 번 적용되었다.
- #41472: v0.26.0 이미지에서 권장 명령으로 동작할 것이라는 기여자 댓글 뒤에 닫혔다. 기전은 댓글 분석과 열린 PR #42420에 있다. ROCm 페이지드 어텐션 커널이 연속 슬롯 배치를 가정해서, 하이브리드 KV 배치에서 엉뚱한 슬롯을 읽었다.
- #42182: #48481로 고쳤다. 비동기 스케줄링에서 새 블록을 0으로 채우는 커널이 NIXL RDMA로 이미 받은 KV를 늦게 덮어썼다.
- #41207: #41460(Transformers v4용)과 transformers#45739(v5용)로 고쳤다. Transformers v4의 상위 클래스 초기화가 `model_type`을 다른 값으로 바꿨다(§: 의존성 동작이 계기).
- #50435: 배포 설정 불일치였다. 깨끗이 다시 배포하자 재현되지 않았다.
- #46347: 보고자가 "Fix already in place"라고만 쓰고 닫았다. 어떤 수정인지는 적지 않았다.
- #49692: 기여자가 E-PD 구성에서 이 정도 편차는 정상이라고 설명했다.
- #48611: #48642로 고쳤다. 결함은 둘이다. (1) 일부 토큰만 처리하는 경로에 전체 배치의 `req_id_per_token`을 넘겨 범위 밖에 썼다(R2, 메모리 오류). (2) 656바이트 `fp8_ds_mla` 캐시 항목을 일반 E4M3처럼 읽었다(R3, 조용한 오답). 조용한 오답을 낸 (2)를 1차 원인으로 잡았다.
- #40373: 90일 stale 알림이 달린 날 보고자가 설명 없이 닫았다.
- #48831: #48901로 고쳤다. 청크 prefill과 torch.compile이 함께 켜지면, 풀링이 생산 커널이 끝나기 전의 버퍼를 읽었다. 풀링 전에 동기화해서 고쳤다. PR은 이를 우회에 가깝다고 했고, inductor의 버퍼 수명 문제를 더 깊은 원인으로 추정했다(§).
- #48324: #48330으로 고쳤다. BF16 활성값과 FP32 RMSNorm 가중치가 섞인 그래프에 대해 융합 패턴이 dtype 일치를 검사하지 않았다.
- #47239: #47381로 고쳤다. MRV2가 요청을 토큰 수로만 정렬해서, 디코드가 prefill로 분류되었다(배치 순서 계약 위반). PR은 정확도 저하를 느린 스텝 때문에 생긴 평가 클라이언트의 타임아웃과 잘림으로 설명했다. 그래서 N4로 했다(¶).
- #47300: #47332로 고쳤다. 결함은 둘이다. `mask_mod`가 있으면 커널이 자체 창 적용을 끄는데, 마스크에 sliding window 조건이 빠져 있었다(R1). 또 청크 안의 상대 `q_idx`를 절대 위치처럼 썼다(R2). 단일 청크에서도 재현되었으므로 창 조건 누락을 1차 원인으로 잡았다.
- #40018: 기여자가 #43781을 수정으로 지목했고, 2026-06-26에 main에서 해결을 확인했다. 이 PR은 RoPE가 입력을 제자리에서 바꾼다는 가정(R1)과 인덱서 캐시 배치를 SHUFFLE로 고정한 것(R3)을 함께 고쳤다(‡). 이슈 본문의 `skip_kv_gather` 분석에 따른 PR(#40049, #46478)은 병합되지 않았다.
- #43602: #43617로 고쳤다(보고자가 작성). 컴파일 워밍업 때 deepstack 입력이 None이어서 디코더 그래프가 그 경로로 특수화되었고, 실제 요청에서 시각 정보가 더해지지 않았다.
- #42007: #42120으로 고쳤다. LoRA 커널이 MoE 준비 단계에서 이미 FP8로 양자화된 은닉 상태를 받았다. 원래 정밀도 값을 따로 보관하도록 바꿨다.

## 3. 범주별 집계

| 범주 | 건수 | 이슈 |
|---|---:|---|
| R1 dropped/ignored meaning | 5 | #52276†, #51063, #41207§, #47300, #40018‡ |
| R2 range/position/offset | 4 | #47087, #51094, #43559, #41472† |
| R3 layout/order/mapping | 4 | #50681, #48611, #48324, #42007 |
| R4 stale/mistimed state | 3 | #42182, #48831§, #43602 |
| N1 numerical/kernel arithmetic | 0 | |
| N2 platform/toolchain | 1 | #48058 |
| N3 tokenizer/template/sampling | 0 | |
| N4 not a bug / user error / no identified fix | 8 | #52644, #29341, #51326, #50435, #46347, #49692, #40373, #47239¶ |
| N5 undeterminable | 0 | |
| 합 | 25 | |

- R1–R4 비율: 16/17 = 94%. 분모는 25건에서 N4 8건을 뺀 17건이다. N5는 0건이다.
- 민감도
  - †와 ‡ 3건을 N5로 옮기면 13/14 = 93%다(N5 3건은 분모에서 빼고 따로 보고). N5를 분모에 넣으면 13/17 = 76%다.
  - 여기에 § 2건까지 N2로 옮기면 11/14 = 79%, N5를 분모에 넣으면 11/17 = 65%다.
  - ¶ #47239를 R3으로 세면 17/18 = 94%다.
- silent: 25건 중 yes 24건(96%), unclear 1건(#52276, 클라이언트는 HTTP 200을 받았지만 서버 로그에 경고가 있었는지 적혀 있지 않다), no 0건. R1–R4 16건 중에서는 yes 15건, unclear 1건이다.
- 'completed'로 닫힌 25건 가운데 10건은 vLLM 저장소의 수정 PR이 병합되지 않은 채 닫혔다. N4 7건, † 2건, 의존성 갱신으로 풀린 #48058이다. N4 중 #47239만 수정 PR이 병합되었다.

## 4. 관찰된 패턴

모델 계열로는 MLA를 쓰는 DeepSeek(R1, V3.2, V4)와 GLM-5.x가 8건, 어텐션과 재귀 상태 계층(GDN, Mamba, 합성곱)을 섞은 하이브리드 모델인 Qwen3.5/3.6, Nemotron-Nano-9B-v2, LFM2가 8건으로 가장 많았고, 하이브리드 모델 중 5건(#47087, #43559, #51094, #42182, #41472)은 결함이 재귀 상태나 하이브리드 KV 배치와 직접 관련되어 있었다. 기능으로는 P/D 분리·KV 커넥터·오프로딩 6건(#52276, #42182, #51094, #50435, #40373, #49692), FP8·NVFP4 양자화 경로 6건(#48058, #48324, #42007, #48611, #40373, #51326), torch.compile·CUDA Graph 경로 4건(#52644, #48831, #43602, #48324), MTP 투기적 디코딩 3건(#47087, #43559, #47239)이 반복되었고, 백엔드로는 ROCm 3건, XPU 1건, CPU 1건이 나왔다. R1–R4로 분류한 16건 중 15건은 오류나 경고 없이 잘못된 출력을 냈다.

## 5. 한계

- 한 사람이 분류했고 교차 검증은 하지 않았다. 표본은 25건이다.
- 표본은 제목 키워드(`mismatch`, `corrupted` 등)와 최근 닫힌 이슈(2026-06-23 ~ 2026-09-12)에 따라 정해졌다. 기간과 키워드를 바꾸면 비율이 달라질 수 있다.
- 'completed' 상태는 수정 여부를 약하게만 알려 준다. 25건 중 10건은 vLLM 수정 PR이 병합되지 않은 채 닫혔다.

## 부록: 제목 판정 목록 (재현용)

<details>
<summary>Y 280건</summary>

11781 11816 12096 12112 12199 12364 12371 13133 13763 13801 13828 14058 14232 14392 14662 15340 15437 15447 15865 16296 16337 16658 16668 16815 16934 17689 17766 17808 18054 18055 18267 18533 18955 19052 19206 19245 19472 19489 19493 19763 19779 20054 20125 20217 20261 20426 20627 21175 21471 21529 21581 22103 22808 23056 23256 23282 23345 23702 23804 23813 23988 24038 24118 24164 24530 25189 25209 25262 25333 25800 25833 25994 26042 26378 26451 27034 27570 27602 27651 27775 28268 28317 28539 28598 28604 28704 28830 28835 28839 28862 29007 29341 29436 29478 29595 29781 30196 30358 30445 30777 30801 30830 30939 31081 31202 31210 31394 31422 31495 31564 31609 31625 31626 31840 31844 31864 31918 32190 32545 32588 32898 33011 33091 33107 33123 33276 33532 33560 33672 33678 33871 34186 34395 34526 34759 34892 35138 35288 35329 35407 35411 35412 35504 35651 35718 35828 35925 35980 36094 36117 36228 36295 36337 36524 36872 36986 36999 37030 37032 37257 37471 37554 37591 37618 37732 37804 37856 38643 38652 38710 38718 38931 39022 39049 39179 39223 39265 39273 39407 39545 39722 40018 40248 40252 40373 41132 41207 41236 41262 41292 41472 41511 41623 42007 42016 42118 42182 42265 42718 42801 42898 43094 43163 43559 43602 43631 43962 43996 44148 44550 44841 45562 45698 45888 45904 46088 46311 46347 46460 46710 47087 47239 47300 47365 47783 48058 48324 48327 48611 48831 48895 48898 49070 49122 49250 49290 49377 49449 49692 49844 49886 49918 50332 50427 50435 50681 50772 50881 51063 51094 51326 51456 52071 52150 52234 52276 52442 52576 52644 53019 53051 53086 53211 53411 53488 53968 54035 54114 54252 54739 54785 54924 55927 55951 56009 56655 57064 57224 57493 58138

</details>

<details>
<summary>B 77건</summary>

12122 13078 13139 13401 14342 15144 15381 16832 17362 19368 19545 19741 21252 21899 22858 23373 23400 25970 26121 27722 28262 28637 28661 29530 29645 30487 30546 30828 30929 31245 32069 32221 32222 32235 33210 33596 33598 34277 34406 35132 35167 35168 35767 35779 37271 39261 39532 40201 40207 40243 40481 40485 40508 40509 40510 40512 40513 40514 40526 40527 41579 42621 43301 43415 45691 46261 46585 46787 46817 46988 50018 50825 52924 54974 56380 56949 57740

</details>

합집합의 나머지 198건은 N이다.
