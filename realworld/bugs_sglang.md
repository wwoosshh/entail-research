# SGLang 출력 정확성 버그 파일럿 조사 (2026-09-22)

대상: GitHub `sgl-project/sglang` 이슈. 읽기 전용 조사이며 `gh api` GET 요청만 썼다.

## 1. 방법과 재현 정보

- 실행: 2026-09-22 12:34 UTC에 검색, 같은 날 이슈·댓글·타임라인·PR을 읽음.
- 검색 질의 10개. 각각 1회, `per_page=100`. total_count가 모두 100 이하라 2페이지는 필요 없었다. `incomplete_results`는 모두 false.

```
gh api -X GET search/issues -f per_page=100 -f q='repo:sgl-project/sglang is:issue in:title KW created:2025-01-01..2026-09-22'
KW = wrong | incorrect | garbage | gibberish | nonsense | accuracy | mismatch | "different output" | corrupted | degraded
```

| KW | total_count |
|---|---:|
| wrong | 49 |
| incorrect | 67 |
| garbage | 23 |
| gibberish | 14 |
| nonsense | 2 |
| accuracy | 98 |
| mismatch | 78 |
| "different output" | 2 |
| corrupted | 12 |
| degraded | 5 |

- 합집합: 이슈 번호 기준 344건(PR 0건, 키워드 중복 6건). 상태: closed/completed 300, open 40, closed/duplicate 2, closed/not_planned 2.
- 출력 정확성 필터(제목 기준, 344건 전체): 통과 183, 경계 41, 탈락 120.
  - 통과: 제목이 생성 결과(텍스트, 토큰, logits/logprobs, 임베딩, 이미지·영상, 도구 호출 내용)가 틀리거나, 깨지거나, 품질이 떨어지거나, 일관되지 않다고 말하는 경우. 평가 정확도 하락도 포함.
  - 탈락: 문서, CI·테스트·벤치마크·메트릭 보고, 잘못된 출력 언급 없는 크래시·shape 오류·hang, 성능, 설치·의존성, API 메타데이터(usage, output_index 등), 메모리·OOM.
  - 경계: 의미 오류를 암시하지만 출력 영향이 제목에 없는 경우(예: "wrong key", "incorrect ForwardBatch parameter", 템플릿 매칭).
- 필터 통과 중 closed/completed: 164건(경계 포함 시 199건).
- 선정: closed/completed 이슈를 closed_at 내림차순으로 43건(2026-09-20 ~ 2026-07-18) 훑어 앞에서부터 25건을 골랐다. 제목으로 15건 탈락. 본문을 읽고 3건 탈락: #29061(프롬프트 렌더링 차이만 보고, 잘못된 출력은 제시하지 않음), #33470(RuntimeError만 발생), #25055(logprob 값은 맞고 응답 필드 정렬 문제). #36371은 제목상 경계였지만 본문에 출력 품질 저하 수치가 있어 포함.
- 분류 규칙
  - "확인된 수정"의 뜻: sglang에 병합된 PR이나 커밋이 이슈와 연결된 경우, 또는 스레드에서 특정 변경(상위 라이브러리 수정, 의존성 업그레이드, main의 특정 PR)이 문제를 해결했다고 밝힌 경우.
  - 비활성 봇이나 작성자가 닫았고 확인된 수정이 없으면 N4로 분류했다. 미병합 PR만 있는 경우도 N4다. 스레드가 제시한 기전은 아래 주석에 "대안"으로 적었다.
  - silent: yes는 잘못된 출력이 사용자에게 도달했고 그 결과에 대해 오류나 경고가 언급되지 않은 경우(HTTP 200, 영상 저장, 평가 완료 등). 같은 이슈에 별도의 크래시 모드가 있어도 적용한다. no는 오류나 경고(로그 경고 포함)가 함께 나온 경우. unclear는 정보가 부족하거나 내부 오류 처리 여부를 알 수 없는 경우.
  - 근거 문구는 이슈나 PR 원문을 그대로 인용했다(15단어 이하).

## 2. 결과 (closed/completed, 종료일 최신순 25건)

| 이슈 # | 제목(축약) | 종료일 | 분류 | silent | 근거 문구 | 출처 URL |
|---|---|---|---|---|---|---|
| [31011](https://github.com/sgl-project/sglang/issues/31011) | Intel XPU: xccl all_reduce(MIN) silently does SUM, TP token-id sync corrupted | 2026-09-20 | N2 | yes | "Issue is resolved with latest oneccl version, so can be closed this PR." | https://github.com/sgl-project/sglang/pull/31895#issuecomment-5407363707 |
| [38605](https://github.com/sgl-project/sglang/issues/38605) | MiniMax-H3 FL2VA corrupted video with layerwise offloading | 2026-09-10 | N4 | yes | "The 64G RAM+3090(24G) seems to RAM is not enough" | https://github.com/sgl-project/sglang/issues/38605#issuecomment-5611616270 |
| [30632](https://github.com/sgl-project/sglang/issues/30632) | InternLM2 accuracy drop after transformers 5.8 → 5.12.1 | 2026-09-08 | N4 | yes | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/30632#issuecomment-5577210590 |
| [37187](https://github.com/sgl-project/sglang/issues/37187) | GPT-OSS + DP attention garbage output (GSM8K 0/128) | 2026-09-06 | R3 | yes | "Since reduce-scatter also sums its inputs, it combined TP replicated copies of `s`" | https://github.com/sgl-project/sglang/pull/37199 |
| [30233](https://github.com/sgl-project/sglang/issues/30233) | PD: aborted prefill makes decode generate garbage from incomplete KV | 2026-09-05 | N4 | unclear | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/30233#issuecomment-5548052373 |
| [30188](https://github.com/sgl-project/sglang/issues/30188) | OpenAI streaming / mixed-batch logprobs wrong or crash | 2026-09-04 | N4 | yes | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/30188#issuecomment-5533904246 |
| [30176](https://github.com/sgl-project/sglang/issues/30176) | LongCat-2.0 `oe_*` config keys mismatch, ngram weights dropped, garbled output | 2026-09-04 | R1 | no | "Because none of the expected keys are present, `use_ngram_embedding` resolves to `False`" | https://github.com/sgl-project/sglang/issues/30176 |
| [24321](https://github.com/sgl-project/sglang/issues/24321) | MiMo-V2.5 NVFP4 garbage tokens on SM120 | 2026-09-01 | N4 | yes | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/24321#issuecomment-5486752406 |
| [29748](https://github.com/sgl-project/sglang/issues/29748) | GPTQ gptq_gemm uninitialized output + atomicAdd race | 2026-08-30 | N4 | yes | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/29748#issuecomment-5465761808 |
| [29683](https://github.com/sgl-project/sglang/issues/29683) | Qwen3-30B-A3B BF16 MoE on MI355X unstable/low GSM8K | 2026-08-29 | N4 | yes | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/29683#issuecomment-5459161102 |
| [36371](https://github.com/sgl-project/sglang/issues/36371) | Mooncake PD accepts mismatched Mamba SSM state dtypes | 2026-08-25 | N4 | yes | "Explicitly aligning both roles to FP32 restored output semantics." | https://github.com/sgl-project/sglang/issues/36371 |
| [28414](https://github.com/sgl-project/sglang/issues/28414) | DeepSeek-V4 slight accuracy issue (tool-call tokens lost) | 2026-08-23 | N4 | yes | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/28414#issuecomment-5383312174 |
| [28685](https://github.com/sgl-project/sglang/issues/28685) | GLM-5.2-FP8 wrong output on gfx950 (aiter bpreshuffle GEMM) | 2026-08-19 | N2 | yes | "Root cause identified — it's a ROCm 7.2 toolchain regression, not a logic bug." | https://github.com/sgl-project/sglang/issues/28685#issuecomment-4747064420 |
| [31833](https://github.com/sgl-project/sglang/issues/31833) | NemotronH extra_buffer accuracy drop (Mamba state tracking) | 2026-08-12 | R2 | yes | "`_init_track_ssm_indices` indexes the packed per-chunk states `h` as a per-request concatenation of chunk grids." | https://github.com/sgl-project/sglang/pull/37836 |
| [27125](https://github.com/sgl-project/sglang/issues/27125) | Wan2.2 T2V native backend mosaic video | 2026-08-11 | R3† | yes | "visual corruption in Wan VAE decoding when Conv3d weights are stored in `channels_last_3d` memory format" | https://github.com/sgl-project/sglang/pull/25985 |
| [31699](https://github.com/sgl-project/sglang/issues/31699) | DeepSeek-V4 DP attention garbage (dp_gather_partial on replicated data) | 2026-08-11 | R3 | yes | "`dp_gather_partial` used where data is already replicated" | https://github.com/sgl-project/sglang/issues/31699 |
| [34227](https://github.com/sgl-project/sglang/issues/34227) | MiniMax-H3 `--use-fsdp-inference` corrupted video/audio | 2026-08-10 | R3 | yes | "the rank-local FSDP fast path read shape-compatible safetensors slices directly and bypassed that loader" | https://github.com/sgl-project/sglang/pull/34294 |
| [33223](https://github.com/sgl-project/sglang/issues/33223) | Kimi K3 low τ³-Banking score | 2026-08-05 | N4 | unclear | "What could be the problem?" | https://github.com/sgl-project/sglang/issues/33223 |
| [33245](https://github.com/sgl-project/sglang/issues/33245) | Quantized DeepSeek-V4 fused wq_a+wkv drops packed weights | 2026-08-04 | N4 | no | "Closing: this was reported while working around a wrong assumption on my side." | https://github.com/sgl-project/sglang/issues/33245#issuecomment-5183269289 |
| [31482](https://github.com/sgl-project/sglang/issues/31482) | DeepSeek-V4 PD HiSparse lower accuracy | 2026-08-03 | R2 | yes | "used host KV page indices for both host and device destination buffers" | https://github.com/sgl-project/sglang/pull/31901 |
| [31490](https://github.com/sgl-project/sglang/issues/31490) | DeepSeek-V4-Flash-FP8 GSM8K regression on gfx950 (after #29275) | 2026-08-02 | R3 | yes | "AITER writes CK-ready transposed scale storage but returns contiguous-looking `[M, G]` metadata." | https://github.com/sgl-project/sglang/pull/31727 |
| [33107](https://github.com/sgl-project/sglang/issues/33107) | flashinfer_cutedsl MoE garbage for MiniMax-M3 NVFP4 | 2026-07-31 | N4 | yes | "we cannot say whether HEAD defaults reproduce this" | https://github.com/sgl-project/sglang/issues/33107 |
| [31641](https://github.com/sgl-project/sglang/issues/31641) | NVFP4 KV cache batched decode corrupted (trtllm_mha, SM120) | 2026-07-21 | R3 | yes | "passes Q to CUDA without stride metadata, but the kernel indexes it as packed rows" | https://github.com/sgl-project/sglang/pull/31667 |
| [25882](https://github.com/sgl-project/sglang/issues/25882) | Incorrect results with EPLB immediate rebalancing (CPU) | 2026-07-21 | N4 | yes | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/25882#issuecomment-5028820098 |
| [25218](https://github.com/sgl-project/sglang/issues/25218) | Kimi-K2.6 wrong tool_call from 2nd round | 2026-07-18 | N4 | unclear | "This issue has been automatically closed due to inactivity." | https://github.com/sgl-project/sglang/issues/25218#issuecomment-5008816733 |

† 수정 귀속은 댓글의 추정("likely fixed from #25985")이다. 아래 주석 참고.

### 이슈별 주석 (종료 방식, 수정, 대안 분류)

- #31011: 비활성 봇 종료. 수정 PR #31895는 작성자가 직접 닫았다. 최신 oneCCL에서 해결됐다는 이유다. 의존성 버전 문제이므로 N2.
- #38605: 작성자가 다음 날 닫았다. 원인을 호스트 RAM 부족으로 추정했고 수정은 없다.
- #30632: 봇 종료. 느린 토크나이저로 되돌리는 PR #30537은 아직 열려 있다(미병합). 대안 N3.
- #37187: PR #37199(2026-09-06 병합)로 수정. MoE 블록이 TP all-reduce를 한 뒤 DP attention 후처리가 같은 그룹에서 reduce-scatter로 한 번 더 합산했다. 그 결과 MoE 출력이 TP배(TP4에서 약 4배)가 됐다. 부분합과 복제본 상태(TP placement)의 불일치로 보고 R3로 분류했다.
- #30233: 봇 종료. PR #30551 미병합. 대안 R2: 1토큰 분량의 KV만 전송됐는데 decode는 전체 길이의 KV를 읽는다.
- #30188: 봇 종료. PR #30187은 유휴 PR 봇이 닫았다(미병합). 대안 N3: 스트리밍 후처리에서 인덱스와 오프셋이 틀린다.
- #30176: 봇 종료지만 수정이 main에 있다. PR #30275(2026-07-07 병합)의 설명은 "Map LongCat 2.0 `oe_*` ngram embedding config fields"이고, 현재 main의 `longcat_flash.py`에 별칭 처리가 들어 있음을 확인했다. 보고는 미병합 브랜치(#30042) 기준이었다. silent=no로 둔 이유는 로더가 "not found in params_dict" 경고를 남겼기 때문이다.
- #24321: 봇 종료. 댓글에서 원인이 밝혀졌다. 평범한 [Q;K;V] 순서의 체크포인트를 TP 교차 배치로 보고 행 단위로 잘랐다. 수정 PR #28100도 제시·검증됐지만 유휴 PR 봇이 닫았다(미병합). 대안 R3.
- #29748: 봇 종료. PR #29749 미병합. 대안 N1(커널 안의 경쟁 상태).
- #29683: 봇 종료. 원인을 비결정적 reduction으로 좁혔지만 수정은 없다. 대안 N5.
- #36371: 작성자가 등록 2분 뒤 닫았고 댓글과 수정은 없다. 대안 R3: 원본의 item length와 stride로 raw 복사해 BF16 상태가 FP32 칸에 들어간다.
- #28414: 봇 종료. 제안된 PR(#28612, #26471, #26766)의 효과는 확인되지 않았다. 대안 N5.
- #28685: 봇 종료지만 원인과 수정이 확인됐다. ROCm 7.2의 LLVM에서 `-amdgpu-coerce-illegal-types` 옵션이 없어졌고, aiter의 플래그 검사기가 이 옵션을 조용히 빼는 바람에 CK 커널이 잘못 컴파일됐다. 수정은 ROCm/rocm-libraries#8639(2026-07-06 병합). 컴파일 옵션이 조용히 사라진 점은 R1과 닮았지만 주 원인이 툴체인이라 N2로 분류했다.
- #31833: 작성자(b8zhong)가 2026-08-12에 "Seems the fix is from BCG"라는 댓글과 함께 닫았다. 실제 수정 PR #37836(2026-09-04 병합)은 이 이슈가 "closed as completed but never landed a fix"였다고 적고 있다. 같은 스레드의 별개 버그(CUDA graph 트랙 버퍼의 갱신되지 않은 꼬리를 읽음, R4형)는 PR #32555로 수정됐다.
- #27125: 봇 종료. 보고자가 main에서 해결됐음을 확인했다. 댓글은 PR #25985(2026-05-22 병합)를 원인 수정으로 지목했다. 보고 버전 v0.5.12는 2026-05-16에 나왔고 이 PR을 포함하지 않는다. PR diff는 channels_last_3d 텐서를 SP all_gather 전에 contiguous로 바꾸고, conv 입력 포맷을 가중치 포맷에 맞춘다.
- #31699: PR #31700(2026-08-11 병합)이 `dp_gather_replicate`로 교체했다. #37187과 같은 부분합/복제본 불일치다.
- #34227: PR #34294(2026-08-10 병합)로 수정.
- #33223: 작성자가 댓글 없이 닫았다. 버그인지도 확인되지 않았다. 대안 N5.
- #33245: 작성자가 닫으면서 원인을 자기 전제 탓으로 돌렸다. 체크포인트를 필요 없이 재패킹했다는 것이다. 수정 PR #33265 미병합. 대안 R1: packed 텐서 이름이 fused 로딩 경로에서 버려진다. silent=no로 둔 이유는 "not found in params_dict" 로그가 남았기 때문이다.
- #31482: PR #31901(2026-08-03 병합)로 수정.
- #31490: PR #31727(2026-08-02 병합)로 수정. 이 회귀를 만든 #29275도 스케일 레이아웃을 고치는 PR이었다.
- #33107: 작성자가 33분 뒤 닫았고 수정은 없다. 대안 N5.
- #31641: PR #31667(2026-07-20 병합)로 수정.
- #25882: 봇 종료. logical expert id를 physical로 재매핑하는 PR #27635 미병합. 대안 R3.
- #25218: 봇 종료. 수정은 없고 다른 이슈 링크만 있다. 대안 N5.

## 3. 분류별 건수와 비율

| 분류 | 건수 | 이슈 |
|---|---:|---|
| R1 | 1 | 30176 |
| R2 | 2 | 31833, 31482 |
| R3 | 6 | 37187, 27125†, 31699, 34227, 31490, 31641 |
| R4 | 0 | |
| N1 | 0 | |
| N2 | 2 | 31011, 28685 |
| N3 | 0 | |
| N4 | 14 | 38605, 30632, 30233, 30188, 24321, 29748, 29683, 36371, 28414, 33223, 33245, 33107, 25882, 25218 |
| N5 | 0 | |

- R1–R4 비율: 9/11 = 82%(95% Wilson 구간 52–95%). 분모는 R1–R4 9건과 N1–N3 2건이다. N4 14건은 분모에서 뺐다. N5는 0건이라 포함 여부에 따라 달라지지 않는다.
- N4 14건의 내역: 비활성 봇 종료 9건, 작성자 종료(설명·수정 없음) 3건, 작성자가 사용자 측 원인으로 종료 2건.
- silent: 25건 중 yes 20건(80%), no 2건, unclear 3건. 분류된 11건 중에서는 yes 10건, R1–R4 9건 중에서는 yes 8건이다. 예외 #30176은 로더 경고가 로그에 남았다.
- 민감도(참고)
  - N4 중 스레드가 코드 수준 기전을 제시한 이슈를 위 "대안"대로 분류하면 R1 2, R2 3, R3 9, N1 1, N2 2, N3 2, N4 1, N5 5가 된다. 이때 R1–R4 비율은 14/19 = 74%이고, N5를 분모에 넣으면 14/24 = 58%다.
  - #27125를 N5로 바꾸면 주 결과는 8/10 = 80%다.

## 4. 관찰된 패턴

- 분류된 R 9건은 모두 한 구성 요소가 만든 값을 다른 구성 요소가 다른 약속으로 읽은 경우였다. 이미 합산된 TP 값을 부분합으로 보고 다시 합산한 경우가 2건(#37187 GPT-OSS, #31699 DeepSeek-V4, 둘 다 DP attention)이다. 나머지는 체크포인트 설정 키와 QKV 행 순서(#30176, #34227), 스케일과 Q의 stride 메타데이터(#31490, #31641), KV 페이지 번호와 Mamba2 청크 오프셋(#31482, #31833), VAE 메모리 포맷(#27125)이었다.
- 모델로는 DeepSeek-V4 계열 5건, MiniMax 계열 3건, Kimi 계열 3건이 반복됐다. 기능으로는 DP attention과 TP 집합 통신(#37187, #31699, #31011), PD 분리 전송(#30233, #36371, #31482), 체크포인트 로딩(#30176, #24321, #34227, #33245), AMD gfx950 경로(#28685, #31490, #29683)가 반복됐다.
- 25건 중 14건이 N4였고 그중 9건은 비활성 봇이 "completed"로 닫은 것이다. 그 9건 중 6건에는 원인 분석과 미병합 수정 PR이 있었다.

## 5. 원자료

임시 폴더 `C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Desktop------ai-compiler\27a9157c-9e06-4d5e-8615-c84cc90eb8bf\scratchpad\sglang\`에 원자료가 있다. 세션이 끝나면 지워질 수 있다.

- `search.py`, `search_meta.json`: 검색 질의와 total_count
- `union.json`: 합집합 344건
- `title_filter.py`, `title_filter.json`: 제목 판정
- `issues/*.json`: 선정 이슈의 본문·댓글·타임라인과 연결 PR

<details>
<summary>제목 판정 목록 (통과 183 / 경계 41)</summary>

통과: 3884, 4158, 4175, 4324, 4771, 4807, 5122, 5402, 5455, 5700, 5702, 5743, 5792, 5874, 6839, 6906, 7041, 7143, 7197, 7641, 7680, 7692, 7936, 7939, 8143, 8402, 8614, 8675, 8736, 8942, 9138, 9457, 9466, 9619, 9771, 9806, 9943, 9944, 10054, 10059, 10138, 10284, 10344, 10626, 11629, 11650, 11696, 11758, 12157, 12198, 12567, 12878, 13011, 13044, 13238, 13469, 13680, 13832, 14382, 15082, 15246, 15357, 15527, 16132, 16246, 16322, 16691, 16834, 16972, 17586, 17593, 17635, 17839, 17887, 17942, 18002, 18210, 18358, 18481, 18610, 18653, 18683, 18831, 18874, 19158, 19449, 19483, 19513, 19971, 20050, 20078, 20420, 20584, 20763, 20791, 20820, 21093, 21132, 21291, 21317, 21372, 21545, 21566, 21614, 21675, 21744, 21843, 21919, 22101, 22130, 22291, 22311, 22472, 22477, 22479, 22671, 22701, 22760, 22764, 23020, 23084, 23463, 23554, 23687, 23729, 23746, 24321, 24440, 24866, 25055, 25187, 25218, 25704, 25742, 25790, 25863, 25882, 26906, 26951, 27000, 27125, 27504, 28414, 28685, 28851, 28979, 29038, 29330, 29683, 29748, 29998, 30176, 30188, 30233, 30632, 31011, 31482, 31490, 31641, 31699, 31833, 32038, 33107, 33187, 33223, 33245, 33360, 34227, 35257, 35564, 35949, 36390, 36423, 36690, 36807, 36938, 37187, 37606, 38573, 38605, 39193, 39626, 39830

경계: 3296, 3486, 4434, 4456, 5498, 5506, 6158, 7913, 8726, 8939, 9100, 10019, 10150, 11004, 11574, 12574, 12589, 12791, 14083, 14196, 15359, 16722, 17227, 17560, 20274, 21171, 21340, 22110, 23797, 24413, 26975, 29054, 29061, 29567, 33470, 33493, 33967, 35440, 36371, 37475, 40735

</details>
