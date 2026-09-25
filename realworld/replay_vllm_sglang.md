# vLLM·SGLang 역할 계열 버그 25건 재연: 어떤 해법이 실제로 막았을까

- 작성일: 2026-09-22
- 입력: `bugs_vllm.md`의 R1–R4 16건과 `bugs_sglang.md`의 R1–R4 9건. 1라운드 분류와 이슈별 메모를 출발점으로 썼다.
- 방법: 25건 모두 이슈 본문·댓글, 수정 PR의 본문과 diff를 다시 읽었다. GitHub에는 `gh api` GET만 보냈다. 버그마다 경계, 잃어버린 사실, 계기를 정했다. 그다음 다섯 해법(S1–S5)이 그 버그를 막았을지 판정했다.
- 목표는 보수적인 적용 범위 추정이다. 해법을 옹호하려는 것이 아니다. 판단이 갈리면 낮은 쪽을 골랐다.

## 1. 판정 기준

### 1.1 표시

- **P**: 서빙 전에 막는다. 빌드, 컴파일, 로드, 초기화 중 하나에서 오류가 난다.
- **D**: 문제가 생긴 순간 런타임에 알린다. 잘못된 값이 소비될 때 오류가 난다.
- **T**: 출시 전 테스트에서 드러난다.
- **–**: 도움이 안 된다.
- **`*`**: 참 선언 전제에서만 성립한다. 이런 버그에서는 결함이 바로 거짓 선언이었다. 설정 필드나 텐서 자신의 stride가 그 예다. 실제 역할 주석이 같은 출처를 베꼈다면 잡지 못한다.
- 1라운드 표시를 그대로 옮긴 것: †는 수정 PR이 병합되지 않았거나 수정 귀속이 댓글의 추정인 경우, ‡는 한 PR이 결함 둘을 함께 고친 경우, §는 더 깊은 원인이 의존성이나 컴파일러에 있을 수 있는 경우다.

### 1.2 해법별 적용 규칙

- **S1 정적 역할 형**: 양쪽 선언이 참이라고 가정한다. 불일치가 빌드·로드 시점에 알 수 있는 역할이면 P다. 설정으로 정해지는 배치, 합산 상태, dtype, 프레임, 정적 범위가 여기에 든다. 실행 중에 정해지는 값은 정적 형으로 볼 수 없어 –다. 청크 끝, 전송 실패, GPU 실행 순서가 그 예다. 어휘에 없는 사실(NEW)도 –로 했다.
- **S2 런타임 역할 sanitizer**: D 아니면 –다. 태그는 Python 수준의 버퍼 경계에 붙는다고 보았다. JIT 커널 안(마스크 술어, Inductor 그래프 내부)은 보지 못한다. 태그 종류는 layout(dtype·양자화 형식 포함), reduction 상태, frame(인덱스 번호 공간 포함), epoch/readiness다.
- **S3 역할 기반 테스트 생성**: 두 경우에 T를 준다.
  1. 계기가 지정된 생성 군에 든다. 군은 블록·청크·창 경계, TP/DP/EP>1, 복제 대 부분합, 제자리 변환 뒤 두 번째 요청, 상대 대 절대 오프셋, 캐시 재사용이다.
  2. 선언된 역할 값(layout, dtype, 모드)을 소비자마다 나열하면 그 조합이 나온다.

  다음은 –로 했다.
  - 타이밍 경쟁, 결함 주입, 의존성 버전이 계기인 경우.
  - 역할 명세에서 만든 기준 구현이 같은 결함을 물려받는 경우. 설정이 틀렸으면 기준도 틀린다.

  필요한 하드웨어(ROCm, SM120, GPU 여러 장)는 CI에 있다고 가정했다.
- **S4 fail-loud**: 소비자가 처리하지 않는 속성이나 값을 조용히 흘려보낸 경우에만 준다. 모르는 것과 지원하지 않는 것이 모두 여기에 든다. 아는 속성을 잘못 처리한 경우는 –다. 초기화·로드·컴파일 시점에 드러나면 P, 특정 요청이 와야 드러나면 D다.
- **S5 선언적 변형 명세**: 변형의 의미를 백엔드마다 손으로 옮기다 생긴 결함에만 P를 준다. 마스크, 창, 위치, prefix 영역이 그런 의미다. 캐시 저장 형식, 페이지 주소 계산, 벤더 커널 호출 규약, 로더, 집합 통신, 스케줄러의 결함은 –다.
- **새 로직**: 실패한 경로가 표현할 수 없던 계산을 수정이 새로 넣었으면 yes다. 커널 기능, 새 데이터 경로, 재계산이 그렇다. 플래그, 조건, 인덱스 번호 공간, 디스패치, 형식 변환, 동기화만 고쳤으면 no다. yes라면 선언만으로는 조용한 오답을 오류나 대체 경로로 바꾸는 데서 그친다.
- **계기**:
  - default: 모델과 하드웨어만 고르면 타는 기본 경로.
  - opt-in: 널리 쓰는 기능 한두 개. prefix caching, MTP, FP8, LoRA, TP>1, DP attention, 다중 GPU가 예다.
  - niche: 드문 모드, 세 가지 이상의 조합, 좁은 입력 조건.

## 2. 결과 표

| repo | issue | 경계 (생산 → 소비) | 사실 | 계기 | S1 | S2 | S3 | S4 | S5 | 새 로직 | 요약 이유 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vllm | [#52276](https://github.com/vllm-project/vllm/issues/52276) | NIXL HMA 수신 워커 → 스케줄러·디코드 | TIME | niche | – | D | – | D | – | no | 수신 실패가 "완료"로 보고됨. ready 태그와 fail-closed만 잡음 (†) |
| vllm | [#51063](https://github.com/vllm-project/vllm/issues/51063) | 체크포인트(top-level tie=true, 별도 lm_head) → 모델 구성·로더 | PROPERTY | default | P\* | – | – | – | – | no | 틀린 것은 설정 선언 자체. 참 선언 전제에서만 S1 |
| vllm | [#41207](https://github.com/vllm-project/vllm/issues/41207) | vLLM 설정 클래스 → Transformers v4 base init → model_type 키 처리 | MAPPING | default | – | – | – | – | – | no | 의존성이 필드 값을 덮어씀. 역할 불일치가 아님 (§) |
| vllm | [#47300](https://github.com/vllm-project/vllm/issues/47300) | 레이어 sliding window·FA4 로컬 q_idx → 손으로 쓴 mm_prefix mask_mod | PROPERTY | default | P | – | T | P | P | no | 창이 mask_mod 경로에서 사라짐, 프레임 혼동도 있음. 커널 안이라 S2 불가 |
| vllm | [#40018](https://github.com/vllm-project/vllm/issues/40018) | native RoPE(새 텐서 반환) → ROCm 인덱서 fast path(제자리 가정); 캐시 SHUFFLE 쓰기 → NORMAL 읽기 | POSITION-FRAME | default | P | D | T | P | – | no | 결함 둘(‡), 둘 다 역할로 보임. 벤더 커널 주변이라 S5 아님 |
| vllm | [#47087](https://github.com/vllm-project/vllm/issues/47087) | 청크 스케줄러 → Mamba prefix cache 해시 | RANGE | opt-in | – | D | T | – | – | no | 청크 끝은 실행 중 산술. epoch 태그만 잡음 |
| vllm | [#51094](https://github.com/vllm-project/vllm/issues/51094) | 오프로드 조회(N−1 토큰) → Mamba 상태 복원(경계 N) | RANGE | niche | P | D | T | P | – | no | 커넥터가 mode "all"을 처리하지 않음 |
| vllm | [#43559](https://github.com/vllm-project/vllm/issues/43559) | 청크 스케줄러 → Mamba prefix cache 해시 | RANGE | opt-in | – | D | T | – | – | no | #47087과 같은 결함·수정 |
| vllm | [#41472](https://github.com/vllm-project/vllm/issues/41472) | 하이브리드 KV 캐시 배치 → ROCm paged decode 커널 | RANGE | default | P | D | T | P | – | yes | 수정 미병합(†). 맞는 대체 백엔드는 있었음 |
| vllm | [#50681](https://github.com/vllm-project/vllm/issues/50681) | MoE SP reduce-scatter → 다음 층 attention(shape로 추론) | REDUCTION | niche | P | D | T | – | – | no | 배치 상태를 선언하면 추론이 필요 없음 |
| vllm | [#48611](https://github.com/vllm-project/vllm/issues/48611) | MLA 캐시(fp8_ds_mla 656B) → 새 dense-MHA 문맥 gather | LAYOUT | opt-in | P | D | T | P | – | yes | 새 소비자가 packed 형식을 모름 |
| vllm | [#48324](https://github.com/vllm-project/vllm/issues/48324) | fusion 패턴 매처 → FlashInfer 융합 커널 | DTYPE | opt-in | P | D | T | P | – | no | 컴파일 시 dtype 검사 누락 |
| vllm | [#42007](https://github.com/vllm-project/vllm/issues/42007) | MoE prepare(FP8 양자화) → MoE LoRA 커널 | DTYPE | opt-in | P | D | T | P | – | yes | 원 정밀도 값을 새로 보관해야 했음 |
| vllm | [#42182](https://github.com/vllm-project/vllm/issues/42182) | KV zeroing 커널 ↔ NIXL RDMA 쓰기 | TIME | niche | – | D | – | – | – | no | 타이밍 경쟁. S2는 호스트 쪽 소유 태그를 가정 |
| vllm | [#48831](https://github.com/vllm-project/vllm/issues/48831) | compile 아래 부분 prefill 스텝 → pooler·다음 스텝 | TIME | default | – | – | T | – | – | no | 원인이 그래프 내부로 추정되고 확정되지 않음(§). 테스트만 잡음 |
| vllm | [#43602](https://github.com/vllm-project/vllm/issues/43602) | compile 워밍업(deepstack=None) → 재사용된 컴파일 그래프 | NEW:SPECIALIZATION | default | – | – | T | D | – | no | 가드 없는 특수화. 무시된 입력을 오류로 바꾸면 잡음 |
| sglang | [#30176](https://github.com/sgl-project/sglang/issues/30176) | 체크포인트 설정(oe_*) → LongcatFlashConfig(ngram_*) → 로더 | MAPPING | default | P\* | – | – | P | – | no | 모르는 키와 안 쓰인 텐서 32개를 오류로 |
| sglang | [#31833](https://github.com/sgl-project/sglang/issues/31833) | Mamba2 chunk scan(배치 전역 격자 h) → radix cache 상태 추적(요청별 격자) | POSITION-FRAME | niche | P | D | T | – | – | yes | 격자 밖 상태는 다시 계산해야 했음 |
| sglang | [#31482](https://github.com/sgl-project/sglang/issues/31482) | HiSparse 할당기(호스트·디바이스 페이지 따로) → Mooncake PD 전송 | RANGE | niche | P | D | T | – | – | no | 번호 공간이 형으로 구분되면 막힘 |
| sglang | [#37187](https://github.com/sgl-project/sglang/issues/37187) | GPT-OSS MoE all-reduce(복제) → DP-attention reduce-scatter | REDUCTION | opt-in | P | D | T | – | – | no | 이미 합산된 값을 다시 합산 |
| sglang | [#27125](https://github.com/sgl-project/sglang/issues/27125) | Conv3d(channels_last_3d) → SP all_gather | LAYOUT | opt-in | P | D | T | D | – | no | 수정 귀속은 댓글 추정(†) |
| sglang | [#31699](https://github.com/sgl-project/sglang/issues/31699) | DSv4 attention(attn-TP 합산 완료) → dp_gather_partial | REDUCTION | niche | P | D | T | – | – | no | #37187과 같은 부류 |
| sglang | [#34227](https://github.com/sgl-project/sglang/issues/34227) | 체크포인트(grouped QKV) → FSDP rank-local 로더 → attention | LAYOUT | niche | P | D | T | P | – | no | 사용자 정의 loader 우회. 수정 자체가 fail-loud |
| sglang | [#31490](https://github.com/sgl-project/sglang/issues/31490) | AITER fused RMS+FP8 quant(전치 저장, 연속처럼 보이는 stride) → CK materializer | LAYOUT | niche | P\* | D\* | T | – | – | no | 거짓말한 것은 텐서 자신의 stride |
| sglang | [#31641](https://github.com/sgl-project/sglang/issues/31641) | TRT-LLM MHA(strided Q) → XQA 커널(packed 가정) | LAYOUT | niche | P | D | T | D | – | no | stride 메타데이터가 버려짐 |

## 3. 버그별 근거

### vLLM

**#52276** DeepSeek-V4 NIXL 수신 실패, 깨진 reasoning. 수정안 [#52232](https://github.com/vllm-project/vllm/pull/52232)는 병합되지 않았다(†). 이슈는 설정 우회로 닫혔다.
- 경계: NIXL HMA 수신 워커 → 스케줄러·디코드. HMA 블록 번호는 풀마다 따로 매겨진다. 그래서 워커가 실패 블록을 보고하지 않았고, 요청은 "수신 완료"로 처리되었다.
- S1 –: 전송 실패는 실행 중 사건이다. 정적 역할이 아니다.
- S2 D: 성공한 전송만 블록에 ready 태그를 붙이면, 디코드 attention이 ready가 아닌 블록을 읽을 때 걸린다. 단, 태그를 `finished_recving` 신호가 아니라 전송 상태에서 붙여야 한다.
- S3 –: 핸드셰이크 실패 같은 결함 주입은 생성 군에 없다.
- S4 D: HMA 경로의 TODO가 실패를 조용히 버렸다. fail-loud면 요청 오류가 된다. 미병합 수정의 제목이 곧 "Fail closed"다.
- S5 –. 새 로직 no: 이 상황에서 최선의 결과는 오류 응답이다.

**#51063** Mistral3 VLM이 top-level 설정으로 tie를 결정했다. 수정 [#51665](https://github.com/vllm-project/vllm/pull/51665).
- 경계: 체크포인트 → vLLM 모델 구성과 로더. 체크포인트의 top-level은 `tie_word_embeddings: true`였지만 실제로는 다른 `lm_head` 텐서가 들어 있었다. 로더는 tie를 결정한 뒤 `lm_head.*`를 건너뛰었다.
- S1 P\*: lm_head 텐서에 참 역할("독립 텐서")을 붙이면 모델 쪽 "tied"와 어긋난다. 그러나 실제로 있던 유일한 선언이 틀린 설정 플래그였다.
- S2 –: tie 여부는 버퍼 태그 종류가 아니다.
- S3 –: 역할 명세에서 만든 기준도 같은 tie=true를 읽는다.
- S4 –: lm_head는 모델별 skip 규칙으로 명시적으로 버려졌다. 모르는 속성이 아니다.
- S5 –. 새 로직 no: untied 경로는 이미 있었다. 수정은 체크포인트 확인과 텐서 비교를 넣었다.

**#41207** DeepSeek-OCR, Transformers 업그레이드 뒤 출력 저하. 수정 [#41460](https://github.com/vllm-project/vllm/pull/41460)(v4)과 transformers#45739(v5). §.
- 경계: vLLM `DeepseekVLV2Config` → Transformers v4 `PretrainedConfig.__init__` → model_type으로 고르는 처리.
  - v4 base init이 `model_type`을 `deepseek_vl_v2`로 되돌렸다.
  - 이 필드로 고르는 처리의 예가 채팅 템플릿 대체표다. `deepseek_ocr`와 `deepseek_vl_v2`에 서로 다른 템플릿을 준다.
- S1 –: 필드 값이 선언 뒤에 덮어써졌다. 필드의 역할은 그대로다.
- S2 –.
- S3 –: 의존성 버전 행렬은 생성 군 밖이다.
- S4 –: 소비자가 버린 속성이 없다. 생산자 쪽에서 값이 바뀌었다.
- S5 –. 새 로직 no.

**#47300** Gemma 4 + FA4 + 이미지 + 긴 입력. 수정 [#47332](https://github.com/vllm-project/vllm/pull/47332).
- 경계: 레이어의 sliding window와 FA4 커널의 `q_idx`(청크 로컬) → 손으로 쓴 mm_prefix `mask_mod`.
  - `mask_mod`가 있으면 커널이 자체 창을 끄는데, 마스크에 창 조건이 없었다.
  - 로컬 `q_idx`를 절대 위치처럼 비교했다.
- S1 P: 레이어는 창을 요구하는데 mask_mod는 causal∨mm_prefix만 선언한다. 커널은 로컬 프레임을 주는데 마스크는 절대 프레임을 기대한다. 둘 다 빌드 시점에 보이는 불일치다.
- S2 –: 두 사실 모두 커널 안의 마스크 술어에 있다. 태그를 붙일 버퍼 경계가 없다.
- S3 T: 창보다 긴 입력에 이미지와 청크 prefill을 넣고 dense 마스크 기준과 비교하면 드러난다. SM90 필요.
- S4 P: `_resolve_causal_local_window`가 커널 창을 끈 뒤 누구도 창을 받지 않았다. 받지 않은 속성은 초기화 오류가 된다.
- S5 P: `(causal ∧ window) ∨ mm_prefix`를 한 번 선언하고 마스크를 생성하면 두 결함이 모두 사라진다. 25건 중 S5가 막는 유일한 경우다.
- 새 로직 no: 마스크 술어만 고쳤다. Triton 경로는 이미 맞았다.

**#40018** ROCM_AITER_MLA_SPARSE prefill 쓰레기 출력. 수정 [#43781](https://github.com/vllm-project/vllm/pull/43781). ‡.
- 경계 (1): RoPE → ROCm 희소 인덱서 fast path. Inductor 기본인 native RoPE는 새 텐서를 반환하는데, fast path는 제자리 변경을 가정하고 원래 q·k를 읽었다.
- 경계 (2): 인덱서 캐시 writer → reader. writer는 SHUFFLE로 고정되어 있었고, reader는 block_size=1이면 NORMAL로 읽는다.
- 사실: 1차는 POSITION-FRAME(q·k에 RoPE가 적용됐는가)이고, 부수 결함은 LAYOUT이다.
- S1 P: q의 역할은 "RoPE 전"인데 인덱서는 "RoPE 후"를 받는다. 다만 제자리 변경을 형에 담아야 한다(typestate). 캐시 layout 불일치는 그대로 보인다.
- S2 D: q의 frame 태그가 RoPE 전으로 남는다. 캐시 layout 태그도 어긋난다.
- S3 T: block_size=1과 기본 compile 설정을 기준과 비교하면 드러난다. ROCm MI300·MI355 CI가 필요하다.
- S4 P: fast path가 RoPE 출력 방식과 캐시 layout을 확인하게 하면 초기화 때 걸린다. 수정이 바로 이 조건을 넣었다.
- S5 –: AITER 벤더 커널과 손으로 합친 fast path다. 변형 명세가 생성할 부분이 아니다.
- 새 로직 no.

**#47087, #43559** Qwen3.5/3.6 하이브리드 + MTP + prefix caching. 둘 다 [#51113](https://github.com/vllm-project/vllm/pull/51113)으로 고쳤다.
- 경계: 청크 스케줄러(`_mamba_block_aligned_split`) → Mamba prefix cache.
  - `cache_blocks`는 슬롯 p를 (p+1)·block_size 시점의 상태로 해시한다.
  - EAGLE이 `last_cache_position`을 한 블록 당긴다. 그 위치를 넘으면 청크가 블록 중간에서 끝나도 허용되었다.
  - 그래서 `state@364`가 `state@1600`으로 해시되었다.
- S1 –: 청크 끝은 공유 토큰 예산에 따라 실행 중에 계산된다. 정렬 여부는 값 수준의 성질이다.
- S2 D: 슬롯에 "몇 토큰까지 반영한 상태인가"를 epoch 태그로 붙이면, 해시할 때 364 ≠ 1600으로 걸린다.
- S3 T: 청크 끝을 블록 경계 밖에 두는 생성 테스트에 캐시 재사용과 동시 요청을 더하면 된다. PR이 추가한 oracle 테스트가 이 모양이다. 단일 요청은 우연히 안전하므로 동시성이 생성 행렬에 있어야 한다.
- S4 –: 생산자의 조건 분기가 불변식을 어겼다. 버려진 속성은 없다.
- S5 –. 새 로직 no.

**#51094** OffloadingConnector + `mamba_cache_mode=all` + 정확한 청크 경계. 수정 [#51100](https://github.com/vllm-project/vllm/pull/51100).
- 경계: 오프로드 조회 → Mamba 상태 복원. 조회는 마지막 토큰을 다시 계산하려고 N−1개로 제한되는데, 복원은 경계 N의 상태를 가져왔다. 그래서 마지막 토큰이 두 번 적용되었다.
- S1 P: mode "all"은 청크 정렬된 적중을 요구하는데, 커넥터는 "align"일 때만 정렬한다. 둘 다 설정으로 정해진다.
- S2 D: 복원한 상태의 epoch N과 재개 위치 N−1이 어긋난다.
- S3 T: 기존 테스트가 정확한 경계를 이미 다뤘지만 "align"만 돌렸다. 캐시 모드를 나열하면 나온다.
- S4 P: 커넥터는 "align"만 알고 "all"은 조용히 기본 처리했다. 처리하지 않는 모드는 초기화 오류가 된다.
- S5 –. 새 로직 no: 조건 한 줄.

**#41472** LFM2 + ROCM_ATTN. 수정안 [#42420](https://github.com/vllm-project/vllm/pull/42420)은 병합되지 않았다(†). "v0.26.0에서 권장 명령으로 동작한다"는 댓글 뒤에 닫혔다.
- 경계: 하이브리드 KV 캐시 배치 → ROCm paged decode 커널. 커널 안에 `slot = block_table·BS + i%BS`가 박혀 있었다.
- S1 P: 캐시가 하이브리드 slot 배치를 선언하면, 연속 배치만 받는 ROCM_ATTN과 어긋난다. 실제로 거짓이었던 것은 플랫폼 전체에 붙은 `support_hybrid_kv_cache = True`다.
- S2 D: 커널 입구에서 캐시의 layout 태그를 확인하면 걸린다.
- S3 T: 하이브리드 배치를 백엔드마다 돌리면 나온다. 기존 하이브리드 테스트는 TRITON_ATTN을 강제해서 문제를 가렸다. ROCm 필요.
- S4 P: 백엔드를 고를 때 하이브리드 배치를 지원하지 않는다며 거부한다. 이미 맞게 동작하던 TRITON_ATTN이 대신 쓰인다.
- S5 –: 손으로 쓴 HIP 커널의 주소 계산이다. decode 커널 자체를 생성하는 경우에만 P다.
- 새 로직 yes: 커널이 slot_mapping을 받도록 바꿔야 했다. 올바른 대체 백엔드는 있었다.

**#50681** Qwen3.6 + EP + 시퀀스 병렬. 수정 [#50685](https://github.com/vllm-project/vllm/pull/50685).
- 경계: MoE 시퀀스 병렬 reduce-scatter → 다음 층 attention 입력. 은닉 상태가 토큰 차원으로 쪼개졌는지를 첫 차원으로 추론했다. TP2 단일 토큰 디코드에서는 전체와 샤드가 모두 한 행이라 구별되지 않는다.
- S1 P: 배치 상태를 선언하면 shape 추론이 필요 없다. 수정이 실제로 모델 전체의 고정 계약으로 바꿨다.
- S2 D: "SP 샤드" 태그를 attention 입구에서 확인한다.
- S3 T: TP/DP/EP>1에서 단일 토큰 디코드. GPU 4장.
- S4 –: 사실이 버려진 것이 아니라 추측되었다.
- S5 –. 새 로직 no.

**#48611** FlashMLA sparse의 dense-MHA 분기와 fp8_ds_mla. 수정 [#48642](https://github.com/vllm-project/vllm/pull/48642).
- 경계: MLA 캐시 writer → 새 dense-MHA 문맥 gather. writer는 `fp8_ds_mla`의 656바이트 packed 항목을 쓰는데, gather는 이를 평범한 E4M3로 읽었다. 부수 결함은 범위 밖 쓰기(RANGE)이고 메모리 오류로 드러났다.
- S1 P, S2 D: 캐시 layout과 gather가 받는 layout이 다르다.
- S3 T: 캐시 재사용(문맥 있는 짧은 prefill), KV 형식 나열, 새 라우팅 경로를 곱하면 나온다. 커널 수준 테스트는 GPU 한 장으로 된다.
- S4 P: gather가 fp8_ds_mla를 지원하는지 선언해야 한다. dtype 디스패치가 이를 E4M3 별칭으로 넘겨 처리된 것처럼 보였다. DCP 경로는 이미 오류를 냈다.
- S5 –: 양자화 캐시 형식의 문제다. 변형의 의미가 아니다.
- 새 로직 yes: packed FP8 gather가 시퀀스 시작점을 받게 했고, decode 전용 메타데이터를 새로 지원했다.

**#48324** FlashInfer allreduce+RMSNorm+quant 융합. 수정 [#48330](https://github.com/vllm-project/vllm/pull/48330).
- 경계: 컴파일 fusion 패턴 매처 → FlashInfer 융합 커널. 활성값은 BF16, RMSNorm 가중치는 FP32(`weight.float() + 1`)였다.
- S1 P: 패턴을 맞출 때 dtype을 검사하면 된다. 같은 검사가 다른 패턴에는 이미 있었다.
- S2 D: 융합 op 입구의 dtype 검사.
- S3 T: TP>1 × 혼합 dtype RMSNorm. 추가된 회귀 테스트가 이 조합이다. GPU 2장.
- S4 P: 융합 op가 FP32 가중치를 지원하지 않는다며 거부한다.
- S5 –. 새 로직 no: 기존 검사를 추가했고, 대체 융합 경로가 있었다.

**#42007** FP8 MoE + LoRA. 수정 [#42120](https://github.com/vllm-project/vllm/pull/42120).
- 경계: MoE modular kernel의 prepare → MoE LoRA 커널. prepare는 활성값을 FP8로 양자화하고 스케일을 따로 두는데, LoRA 커널은 BF16을 기대했다.
- S1 P, S2 D: 입력의 dtype·양자화 형식이 다르다. 한 LoRA 커널은 실제로 "Unsupported lhs dtype"으로 죽었고, 다른 두 커널은 조용히 계산했다.
- S3 T: 입력 양자화 형식을 LoRA 소비자마다 나열하면 나온다.
- S4 P: 초기화 때 FP8 입력을 지원하지 않는다며 거부한다.
- S5 –.
- 새 로직 yes: 원 정밀도 활성값을 따로 보관해야 했다. DP/EP에서는 양자화를 뒤로 미뤄야 했다.

**#42182** Qwen3.5 P/D + async scheduling. 수정 [#48481](https://github.com/vllm-project/vllm/pull/48481).
- 경계: 새 블록을 0으로 채우는 커널 ↔ 같은 블록에 쓰는 NIXL RDMA. 비동기 스케줄링에서는 zeroing 커널이 이전 forward 뒤에 줄을 서서, RDMA가 먼저 쓴 KV를 지웠다.
- S1 –: GPU 작업과 RDMA의 순서는 실행 중에 정해진다.
- S2 D: 호스트 쪽에서 블록에 "원격 쓰기 대기" 태그를 붙이면, zeroing을 요청할 때 결정적으로 걸린다. 데이터를 읽는 쪽의 검사만으로 잡으려면 RDMA가 태그도 함께 옮겨야 한다.
- S3 –: 타이밍에 달려 있다. 32문항은 멀쩡했고 1319문항에서 무너졌다.
- S4 –: 동기 스케줄링에서는 맞던 순서 가정이다. 모르는 속성이 아니다.
- S5 –. 새 로직 no.

**#48831** Qwen3-Reranker, 8K 토큰 초과. 수정 [#48901](https://github.com/vllm-project/vllm/pull/48901). §.
- 경계: torch.compile 아래의 부분(청크) prefill 스텝 → pooler와 다음 스텝. 끝나지 않은 커널의 버퍼가 재사용되거나 읽혔다. 수정은 부분 prefill 스텝에 동기화를 넣었고, PR 스스로 우회에 가깝다고 했다.
- S1 –.
- S2 –: 수정자가 손상 위치를 그래프 내부로 좁혔다. Inductor의 버퍼 수명 문제로 추정했다. Python 수준 태그로는 보이지 않는다. 원인이 다음 스텝의 영구 입력 버퍼 재사용이라면 D가 될 수 있지만, 확인되지 않았다.
- S3 T: 청크 경계에서 compile 기본값으로 비청크 기준과 비교하면 드러난다. 기존 테스트는 eager로만 돌았다.
- S4 –, S5 –. 새 로직 no: 동기화.

**#43602** Qwen3-VL 정확도 저하. 수정 [#43617](https://github.com/vllm-project/vllm/pull/43617).
- 경계: compile 워밍업(deepstack 입력이 None) → 가드를 다시 확인하지 않고 재사용되는 컴파일된 디코더 그래프. 실제 요청이 가져온 deepstack 텐서는 무시되었다.
- 사실: NEW:SPECIALIZATION.
- S1 –: 그래프가 어떤 입력 모양에 특수화되었는지는 어휘에 없는 역할이다.
- S2 –: 어떤 태그도 다르지 않다. 텐서가 그냥 읽히지 않는다.
- S3 T: "워밍업 뒤 첫 실제 요청"(캐시 재사용 군)을 eager 기준과 비교하면 드러난다.
- S4 D: 특수화된 그래프가 처리하지 않는 입력이 오면 오류로 만든다. Dynamo 가드를 켜 두는 것과 같다.
- S5 –. 새 로직 no.

### SGLang

**#30176** LongCat-2.0 `oe_*` 설정 키. 수정 [#30275](https://github.com/sgl-project/sglang/pull/30275). 이 지원 PR에 별칭 처리가 들어 있다. 보고는 미병합 브랜치(#30042) 기준이었다.
- 경계: 체크포인트 설정(`oe_*`) → `LongcatFlashConfig`(`ngram_*`, `emb_*`) → 모델 구성과 로더. `use_ngram_embedding`이 False가 되었고, 가중치 32개가 "not found in params_dict" 경고만 남기고 버려졌다.
- S1 P\*: 키 이름 대신 역할로 묶거나, 받을 곳이 없는 텐서 역할을 오류로 하면 드러난다. 다만 체크포인트의 역할 주석은 별칭을 빠뜨린 바로 그 어댑터가 쓰게 된다.
- S2 –: 적재되지 않은 텐서에는 태그를 붙일 버퍼가 없다.
- S3 –: 역할 명세에서 만든 기준도 같은 설정을 읽는다.
- S4 P: 모르는 `oe_*` 키와 소비되지 않은 텐서 32개가 로드 오류가 된다. 보고자가 제안한 두 번째 수정안이 이것이다.
- S5 –. 새 로직 no: 별칭.

**#31833** NemotronH extra_buffer. 수정 [#37836](https://github.com/sgl-project/sglang/pull/37836). 이슈가 닫힌 뒤에 병합되었다.
- 경계: Mamba2 chunk scan 커널 → mamba radix cache 상태 추적. 커널이 내는 packed 상태 `h`는 배치 전체를 편 하나의 격자 위에 있다. 그런데 `_init_track_ssm_indices`는 요청별 격자로 색인했다.
- S1 P: `h`의 프레임(배치 전역 격자)을 선언하면, 요청별 프레임을 기대하는 소비자와 어긋난다. 프레임은 백엔드마다 정적이다. FLA는 요청별 격자라서 맞았다.
- S2 D: `h`의 frame 태그.
- S3 T: 청크에 정렬되지 않은 시작점을 가진 다중 요청 배치를 요청별 재계산과 비교한다. PR의 테스트가 이 모양이다.
- S4 –, S5 –.
- 새 로직 yes: 원하는 상태가 `h`에 없으므로 `chunk_state_varlen`으로 다시 계산해야 했다.
- 참고: AIME·GPQA·GSM8K 점수 차이는 잡음 범위 안이었다. 캐시 적중 logprob KL은 153배 줄었다.

**#31482** DeepSeek-V4 PD HiSparse. 수정 [#31901](https://github.com/sgl-project/sglang/pull/31901).
- 경계: 디코드 쪽 HiSparse 할당기 → Mooncake PD 전송. 할당기는 호스트 페이지와 논리 디바이스 페이지를 따로 할당하는데, 전송은 디바이스 버퍼에도 호스트 페이지 번호를 썼다.
- S1 P: 호스트 페이지 번호와 디바이스 페이지 번호를 다른 형으로 두면 컴파일 오류가 난다.
- S2 D: 인덱스 배열의 번호 공간 태그.
- S3 T: 호스트와 디바이스 번호가 다른 할당 상태를 만들어 비교한다. 할당기 단위 테스트로 된다.
- S4 –: 전송 경로는 HiSparse를 알고 있었다. 번호 공간을 잘못 썼을 뿐이다.
- S5 –. 새 로직 no: 디바이스 번호를 따로 전달.

**#37187** GPT-OSS + DP attention. 수정 [#37199](https://github.com/sgl-project/sglang/pull/37199).
- 경계: GPT-OSS MoE 블록 → DP-attention 후처리 reduce-scatter. MoE 블록은 TP all-reduce로 이미 합산해 복제 상태였다. reduce-scatter는 이를 부분합으로 보고 다시 합산해서 출력이 TP배가 되었다.
- S1 P, S2 D: 복제 대 부분합.
- S3 T: TP/DP/EP>1과 복제 대 부분합이 모두 생성 군에 있다.
- S4 –: 합산 책임을 정하는 중앙 판정 함수가 이미 있었는데 GPT-OSS가 쓰지 않았다. 모르는 속성이 아니다.
- S5 –. 새 로직 no.

**#27125** Wan2.2 T2V 모자이크. 수정 [#25985](https://github.com/sgl-project/sglang/pull/25985). 귀속은 댓글의 추정이다(†).
- 경계: Conv3d → SP all_gather와 다음 conv. 가중치가 `channels_last_3d`여서 출력도 그 형식이었는데, 소비자는 기본 연속 배치를 가정했다.
- S1 P: 메모리 형식은 로드 시점에 정해진다.
- S2 D: stride는 런타임에 보인다.
- S3 T: SP>1과 메모리 형식을 곱해 단일 GPU 기준과 비교한다.
- S4 D: gather가 기본이 아닌 메모리 형식을 거부하면 첫 디코드에서 걸린다.
- S5 –. 새 로직 no: 형식 변환.

**#31699** DeepSeek-V4 DP attention, attn-TP>1. 수정 [#31700](https://github.com/sgl-project/sglang/pull/31700).
- 경계: attention 출력 → `dp_gather_partial`. attention 출력은 attn-TP로 이미 all-reduce되어 복제 상태였는데, `dp_gather_partial`은 이를 부분합으로 보고 합산했다.
- S1 P, S2 D, S3 T: #37187과 같은 이유다.
- S4 –, S5 –. 새 로직 no: `dp_gather_replicate`로 교체.

**#34227** MiniMax-H3 `--use-fsdp-inference`. 수정 [#34294](https://github.com/sgl-project/sglang/pull/34294).
- 경계: 체크포인트 → rank-local FSDP fast path 로더 → attention.
  - 체크포인트는 QKV 행을 헤드별로 묶어 저장한다.
  - fast path는 모델의 QKV 재배열 weight_loader를 우회했다.
  - attention은 Q, K, V를 이어 붙인 배치를 기대한다.
- S1 P: 적재된 파라미터의 layout 역할이 "grouped"로 남아, "concat"을 요구하는 소비자와 어긋난다.
- S2 D: 태그를 바꾸는 재배열이 우회되었으므로 태그가 "grouped"로 남는다.
- S3 T: 샤딩 적재와 표준 적재가 같은 값을 내는지 비교한다.
- S4 P: fast path가 파라미터의 사용자 정의 loader 속성을 확인하게 하면 거부하거나 대체 경로로 간다. 수정이 실제로 "지원하지 않는 사용자 정의 loader면 전체 loader로 돌아간다"를 넣었다.
- S5 –. 새 로직 no.

**#31490** DeepSeek-V4-Flash FP8 gfx950 회귀. 수정 [#31727](https://github.com/sgl-project/sglang/pull/31727).
- 경계: AITER fused RMSNorm+FP8 quant → CK materializer와 FP8 GEMM.
  - `transpose_scale=True`면 AITER는 스케일을 전치해서 저장한다.
  - 그런데 반환하는 `[M, G]` 메타데이터는 연속 배치처럼 보인다.
  - 소비자는 메타데이터를 믿었고, M>1에서 스케일이 다른 토큰에 붙었다.
- S1 P\*: 참 layout 선언이면 불일치가 보인다. 거짓말을 한 것은 텐서 자신의 stride였다.
- S2 D\*: stride가 아니라 `transpose_scale=True`에서 layout 태그를 만들어야 잡힌다.
- S3 T: 스케일 layout 생산자마다 M>1로 시험하면 드러난다. 회귀를 만든 PR은 GLM으로만 검증되었다. gfx950 필요.
- S4 –: 선언된 속성이 없었다. 소비자는 메타데이터를 그대로 믿었다.
- S5 –. 새 로직 no: 복사 없는 stride 뷰.

**#31641** NVFP4 KV + trtllm_mha, SM120. 수정 [#31667](https://github.com/sgl-project/sglang/pull/31667).
- 경계: TRT-LLM MHA 백엔드 → FlashInfer XQA 래퍼와 커널. 백엔드가 넘긴 Q는 fused QKV에서 잘라 낸 뷰라서 stride가 6144였다. 래퍼는 stride 없이 Q를 넘겼고, 커널은 packed 행으로 읽었다.
- S1 P: Q가 packed인지가 역할로 드러난다.
- S2 D: 래퍼 입구의 stride 검사. bs=1은 torch 기준으로 연속이라 통과한다.
- S3 T: bs>1이면 드러난다. 이 조합의 CI 테스트가 있었지만 한 번도 끝까지 돌지 못했다(OOM, 건너뜀). SM120 필요.
- S4 D: 래퍼가 지원하지 않는 stride를 버리지 않고 거부한다.
- S5 –. 새 로직 no: `.contiguous()`.

## 4. 합계

| 해법 | P | D | T | – | 비고 |
|---|---:|---:|---:|---:|---|
| S1 정적 역할 형 | 18 | 0 | 0 | 7 | P 중 3건은 `*` (#51063, #30176, #31490) |
| S2 런타임 sanitizer | 0 | 19 | 0 | 6 | D 중 1건은 `*` (#31490). #52276·#42182는 태그를 올바른 층위에 붙인다는 가정 |
| S3 역할 기반 테스트 생성 | 0 | 0 | 20 | 5 | T 중 10건은 ROCm, SM120, GPU 2장 이상 중 하나가 필요 |
| S4 fail-loud | 9 | 4 | 0 | 12 | |
| S5 선언적 변형 명세 | 1 | 0 | 0 | 24 | #47300만 |

합집합:
- 적어도 한 해법이 P나 D를 준 버그: **23/25**. `*` 표시를 빼면 **21/25**다. #51063과 #31490이 빠진다.
- 적어도 한 해법이 P를 준 버그(서빙 전 차단): 18/25. `*`를 빼면 16/25다.
- D로만 잡히는 버그: 5건(#52276, #47087, #43559, #42182, #43602).
- T로만 잡히는 버그: 1건(#48831). T까지 넣으면 24/25다.
- 어떤 해법도 잡지 못하는 버그: 1건(#41207).
- 새 로직이 필요했던 버그: 4건(#41472, #48611, #42007, #31833).
- 계기별 P·D 적용: default 8건 중 6건, opt-in 7건 중 7건, niche 10건 중 10건이다. 놓친 두 건(#41207, #48831)은 모두 기본 경로에서 났다.

사실별 교차표 (건수 / S1 P / S2 D / S3 T / S4 P·D / S5 P):

| 사실 | 건수 | S1 P | S2 D | S3 T | S4 P·D | S5 P |
|---|---:|---:|---:|---:|---:|---:|
| LAYOUT | 5 | 5 (`*` 1) | 5 (`*` 1) | 5 | 4 | 0 |
| DTYPE | 2 | 2 | 2 | 2 | 2 | 0 |
| REDUCTION | 3 | 3 | 3 | 3 | 0 | 0 |
| POSITION-FRAME | 2 | 2 | 2 | 2 | 1 | 0 |
| RANGE | 5 | 3 | 5 | 5 | 2 | 0 |
| TIME | 3 | 0 | 2 | 1 | 1 | 0 |
| PROPERTY | 2 | 2 (`*` 1) | 0 | 1 | 1 | 1 |
| MAPPING | 2 | 1 (`*` 1) | 0 | 0 | 1 | 0 |
| NEW:SPECIALIZATION | 1 | 0 | 0 | 1 | 1 | 0 |
| 합 | 25 | 18 | 19 | 20 | 13 | 1 |

## 5. 새 어휘

- **NEW:SPECIALIZATION** (#43602): 컴파일된 산출물이 추적 시점의 입력 모양을 전제로 특수화된 사실. 이 버그에서는 "deepstack 입력이 없다"가 그 전제였다. 재사용할 때 이 전제를 다시 확인하지 않았다. 가장 가까운 기존 어휘는 TIME(낡은 산출물의 재사용)이다. 그러나 이 사실은 시점보다 "무엇을 가정하고 만들었는가"에 가깝다. 4주차 틀의 "추측된 사실"에 해당한다.
- 기존 어휘를 넓혀 쓴 곳:
  - #52276 TIME: 전송 실패 상태를 readiness로 보았다. 따로 떼면 FAILURE-STATUS 같은 이름이 된다.
  - #40018 POSITION-FRAME: q에 RoPE가 적용됐는가를 frame으로 보았다. 기전은 제자리 변경 가정이라 EFFECT/ALIASING으로 볼 수도 있다.
  - #31482 RANGE: 페이지 번호 공간(호스트 대 디바이스)을 RANGE로 보았다. MAPPING으로 볼 수도 있다.
  - #41207 MAPPING: 필드 값이 덮어써진 경우다. 역할 불일치라기보다 값의 변형이다.

## 6. 관찰

- S1과 S4는 빌드·로드 시점에 알 수 있는 사실에 강하다. LAYOUT·DTYPE·REDUCTION·POSITION-FRAME 12건은 S1과 S2가 모두 잡는다. 반면 TIME 3건은 S1이 하나도 못 잡는다. 실행 중에 정해지는 사실 5건(#47087, #43559, #42182, #52276, #43602)은 런타임 태그나 fail-loud 오류로만 잡힌다. 그중 #52276과 #42182는 태그를 올바른 층위에 붙인다는 가정에 기댄다.
- S5는 1건(#47300)만 막는다. 나머지 24건은 변형의 의미가 아니라 배관에서 생겼다. 로더, 집합 통신, 스케줄러, 캐시 주소 계산, 벤더 커널 호출 규약이 그 배관이다. 선언적 명세의 가치는 "새 변형의 지원 시간"에 있고, 이 표본의 오답 대부분을 막는 수단은 아니다.
- 어떤 해법도 P·D로 잡지 못하는 버그가 2건 있다.
  - #41207: 의존성의 기반 클래스 초기화가 설정 값을 덮어썼다. 역할 불일치가 아니다.
  - #48831: compile과 chunked prefill이 겹칠 때의 버퍼 재사용 경쟁이다. 수정자도 근본 원인을 확정하지 못했고, 테스트만 잡는다.

  #51063과 #31490은 참 선언 전제에서만 잡힌다. 결함이 바로 거짓 선언이었기 때문이다(설정의 tie 플래그, 텐서 자신의 stride). "선언이 참이다"라는 가정이 일을 대신한 경우는 따로 세어야 한다.
- 새 계산이 필요했던 4건(#41472, #48611, #42007, #31833)에서는 선언이 조용한 오답을 오류나 대체 경로로 바꿀 뿐이다. 빠르고 맞는 경로는 여전히 손으로 써야 했다.
- S3의 T 20건은 생성된 조합이 알맞은 하드웨어에서 실제로 돈다는 가정에 기댄다. 그중 10건은 ROCm, SM120, GPU 2장 이상이 필요하다. 4건(#31641, #41472, #48831, #51094)은 가까운 테스트가 이미 있었다. 하지만 꺼져 있었거나 다른 백엔드, eager, 다른 모드로만 돌았다. 이 경우에 부족했던 것은 테스트 설계보다 실행 범위다.

## 7. 한계

- 한 사람이 판정했다. P와 D의 경계, 계기 분류, S3 생성 범위의 해석은 판단이다. 판단이 갈리면 낮은 쪽을 골랐다.
- 1라운드의 불확실성을 그대로 이어받는다.
  - 수정 PR 미병합: #52276, #41472.
  - 댓글의 추정 귀속: #27125.
  - 결함 둘을 함께 고침: #40018.
  - 더 깊은 원인 가능: #41207, #48831.
- 모든 해법은 선언이 존재한다고 전제한다. 누가 그 선언을 쓰는지, 틀리게 쓸 확률이 얼마인지는 따지지 않았다. `*` 표시는 그 전제가 결과를 좌우하는 경우만 표시한 것이다.
- 25건은 역할 계열로 분류된 버그만이다. 같은 표본에서 원인이 확인된 나머지 3건(N2 툴체인: vLLM #48058, SGLang #31011, #28685)은 분석하지 않았다. 원인이 확인된 출력 오류 28건을 분모로 하면 P·D 적용 범위는 많아야 23/28이다.
- S3의 T는 조합 폭발과 하드웨어 비용을 무시한 상한이다.
