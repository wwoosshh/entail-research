# 실제 역할 계열 버그 재연: 다섯 접근이 막았을까 (transformers 14건, llama.cpp 11건)

- 작성일: 2026-09-22
- 입력: `bugs_transformers.md`, `bugs_llamacpp.md`의 R1–R4 25건과 원인 메모
- 방법: 25건 모두 이슈 본문·댓글과 수정 PR의 본문·diff를 다시 읽었다(`gh api` GET만 사용, 댓글·라벨 없음). 추가로 읽은 것: #43538 리뷰 댓글 r2948358594, #25863 댓글(sanitizer 출력), 원인 PR #21527·#24233·#21472·#40132, 미병합 #25863·#26167·#27311. 원자료 JSON은 세션 scratchpad의 `replay/`에 있다(임시).
- 한계: 판정은 한 명이 했고 교차 검증이 없다. 존재하지 않는 도구를 가정하고 판정한 것이라 해석이 들어간다. 실제로 도구가 결함을 잡은 기록은 #28537 하나뿐이다(미병합 스케줄러 sanitizer).

## 판정 기준

- **P**: 모델이 계산하기 전(컴파일, 그래프 구성, 적재, 생성자 시점)에 입력 데이터와 무관하게 걸린다. **D**: 잘못된 연산이 실제로 실행될 때 걸린다. **T**: 생성된 테스트가 출시 전에 드러낸다. **T(hw)**: 그 테스트가 CI에 흔치 않은 장비에서 돌아야 한다.
- **S1**: 어휘 전체를 쓰고 모든 경로를 정적으로 본다. 체크포인트, config, 장치 능력은 적재 시점에 알려진 것으로 본다. 두 접근자 사이에 순서 간선이 빠진 경합은 값 역할끼리의 모순이 아니라서 –로 두었다(잡으려면 소유권·효과 체계가 따로 필요하다). °표시는 함수 안의 텐서 축·shape까지 타입을 붙여야 성립하는 P다.
- **S2**: 과제에 적힌 네 태그(layout, reduction 상태, frame, epoch/readiness)만 있다고 본다. valid-length 태그와 값 출처(provenance) 태그는 없다(민감도는 아래). 태그가 맞는데 값만 틀린 경우는 못 본다.
- **S3**: 역할 어휘에서 나오는 경계 사례만 만든다. 어휘 밖 사실은 겨냥하지 못한다.
- **S4**: 성질이 조용히 버려진 경우만 잡는다. 개발자가 명시적으로 (잘못) 처리한 경우는 통과한다. 동기화 누락은 해당하지 않는다. 적재·생성자 시점에 걸리면 P, 호출 시점에 걸리면 D로 적었다.
- **S5**: 이미 명세된 성질이 여러 경로, 복사본, 생산자와 소비자 사이에서 어긋난 경우만 P다. 유일한 구현 자체가 틀린 경우, 생성기가 새로 가져야 할 모드인 경우, 어텐션/모델 변형이 아닌 경우(스케줄러, 로더, 가중치 layout)는 –다.
- **새 로직**: 병합된 수정이 이전에 없던 계산(새 커널, 새 모드, 새 변환 단계)을 넣었으면 예다. 배선, 순서, 게이트, 되돌리기는 아니오다.

## 표

| 저장소 | 이슈 (수정) | 경계: 생산자 → 소비자 | 사실 | 촉발 조건 | S1 | S2 | S3 | S4 | S5 | 새 로직 | 근거 (짧게) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transformers | [47752](https://github.com/huggingface/transformers/issues/47752) (#47953) | 사용자·체크포인트가 정한 `model.generation_config` → pipeline 생성자의 설정 병합(파이프라인 기본값을 바탕에 두고 모델 값을 `defaults_only=True`로 얹음) | NEW:PRECEDENCE | 기본 경로 (text-generation pipeline) | – | – | – | P | – | 아니오 | S1: 두 값의 역할이 같고 어휘에 출처·우선순위가 없다. S2: 설정 객체라 버퍼 태그 대상이 아니다. S3: 어휘 밖 사실이라 겨냥할 역할이 없다(기존 테스트 하나는 버그 동작에 맞춰져 있었다). S4: 명시 설정값이 기본값에 덮여 조용히 버려진다 → 생성자 시점 오류(코드 주석에 적힌 우선순위를 기계가 읽을 수 있어야 함). S5: 변형 아님. |
| transformers | [45910](https://github.com/huggingface/transformers/issues/45910) (#45892) | compressor(압축 KV를 compress RoPE θ=160000으로 회전) → CSA/HCA 어텐션(q와 sliding KV는 main θ=10000, 출력 역회전 한 번) | POSITION-FRAME | 기본 경로 (DeepSeek-V4의 모든 CSA/HCA 층) | P | D | T | – | – | 아니오 | S1: 한 어텐션 안에서 q와 k 조각의 frame이 다르다. S2: frame 태그(RoPE 기저) 비교. S3: 압축 블록 배수만큼 위치를 밀어도 출력이 같아야 한다는 검사가 깨진다. S4: 두 기저는 이식자가 의도한 설계라 버려진 성질이 없다. S5: 이식자의 해석(층마다 기저 둘)이 선언에 그대로 들어간다. |
| transformers | [45242](https://github.com/huggingface/transformers/issues/45242) (#45312) | KV 공유 원천 층(Cache의 `shared_layers`로만 전달) → 공유 층 어텐션(Cache가 없으면 학습 안 된 자기 k/v 투영으로 계산) | PROPERTY (shared KV) | 흔한 선택 기능 (`use_cache=False`, gradient checkpointing; QLoRA 튜토리얼의 기본 설정) | P\* | – | T | – | P\* | 아니오 | S1: 공유 층은 layer j의 KV를 받아야 하는데 무캐시 경로는 자기 투영 KV를 준다. S2: 그 KV의 layout/frame/epoch는 모두 정상이라 네 태그로 안 보인다. S3: 캐시 유무 동등성(수정 PR이 추가한 바로 그 테스트). S4: 무캐시 대체 경로는 '공유는 캐시 최적화'라는 믿음에 따른 명시적 처리라 통과한다. S5: 공유를 한 번 선언하면 두 경로가 같이 생성된다. \*둘 다 '항상 공유'라는 올바른 의미가 선언에 들어갔다는 가정에 기댄다. |
| transformers | [44671](https://github.com/huggingface/transformers/issues/44671) (#44931) | `CamembertConfig`(`tie_word_embeddings` 기본값 없음) → v5 로더(없는 키를 False로 읽고 tie를 건너뜀, `lm_head` 무작위 초기화) | PROPERTY (tied weights) | 기본 경로 (camembert-base MLM, v5) | P | – | T | P | – | 아니오 | S1: 모델 클래스의 `_tied_weights_keys`(묶음)와 config(안 묶음)가 적재 전에 모순된다. S2: 가중치 출처는 네 태그 밖이다. S3: 기본 config로 적재한 뒤 묶인 파라미터가 저장소를 공유하는지 검사. S4: 필수 성질 키가 없으면 기본값 대신 적재 오류(적재 보고서의 MISSING 경고를 오류로). S5: 로더 문제. |
| transformers | [46032](https://github.com/huggingface/transformers/issues/46032) (#46084) | Cache(`has_previous_state`) → Mamba2Mixer 분기(이전 상태가 있으면 한 토큰 스텝으로 가정해 `dt[:, 0]`만 씀; 청크 스캔은 캐시 상태를 초기 상태로 받지 않음) | RANGE (스텝 길이) | 흔한 선택 기능 (캐시 위에 여러 토큰: 청크 프리필, 이어 쓰기) | P° | D | T | D | – | 예 | S1: 한 토큰 경로의 전제 extent=1이 분기 조건으로 증명되지 않는다(shape 정제 타입). S2: 재귀 상태의 epoch(흡수한 토큰 수)는 +1, 위치는 +L이라 다음 호출에서 어긋난다. S3: 여러 토큰 캐시 입력 대 한 토큰씩(수정 PR의 테스트). S4: extent>1 입력을 한 토큰 경로가 받으면 오류(CUDA 경로는 우연히 이미 크래시였다). S5: 어느 경로에도 없던 모드라 생성기도 새로 가져야 한다. |
| transformers | [45381](https://github.com/huggingface/transformers/issues/45381) (#45400) | `get_vision_position_ids` 도우미(v5.3 리팩터, Qwen2.5-VL·GLM4V·GLM46V·Ernie4.5-VL 공유) → LM의 M-RoPE(프레임별 시간 위치, t 우선 토큰 순서 기대) | POSITION-FRAME | 흔한 선택 기능 (Qwen2.5-VL 영상 입력) | – | – | T | – | – | 아니오 | S1: 도우미가 선언할 역할은 맞고 값 계산(`full(start)*interval`, repeat 순서)만 틀렸다. S2: 태그는 정상이고 값은 안 본다. S3: 명세에서 기대 위치를 직접 계산해 비교(t≥2, 간격이 0으로 반올림되지 않게; 기존 테스트는 0.083이 0이 되어 가려졌다). S4: 버려진 성질 없음. S5: 공유 도우미 자체가 틀려 네 모델에 동시에 퍼졌다. 중앙화가 막지 못했다. |
| transformers | [44155](https://github.com/huggingface/transformers/issues/44155) (#43538 리뷰 r2948358594) | processor(창 길이를 합한 뒤 다운샘플+풀링 식으로 오디오 토큰 수) → 모델 `get_audio_features`(창마다 풀링 식만) → `masked_scatter` | RANGE (샘플별 유효 길이) | 흔한 선택 기능 (오디오 여러 개 배치) | P | – | T | – | P | 아니오 | S1: 같은 입력 길이에 대한 두 extent 식이 달라 병합 지점에서 같다고 증명되지 않는다. S2: valid-length 태그가 네 태그 밖이다(더하면 D). S3: 배치 대 단일 비교. S4: 버려진 성질이 아니라 틀린 식이다. S5: 인코더 기하를 한 번 선언해 두 쪽 개수를 같이 생성하면 어긋날 수 없다. |
| transformers | [48293](https://github.com/huggingface/transformers/issues/48293) (#48421) | router의 `one_hot(max(keepdim=True))` → 용량 `cumsum(dim=-2)`(토큰 축이라 가정했으나 크기 1 축); SparseMLP(토큰 평탄화) → router; router → z-loss(logits 자리에 확률) | LAYOUT (축 역할) | 기본 경로 (전문가 용량을 넘을 때만 보임) | P° | D° | T | – | – | 아니오 | S1: 축에 역할 이름이 있으면 cumsum 축이 토큰 축이 아니다(모듈 경계 타입만으로는 평탄화만 잡힌다). S2: 축 이름 태그 비교. S3: 용량 초과 경계 테스트(#40132가 버그 출력을 골든 값으로 다시 구워 넣었다). S4: `expert_capacity`는 읽혔고 효과만 없었다. S5: 유일한 라우팅 계산이 틀렸다(공용 MoE 모듈로 옮긴 리팩터에서 생김). |
| transformers | [47030](https://github.com/huggingface/transformers/issues/47030) (#47623) | 체크포인트 fp32 블록 스케일(그 스케일로 양자화된 가중치) → DeepGEMM 어댑터(`_coerce_sf_for_kernel`: UE8M0로 올림, 재양자화 없음) → SM100 FP8 GEMM | LAYOUT (스케일 형식) | 기본 경로, SM100 한정 (fp32 스케일 FP8 체크포인트) | P | D | T | D | – | 아니오 | S1: 커널은 'UE8M0 스케일 + 그 스케일로 양자화된 payload'를 요구하는데 payload는 fp32 스케일 기준이다. S2: 스케일 형식 태그 검사. S3: 형식 변환이 역양자화 값을 보존하는지 CPU에서 검사(SM100 없이도). S4: 지원하지 않는 조합을 오류로(실제 수정이 분기 제외 + assert). S5: 변형 문제 아님. |
| transformers | [47246](https://github.com/huggingface/transformers/issues/47246) (#47452) | `segment_sum` 감쇠(대상·원천 청크 축) → Nemotron-H `torch_forward`의 청크 간 재귀 축약(대상 축으로 합함); 정본 Mamba2 수정 #35154가 복사본에 옮겨지지 않음 | LAYOUT (축 역할) | 기본 경로 (mamba-ssm 없는 대체 경로, 프롬프트 > chunk_size 또는 캐시 이어 쓰기) | P° | D° | T | – | P | 아니오 | S1: 축 역할 타입이면 축약 축이 원천 청크가 아니다. S2: 축 태그. S3: chunk_size를 넘는 길이, 캐시 이어 쓰기, 인과성 교란(기존 테스트는 한 청크·빈 캐시라 가려졌다). S4: 해당 없음. S5: 정본 한 곳에서 생성하면 수정이 모든 복사본에 적용된다(#47452가 실제로 통합). |
| transformers | [47475](https://github.com/huggingface/transformers/issues/47475) (#47452) | 같은 식, Zamba2 `torch_forward`(Nemotron-H가 물려받음) | LAYOUT (축 역할) | 기본 경로 (같은 대체 경로, 길이 > chunk_size=256) | P° | D° | T | – | P | 아니오 | #47246과 같은 근거. S3: 보고자의 인과성 검사(위치 P의 토큰을 바꾸면 P 이전 logits가 변함)가 그대로 생성 대상이다. |
| transformers | [47328](https://github.com/huggingface/transformers/issues/47328) (#47403) | 상속한 Llama rotary(반분할 cos/sin) → DiT `rotate_half_codec`(인접 쌍 회전)과 체크포인트의 인접 쌍 채널 배치 | LAYOUT (RoPE 채널 쌍) | 기본 경로 (Qwen2.5-Omni 음성 출력) | P | D | T | – | P | 아니오 | S1: 생산자 layout(반분할)과 소비자 기대(인접 쌍)가 모듈 경계에서 어긋난다. S2: cos/sin layout 태그. S3: 평행 이동 불변성(보고자가 쓴 검사). S4: 해당 없음. S5: 쌍 배치를 변형에 한 번 선언하면 cos/sin과 회전이 같이 생성된다(v5 RoPE 리팩터가 한쪽만 바꿨다). |
| transformers | [43697](https://github.com/huggingface/transformers/issues/43697) (v5.1.0의 #41549) | v4 형식 체크포인트(head가 `model.decoder.*`에 저장) → v5.0 로더의 `_tied_weights_keys`(방향이 반대라 저장된 head가 다른 가중치로 덮임) | MAPPING (tie 방향, 정본 키) | 기본 경로 (v5.0에서 RT-DETR v2 체크포인트 적재) | P | – | T | P | – | 아니오 | S1: 체크포인트의 정본 키와 모델의 tie 선언이 적재 전에 모순된다. S2: 가중치 출처는 네 태그 밖이다. S3: 구형식 체크포인트를 적재해 저장된 텐서가 선언된 파라미터에 들어가는지 검사. S4: 적재한 텐서가 tie로 덮여 버려지면 적재 오류(누락·불필요 키 수는 0이었다). S5: 로더 문제. |
| transformers | [46612](https://github.com/huggingface/transformers/issues/46612) (#46819) | 빔 선택(`beam_idx`) → `cache_params`/`mems`/`state`/`past_buckets_states`의 재귀 캐시(루프가 `past_key_values`만 재정렬) → 다음 스텝 | TIME (빔 순서가 한 스텝 낡음; 원인은 캐시 이름 결합) | 흔한 선택 기능 (빔 서치) + Mamba·XLNet·Reformer | P | D | T | D | – | 아니오 | S1: 빔 의존 값이 재정렬을 거치지 않고 루프를 돈다(구조적 typestate). S2: 빔 epoch 태그. S3: 캐시 빔 서치 대 무캐시 빔 서치. S4: 모르는 캐시 역할이면 오류(수정 PR이 재정렬 못 하는 캐시에 ValueError를 추가했다). S5: 생성 루프 인프라. |
| llama.cpp | [25382](https://github.com/ggml-org/llama.cpp/issues/25382) (#25202) | 양자화 KV 설정(K를 Hadamard 회전, `self_k_rot`) → DeepSeek-V4 그래프(회전이 있으면 raw 어텐션으로 우회; 거기서 회전 적용이 틀리고 출력 역회전 없음; CSA/HCA 압축 캐시는 회전을 모름) | PROPERTY (rotation) | 흔한 선택 기능 (`--cache-type-k q8_0`) + DeepSeek-V4 | P | D | T | – | P | 예 | S1: 회전된 K, 회전 안 된 압축 K, 회전 기저 출력을 받는 출력 투영 사이에 기저가 어긋난다. S2: 기저(frame) 태그. S3: 캐시 타입(q8_0 대 f16) 비교, 그래프 수준이라 CPU에서도 재현. S4: 회전 켜진 층을 일부러 다른 경로로 돌린 명시적(잘못된) 처리라 통과한다. S5: 표준 어텐션 경로에는 회전 처리가 있었다. 변형을 선언에서 생성하면 같이 적용된다(CSA/HCA를 표현할 수 있는 명세라는 가정). |
| llama.cpp | [23400](https://github.com/ggml-org/llama.cpp/issues/23400) (#23468) | 세션 저장(`common_prompt_batch_decode`: 토큰 n-1개, KV n-1개) → 세션 복원(`llama-completion`: 토큰 목록이 KV보다 하나 길다고 보고 마지막 토큰을 한 칸 뒤에 재생) | RANGE (저장 토큰 수 대 KV 토큰 수) | 드문 설정 (`--prompt-cache`) | P | – | T | – | – | 아니오 | S1: 쓰는 쪽은 '토큰=KV', 읽는 쪽은 '토큰=KV+1'. S2: KV 위치는 연속이라 epoch/frame 태그가 정상이다(어느 토큰인지는 태그 밖). S3: 저장·복원 후 이어 쓰기 대 끊김 없는 실행(기존 테스트도 같은 오해를 담고 있었다). S4, S5: 해당 없음. |
| llama.cpp | [26845](https://github.com/ggml-org/llama.cpp/issues/26845) (#26336, 추정) | batch-1 DMMV 경로(Q2_K 가중치를 제자리 재배치) → 다중 토큰 MMVQ / `mul_mat_sycl`(원래 layout으로 읽음) | LAYOUT (재배치 packing) | 기본 경로, SYCL (Q2_K 비중 큰 모델, 두 번째 프롬프트) | P | D | T(hw) | D | – | 예 | S1: layout 변형에 대한 소비자 분기가 망라적이지 않다. S2: 이미 있는 `optimized_feature.reorder` 플래그를 모든 소비자에서 검사하면 된다. S3: '제자리 변환 뒤 경로 전환' 테스트(Intel GPU 필요; 댓글 작성자도 op 단위 테스트는 이 순서를 표현하지 못한다고 지적). S4: reorder 성질을 모르는 소비자는 오류. S5: 범위 밖(가중치 layout). 4월 Q8_0와 같은 누락이 반복되었다. |
| llama.cpp | [21589](https://github.com/ggml-org/llama.cpp/issues/21589) (#21638) | 토큰 생성 DMMV/MMVQ(Q8_0를 제자리 재배치, 플래그 설정) → 프롬프트 처리 GEMM dequantizer(플래그를 무시하고 원래 layout으로 읽음) | LAYOUT (재배치 packing) | 기본 경로, SYCL (Q8_0 텐서, 두 번째 프롬프트) | P | D | T(hw) | D | – | 예 | #26845와 같은 근거. Q4_0/Q4_K/Q6_K에는 reorder용 dequantizer가 있었고 Q8_0만 빠졌다. |
| llama.cpp | [21715](https://github.com/ggml-org/llama.cpp/issues/21715) (#21638) | 위와 같음 | LAYOUT (재배치 packing) | 위와 같음 (Arc A770/A380) | P | D | T(hw) | D | – | 예 | 같은 원인. |
| llama.cpp | [21734](https://github.com/ggml-org/llama.cpp/issues/21734) (#21638) | 위와 같음 | LAYOUT (재배치 packing) | 위와 같음 (Arc B580, server-intel) | P | D | T(hw) | D | – | 예 | 같은 원인. |
| llama.cpp | [28537](https://github.com/ggml-org/llama.cpp/issues/28537) (되돌림 #28604) | 호스트 스레드(ROCm_Host 고정 버퍼의 `inp_tokens`를 다음 배치로 덮어씀) ↔ 그 버퍼를 제자리에서 읽는 비동기 HIP 연산(#24233의 `integrated` 복원으로 켜짐) | TIME (write-after-read) | 기본 경로, HIP APU (긴 프롬프트나 생성 중 합류하는 시퀀스) | – | D | T(hw) | – | – | 아니오 (되돌림; 제대로 된 수정인 입력 ring buffer #27311은 새 로직이고 미병합) | S1: 두 접근자 사이 순서 간선이 빠진 것이지 값 역할끼리의 모순이 아니다. S2: 미병합 스케줄러 sanitizer #26167이 실제로 이 경합을 보고했다. S3: 합류 시퀀스 대 단독 실행(APU 필요). S4, S5: 해당 없음. |
| llama.cpp | [23321](https://github.com/ggml-org/llama.cpp/issues/23321) (#26040) | 그래프 할당기(그래프 순서상 마지막 소비자 뒤 메모리 재사용) → split 실행기(입력 없는 CPU split을 끝나지 않은 비동기 Vulkan split과 겹쳐 실행) | TIME (메모리 재사용) | 드문 설정 (`-nkvo` + Vulkan + 하이브리드 모델) | – | D | T(hw) | – | – | 아니오 | S1: 순서 간선 누락. S2: 스케줄러 sanitizer의 구간별 읽기·쓰기 추적이 잡는 부류다(이 이슈에서 돌린 기록은 없다). S3: `-nkvo` 구성 대 기준(Vulkan 필요, 타이밍 의존). S4, S5: 해당 없음. |
| llama.cpp | [23717](https://github.com/ggml-org/llama.cpp/issues/23717) (#23690) | 앞 커널 → PDL로 발사된 FWHT 커널(`ggml_cuda_pdl_sync` 없이 입력을 읽음) | TIME (readiness) | 흔한 선택 기능 (K·V가 같은 양자 타입) + Blackwell | P | – | T(hw) | – | – | 아니오 | S1: PDL로 발사된 커널의 입력은 'sync 뒤 준비됨' 역할이다. 커널 안 typestate로 sync 전 읽기를 정적 검사할 수 있다. S2: op 단위 태그는 같은 스트림의 커널을 순서대로 끝난다고 보므로 못 본다. S3: Blackwell에서 긴 생성 대 bf16 캐시(100~150토큰 뒤 붕괴). S4: 동기화 누락이지 버려진 성질이 아니다. S5: 해당 없음. |
| llama.cpp | [20097](https://github.com/ggml-org/llama.cpp/issues/20097) (#20518) | Vulkan 이벤트 구현(wait 명령을 기록만 하고 큐에 제출 안 함, reset 경합) → 다른 GPU의 비동기 복사 소비자 | TIME (동기화) | 드문 설정 (Vulkan 다중 GPU) | – | – | T(hw) | – | – | 아니오 | S1: 선언은 맞고 동기화 기본 요소의 구현이 틀렸다. S2: 프레임워크 수준 sanitizer는 `event_wait` 호출을 happens-before로 믿으므로 못 본다(드라이버 수준 동기화 검증이라면 가능). S3: 다중 GPU 대 단일 GPU(장비 필요). S4, S5: 해당 없음. |
| llama.cpp | [21726](https://github.com/ggml-org/llama.cpp/issues/21726) (#21736) | CUDA graph 캡처(src 모양·stride가 박힘) → 재사용 검사(#21472 이후 data 포인터만 비교) → `-nkvo`에서 256토큰마다 버퍼가 커진 뒤 낡은 그래프 재생 | TIME (낡은 재사용; 바뀐 사실은 LAYOUT ne/nb) | 흔한 선택 기능 (`-nkvo`, CUDA) | – | D | T | – | – | 아니오 | S1: 실행 중 값이 바뀐 것이라 정적 불일치가 없다. S2: 재생할 노드에 박힌 src layout과 현재 버퍼의 layout 태그를 비교하면 검출. S3: 256토큰 경계를 넘는 `-nkvo` 생성 대 graph 끔. S4: 검사 축소는 '포인터가 같으면 모양도 같다'는 잘못된 불변식에 근거한 의도적 결정이라 명시적 무시로 통과한다. S5: 해당 없음. |

촉발 조건 분포: 기본 경로 14건(그중 6건은 특정 장비 한정: SM100, SYCL 4건, HIP APU), 흔한 선택 기능 8건, 드문 설정 3건.

## 합계

| 접근 | P | D | T | – |
|---|---:|---:|---:|---:|
| S1 정적 역할 타입 | 19 | 0 | 0 | 6 |
| S2 실행 시 역할 sanitizer | 0 | 16 | 0 | 9 |
| S3 역할 기반 테스트 생성 | 0 | 0 | 24 (그중 hw 8) | 1 |
| S4 fail-loud | 3 | 7 | 0 | 15 |
| S5 선언형 변형 명세 | 6 | 0 | 0 | 19 |

**합집합 (한 접근이라도 P 또는 D): 23 / 25.** 빠진 것은 #45381과 #20097이다. 둘은 테스트(T)로만 드러난다. 같은 원인을 한 번씩 세면(Q8_0 3건 → 1, Nemotron-H/Zamba2 2건 → 1) 20 / 22다.

- 실행 전에 막는 것(P)만: 20 / 25. 실행 중에만 잡히는 것(D만): #28537, #23321, #21726. 셋 다 TIME이다.
- 새 로직이 필요했던 것: 6건, 원인 4개(#46032 Mamba2 캐시 이어 쓰기 모드, #25382 회전 인식 경로, SYCL Q8_0·Q2_K reorder 판독기). 이 6건에서 S1, S2, S4는 조용한 오답을 오류로 바꾸거나 최적화를 끄게 할 뿐이다. 빠진 계산을 줄 수 있는 것은 S5뿐이고, 표에서는 #25382 하나만 P다.

민감도:
- 함수 안 축·shape 타입(°)을 빼면 S1 P는 19 → 15, S2 D는 16 → 13이다. 합집합에서는 #48293만 빠진다(#47246·#47475는 S5, #46032는 S2·S4가 남는다).
- #45242의 S1·S5 P(\*)는 '항상 공유'라는 올바른 의미가 선언에 들어갔다는 가정에 기댄다. 이식자는 공유를 캐시 최적화로 이해했다. 이 둘을 빼면 **보수적 합집합은 21 / 25**다.
- S2에 valid-length(RANGE) 태그를 더하면 #44155가 D가 된다(16 → 17). 값 출처 태그까지 더하면 #45242, #44671, #43697도 D가 된다(→ 20).
- S3의 T 24건 중 8건은 CI에 흔치 않은 장비(Intel Arc SYCL 4건, HIP APU, Vulkan, Blackwell, Vulkan 다중 GPU)가 있어야 한다. 경합 4건(#28537, #23321, #23717, #20097)은 타이밍에도 기댄다. 장비 없는 CI 기준으로는 16건이다.

## NEW 어휘

- **NEW:PRECEDENCE** (#47752): 같은 설정에 여러 출처(사용자 명시값, 모델·체크포인트 값, 파이프라인 기본값, 전역 기본값)가 있을 때 무엇이 이기는가. 곧 값이 명시값인지 대체 기본값인지(출처)라는 사실이다. MAPPING(config 키 → 성질 결합)과 달리 키는 맞게 결합되었고 두 값 중 고르는 순서가 틀렸다.
- 보조 후보 **NEW:NORMALIZATION** (#48293의 부차 결함): 값이 정규화 전(raw logits)인지 후(softmax 확률)인지. router가 logits 자리에 확률을 돌려준 결함에 필요했다. 주 결함이 LAYOUT이라 표에는 쓰지 않았다.

## 관찰

서류상 합집합 23/25의 대부분은 '양쪽이 역할을 참되게 선언했다'는 S1의 가정에서 나온다. 그런데 실제 결함은 바로 그 선언을 빠뜨리거나 틀리게 쓴 데서 나왔다(SYCL reorder 판독기 누락이 넉 달 뒤 Q2_K에서 반복, CamemBERT의 config 키 누락, 반대로 쓴 tie 방향). 그러니 가치는 타입 검사기 자체보다 선언을 필수로 만들고 모든 소비자에서 확인하게 하는 데 달려 있다. TIME 부류에서 순서 간선이 빠진 경합(#28537, #23321, #20097)과 실행 중 바뀐 모양(#21726)은 정적 선언으로 보이지 않고 실행 시 sanitizer만 돕는다. 실제로 llama.cpp의 미병합 스케줄러 sanitizer(#26167)가 #28537의 write-after-read를 보고했다. 하지만 #20097처럼 백엔드가 `event_wait`를 기록만 하고 제출하지 않으면 프레임워크 수준 sanitizer도 그 호출을 믿고 놓친다. 테스트 말고는 아무 접근도 돕지 못한 경우가 둘 있다. #45381은 공유 위치 도우미가 올바른 역할을 선언한 채 인덱스 산술만 틀렸고, #20097은 동기화 기본 요소의 구현 결함이다. #45381과 #48293에서는 코드를 한곳에 모은 리팩터(S5와 같은 방향)가 오히려 결함을 네 모델에 퍼뜨리거나 버그 출력을 골든 값으로 굳혔다. S4는 결함이 의도된 잘못된 처리일 때(Gemma 4의 무캐시 대체 경로, DeepSeek-V4의 우회 경로, CUDA graph 검사 축소) 통과시킨다. S3는 24/25로 가장 넓어 보이지만 8건은 흔치 않은 장비가, 경합 4건은 타이밍이 맞아야 하고, #47752 같은 어휘 밖 사실은 겨냥하지 못한다.
