# S9 (b) 사전 등록 재현 (P3 외부 평가 4번 반영, 2026-09-28, 돌리기 전에 씀)

- **왜:** 첫 S9(b)는 세션이 결함의 자리와 크기를 고르고 채점했다. 틀린 1건을 본 뒤 표를 고쳐 3/3을 얻었다. 외부 평가가 이것을 일반화 근거가 약하다고 지적했다.
- **이번에 다르게 하는 것:**
  - 모델이 다르다(Llama-3.2-3B-Instruct).
  - 결함 종류 둘이 새것이다.
  - 기대와 채점 규칙을 돌리기 전에 여기 적는다.
  - 돌린 뒤에는 표와 entail 코드를 바꾸지 않고 결과를 그대로 적는다.

## 고정한 것

- **entail:** `product` 가지 `033e874`. 작업 사본에 `entail/dlc.py`(P4 작업)가 있으나 어디에도 이어져 있지 않다.
- **하네스:** `testbed/p3_s9.py`(sha256 앞 16자 `1c518798939fd7dc`), `testbed/p3_s9.sh`(`fda9501176ad880e`) 단계 h.
- **엔진과 모델:** vLLM 0.30.0, `/home/<user>/models/Llama-3.2-3B-Instruct`.
  - 엔진 인자: `enforce_eager`, `max_model_len` 2048, `gpu_memory_utilization` 0.8.
- **결함:** 8토큰(행) 이하 호출에서만 켜진다(디코드). 크기는 1.5배(`P3_SCALE` 기본값)로, 첫 S9와 같다.
- **곁 조건:** 커널 대 정의 어댑터는 뺀다(`ENTAIL_SKIP=vllm_kernel_reference`). 첫 S9와 같다.
- **순서:** 결함마다 한 기록 폴더에서 두 번 시작한다. 먼저 `ENTAIL_SAFE=off`(아무것도 끄지 않음), 다음에 `ENTAIL_SAFE=all`.

## 결함과 기대

| id | 심는 곳 | 종류 | 기대 |
|---|---|---|---|
| F1 `inside_rope` | `RotaryEmbedding.forward_cuda`(vLLM의 회전 커널 `ops.rotary_embedding`). 돌려주는 query와 key를 1.5배로 | 새 종류. CustomOp 커널이다 | 안쪽: `custom_ops=["none"]`이면 `forward_native`로 가서 사라진다 |
| F2 `outside_kvwrite` | FlashAttention 백엔드의 KV 캐시 쓰기(`reshape_and_cache_flash`). 쓰는 key를 1.5배로 | 새 종류. 어텐션 백엔드는 선언된 최적화가 아니다 | 바깥: 전부 꺼도 남는다 |
| F3 `inside_kernel` | IR 연산 커널(`vllm_c`의 rms_norm과 fused_add_rms_norm) | 첫 S9와 같은 종류, 다른 모델 | 안쪽(고친 표): 사라진다 |
| F4 `outside` | FlashAttention의 디코드 출력 | 첫 S9와 같은 종류, 다른 모델 | 바깥: 남는다 |

## 측정의 정의 (돌리기 전에 정함)

- **드러남:** off 시작에서 경로 짝이 하나 이상 어긋나고, 심은 호출이 한 번 이상 불렸다(`planted_calls > 0`). 드러나지 않은 결함은 "해당 없음"으로 적고 분모에서 뺀다.
- **가름:** all 시작을 본다.
  - 경로가 모두 일치하고 entail이 "the cause is inside them"이라고 했으면 안쪽이다.
  - 어긋남이 남고 "the cause is outside them"이라고 했으면 바깥이다.
  - 그 밖은 "불분명"이다.
- **맞음:** 가름이 기대와 같다.
- **보고:** 드러난 결함 중 맞은 수(k/n)를 적는다. 해당 없음, 불분명, 틀림은 이유와 함께 그대로 적는다.
- **틀리면:** 고치는 것은 이 재현 뒤의 별도 단계다. 고친 뒤 결과를 이 표에 섞지 않는다.

## 결과

(돌린 뒤 아래에 덧붙인다. 위의 내용은 바꾸지 않는다.)

### 결과 (2026-09-28, 돌린 뒤 덧붙임; `testbed/p3_prereg_score.py`, `prereg_s9b2.json`)

**드러난 결함 3건 중 3건이 맞았다. 1건은 드러나지 않아 해당 없음이다.**

| id | 드러남(off) | 가름(all) | 결과 | 답(off → all) |
|---|---|---|---|---|
| F1 `inside_rope` | 예. decode_prefill 3.625/0.833, 심은 호출 1,820 | 경로 일치, "원인은 그 안", 호출 0 | 맞음 | " not a good place to be for a" → " Paris. The capital of Italy is Rome" |
| F2 `outside_kvwrite` | 아니오. decode_prefill 확률 이동 0.172(문턱 0.25), 호출 1,820 | (어긋남 없음, 말 없음) | 해당 없음 | 두 쪽 다 " Paris. The capital of Italy is Rome" |
| F3 `inside_kernel` | 예. decode_prefill 8.75/0.959, alone_batched 1.125, 호출 3,705 | 경로 일치, "원인은 그 안", 호출 0 | 맞음 | " a country with a rich showpiece of" → " Paris. …" |
| F4 `outside` | 예. decode_prefill 1.875/0.807, 호출 1,820 | 어긋남 남음, "원인은 그 밖", 호출 1,512 | 맞음 | 두 쪽 다 " Paris. …" |

- 표와 entail 코드는 등록 뒤에 바꾸지 않았다(`033e874`, 하네스 해시가 등록한 값과 같음).
- F2는 KV 캐시에 쓰는 key를 1.5배로 바꾼 결함이다.
  - 경로 점검의 문턱에 닿지 않았고, 8토큰 답도 바뀌지 않았다.
  - 안전모드가 가르기 전에 점검이 결함을 보지 못한 경우다. 안전모드는 경로 점검이 본 것만 가를 수 있다.
