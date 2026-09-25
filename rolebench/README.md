# rolebench: 역할 계열 결함 재현 벤치마크

- 로드맵 1단계. 측정 정의는 `PROTOCOL.md`, 결과 표는 `results/results.md`(`run_all.py`가 만듦)에 있다.
- 실행 방법: WSL gpu 환경에서 아래를 돌린다. 사례마다 별도 프로세스로 실행된다.
  ```bash
  cd ~/ai_compiler/rolebench && python run_all.py
  ```

## 사례

| 번호 | 사실 | 결함의 모양 | 원래 이슈 | 종류 | 증상 |
|---|---|---|---|---|---|
| 01 | LAYOUT | 한 경로가 양자화 가중치를 재배치했는데, 다른 경로가 원래 형식으로 읽음 | [llama.cpp #21589](https://github.com/ggml-org/llama.cpp/issues/21589), [#26845](https://github.com/ggml-org/llama.cpp/issues/26845) | 기전 모사 | 깨진 출력(NaN 포함) |
| 02 | LAYOUT | 블록 스케일을 2의 거듭제곱으로 올리고 FP8 데이터는 재양자화하지 않음 | [transformers #47030](https://github.com/huggingface/transformers/issues/47030) | 기전 모사 | 그럴듯한 오답 |
| 04 | REDUCTION | 이미 합산되어 복제된 값을 부분합으로 보고 다시 합산 | [SGLang #37187](https://github.com/sgl-project/sglang/issues/37187), [#31699](https://github.com/sgl-project/sglang/issues/31699) | 기전 모사(gloo 2프로세스) | 깨진 출력 |
| 05 | FRAME | 마스크 함수가 청크 상대 질의 위치와 절대 키 위치를 비교 | [vLLM #47300](https://github.com/vllm-project/vllm/issues/47300) | 기전 모사 | 깨진 출력 |
| 06 | PROPERTY | 사용자 마스크를 주면 커널 자체의 윈도가 꺼지는데, 마스크에 윈도 조건이 없음 | [vLLM #47300](https://github.com/vllm-project/vllm/issues/47300) | 기전 모사 | 깨진 출력 |
| 07 | PROPERTY | 설정은 가중치 묶음을 선언했는데 체크포인트에는 별도 출력층이 있음 | [vLLM #51063](https://github.com/vllm-project/vllm/issues/51063) | 기전 모사, transformers 관찰 | 그럴듯한 오답 |
| 08 | PROPERTY | 기본 SDPA 경로가 어텐션 softcap을 버림 | 로컬 발견, `issue_track/gemma2_softcap` | 실엔진(transformers) | 그럴듯한 오답 |
| 09 | TIME | CUDA Graph 재사용 검사가 포인터만 비교해, 길이가 바뀐 뒤 낡은 그래프를 재생 | [llama.cpp #21726](https://github.com/ggml-org/llama.cpp/issues/21726) | 기전 모사 | 깨진 출력 |
| 10 | TIME | flex 마스크가 위치 카운터를 참조로 들고 있다가, 제자리 증가 뒤에 읽음 | 4주차 발견(`phase0/week4/WEEK4_NOTES.md` 7.2절) | 실엔진(transformers) | 그럴듯한 오답 |
| 11 | SPECIALIZATION | 워밍업 때 빈 입력으로 특화된 컴파일 결과를 가드 없이 재사용 | [vLLM #43602](https://github.com/vllm-project/vllm/issues/43602) | 기전 모사 | 그럴듯한 오답 |
| 12 | RANGE | 세션 복원이 토큰 수(KV 수보다 1 많음)로 위치를 정해 한 칸 밀림 | [llama.cpp #23400](https://github.com/ggml-org/llama.cpp/issues/23400) | 기전 모사 | 그럴듯한 오답 |
| 14 | TIME | 빔 서치가 표준 이름의 캐시만 재정렬하고 순환 상태는 옛 순서로 둠 | [transformers #46612](https://github.com/huggingface/transformers/issues/46612) | 기전 모사 | 그럴듯한 오답 |
| 15 | MAPPING | 설정 키가 알 수 없는 이름으로 적혀 조용히 무시됨 | [SGLang #30176](https://github.com/sgl-project/sglang/issues/30176)(기전), 소비자는 transformers | 실엔진(transformers) | 그럴듯한 오답 |
| 16 | DTYPE | FP8로 양자화된 활성값을 BF16으로 알고 스케일 없이 씀 | [vLLM #42007](https://github.com/vllm-project/vllm/issues/42007) | 기전 모사 | 깨진 출력 |
| 03 | LAYOUT | 융합 QKV에서 자른 strided Q를 packed로 가정한 커널이 읽음 | [SGLang #31641](https://github.com/sgl-project/sglang/issues/31641) | 기전 모사(Triton) | 깨진 출력 (GPU 실행 대기) |
| 17 | PROPERTY | SGLang torch_native 어텐션 백엔드가 선언된 logit cap을 적용하지 않음 | 로컬 발견(코드 읽기, 2026-09-23) | 실엔진(SGLang 0.5.20) | 그럴듯한 오답 (GPU 실행 대기) |

## 읽을 때 주의

- 기전 모사는 원래 엔진이 아니라 같은 기전을 최소 코드로 다시 만든 것이다. 실엔진 재현이라고 부르지 않는다.
- 결과 표의 "최상위 토큰 일치"는 출력이 로짓일 때만 뜻이 있다(07, 08, 10). 다른 사례에서는 마지막 축의 argmax를 비교한 값이라 해석하지 않는다.
- 01의 결함 판은 스케일 바이트 자리에서 int8 데이터를 fp16으로 읽어 NaN이 나온다. 차이 지표가 NaN이면 "틀림"으로 판정된다.
- 02의 수정 판은 FP8을 다시 양자화하므로 참조와 조금 다르다. `fp8` 허용 오차로 판정한다(PROTOCOL 개정 1).
- 08은 작은 무작위 모델에서 q·k를 키워 softcap이 걸리게 만든 것이다. 실제 Gemma 2 2B·9B에서의 영향은 `issue_track/gemma2_softcap/RESULTS.md`에 따로 잰다. 2B에서는 커널 잡음 수준이었다.

- 15는 transformers가 모르는 키(`rope_scale`)를 경고 없이 속성으로 저장하고, RoPE는 기본값으로 두는 것을 관찰했다(probe 기록).
- 17은 실제 2B 설정(softcap 50)에서는 softcap이 거의 걸리지 않아서, 걸리도록 softcap 5.0을 선언한 설정 사본으로 시험한다. 확인하려는 것은 엔진이 선언된 성질을 지키는가다. 모든 경로가 같은 설정을 쓴다.

## 기존 탐지 수단 기록 (로드맵 1.6, `baselines.json`)

- 판정은 가설을 모르는 평가자 에이전트가 했다(2026-09-23). 모두 측정이 아니라 판단이다.
- 로컬 발견(08, 10, 17)의 설명은 연구 쪽이 썼다.

| 질문 | 예 | 아니오 | 불확실 |
|---|---|---|---|
| (b) 표준 평가(vLLM 야간 평가: H200·B200·MI300X·MI355X, GSM8K·GPQA·AIME·BFCL, 기본 설정)가 발동 조건을 건드리는가 | 0 | 14 | 2 |
| (c) 건드린다면 점수가 잡음 이상 움직이는가 | 10 | 1 | 5 |
| (d) 그 경로에서 실무상 참조 구현과 비교할 수 있는가 | 13 | 0 | 3 |

- **표준 평가가 놓치는 이유:** 발동 조건이 특정 장비이거나, 기본값이 아닌 기능이거나, 특정 모델이다.
  - 장비: Intel Arc, SM120, SM100
  - 기능: DP 어텐션, LoRA, 이미지 입력, 빔 서치, 세션 파일, `-nkvo`, flex
  - 모델: 재수출된 체크포인트, 멀티모달
- **참조 비교는 대부분 가능하다.** 다만 그 장비와 기능 조합에서 실제로 돌려야 잡힌다.
- **"둘 다 놓침":** PROTOCOL 6절의 정의(참조가 없음)로는 0건이다. 참조가 있는지 불확실한 사례는 3건(03, 15, 16)이다.
- **자료로 읽히는 것 (판단 아님):**
  - 이 사례들에서 탐지가 실패하는 원인은 "참조가 없음"이 아니다. "발동 조건 조합이 평가와 비교에 들어가지 않음"이다.
  - 서버 시작 때의 선언 검사는 사용자가 실제로 켠 조건에서 돈다. 그래서 조건 조합을 미리 늘어놓을 필요가 없다. 이 차이가 실제로 얼마나 되는지는 2단계에서 잰다.

## 검출 판(instrumented.py)을 읽을 때 주의

- 사례마다 `instrumented.py`가 있으면, 결함 판과 수정 판을 entail 선언을 붙여 다시 돌린다.
- 결함 판에서만 오류가 나면 검출로 본다(`PROTOCOL.md` 4절). 결과는 `results/detection.md`에 있다.
- **과대평가 위험:** 검출 판은 결함을 알고 난 뒤에 쓴 것이다. 버그를 본 뒤에 그 버그를 잡는 시험을 쓰는 것과 같은 구조라서, 이 표의 검출률은 부풀려져 있을 수 있다.
- 이 위험은 세 가지로 보완한다.
  - 선언을 결함별로 쓰지 않고 경계별로 쓴다. 판독기는 받는 형식을, 커널은 능력을, 소비자는 받는 합산 상태를 선언한다.
  - 정상 실행에서의 오탐을 잰다.
  - 같은 선언으로 실제 엔진에서 모르는 결함을 찾아본다(2단계). 이것이 검출력의 진짜 시험이다.

## 남은 일 (로드맵 1단계)

- #3, #17은 GPU가 빈 뒤 실행한다(`run_gpu_queue.sh`).
- 1.6 사례마다 기존 탐지 수단(표준 평가, 참조 비교)이 잡는지 기록한다. 가설을 모르는 평가자에게 판단을 맡겼다(진행 중).

## 기록: 결과 파일을 덮어쓴 사고 (2026-09-23 15:33~15:35)

- **무엇이 일어났나.** entail 범용 장치 추출(`entail/DESIGN.md` 7절)의 회귀 시험을 돌리다 사고가 났다.
  - 원래는 복사본(`~/regress_A`)에서 돌리려던 실행이었다. 스크립트의 경로가 잘려 `cd`가 실패했다. `sed 's/
$//'`가 wsl.exe를 거치며 줄 끝의 `r`을 지웠기 때문이다.
  - 그 결과 이 폴더에서 돌았다. 사례 01~10의 `results/*.json`이 새 entail로 같은 절차를 다시 돈 결과로 바뀌었다.
  - 사례 11~17의 JSON, `results.md`, `g1_check.json`은 바뀌지 않았다.
- **대조.**
  - `detection.md`는 새 JSON으로 다시 만들어졌지만, 원래 표와 22줄 모두 같다.
  - 바뀐 10건 모두 재현 '예', 검출 '예', 수정 판 무오류로 원래와 같고, 결함 판 오류 문구도 같다.
- **잃은 것.** 사례 01~10의 원래 실행의 원시 수치다(시간 등). 요약은 `results.md`에 그대로 남아 있다.
- **이후.** 회귀 시험은 복사본에서만 돌리고, `cd`가 실패하면 멈추게 했다.
