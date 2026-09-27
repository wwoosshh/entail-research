# 시험 문제 모음 (ROADMAP M0.4)

- 라이브러리가 풀어야 할 문제들이다(`LIBRARY_DESIGN.md` 9절).
  - 결함을 새로 찾는 데 쓰지 않는다.
  - 설계가 원인 부류를 막는지, 그리고 의미가 깨진 곳을 짚는지 확인하는 데만 쓴다.
- **기대 판정은 설계가 요구하는 것이고, 아직 잰 값이 아니다.**
  - 해소 수단이 등록되어 있으면 `resolved`(원칙 7, 해소가 먼저)이고, 없으면 `refused`다.
  - M5.4부터 기본 정책에서는 해소 수단이 없으면 `broken`이다: 오류로 보고하고 멈추지 않는다. `refused`는 엄격 정책(`ENTAIL_ON_BROKEN=stop`)의 판정이다. M5.3까지의 결과 칸은 멈추는 정책에서 잰 것이다.
  - 측정 결과는 담당 단계에서 결과 파일로 남기고, 이 표의 "결과" 칸에 적는다.
- **열 뜻**
  - 사실: `LIBRARY_DESIGN.md` 6절의 어휘 이름
  - 자리: 적재, 경계(코드 서명), 그릇, 요청, 재사용
  - 단계: 이 문제를 통과시켜야 하는 `ROADMAP.md` 단계
- 작성: 2026-09-23. 결과 칸은 M3.5, M4.2, M4.3(2026-09-23), M5.1, M5.2, M5.3, M5.4, M5.5, M6.3, M7.3(2026-09-24)에서 채웠다.
- M9.1(2026-09-24): 1~4절의 문제를 최종 코드로 모두 다시 쟀다(ComfyUI 두 건 fd-m7, fd-lora는 연구자 설치가 필요해 M6.3 결과를 그대로 둠).
  - 문제별 결과: `testbed/results/m91/problems.json`, 요약 `testbed/results/m91/SUMMARY.md` S2·S8 절
  - 각 단계의 결과와 판정 칸끼리 비교: 같음 75, 다름 19개 파일(`testbed/results/m91/s2.json`). 다른 칸은 모두 M5.4 정책 변경에서 왔다(`refused`→`broken`, 400→200). 엄격 정책으로 다시 잰 적재·그릇 문제는 `refused`였다(`results/m91/rerun_strict`).
  - 사후 탐지와 나란히 잰 결과(S7, M9.2)는 `testbed/results/m92/S7.json`에 있다.

## 1. 재현 벤치마크 (`rolebench/cases/`)

각 폴더의 `case.py`에 결함 판, 수정 판, 기준이 있다. `instrumented.py`는 옛 선언 방식으로 붙인 판이다.

| ID | 사실 | 결함 | 자리 | 기대 판정 | 해소 수단(있을 때) | 단계 | 결과 |
|---|---|---|---|---|---|---|---|
| rb-01 | Layout | 재배치한 양자화 가중치를 원래 형식으로 읽음 | 경계 | `refused` | 역재배치 변환이 등록되면 `resolved` | M4 | M4.3: 결함 판 `refused`, 수정 판 `pass`이고 출력이 맞음(생산자 서명만, 손 태그 0). `testbed/results/m43/SUMMARY.md` |
| rb-02 | Layout(스케일 형식) | 스케일을 2의 거듭제곱으로 올리고 데이터는 재양자화하지 않음 | 적재, 경계 | `refused` | — | M3, M4 | M3.5 적재: 결함 판 `refused`, 수정 판 `pass`(기전 모사, 사례의 커널을 시험용 표로 선언). `testbed/results/m3/SUMMARY.md`. M4.3 경계: 결함 판 `refused`, 수정 판 `pass`이고 출력이 fp8 허용 오차 안. `testbed/results/m43/SUMMARY.md` |
| rb-03 | Layout(stride) | strided Q를 packed로 가정한 커널이 읽음 | 경계 | `resolved` | 연속 배치로 변환 | M4 | M4.3: 결함 판 `resolved`(연속으로), 기준과 최대 차이 2.9×10⁻⁶. 수정 판 `pass`. `testbed/results/m43/SUMMARY.md` |
| rb-04 | Reduction | 이미 합산된 복제값을 다시 합산 | 경계 | `refused` | — | M4 | M4.3: 결함 판 `refused`, 수정 판 `pass`이고 출력이 맞음(한 프로세스가 두 랭크를 대신함). `testbed/results/m43/SUMMARY.md` |
| rb-05 | Positions | 청크 상대 위치와 절대 위치를 비교 | 경계 | `resolved` 또는 `refused` | 기준점 변환 | M4 | M4.3: 결함 판 `resolved`(절대 위치로), 기준과 최대 차이 4.8×10⁻⁷. 수정 판 `pass`. `testbed/results/m43/SUMMARY.md` |
| rb-06 | ModelProps(sliding_window) | 사용자 마스크가 커널의 창을 끄는데 마스크에 창 조건이 없음 | 적재 | `resolved` | 창을 지키는 경로로 보냄 | M3 | M3.5: `resolved`(창을 마스크에 넣는 경로로), 그 출력이 기준값과 맞음(기전 모사). `testbed/results/m3/SUMMARY.md` |
| rb-07 | ModelProps(tie) | 설정은 묶음을 선언했는데 체크포인트에 별도 출력층이 있음 | 적재(선언 대 데이터) | `refused`(선언이 틀림) | 정책이 허락하면 검증된 값으로 `resolved` | M3 | M3.5: 결함 판 `refused`(정적, 실행 모두), 수정 판 `pass`(실엔진 transformers). `testbed/results/m3/SUMMARY.md` |
| rb-08 | ModelProps(softcap) | transformers 기본 SDPA가 softcap을 버림 | 적재 | `resolved` | eager 경로로 보냄 | M3 | M3.5: `resolved`(sdpa→eager), 기준값과의 최대 차이 1.77→2.8×10⁻⁶(실엔진 transformers). `testbed/results/m3/SUMMARY.md` |
| rb-09 | Epoch, Assumed | 모양이 바뀐 뒤 낡은 CUDA Graph를 재생 | 재사용(그릇) | `resolved` 또는 `refused` | 다시 캡처 | M5 | M5.5(사례 코드 그대로, 사례의 그래프 캐시에 건 하네스): 기본 정책에서 결함 판 `resolved`(다시 캡처), 출력이 기준과 맞음(최대 차이 6×10⁻⁸). 고치지 않는 정책(`ENTAIL_POLICY=refuse`)에서 `broken`, 출력은 끔처럼 틀림(최대 0.42). 수정 판은 판정 없이 맞음. `testbed/results/m55/SUMMARY.md` |
| rb-10 | Epoch | 위치 카운터를 참조로 들고 있다가, 제자리 증가 뒤에 읽음 | 경계 | `refused` | 호출 시점 값으로 묶기가 등록되면 `resolved` | M4, M5 | M4.3(기전 모사): 결함 판 `refused`(마스크와 오프셋의 Epoch가 다름), 수정 판 `pass`. `testbed/results/m43/SUMMARY.md`. M5.2(설치된 transformers 경로): 기본 정책에서 `resolved`(넘겨주는 순간의 값으로 묶음, 기준과 최대 6×10⁻⁷), 거부 정책에서 `refused`, 수정 판 `pass`. `testbed/results/m52/rb10.json` |
| rb-11 | Assumed | 워밍업의 빈 입력으로 특화된 결과를 재사용 | 재사용 | `resolved` 또는 `refused` | 다시 컴파일 | M5 | M5.5(사례의 컴파일 호출에 건 하네스): 기본 정책에서 결함 판 `resolved`(다시 컴파일), 출력이 기준과 맞음(최대 차이 5.7×10⁻⁶). 고치지 않는 정책에서 `broken`, 출력은 틀림(최대 6.39). 가드를 지키는 수정 판은 torch가 스스로 다시 확인해서 판정이 없음. `testbed/results/m55/SUMMARY.md` |
| rb-12 | KvExtent | 세션 복원이 한 칸 밀림 | 그릇 | `refused` | — | M5 | M5.5(사례의 복호 단계에 건 하네스): 기본 정책에서 결함 판 `broken`(보고하고 진행, 출력은 끔처럼 틀림, 최대 1.23), 엄격 정책에서 `refused`. 수정 판 `pass`, 출력 일치. `testbed/results/m55/SUMMARY.md` |
| rb-14 | Epoch | 빔 서치가 일부 캐시만 재정렬 | 그릇 | `refused` | — | M5 | M5.5(사례의 빔 루프에서 상태를 그릇에 담은 하네스, 끔에서 사례 자체 출력과 같음): 기본 정책에서 결함 판 `broken`(어긋난 읽기 7회를 세고 한 번 기록, 출력은 끔처럼 틀림), 엄격 정책에서 `refused`. 수정 판 `pass`. `testbed/results/m55/SUMMARY.md` |
| rb-15 | Coverage(설정 키) | 알 수 없는 이름의 설정 키가 조용히 무시됨 | 적재 | `refused` | 별칭이 대응표에 있으면 `resolved` | M3 | M3.5: 결함 판 `refused`(정적, 실행 모두), 수정 판 `pass`이고 출력이 맞음(실엔진 transformers). `testbed/results/m3/SUMMARY.md` |
| rb-16 | Quantized | FP8 활성값을 BF16으로 알고 스케일 없이 씀 | 경계 | `resolved` 또는 `refused` | 역양자화 | M4 | M4.3: 결함 판 `resolved`(역양자화), 출력이 fp8 허용 오차 안. 수정 판 `pass`. `testbed/results/m43/SUMMARY.md` |
| rb-17 | ModelProps(softcap) | SGLang torch_native 백엔드가 logit cap을 적용하지 않음 | 적재 | `resolved` | triton 백엔드로 보냄 | M3 | M3.5: `resolved`(torch_native→triton), triton과 로그확률 차이 0. 결함 판은 평균 0.716, 최대 5.97 다름(실엔진 SGLang). `testbed/results/m3/rb17/compare.json` |

## 2. 실제 환경 측정 (`issue_track/`, `entail/audits/`)

| ID | 사실 | 결함 | 근거 | 자리 | 기대 판정 | 단계 | 결과 |
|---|---|---|---|---|---|---|---|
| fd-rope | Rotary | 실행 시점에 `rope_scaling`을 넘기면 `rope_theta`가 사라짐. vLLM에서 GSM8K 379→279/500 | `issue_track/rope_override/` | 적재 | `resolved`(옛 이름을 새 자리로 변환) | M3 | M3.5: `resolved`, GSM8K 380/500. 같은 날 대조군 380/500. `issue_track/rope_override/results/LC_entail_v2.json` |
| fd-softcap | ModelProps(softcap) | Gemma 2 실모델. transformers sdpa와 paged 경로, SGLang 백엔드 3종 | `issue_track/gemma2_softcap/`, `entail/audits/ADAPTER_PILOT.md` | 적재 | `resolved`(지키는 백엔드로) | M3 | M3.5: transformers sdpa→eager, SGLang flashinfer·torch_native·flex_attention→triton은 `resolved`. transformers paged는 지키는 paged 커널이 없어 `refused`. `testbed/results/m3/SUMMARY.md` |
| fd-m7 | Prediction | 파일 메타데이터가 v를 선언했는데 ComfyUI가 eps로 돌림 | `issue_track/comfyui_field_test/VPRED_PROTOCOL.md` M7 | 적재 | `resolved`(샘플러를 v로) | M6 | M6.3 ComfyUI 0.34.1(연구자 설치): 끔은 작성자 설정과 83~95/255 다름. 켬 `resolved`(v, zsnr은 ComfyUI 값), 그 기준과 같음·0.16·0.14/255. diffusers 0.40(단일 파일 적재도 선언을 읽지 않음): 켬 `resolved`, 기준과 화소 단위로 같음(3/3), 끔은 90~95/255 다름. `testbed/results/m63/SUMMARY.md` |
| fd-lora | Coverage(LoRA) | 다른 기반 모델용 LoRA가 모델에 닿지 않음 | `issue_track/comfyui_field_test/README.md` 4절 | 적재 | `refused` | M6 | M6.3 ComfyUI(waiIllustrious v170에 Anima LoRA 0.9): 끔은 LoRA 없는 그림과 화소 단위로 같음(아무것도 적용되지 않음). 켬은 `broken`(280개 중 0개, LoRA의 학습 기반을 알림), 그림은 끔과 같음. 엄격 정책은 LoRA 노드에서 `refused`. diffusers 0.40은 이 LoRA에 스스로 오류를 냄(`NoMatchingPeftModuleError`), 조용한 실패가 아님. `testbed/results/m63/SUMMARY.md` |
| fd-kv | KvExtent | KV 캐시가 토큰 하나를 잃음(심은 결함) | `entail/audits/CACHE_CONTRACT.md` | 그릇 | `refused` | M5 | M5.1: 세 엔진 모두 `refused`. transformers(Qwen3-4B, 모든 층에서 토큰 하나를 뗌), vLLM(블록 표가 한 블록 적음), SGLang(장부가 한 칸 적음). 정상 실행은 거부 0. `testbed/results/m51/SUMMARY.md`. M5.3: vLLM 어댑터가 접두 캐시 적중 토큰을 빠뜨려 되풀이된 프롬프트에서 거짓 거부한 것을 찾아 고침. 고친 뒤 서버 실측에서 할당 128회 통과. `testbed/results/m53/SUMMARY.md`. M5.4(기본 정책): vLLM의 심은 결함 실행이 멈추지 않고 `broken`으로 기록됨(요청마다 1건, 할당 96회 중 62회 셈). 출력이 entail을 끈 실행과 같아서, 이 심은 결함(`get_block_ids`의 반환값을 한 칸 줄임)은 계산 경로에 닿지 않는 장부의 어긋남임을 알았다. `testbed/results/m54/SUMMARY.md` |
| fd-identity-mech | Identity(제안, 어휘 v5) | 스트리밍 세션 갱신이 토큰을 자르고 `block_hashes`는 덧붙이기만 해, 저장된 해시가 옛 토큰의 것으로 남음(vllm#49377·#49449, 열림, 수정 PR 셋 미병합) | `realworld/CODEBOOK_v2.md` 3절, `testbed/r4/identity_probe.py` | 그릇 | `resolved`(첫 어긋난 해시부터 버리고 다시 만듦), 해소 수단 없으면 `broken` | R4 → M14 | **M14(2026-09-25): `resolved`.** 기전 재현(GPU 불필요)과 실제 어댑터 재현(실제 Request): 불변식 검사가 낡은 해시를 0번에서 잡고, 어댑터가 `identity_recompute`로 되돌려 열쇠가 지금 토큰의 것이 되며 옛 프롬프트의 다른 요청과 공유 안 함. 정상 갱신은 판정 0. 비용: 마지막 블록 3.0 µs. `testbed/results/r4/identity_probe.json`, `adapter_check.json`, `SUMMARY.md` |
| fd-identity-e2e | Identity(제안) | 위 결함을 끝까지: SmolLM2-135M, prefix 캐시, 옛 토큰과 같은 프롬프트의 요청이 새 토큰의 KV를 받음 | `testbed/m10_e3/vl49449.py` | 그릇 | `resolved`(후보 첫 토큰 = 재계산 참조) 또는 `container:vllm.request.block_hashes`에서 `broken` | R4 → M14 | **M14(2026-09-25): `resolved`, 출력이 고쳐짐.** 끔: 후보 `,`(28), 캐시 16, 참조 `
| fd-tokentype | TokenType(어휘 v6) | vLLM 점수 경로가 cross-encoder 패딩에 토크나이저의 pad 종류(0) 대신 마지막 토큰의 종류(1)를 줌; /rerank 점수가 바뀜(vllm#58138) | `testbed/m10_e3/vl58138.py` | 요청 | `broken`(소비자가 담을 수 없음: 능력표 행), 멈춤 정책에서 `refused` | M15.1 | **M15.1(2026-09-25): 검출.** 기본: `broken` 4건(능력표 note), 요청 진행, 점수는 틀린 그대로. 멈춤: 패딩 없는 요청 정상, 패딩 요청은 점수 전에 거부(HTTP 500, entail 문구). 첫 구현의 수리(패딩에 0)는 vLLM 압축(첫 1 위치)이 못 담아 400을 냈고, 그래서 능력표 행(honours false, measured)으로 옮김. `testbed/results/m15/SUMMARY.md` |
| fd-tile | KernelConfig(어휘 v6) | 블록 FP8 Triton matmul의 K 타일(64)이 양자화 블록(32)보다 커 스케일 포인터가 어긋남; 288 대신 64(sglang#39626) | `testbed/m10_e3/sg39626.py` | 적재(커널 설정) | `resolved`(타일을 블록으로 묶음) | M15.2 | **M15.2(2026-09-25): `resolved`.** on: 타일 32로 묶여 288(맞음), `resolved` 1건. off: 64(틀림). 배포 설정 1,887항목은 위반 0. `testbed/results/m15/SUMMARY.md` |
| fd-vocab | Vocab(어휘 v6) | 폴더가 두 어휘(vocab.txt 100,000 = 모델, tokenizer.json 32,000)를 들고 transformers 5가 tokenizer.json을 취해 틀린 id(transformers#48967) | `testbed/m10_e3/tf48967.py` (리비전 d8ce98a), `entail check` | 적재(토크나이저) | `broken`(모델의 것인 출처를 이름), 멈춤 정책 `refused` | M15.3 | **M15.3(2026-09-25): 검출.** 기본 `broken` 1건, 멈춤은 첫 id 전에 거부(exit 1). `entail check`: 옛 스냅숏 exit 1, 고친 리비전·Qwen3-4B pass. `testbed/results/m15/SUMMARY.md` |
| fd-tile-moe | KernelConfig | SGLang 0.5.20 배포 fused-MoE 설정(H100, E=512,N=256, fp8 블록 [128,128])의 BLOCK_SIZE_K=256 > 128; 커널이 타일마다 스케일 하나를 고름 → 뒤 절반이 앞 블록 스케일. **미보고** | `testbed/m15/moe_tile.py` | 적재(MoE 커널 설정) | `resolved`(타일을 블록으로) | M15.7 | **M15.7(2026-09-25): `resolved`.** 커널 수준: off 256(틀림) / on 512(맞음). `testbed/results/m15/SUMMARY.md` |
`(198), reproduced. 켬: 후보 `
`(198), 캐시 0, 참조와 같음, `container:vllm.request.block_hashes`에서 `resolved`(index 0). entail 1.0.0~1.0.2는 통과시켰다(E3 놓침). `testbed/results/r4/SUMMARY.md`, `e2e_off.json`·`e2e_on.json` |
| fd-shift | Layout(적재 원본) | 적재 중 가중치 행을 밂(심은 결함) | `entail/audits/D_LEDGER.md` 결과 5 | 적재 | `refused` | M3 | M3.5: `refused`(vLLM, 가중치 145개 대조, 심은 1개를 짚음). 사실은 `Coverage`로 표현했다(적재기가 받은 체크포인트 값 대 제자리에 놓인 값). `testbed/results/m3/SUMMARY.md` |
| fd-repack | Layout, Coverage | 재포장 뒤 가중치를 망가뜨림(심은 결함: 행 한 칸 밀기, 전치, strided 뷰). 모양과 dtype이 그대로면 조용하다 | `entail/audits/D_LEDGER.md` 결과 4, 4-1 | 적재기 경계 서명 | `refused`. `use_data` 정책에서 strided는 `resolved`(연속으로) | M4.2 | M4.2: 셋 모두 서버가 답하기 전에 `refused`(vLLM, Qwen3-4B, 가중치 144개씩). `use_data`에서 strided 144개는 `resolved`, 출력 정상. 검사를 끈 행 밀기는 깨진 출력. `testbed/results/m42/SUMMARY.md` |
| fd-vae | LatentScale | diffusers가 VAE만 든 단일 파일을 SD1.5로 읽어, SDXL VAE가 SD1.5의 VAE 설정(배율)을 받음 | diffusers 0.40 `loaders/single_file_utils.py`(`infer_diffusers_model_type`), 오프라인 실행에서 SD1.5 저장소 설정을 찾음(M6.1) | 적재 | 모델 쪽 배율을 선언하면 `resolved`(선언한 배율로) | M6 | M6.3 diffusers 0.40: 단독 적재한 SDXL VAE가 SD1.5의 배율 0.18215를 받음(모델은 0.13025). 끔은 기준과 15~16/255 다름. 켬은 선언 파일(모델의 배율)로 `resolved`, 기준과 화소 단위로 같음(3/3). 관찰 정책은 `broken`. `testbed/results/m63/SUMMARY.md` |
| fd-leak | (엔진 전용) | ComfyUI 동적 VRAM의 설정 누수(#16490) | `issue_track/comfyui_field_test/VPRED_PROTOCOL.md` M6 | ComfyUI 어댑터 | 엔진 결함의 우회라 핵심 판정에 넣지 않음 | M6 | |

## 3. 시장 사례 (`reinvestigation/market_incidents.md`, 모사해서 쓴다)

| ID | 사실 | 사례 | 자리 | 기대 판정 | 단계 | 결과 |
|---|---|---|---|---|---|---|
| mk-L03 | ModelProps(softcap) | Gemma 2 출시 때 여러 엔진이 softcap을 빠뜨림 | 적재 | `resolved` | M3 | M3.5: fd-softcap과 같은 기전이라 그 결과로 봄 |
| mk-L05 | Valid, Origin | Ollama 기본 문맥 길이가 긴 입력을 조용히 자름 | 요청 | 기본값을 쓴다고 기록하고, 뜻을 바꾸는 사실이면 `unknown`→`require` | M5 | M5.5(기전을 줄인 모사, 새 규칙 `request_contract.window`): 기본 문맥 2048, 프롬프트 10983토큰, 모델 선언 32768이면 `resolved`(문맥을 넓힘), 답이 맞음. 끔은 2048토큰만 보고 틀림. 사용자가 정한 문맥이나 여유 없는 모델은 `broken`(엄격 정책 `refused`). 들어가는 길이는 판정 없음. 기대 판정(`unknown`)과 다르다. 요청 경계에서는 세 사실을 모두 알아서 판정으로 끝난다(`LIBRARY_DESIGN.md` 11절 M5.5 (1)). `testbed/results/m55/SUMMARY.md` |
| mk-L07 | Coverage(요청 인자) | 옛 vLLM이 `reasoning_effort`를 무시함 | 요청 | `refused` | M5 | M5.3(기전, vLLM 0.30 + Qwen3-4B): 요청이 준 설정을 아무도 읽지 않는 경우 셋이 끔에서 200으로 조용히 지나갔고 켬에서 `refused`(400, 생성 전). vLLM 0.30이 모르는 필드 `guided_json`, 오타 설정 `enable_thinkng`, Qwen3 템플릿이 읽지 않는 `reasoning_effort` 수준(low). 모사는 M5.5. `testbed/results/m53/SUMMARY.md`. M5.5(모사, vLLM 0.30 + Qwen3-4B, `reasoning_effort`를 읽게 고친 템플릿, 서버가 그 설정을 빠뜨리게 심음): 끔은 200, 프롬프트 27→17토큰으로 조용함. 켬은 200과 `broken` 기록, 엄격 정책은 400. 결함이 없으면 켬도 판정 없음. 이 모사에서 요청 설정 읽기의 빈틈을 찾아 고쳤다. `testbed/results/m55/SUMMARY.md` |
| mk-L11 | Template(도구 호출 형식) | 업체마다 도구 호출 형식 처리가 다름 | 요청 | `refused` 또는 `resolved` | M5 | M5.3(vLLM 0.30 + Qwen3-4B, 선언 hermes, 서버 파서 pythonic): 끔은 도구 호출이 본문 텍스트로 돌아옴. 켬은 시작 때 `resolved`(hermes로 바꿈, 구조화된 호출). 거부 정책은 시작 `refused`. `testbed/results/m53/SUMMARY.md` |
| mk-L13 | Template(사고 기록) | OpenAI 호환 통합이 이전 사고를 버림 | 요청 | `refused` | M5 | M5.3(선언을 심은 기전, vLLM 0.30 + Qwen3-4B에 keep 선언): 사고를 넘긴 요청은 `pass`, 뺀 요청은 `refused`(400). Qwen3의 실제 템플릿은 이전 사고를 버려 출력 차이는 없다. 모사는 M5.5. `testbed/results/m53/SUMMARY.md`. M5.5(모사, 이전 사고를 남기게 고친 템플릿과 keep 선언): 사고를 뺀 요청은 끔에서 200, 프롬프트 67→48토큰으로 조용함. 켬은 200과 `broken` 기록, 엄격 정책은 400. 사고를 넘긴 요청은 `pass`. 이 작은 예에서 답은 같았다. `testbed/results/m55/SUMMARY.md` |
| mk-I01 | Coverage(LoRA 키) | 다른 형식의 LoRA 키가 적용되지 않음 | 적재 | `refused` 또는 `resolved`(키 대응) | M6 | M6.3(모사: 연구자의 SDXL LoRA를 PEFT가 감싼 모델의 이름 base_model.model.*로 바꿈. ComfyUI 0.34.1은 이 접두어를 SD3·PixArt에만 대응): ComfyUI와 diffusers 모두 아무것도 적용하지 않음(그림이 LoRA 없는 것과 같음, 3/3씩). 켬 `broken`(1348개 중 0개), 엄격 정책 `refused`. 키 대응 해소는 등록하지 않았다. `testbed/results/m63/SUMMARY.md` |
| mk-I04 | Prediction | v 예측 모델이 과포화됨 | 적재 | `resolved` | M6 | M6.3(diffusers 0.40 단일 파일 적재가 표지 키를 읽지 않는 것으로 모사, NoobAI-XL-Vpred): 끔은 eps로 돌아 기준과 55~83/255 다름. 켬 `resolved`(v와 zsnr로), 표지대로 손으로 맞춘 기준과 화소 단위로 같음(3/3). 관찰 정책은 `broken`, 그림은 끔과 같음. `testbed/results/m63/SUMMARY.md` |

## 4. 위치 짚기 (S8, `LIBRARY_DESIGN.md` 12절)

| ID | 심는 곳 | 기대 보고 | 단계 | 결과 |
|---|---|---|---|---|
| loc-boundary | 1~3절 문제 가운데 경계 결함 하나 | 그 경계를 문제 영역으로 짚음 | M7 | M7.3: 7건 모두 그 경계를 짚었다. 프로세스 안(`entail.locate()`)과 기록 파일(`entail locate`)이 같은 답이었다. 적재: RoPE 덮어쓰기(Qwen3-4B, 관찰 정책·디버그 모드. transformers 5.17에서는 모델을 만들 때 오류로 드러남), 설정 키 오타(yarn 스케일링을 `rope_scalling`으로 적음, 조용한 오답). 그릇: KV 토큰 잃기. 코드 경계: rb-01, rb-04. 경계 사이 연산: transpose를 연산 이름까지 짚음. `testbed/results/m73/SUMMARY.md` |
| loc-inside | 경계의 의미는 온전하게 두고, 커널 계산만 틀리게 만듦(새로 만들 것) | "모든 경계 온전"이라고 보고하고, 의심 구간을 그 커널의 계층으로 좁힘 | M7 | M7.3(Qwen3-4B, 진단 모드, 계층 셋을 float32 참조와 층마다 비교): sdpa 어텐션의 softmax 스케일 ×1.15, MLP의 GELU를 심음. 둘 다 "모든 경계 온전(4개)"과 함께 그 계층으로 좁혔고, 다른 두 계층은 풀렸다(차이 0.135와 0.813, 정상 최대 0.0078). 정상 실행은 의심 구간이 없었다. 함수의 첫 호출만 비교하는 처음 방식은 ×1.10을 놓쳐서 층마다 비교로 고쳤다. `testbed/results/m73/SUMMARY.md`, `tolerance.json` |
| loc-unchecked | 검사하지 못한 경계 하나를 포함 | 그 경계와 양옆 계층을 의심 구간으로 남김 | M7 | M7.3(gemma-2-2b-it, 능력표에 없는 어텐션 구현 "planted", 스케일 ×2): 어텐션 경계가 "검사 못 함"으로 남았고, 그 경계와 양옆(config.json의 softcap·창 선언, `transformers.attention.planted`)이 첫 의심 구간이 되었다. 스케일 ×1.15에서는 출력이 같았다(심은 함수 832회 호출). `testbed/results/m73/SUMMARY.md` |

## 5. 정상 기준 (S3 오탐)

| 모델 | 엔진 | 기대 |
|---|---|---|
| Qwen3-4B, Llama-3.2-3B, Gemma 2 2B | transformers, vLLM, SGLang | `refused` 0, 잘못된 `resolved` 0. M3.5 적재: 실엔진 9회 `refused` 0, `resolved` 2(Gemma 2: sdpa→eager, flashinfer→triton, 둘 다 능력표의 측정 칸대로). 정적 검사 27조합(로컬 LLM 9개) `refused` 0. M4.2 적재기 서명: vLLM Qwen3-4B 양자화 없음·온라인 fp8 2회 `refused` 0(가중치 144개씩 통과) |
| SDXL 계열(연구자 체크포인트), Anima | ComfyUI, diffusers | `refused` 0, 켜고 끈 그림이 같음. M6.3 diffusers 0.40(waiIllustrious v160, 선언 없음): 켜고 끈 그림이 화소 단위로 같음(3/3). 판정은 모름 둘(예측 방식, 잠재 배율)뿐이고 깨짐·거부·해소 0. ComfyUI 쪽은 허락을 기다림. `testbed/results/m63/SUMMARY.md` |
