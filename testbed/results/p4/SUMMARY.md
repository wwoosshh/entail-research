# P4: 공식 DLC (제품 트랙, 2026-09-28)

- **자:** `LIBRARY_DESIGN.md` 8절 S10과 13.7절, 로드맵 P4의 완료 기준(S10, 핵심의 시험이 그대로 통과)이다.
- **entail:** `product` 가지 `9c77291`(로컬)이다.
- **DLC:** `dlc/comfyui`(배포 이름 `entail-dlc-comfyui` 0.1.0)이다.

## 만든 것

- `entail/dlc.py`: 진입점 무리 `entail.dlc`에서 DLC를 찾는다. DLC의 형식은 속성 일곱이다: `name`, `version`, `requires`, `engines`, `targets`, `facts`, `nodes`.
  - 판 범위가 맞지 않거나, 어휘에 없는 사실을 대거나, 이름이 틀리면 붙이지 않고 이유를 한 번 말한다.
  - 설치는 핵심이 한다. 실패하면 `dlc:<이름>.install`에 적고(DLC의 첫 사실로 unknown, 사실이 없으면 말한 줄) 실행은 이어진다. 두 번 실패한 항목은 그 프로세스에서 뗀다. 디버그 모드는 예외를 낸다.
- **시작 훅:** entail이 켜졌을 때만 DLC를 찾는다. `ENTAIL_DLC=off`로 떼고, `ENTAIL_ONLY`와 `ENTAIL_SKIP`이 DLC 항목에도 걸린다.
- **노드 모델:** DLC 노드는 핵심의 나머지 받이 앞에 들어간다. `entail doctor`가 DLC와 그 항목을 보인다.
- **첫 DLC:** ComfyUI #16490 수리를 핵심에서 옮겼다(git이 이름 바꿈으로 인식: 93%). 핵심의 `ENGINE_SPECIFIC`는 비었고, 시험이 핵심에 엔진 전용 수리가 없음을 지킨다.
- **시작 훅이 한 번만 켜지게 고쳤다.** 같은 파일이 두 번 들어와도(.pth의 `entail.adapters.autoinstall.sitecustomize`, PYTHONPATH의 `sitecustomize`) 가져오기 감시자는 하나다.
  - 아래 ComfyUI 측정에서 드러났다. 그 환경에는 entail 0.3.0의 .pth가 남아 있어, 모든 줄이 두 번 찍혔다.
  - 수리 함수들이 이미 겹쳐 걸기를 막고 있어서 그림에는 영향이 없었다(아래 12/12).

## S10

1. **붙이고 떼도 핵심의 시험이 그대로다.** gpu 환경에서 DLC를 PYTHONPATH의 폴더로 붙인 채로도(설치하지 않음), 뗀 채로도 61개 파일 572 검사가 통과했다. DLC 자신의 시험 4개도 통과했다.
2. **ComfyUI 측정이 그대로다.** ComfyUI 0.34.1(`E:\ComfyUI\ComfyUI-new`로 옮겨져 있었음)에서 M6의 두 순서를 다시 돌렸다. NoobAI-XL-Vpred를 그대로 한 번, ModelSamplingDiscrete(v_prediction, zsnr=false) 노드를 걸고 한 번, 시드 셋이다. 그림은 M6의 그림(`_g2`, 그때 핵심 안의 수리; 새 세션과 화소 단위로 같았던 것)과 대조했다(`testbed/p4_comfy_compare.py`, `comfy_m6.json`).

   | 조건 | M6 그림과 화소 단위로 같음 | 수리가 한 말 |
   |---|---|---|
   | DLC 붙임, 그대로→노드 | 6/6 | 한 번: 다른 객체의 일정(4518.8)을 제 객체(14.6)에 쓰려던 것을 돌려줌 |
   | DLC 붙임, 노드→그대로 | 6/6 | 한 번: 14.6 → 4518.8, 돌려줌 |
   | DLC 뗌(`ENTAIL_DLC=off`), 그대로→노드 | 3/6. 노드 경우 셋이 다름(평균 58.73, 12.37, 20.27/255) | 없음 |
   | DLC 뗌, 노드→그대로 | 3/6. 그대로 경우 셋이 다름(64.39, 12.37, 20.27/255) | 없음 |

   - DLC를 떼면 #16490 누수가 돌아온다. 58.73과 64.39는 M6에서 entail을 껐을 때의 값과 같다.
   - 수리가 이제 DLC에서 온다.
3. **정상 실행에서 DLC의 broken·refused는 0이다.** 네 세션의 기록(ComfyUI 폴더의 `entail_logs/record-2026-09-28.jsonl`, 82줄)에서 DLC가 남긴 것은 수리의 말한 줄(`dlc:comfyui.schedule`) 둘뿐이다.
   - 세션마다 있는 `load:comfyui.prediction` broken 1건은 핵심의 ComfyUI 어댑터가 낸 것이다. 측정이 일부러 건 노드(zsnr=false)가 모델의 선언(ztsnr)과 다르다는 설계대로의 알림이고, DLC와 무관하다.
   - 다만 이 측정은 결함 재현용 워크플로다. DLC의 오탐을 넓게 재는 정상 실행(다른 모델, 다른 워크플로)은 하지 않았다.

## 쓴 파일(작업공간 밖)

- ComfyUI 폴더(`E:\ComfyUI\ComfyUI-new`) 안:
  - `output/entail_test/vpred_noob_*_on_dlc_*.png`, `*_on_nodlc_*.png`(24장; M6 때와 같은 시험용 하위 폴더)
  - `entail_logs/`(기록, 로그, 토크나이저·어휘 캐시; entail의 기본 기록 폴더)
- ComfyUI의 가상 환경은 바꾸지 않았다. entail과 DLC는 PYTHONPATH로만 붙였다.
- 임시 폴더: 세션 임시 폴더의 `dlc_site_win`(Windows용 DLC 배치)과 WSL의 `/tmp/dlc_site`, `/tmp/s10_*.txt`.

## 한계

- DLC의 SDK(검증기가 판정을 내는 모양)와 시간 예산은 P5에서 정한다. 지금의 DLC는 설치 함수로 엔진 함수를 감싸는 형태뿐이다.
- DLC 패키지는 PyPI에 올리지 않았다(공개는 허락 뒤).

## 외부 평가(에이전트 1, 73,192 토큰)와 반영

**판정은 조건부 적합이다.** P5 전에 보완할 것 넷을 받았고, 넷 모두 반영했다.

1. **두 번째 독립 패키지로 형식과 노드 id 충돌을 시험하라.** P5의 창작마당 예제를 설치할 수 있는 실제 패키지로 만들고, ComfyUI DLC와 함께 붙여 노드가 둘 다 들어오는지 시험한다(P5에서). 지금까지 둘 이상의 노드 충돌은 가짜 배포본으로만 시험했다.
2. **재현용이 아닌 정상 워크플로에서 오탐을 재라 → `testbed/p4_comfy_normal.py`, `comfy_normal.json`.** 두 워크플로를 entail 끔, 그리고 켬과 DLC 붙임으로 시드 셋씩 돌렸다.

   | 워크플로 | 끔과 같은 그림 | DLC가 한 말과 판정 | broken·refused | 반복된 줄 |
   |---|---|---|---|---|
   | Anima(연구자의 주력: Anima DiT + nagito LoRA 0.8, 832×1216, 30단계) | 3/3 | 0 | 0 | 0 |
   | waiIllustrious v1.6(eps SDXL) | 3/3 | 0 | 0 | 0 |

   - 핵심의 판정은 pass와 unknown뿐이었다(Anima pass 4·unknown 3, wai pass 1·unknown 3).
   - 반복된 줄 0은 시작 훅이 한 번만 켜지게 한 고침이 실제 환경(.pth와 PYTHONPATH가 함께 있음)에서도 선다는 것이다.
3. **환경 충돌을 체계적으로 훑어라.**

   | 경우 | 본 곳 | 상태 |
   |---|---|---|
   | 시작 훅이 두 번 들어옴(.pth와 PYTHONPATH의 `sitecustomize`) | ComfyUI(0.3.0의 .pth가 남아 있음) | 고침(`9c77291`). 실측에서 반복 줄 0 |
   | PYTHONPATH의 훅이 환경의 `sitecustomize`를 가림 | WSL 우분투(`/usr/lib/python3.12/sitecustomize.py`, apport) | 고침(`1da7ede`): 가린 것을 이어 실행한다. WSL에서 확인 |
   | 한 환경에 entail 판이 둘(설치된 것과 개발 경로) | ComfyUI(0.3.0 설치와 작업 사본) | 경로 순서대로 한 판만 들어온다. 옛 판의 핵심은 DLC를 모르므로, 옛 판이 이기면 DLC는 조용히 빠진다. `entail doctor`로 보인다 |
   | DLC의 핵심 판 범위 밖 | 시험 | 붙이지 않고 한 번 말한다 |
   | 두 배포본이 같은 DLC 이름 | 시험 | 앞의 것만 붙는다 |
   | 진입점 이름과 DLC 이름이 다름 | 시험 | 붙이지 않는다(`1da7ede`) |
   | 엔진의 자식 프로세스 | vLLM(P2 실측) | 환경 변수를 물려받아 한 실행으로 묶인다 |
   | 엔진과 `entail serve`의 기록 폴더가 다름 | 보지 않음 | 서버가 읽는 폴더를 찍는다. 맞추는 것은 사용자다 |
   | 경로에 ASCII가 아닌 글자(`프로젝트`) | Windows ComfyUI, WSL | 됨 |
4. **외부 코드의 신뢰 경계를 정하라(`1da7ede`, 설계 13.7절).**
   - `ENTAIL_DLC=이름,이름`이면 그것들만 붙는다. 목록에 없는 진입점은 import하지 않는다.
   - DLC의 진입점은 제 이름을 가져야 하므로, 목록이 말한 대로 된다.
   - entail은 DLC를 격리하지 않는다. DLC는 프로그램의 프로세스에서 프로그램의 권한으로 돈다.
   - 핵심이 막는 것은 실행을 깨는 것(실패는 기록하고 올리지 않음)과 핵심의 규칙을 바꾸는 것이다. 무엇이 도는지는 사용자가 설치하고 목록에 올린 것으로 정한다.
   - 창작마당(P5)도 같은 경계를 쓴다.
- **과장 지적도 반영했다.** 이 요약의 S10 3항("DLC의 broken·refused 0")은 결함 재현 워크플로에서 잰 값이다. 위 2의 정상 워크플로 둘이 표본을 넓혔지만, 여전히 ComfyUI 한 설치의 두 워크플로다. "공식 DLC"는 아직 PyPI에 없고 사례는 하나다.
