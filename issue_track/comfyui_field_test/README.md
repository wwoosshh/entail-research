# 현장 시험: 연구자의 실제 ComfyUI 환경에 entail 설치 (2026-09-23)

> **2026-09-23 재정립.** 이 현장 시험에서 ComfyUI 전용 검사를 덧붙인 것은 라이브러리 방향에서 벗어난 작업이었다. 측정한 사례(M7, LoRA, 설정 누수)는 시험 문제로 쓴다(`LIBRARY_DESIGN.md` 9절). 설정 누수 가드는 ComfyUI 자체 결함(#16490)을 우회하는 것이라 핵심에 두지 않는다.

## 판정

- **(M7, 반대 결과) 표지 없이 실제로 배포된 v-pred 모델에서 entail 0.3.0은 놓쳤다.**
  - 대상은 AstolfoCarmix-VPredXL AC-Evo 2.5EP이고, ComfyUI 이슈 #12579가 가리킨 모델이다. `v_pred` 키가 없어 ComfyUI는 EPS로 판정하고, 그림이 망가진 채 "성공"으로 끝난다.
  - entail의 첫 호출 판정값은 0.94(eps 쪽)였다. 그래서 켜고 끈 그림이 같았다.
  - 파일 메타데이터에는 `prediction_type = v`가 적혀 있지만, ComfyUI도 entail도 읽지 않는다.
  - 이 모델은 목표를 eps에서 v로 바꾸는 도중이다. 작성자가 ComfyUI용 전용 워크플로를 따로 낸다(`VPRED_PROTOCOL.md` M7).
- **(0.3.0 가드) ComfyUI 동적 VRAM의 설정 누수도 막는다.**
  - 샘플링 노드를 쓴 뒤 같은 체크포인트를 노드 없이 돌리면 다른 그림이나 검은 화면이 나왔다. 반대로, 노드 없이 먼저 돌린 뒤에 쓴 노드는 설정이 무시됐다. 이 결함은 entail과 무관하다.
  - entail을 켜면 두 방향 모두 새 세션과 화소 단위로 같아졌다. 다른 실행은 켜고 끈 그림이 같았고 속도도 같았다(`VPRED_PROTOCOL.md` M6).
- **(0.3.0 추가) 예측 방식(v-pred) 표지를 잃은 모델도 해소한다(5절).**
  - 표지 없는 사본을 entail 없이 돌리면 색 잡음이나 검은 화면이 나왔다(정상 대비 67~102/255).
  - entail을 켜면 거의 정상으로 돌아왔다(12~20/255).
  - 비간섭은 첫 그림까지 포함해 화소 단위였다.
- **(0.2.0 추가) ComfyUI에서도 예방이 된다.** 첫 ComfyUI 검사(LoRA와 모델의 불일치)를 넣었다(4절).
  - 잘못된 조합은 샘플링 전에 이유와 함께 멈췄다.
  - 맞는 조합 22개는 오탐 없이 통과했다.
  - 켜고 끈 그림은 같았다.
- **설치와 비간섭은 통과했다.** 실제로 쓰는 ComfyUI 환경에서 entail을 켜도 그림이 비트 단위로 같았고, 속도 차이는 잡음 수준이었다.
- **(0.1.0 시점) 검사 범위는 비어 있었다.** 이 환경의 이미지 생성 경로에서 entail이 검사하거나 해소한 것은 0건이다. ComfyUI는 모델을 transformers가 아니라 자체 코드로 돌리기 때문이다.
- 그래서 ComfyUI 쪽에서 지켜야 할 뜻을 따로 정의해야 했다(3절). 첫 검사는 4절에서 넣었다.

## 1. 환경

| 항목 | 값 |
|---|---|
| 위치 | `Desktop\ComfyUI\ComfyUI-new` |
| 버전 | ComfyUI v0.34.1, Windows 네이티브, `.venv` Python 3.12.10 |
| 주요 패키지 | torch 2.13.0+cu130, transformers 5.16.1, diffusers 0.40.0 |
| 커스텀 노드 | 19개. TIPO 프롬프트 LLM은 llama.cpp(GGUF)로 돈다 |
| 모델 | SDXL 계열(Illustrious, NoobAI), Anima, Qwen-Image-Edit(GGUF), Hunyuan3D |

## 2. 한 일과 결과

- **설치.** GitHub에서 `pip install`로 설치했다. 설치 전후 `pip freeze`를 비교했더니 늘어난 것은 `entail-ai` 하나뿐이었다(`comfy_freeze_before.txt`, `comfy_freeze_after.txt`).
- **`entail doctor`.** 시작 훅이 제자리에 있었다. transformers 5.16.1은 시험한 판(5.12.1, 5.17.0)이 아니라고 표시했다.
- **예제.** 이 환경의 transformers 5.16.1로 돌려도 결함이 재현됐다(`rope_theta DROPPED`). `ENTAIL=load`로 켜면 해소됐다(`rope_theta kept`).
- **실제 생성.** 평소 옵션(`--fp16-vae`)에 포트 8189로 실행했다.
  - 모델은 waiIllustriousSDXL v160, 1024², 20 steps, euler로 두고 시드 3개를 썼다.
  - 순서가 결과를 흐리지 않도록 켠 실행을 먼저 했다(`comfy_entail_test_v2.json`).

| | 서버 시작 | 1회차(적재 포함) | 2·3회차 | 그림 |
|---|---|---|---|---|
| entail 켬 | 15.1 s | 8.12 s | 5.07, 5.05 s | 끈 실행과 **3/3 비트 단위로 같음** |
| entail 끔 | 16.1 s | 8.39 s | 5.11, 5.09 s | |

- **켠 실행의 로그.** `[entail] installed` 줄은 2줄이었다(`rope_alias`, `transformers_adapter`). 해소와 오류는 0건이었다. 끈 실행의 로그에는 entail 줄이 없었다.
- **측정 설계의 실수.** 첫 판은 같은 프롬프트를 두 번 보냈는데, ComfyUI가 캐시로 답해 반복 측정이 되지 않았다(0.001 s). 순서도 끈 실행이 먼저라 파일 캐시 효과가 섞였다. 둘 다 고쳐서 다시 쟀다(v2).
- **직접 실행용 파일.** `Desktop\ComfyUI\entail-검증\`에 배치 파일 두 개와 안내 파일을 두었다. 둘 다 cmd로 실행해 확인했다.

## 3. ComfyUI에서 지켜야 할 뜻 (다음 후보)

- **예측 방식 선언(v-prediction).**
  - ComfyUI는 체크포인트 안에 `v_pred` 키가 있는지만으로 판단한다(`comfy/supported_models.py:227`). 병합 과정에서 이 표지가 빠지면, ComfyUI는 v-pred 모델을 경고 없이 eps로 돌린다.
  - 보유 파일 20개의 헤더를 읽었다. 표지가 있는 것은 NoobAI-XL-Vpred 하나였고, 그 모델의 Civitai 메타데이터와도 맞았다. **현재 어긋남은 없다.**
  - 검사를 만든다면 표지 키와 모델 메타데이터(`modelspec.prediction_type`, Civitai 정보)를 대조하는 형태가 된다.
- **텍스트 인코더의 RoPE 기준값.** `comfy/text_encoders/llama.py`는 모델별 `rope_theta`를 코드 안에 상수로 박는다. 체크포인트 파일에는 설정이 없으므로, 같은 구조의 다른 기준값 체크포인트가 오면 조용히 틀린다. 아직 쟀지 않은 가설이다.
- 이 후보들을 entail에 넣을지는 연구자가 정한다. 넣는다면 먼저 실제로 어긋나는 사례가 있는지부터 잰다.

## 4. 첫 ComfyUI 검사: 모델에 닿지 못하는 LoRA (0.2.0)

**사례를 먼저 쟀다(entail 끔, `lora_off.json`).** 연구자의 `나기토화풍-WAI`와 같은 구성으로 돌렸다. 모델은 waiIllustriousSDXL v170이고, LoRA는 `LoraLoaderModelOnly`에 강도 0.9로 붙였다.

| LoRA | 실행 | LoRA 없음 대비 그림 변화(평균, 0~255) |
|---|---|---|
| `nagito_illustrious_v3`(맞음) | 성공 | 35.2 |
| `nagito_anima_e10`(Anima용) | **성공** | **0.8** |

- 화면에는 아무 표시가 없었다. 콘솔에만 `lora key not loaded`가 840줄 나왔다(`comfy/lora.py:93`).
- 연구자는 같은 캐릭터 LoRA를 기반 모델마다 따로 두고 있다(illustrious, wai, anima). 그래서 엇갈려 끼우는 일이 실제로 일어날 수 있다.

**검사.** `entail/adapters/comfyui.py`는 `comfy.sd.load_lora_for_models`를 감싼다. 이 함수는 기본 LoRA 노드와 대부분의 커스텀 로더가 부른다.
- LoRA 모듈 가운데 모델(그리고 텍스트 인코더)에 닿는 수를 ComfyUI 자신의 대응표로 센다.
- 0이면 고칠 방법이 없으므로 이유를 붙여 멈춘다. 이유에는 LoRA가 선언한 학습 기반과 실제로 만난 모델이 들어간다.
- 일부만 닿으면 한 줄로 알리고 계속한다.
- 훅은 `comfy.sd`가 실행을 마친 직후에 설치된다. 그 시점에는 `comfy.sd`가 아직 패키지 속성으로 연결되지 않아서 첫 판은 설치에 실패했다. 모듈을 `sys.modules`에서 직접 가져오도록 고쳤다. 설치에 실패한 동안에도 ComfyUI는 정상으로 돌았다.

**결과.**

| 조합 | entail 끔 | entail 켬 |
|---|---|---|
| 맞는 SDXL LoRA | 적용됨 | 적용됨. 그림이 끈 실행과 **화소 단위로 같음** |
| LoRA 없음 | - | 끈 실행과 화소 단위로 같음 |
| Anima LoRA + SDXL 모델 | 성공 표시, 효과 없음 | **샘플링 전 멈춤**: "none of its 280 modules match a weight of the loaded SDXL model. It declares it was trained for anima-preview/lora / anima" |
| SDXL LoRA + Anima 모델 | - | **멈춤**: "none of its 788 modules match ... Anima model. It declares ... stable-diffusion-xl-v1-base/lora" |
| SDXL LoRA 21개 전부(모델과 텍스트 인코더) + SDXL | - | 21개 모두 "all modules reach", 오탐 0 |
| Anima LoRA + Anima 모델 | - | "all 280 model modules reach", 오탐 0 |

- 개발 경로(PYTHONPATH)로도, 설치된 0.2.0만으로도 결과가 같았다(`lora_on.json`, `lora_installed.json`, `lora_scan.json`).
- 단위 시험은 파일 11개, 검사 69개가 통과한다.
- **한계:** 이번 검사는 LoRA와 모델이 구조적으로 맞는지까지만 본다. 구조는 같지만 학습 기반이 다른 경우(예: Illustrious용 LoRA를 NoobAI 모델에)는 잡지 못한다. 그런 조합은 실제로 어느 정도 작동하기 때문에, 막을 기준 자체가 데이터로 정해져 있지 않다.

## 5. 두 번째 ComfyUI 검사: 예측 방식 표지 (0.3.0)

측정 정의와 전체 결과는 `VPRED_PROTOCOL.md`에 있다. 요약하면 다음과 같다.

| 경우 | entail 끔 | entail 켬 |
|---|---|---|
| v-pred 모델, 표지 있음 | 정상 | 정상, 끈 그림과 화소 단위로 같음 |
| **v-pred 모델, 표지 없음(사본)** | "성공"이지만 색 잡음이나 검은 화면(정상 대비 67~102/255) | **해소**: "sampling it as v_prediction", 정상 대비 12~20/255 |
| v-pred 모델에 `ModelSamplingDiscrete(eps)`를 명시 | 잡음이나 검은 화면 | **멈춤**, 이유 표시 |
| eps 모델 | 정상 | 정상, 첫 그림까지 화소 단위로 같음 |
| eps 모델에 `v_prediction` 노드가 남음 | "성공"이지만 회색 단색(정상 대비 61~64/255) | **멈춤**, 이유 표시(3/3) |

- 판별은 샘플링의 첫 모델 호출로 한다. 추가 계산은 없다.
- 탐침을 따로 돌리는 첫 설계는 첫 그림을 0.6/255 바꿨다. 그래서 관찰 방식으로 바꿨다.
- 사본은 `models/checkpoints/entail_test/`에 있다(약 7 GB). 연구자가 직접 확인해 본 뒤 지워도 된다.

**ComfyUI 자체 결함(entail과 무관).** `ModelSamplingDiscrete` 노드를 쓴 실행 뒤에 같은 체크포인트를 노드 없이 돌리면, 노드 설정이 남아 다른 그림이나 검은 화면이 "성공"으로 나왔다.
- entail을 꺼도 수치가 같았다. `--disable-dynamic-vram`으로 켜면 거의 사라졌다(같음, 또는 1.4/255 이하).
- 처음 만든 0.3.0은 이것을 잡지 못했다. entail이 멈춘 뒤 노드를 빼고 다시 돌리면 이 결함에 걸렸다.
- 원인은 로더가 버퍼를 경로 이름으로 백업해서 바꿔 끼운 객체에 되돌리는 데 있었다. 그래서 0.3.0에 가드를 넣었다. 백업을 원래 객체에 돌려주고, 첫 모델 호출에서 한 번 더 확인한다.
- 결과: 노드를 쓴 뒤의 실행이 새 세션과 화소 단위로 같아졌다. 반대 방향(노드 설정이 무시되던 것)도 같아졌다. Anima 등 다른 실행은 켜고 끈 그림이 같았다(`VPRED_PROTOCOL.md` M6).
- 자세한 수치는 `VPRED_PROTOCOL.md` 4절 끝에 있다.
