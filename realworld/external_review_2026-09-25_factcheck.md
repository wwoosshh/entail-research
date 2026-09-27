# 외부 평가(2026-09-25, 별도 환경 `C:/Entail`, 1.0.2 체크아웃) 사실 확인

연구자가 다른 도우미와 나눈 대화(코드 검토 → 가치 평가 → 방향 → 방화벽 목표 → 증상 중심 목표)의 사실 주장을 코드, 결과 파일, GitHub에서 대조했다. 확인 방법: 작업 사본(ba94290)에서 코드 읽기, WSL `~/venvs/ci`(Python 3.12.3, torch 2.14 CPU, transformers 5.17.0)에서 재현, `testbed/results/`의 요약 파일, `gh`(읽기 전용).

## 1. 코드 검토 문단

| 주장 | 판정 | 근거 |
|---|---|---|
| 1.0.2, Python 57개 파일·약 11,600줄, 테스트 38개 파일·336개 | 맞음 | 57개 파일, 11,561줄; 38개 파일, `run_all.sh` 기준 336 checks(pytest 수집 기준 344 함수) |
| 구성 표(facts/readers/sources/manifest, contracts/policies/load, adapters, record/diagnose/pytest_plugin, frontend) | 맞음 | 모듈이 모두 있다 |
| `sitecustomize.py:100`, `contracts.py:191`이 중심 | 맞음 | 100행 `_PatchAfterImport`, 191행 `decide` |
| 어댑터에 규칙이 섞이지 않는지 검사하는 구조 테스트 | 맞음 | `tests/test_adapter_rules.py` |
| **문제 1** 없는 모델 경로에 `entail check` → 종료 코드 0; 최소 설정 + `--attention does_not_exist` → 0 | **재현됨** | `/nonexistent/model` → "declares nothing", exit 0. 최소 config → exit 0. Qwen3-4B(실제 모델) + `does_not_exist` → exit 0. `_check`의 2는 "what the consumer uses" 규칙의 unknown이 있을 때만 나온다(`cli.py:145`). 경로·백엔드 이름은 검증되지 않는다 |
| **문제 2** 오류 판정이 전역 리스트에 누적. 동일 오류 요청 2,000회 → 판정 2,000개, 약 1.95 MiB | **재현됨** | `request_contract._settle` → `load.enforce(out)` → `LEDGER.extend` (`load.py:1006`). `once_for` 중복 제거는 M6.2 샘플링 경계에만 쓰인다. `window()` 2,000회(같은 요청) → 판정 2,000개, tracemalloc 2.09 MiB; 서로 다른 요청 2,000회 → +2,000개, 1.98 MiB. 한 줄씩 출력·기록도 된다. KV 경계는 `_tally.first`로 요청당 한 번만 기록한다(`kv_contract._refuse`) |
| **문제 3** 능력표의 엔진 버전을 조회·라우팅 조건으로 안 씀 | 맞음 | `caps.lookup`은 consumer·fact로만, `caps.route`는 evidence로만 고른다. `version` 필드는 기록용. 검사되는 것은 어휘 판(`vocab_version`)뿐 |
| 테스트 파일 21개 정상, 17개 의존성 부재로 미완료 | 평가자 환경의 사정 | 우리 환경(ci venv) `bash tests/run_all.sh`: 38개 파일 PASS, 336 checks, exit 0 |
| CI는 Linux·CPU 중심 | 맞음 | ubuntu-latest, CPU torch, transformers 5.17.0, Python 3.10/3.12 |

추가로 발견한 것(평가에는 없음): `python -m pytest tests`처럼 **한 프로세스**로 전체를 돌리면 30개가 실패한다(306 통과). 원인은 `test_adapter.py`, `test_boundary.py`, `test_cache_contract.py`가 모듈 적재 시점에 `ENTAIL_ON_BROKEN=stop`, `ENTAIL_UNKNOWN=require`를 `os.environ`에 넣고 되돌리지 않아 뒤 파일들이 "보고하고 계속" 대신 멈추기 때문이다. 파일 단독·`run_all.sh`(파일마다 새 프로세스; CONTRIBUTING과 CI가 쓰는 방법)로는 모두 통과한다. 코드 결함이 아니라 테스트 격리 문제이며, 기여자가 `pytest`를 그대로 치면 겪는다.

## 2. 가치 평가 문단

| 주장 | 판정 | 근거 |
|---|---|---|
| SGLang에 soft-capping을 잘못된 API 단계에 넘겨 무시되는 문제가 외부 사용자에 의해 보고됨 | 맞음 | sgl-project/sglang#33915 (2026-08-07, hijohnnylin, open, 댓글 0): FlashInfer가 `logit_cap`을 `plan()`이 아닌 deprecated `forward()`에 넘겨 무시. 우리 `caps.json`과 감사 문서가 인용한다 |
| "180개 중 64개"는 설정 수준의 노출 조사이고 이용자 비율은 재지 않았다 | 맞음 | E1은 config 구성 검사. 내려받기 가중 22%는 모델의 내려받기 비중이지 이용자 비율이 아니다 |
| RoPE 표: 379 / 273 / entail 복원 376 | 맞음 | `PUBLIC_CLAIMS.md` W2, `E4_USE_PATTERNS.md`: 379 → 273, entail 켬 376~380 (`m91/SUMMARY.md` fd-rope) |
| softcap: 198/500 답 변화, 정답 차이 3문항(313 대 316) | 맞음 | `m91/SUMMARY.md` S7 fd-softcap: 313/500, 316/500, p = 0.6636, 198 |
| E3: 12건 시도, 8건 재현, 0건 검출, 중복 제외 7종 | 맞음 | `E3_SUMMARY.md`: 73 screened, 12 passed, reproduced 8, detected 0. #24(vllm#49377)와 #37(vllm#49449)이 같은 결함·코드 경로 |
| 놓친 것에 캐시 식별, 토큰 종류, 토크나이저 어휘가 있다 | 맞음 | E3 분류: "no fact for cache identity", "no fact for token types"(vllm#58138), "no tokenizer fact"(transformers#48967) |
| 후속 판에서 미검출이 해결됐다는 근거 없음 | 맞음 | m11은 E3 #56(DSPARK 오탐)만 다시 돌렸다. 검출 0은 그대로 |
| "102회 오탐 0": 실행당 프롬프트 3개, 16토큰; 81회는 기존 집합, 새 표본 8개 모델·21회; 일부는 확정 판정을 unknown으로 바꾼 결과 | 맞음 | `E2_SUMMARY.md`: "Each run: 3 prompts, 16 greedy tokens"; m11 SUMMARY: fresh 8 models 21 runs; unknown 69줄 |
| 적재 비용 약 1%는 자동 수정 뒤 실행 비용이 아님. softcap 백엔드 전환 1.13배·1.18배 | 맞음 | `E4_USE_PATTERNS.md`: SGLang triton 1.13배, transformers eager 1.18배 |
| probe는 출력 차이로 "읽는다"를 판정하며 코드가 한계를 인정 | 맞음 | `probes.py` 머리말 4항: "a drop on one path can hide behind a difference" |
| jaxtyping은 shape·dtype 검사 | 맞음 | 일반 사실 |

## 3. 증상 중심 목표 문단

| 주장 | 판정 | 근거 |
|---|---|---|
| vLLM: 동시 요청 때 잘못된 KV 캐시 사용 보고, 오래된 캐시 추적 정보 수정 PR 병합 | 맞음 (평가자가 2차로 링크를 줌) | vllm#18955 (2025-05-30, z7d1, 닫힘): 높은 동시성에서 KV 캐시가 덮여 뒤 요청에 반복 문자·횡설수설. 수정 PR #18957 (2025-06-16 병합, quanliu1991): `ComputedBlocksTracker`의 오래된 `num_new_tokens_cached`로 잘못된 블록 배정. 고친 파일은 `vllm/core/block_manager.py`, `scheduler.py`로 **V0 엔진**이며, 이 코드 경로는 현재 vLLM(0.30)에 없다 |
| vLLM: 캐시 읽기·쓰기 순서 충돌, 보고자 실험 142/3,300 → 0/3,000 | 맞음 (평가자가 2차로 링크를 줌) | vllm#47282 (2026-07-01, Saddss, 열림, 댓글 4): `VLLM_USE_SIMPLE_KV_OFFLOAD=1`(CPU 오프로드 커넥터)의 load 경로에서 스트림 간 write-after-read 경합. 증상은 반복 + 여러 문자 체계 섞임. 보고자의 수정으로 142/3300(4.3%) → 0/3000. 특정 기능 경로(오프로드)에서만 난다 |
| vLLM 캐시 검사는 용량(토큰 수 대 확보 공간)만 본다. 블록 ID를 읽지만 판정은 개수·크기로 한다 | 맞음 | `vllm_cache_contract.read_choice`: `len(ids) * size`로 held를 만든다(`KvExtent(held, needed, granularity)`) |
| 이 어댑터에는 복구 동작이 없고 기본 정책은 기록하며 계속 | 맞음 | `handles()`가 `{}`; `kv_contract._refuse`는 기본 정책에서 broken으로 기록하고 계속 |
| PyTorch는 재컴파일 한도에서 eager로 돌아간다; seL4는 가정을 명시한다 | 맞음 | 일반 사실 |

## 4. 결론

- 평가의 사실 주장은 모두 맞다(vLLM 캐시 이슈 두 건은 평가자가 2차 답변에서 링크를 주어 확인했다). 코드 문제 세 가지(CLI 종료 코드, 판정 누적, 능력표 버전)는 재현·확인됐다.
- 두 vLLM 사례는 엔진 내부 상태의 경합(V0 스케줄러의 오래된 추적 정보, 오프로드 커넥터의 스트림 경합)이다. "추론 내부 오류를 고치면 이상 출력이 준다"는 근거는 되지만, 선언된 사실이 소비자에 닿지 못한 사례는 아니다. `LIBRARY_DESIGN.md` 2절("계층 내부는 보지 못한다", "증상 자체의 제거는 하지 않는다")과 E3의 분류(stale state는 "넓은 정의로는 부류 안, 캐시 정체 사실은 없음")에 비추면, 캐시 정체·시점·소유를 사실로 선언하는 일은 R4 코드북 v2(소유·수명, 정체·시점)의 연구 범위이고, 스트림 경합 자체는 범위 밖이다.
- 평가자의 완료 기준 "경고 출력이나 요청 중단만으로는 성공으로 인정하지 않는다"는 S2(해소할 수 없으면 정확히 `broken`으로 보고하고 멈추지 않는다)와 연구자의 "보고하고 계속" 결정(M5.4)과 다르다. 채택 여부는 연구자가 정한다.
- 새로 확인한 것: 단일 프로세스 pytest에서 30개 순서 의존 실패(테스트 격리). CONTRIBUTING과 CI는 파일별 실행이라 통과한다.
- 평가의 의견(범위 축소, 보호 모드 정책, 증상 기준 검증)은 사실 확인 대상이 아니다. 기준 문서와 대조하려면 연구자 지시가 필요하다.
