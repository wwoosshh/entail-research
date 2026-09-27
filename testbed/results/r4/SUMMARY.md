# R4 / M14: Identity 사실과 vLLM 블록 해시 계약 — 측정

- 날짜: 2026-09-25. 환경: WSL, vLLM 0.30.0(`~/venvs/vllm`), RTX 4070 Ti. 단위 시험은 CPU(`~/venvs/ci`).
- 사실: `Identity`(TIME, 어휘 v5). 규칙: `identity_stale`(`entail/identity_contract.py`). 어댑터: `entail/adapters/vllm_identity.py`(`Scheduler._update_request_as_session`를 감쌈).
- 시험 문제: fd-identity-mech, fd-identity-e2e(`testbed/PROBLEMS.md` 2절).

## 1. 기전 (GPU 불필요) — `identity_probe.py` → `identity_probe.json`

| 항목 | 값 |
|---|---|
| 갱신 뒤 저장 해시가 옛 토큰의 것인가 | 예 (재현) |
| 옛 프롬프트의 다른 요청이 열쇠를 공유하는가 | 예 (틀린 적중 가능) |
| 불변식 검사가 잡는 첫 지점 | 블록 0 |
| 해소 뒤 열쇠 | 지금 토큰의 것, 공유 없음 |
| 정상 갱신 | 조용 |
| 비용 | 마지막 블록 3.0 µs, 256블록 사슬 282 µs |

## 2. 어댑터 (실제 vLLM 객체, GPU 불필요) — `adapter_check.py` → `adapter_check.json`

- `read_choice`가 저장 열쇠와 지금 토큰의 열쇠를 정확히 읽음(hex).
- 판정: `resolved`, handle `identity_recompute`, target 0.
- 해소 뒤 블록 0 열쇠 = 새 토큰의 해시.
- 정상 갱신은 판정 0개.
- `ok: true`.

## 3. 끝까지 (GPU) — `testbed/m10_e3/vl49449.py`, SmolLM2-135M-Instruct, prefix 캐시

| entail | 후보 첫 토큰 | 캐시 적중 | 참조 첫 토큰 | reproduced |
|---|---|---|---|---|
| off | 28 (`,`) | 16 | 198 (`\n`) | true (틀린 출력) |
| on | 198 (`\n`) | 0 | 198 (`\n`) | **false (고쳐짐)** |

- 켬에서 `container:vllm.request.block_hashes`가 `resolved`(index 0, "forget the stale identities and let the store remake them")를 기록했다. 후보 요청의 거짓 캐시 적중(16토큰)이 막혀 재계산됐고, 참조와 같은 토큰이 나왔다.
- 이 결함(vllm#49377·#49449)은 열려 있고 수정 PR 셋은 미병합이라 vLLM 0.30.0에 그대로 있다. entail 1.0.0~1.0.2는 통과시켰다(E3 놓침). M14가 처음으로 잡고 되돌린다.

## 4. 단위 시험

- `tests/test_identity.py`(핵심 규칙, 사실, 해소·거부·멈춤), `run_all.sh`: 39개 파일, 342 checks, exit 0(1.0.2의 38개·336에서 +1 파일·+6).
- 어휘 완전성(`test_contracts.py` 표본), 구조 검사(`test_adapter_rules.py`: `vllm_identity`가 규칙 없음), 판본(`test_facts_v1.py`) 통과.

## 5. 아직 안 잰 것 (M14 완료 기준의 나머지)

- **S3 정상 실행 오탐:** 훅은 `_update_request_as_session`(스트리밍 세션)만 감싸므로 보통 생성은 호출하지 않는다. 구조상 보통 실행의 오탐은 0이지만, 38개 모델 정상 실행 대조로 재지는 않았다.
- **S4 상시 비용:** 훅이 세션 갱신 때만, 잘린 지점부터의 블록만 대조한다(꼬리 3.0 µs/블록). 보통 실행에는 훅이 걸리지 않는다. E2 상시 배수로 재지는 않았다.
- **다른 엔진:** SGLang radix 캐시의 같은 모양 낡음은 재지 않았다(R4.4).
