# entail 연구: 의미가 사라지지 않는 AI 실행

> **For readers of the entail repository (English).** This is the research workspace behind
> [entail](https://github.com/wwoosshh/Entail): the theory (`THEORY.md`), the library design (`LIBRARY_DESIGN.md`),
> the roadmap with every milestone's completion record (`ROADMAP.md`), the measurement scripts (`testbed/`,
> `sweep/`, `rolebench/`) and their result files (`testbed/results/`, `issue_track/`, `realworld/`, `phase0/`),
> and the reinvestigation of the earlier claims (`reinvestigation/`). Most of the text is Korean. Every number in
> entail's README traces to a result file here; `testbed/results/m10/PUBLIC_CLAIMS.md` lists which file backs which
> claim, and `testbed/M10_PROTOCOL.md` defines the measurements. Personal paths were replaced by `<workspace>` and
> `<user>` before publishing (`publish_workspace.py`); model weights are not included. Reproducing the GPU runs
> needs the environment described in `env/SETUP.md` (one RTX 4070 Ti, WSL2). License: Apache-2.0, as entail.

AI 실행 스택(LLM 추론 엔진, 이미지 생성 파이프라인)에서 값의 의미가 부품 경계에서 사라지거나 어긋나 경고 없이 틀린 결과가 나오는 문제를 다룬다. 해법은 의미를 프런트엔드의 타입 선언처럼 명시적으로 선언하고, 강제하고, 보존하는 것이다. 이 연구는 그 원인 부류를 구조적으로 막는 라이브러리 entail을 만든다.

## 읽는 순서

1. `THEORY.md`: 이론. 연구자 원문, 명제, 재조사 판정
2. `LIBRARY_DESIGN.md`: 라이브러리 설계. 해결할 것, 하지 않을 것, 원칙, 구조, 사실 어휘, 판정 규칙, 성공 기준
3. `ROADMAP.md`: 단계(M0~M9), 완료 기준, 현재 위치, 작업 규칙
4. `reinvestigation/README.md`: 재조사의 목록과 상태, 옛 조사의 사실 검증

재정립 이전의 문서(`RESEARCH_PLAN.md`, `EXECUTION_PLAN.md`, `REPORT_2026-09-23.md`, `realworld/`, `phase0/`)는 기록으로 둔다. 방향은 위 문서를 따른다.

## 지금 있는 것

| 무엇 | 어디 | 상태 |
|---|---|---|
| 라이브러리 작업 사본 | `entail/` (공개 저장소 github.com/wwoosshh/Entail, PyPI `entail-ai` 1.0.1) | 로드맵 M0~M9를 마치고 1.0.0을 공개했다(2026-09-24). M11에서 규모에서 드러난 틀린 경보를 고쳐 1.0.1을 공개했다(2026-09-25, `ROADMAP.md`) |
| 시험 문제: 재현 벤치마크 16건 | `rolebench/` | 사용 중 |
| 시험 문제: 실제 환경 사례 | `issue_track/` | 사용 중 |

새 Claude 세션은 `CLAUDE.md`를 자동으로 읽는다.
