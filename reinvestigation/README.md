# 재조사 (2026-09-23)

- **왜 했나:** 연구자 지시 "내가 말한 최초이론에 맞게 문서를 전부 정리하고 조사를 다시 진행해"에 따랐다. 앞선 시장 조사와 연구 방향 검토의 사실을 의심할 수 있어서, 이론(`../THEORY.md`)을 먼저 고정하고 새로 조사했다.
- **어떻게 끝났나:** 에이전트를 너무 많이 돌려 토큰을 과하게 썼다. 연구자 지시로 모두 멈췄고, 남은 결과로 판정했다. 판정은 `../THEORY.md` 4절에 있다.
- GitHub에는 읽기(GET)만 했다.

| 파일 | 내용 | 상태 |
|---|---|---|
| `audit/bug_data_audit.md`, `.json` | 옛 버그 파일럿의 수치 재계산, 표본 30건 대조, 엔진 주장 15개 | 완결 |
| `audit/survey_audit.json` | 옛 시장 조사의 사실 146개 확인 | 중간 저장본. md 보고서와 결론 점검이 없다 |
| `audit/prior_work_audit.json` | 옛 선행 연구·새로움 문서의 사실 88개 확인 | 중간 저장본. md 보고서와 새로움 점검이 없다 |
| `performance_side.md` | 이론의 성능 쪽 주장 검증 | 완결 |
| `market_incidents.md`, `.json` | 실행 소프트웨어 때문에 품질이 떨어진 시장 사례 28건(LLM 20, 이미지 8) | 완결 |
| `feasibility.md` | 실현 가능성: 이론적 한계, 붙이는 수단, 전파, 비용, 채택, 이미지 쪽, 로컬 측정, 판정 | 판정 절(9절)까지 저장됨 |
| `prior_attempts.md`, `.json` | 유사 시도: 범주 1~6과 소결 | 중간 저장본. 범주 7(일반 소프트웨어)과 종합(0절)이 없다 |
| `market_sample/` | 새 이슈 표본(가설을 모르는 기록자) | 아래 표 |

`market_sample/`

| 파일 | 내용 | 상태 |
|---|---|---|
| `CODEBOOK_R.md`, `CODEBOOK_R_rater.md`, `analyze.py` | 평정 규칙과 집계 정의. 사례를 보기 전에 썼다 | 평정은 하지 못함 |
| `make_llm_sample.py`, `collect_image.py` 등 | 표본 추출. 규칙과 시드는 파일 머리말에 있다 | 완결 |
| `cases_L1.json`, `cases_L2.json` | LLM 엔진 이슈 80건 기록 | 완결 |
| `cases_I1.json`, `cases_I2.json` | 이미지 생성 이슈 50건 기록(diffusers 25, ComfyUI 25) | 완결 |
| `cases_L4.json`, `cases_S1.json` | 확대 표본 기록 | 도중에 멈춤. 집계에 쓰지 않았다 |
| `descriptive_stats.json` | 완결된 기록 130건의 기술 통계. 출력 문제 수, 원인 확인 수, 경고 여부, 증상 | 완결 |
