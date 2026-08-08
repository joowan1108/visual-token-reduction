# Subagent allocation plan

## 원칙

- Agent 하나당 artifact 하나를 책임진다.
- 문헌 조사와 read-only 코드 분석만 병렬 실행한다.
- Application source는 승인 후 `hypothesis_implementer` 한 명만 수정한다.
- Raw result가 완성되기 전에는 결과 해석 agent를 실행하지 않는다.

## 배분

| 순서 | Agent | 입력 | 단일 출력 | 완료 조건 |
|---:|---|---|---|---|
| 1A | `paper_researcher` | `00_hypothesis.md`, reference URL | `01_literature_review.md` 갱신 | 직접 지지·반례·차이를 primary source로 구분 |
| 1B | `reference_impl_reader` | `00_hypothesis.md`, local HEAD, dataset repo | `02_implementation_map.md` 갱신 | train/inference/data/checkpoint 경로와 chunk mismatch 해결안 확인 |
| 2 | Primary orchestrator | `00`–`02`, `final_idea.md` | `03_experiment_plan.md` | metric, seed, budget, 비교 조건, 통계, 반증 기준 동결 |
| 3 | `hypothesis_implementer` | 승인된 `00`–`03` | source diff, tests, `04_change_log.md` | 최소 diff와 CPU smoke test 통과 |
| 4 | Primary orchestrator | 구현 diff와 tests | review 결과 | source flow, 회귀, config/checkpoint 호환성 확인 |
| 5 | `paper_method_evaluator` | 승인 계획, 구현, raw run manifest | `05_method_evaluation.md` | baseline/intervention protocol 일치 여부 평가 |
| 6 | `results_analyst` | 완료된 raw results와 `03` | `06_analysis.md` | supported/unsupported/inconclusive 판정 |

## 병렬 실행

`paper_researcher`와 `reference_impl_reader`는 후보 논문·repository가 이미 `00_hypothesis.md`에 있으므로 동시에 실행할 수 있다. 그 외 단계는 앞 단계가 끝난 뒤 순차 실행한다.

```text
paper_researcher ───────┐
                       ├─> orchestrator plan/approval
reference_impl_reader ─┘             │
                                     v
                           hypothesis_implementer
                                     │
                                     v
                           paper_method_evaluator
                                     │
                                     v
                              results_analyst
```

## Agent별 금지 범위

- Researcher/reader: source와 raw results 수정 금지
- Implementer: 실험 metric·seed·budget 변경 금지
- Evaluator: 학습 재실행·checkpoint 선택 변경 금지
- Analyst: primary metric 재정의·결측 run 대체 금지

## 현재 gate

사용자가 2026-08-08에 경계당 pair sample 하나의 amendment와 P1 구현·검증을 승인했다. `hypothesis_implementer`만 application source를 수정하고, orchestrator가 diff와 CPU tests를 검토한다. 별도 승인 질문은 하지 않는다.
