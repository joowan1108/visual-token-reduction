# 03. 사전등록 실험 계획: Focus Token Selection for SmolVLA

## 목적과 고정 가설

SmolVLA action expert의 초기 context aggregation은 dense로 유지하고, 후기 visual-prefix cross-attention에서 현재 noisy-action query와 관련 높은 visual patch token만 카메라별로 선택하면 dense baseline보다 LIBERO 40-task macro episode success rate를 높이면서 expert attention 연산을 줄일 수 있는지 검증한다.

이 실험은 FocusVLA 전체 방법의 재현이 아니라 SmolVLA late expert cross-attention hard top-k ablation이다.

## `00_hypothesis.md`의 모호성 해소

아래 해석을 결과 관찰 전에 고정한다.

- 원문의 반복된 비교 예산에 따라 각 run은 **10,000 optimizer steps**다. `planned training: 100,000 steps`는 이번 primary run budget으로 사용하지 않고 cosine scheduler의 decay horizon으로만 사용한다.
- 원문의 구체적인 training initialization을 따라 SmolVLM2 pretrained VLM과 seed별로 새로 초기화한 SmolVLA expert를 사용한다. `lerobot/smolvla_base`와 `sungkyunner/smolvla_libero_baseline`은 config/processor 및 재현 sanity reference이며 primary 초기 가중치나 primary 통계에 넣지 않는다.
- layer 번호는 코드 기준 0-indexed다. 8–15층 중 실제 visual-prefix cross-attention인 9, 11, 13, 15층만 sparse하게 한다.

이 해석을 바꾸려면 구현 전에 사용자가 수정 계획을 승인해야 하며, 결과를 본 뒤에는 바꾸지 않는다.

## 비교 조건

| 조건 | Camera별 visual keep ratio | 역할 |
|---|---:|---|
| Dense | 100% | baseline |
| Focus-50 | 50% | 유일한 primary intervention |
| Focus-25 | 25% | 과도 pruning sensitivity |
| Focus-75 | 75% | 완만한 pruning sensitivity |

모든 Focus 조건은 다음 알고리즘을 공유한다.

- expert layer 0–7은 dense다.
- layer 9, 11, 13, 15의 visual-prefix cross-attention만 sparse다. layer 8, 10, 12, 14의 self-attention은 변경하지 않는다.
- 각 대상 layer와 Euler denoising step에서 현재 noisy-action expert query와 visual key의 scaled QK logit을 계산하고 head와 action-query 차원을 산술평균한다.
- camera별 valid patch token을 독립 ranking하고 `max(1, ceil(ratio * N_valid))`개를 선택한 뒤 원래 spatial index 순으로 정렬한다. fully masked camera는 0개다.
- image special, language, state, action token은 제거하지 않는다.
- 선택 visual position의 K와 V를 함께 조회하지만 full prefix와 `DynamicCache`는 자르지 않는다.
- 별도 router와 신규 trainable parameter를 추가하지 않는다.
- flow-matching loss, noise/time sampling, Euler integrator는 변경하지 않는다.
- `compile_model=False`를 모든 조건에서 사용한다.

## 데이터와 학습 조건

- Dataset: `HuggingFaceVLA/libero`, Spatial/Object/Goal/Long 4 suites의 전체 training episodes
- Camera input: agent/top view와 wrist view; 실제 dataset key 순서를 resolved config에 기록
- Train seeds: `1000`, `1001`, `1002`
- Runs: 4 conditions × 3 seeds = 12
- Steps/run: 10,000; 총 120,000 optimizer steps
- Batch size: 4
- Augmentation: disabled
- Initialization: 동일 pretrained SmolVLM2 VLM + seed별 paired 신규 expert initialization
- Noise: `Normal(0,1)`
- Flow time: `Beta(1.5,1.0) * 0.999 + 0.001`
- Optimizer: AdamW, lr `1e-4`, betas `(0.9,0.95)`, eps `1e-8`, weight decay `1e-10`
- Scheduler: cosine, warmup 1,000 steps, decay horizon 100,000 steps, final lr `2.5e-6`
- Checkpoints: 2,000, 4,000, 6,000, 8,000, 10,000 steps
- Gradient clipping과 명시하지 않은 model/training defaults: local baseline resolved config 그대로

같은 train seed 안에서는 조건 간 expert initial state, dataset order, sampled noise/time을 paired로 맞춘다. 실행 전 initial state-dict hash와 resolved config를 저장한다. 조건 간 차이는 keep ratio와 selection 경로뿐이어야 한다.

## 평가 예산과 pairing

각 평가 checkpoint는 suite당 10 tasks × task당 20 episodes, 총 800 episodes/run이다. 각 task의 environment 및 action-noise episode seed는 `0..19`로 고정하고 모든 조건/checkpoint에서 paired로 재사용한다.

- Dense와 Focus-50: 5 checkpoints × 3 train seeds × 800 = 12,000 episodes/condition, 합계 24,000
- Focus-25와 Focus-75: 10K final checkpoint만 3 train seeds × 800 = 2,400 episodes/condition, 합계 4,800
- 총 평가 예산: 28,800 episodes

중단/실패 episode도 기존 LIBERO evaluator의 success 판정대로 0으로 포함한다. infrastructure failure는 원인과 재시도 횟수를 기록하고, 성공한 episode만 선택적으로 남기지 않는다. 필수 paired episode가 복구되지 않으면 해당 run을 결측으로 둔다.

## 지표

### Primary metric

10,000-step checkpoint의 LIBERO 40-task macro episode success rate다. 먼저 각 task의 20-episode success rate를 계산하고, 40개 task를 동일 가중 평균한다. Primary comparison은 paired Dense 대 Focus-50 하나뿐이다.

### 최소 의미 효과와 성공 조건

가설을 지지하려면 다음을 모두 만족해야 한다.

1. Focus-50 − Dense의 3-seed macro success-rate 차이가 **최소 +1.0 percentage point**다.
2. 아래 paired hierarchical bootstrap 95% CI의 하한이 0보다 크다.
3. 3 train seeds 중 적어도 2개에서 차이가 양수다.
4. Focus-50의 measured late cross-attention FLOPs와 effective visual KV token 수가 Dense보다 작다.

1.0pp는 FocusVLA와 VLA-Pruner의 50% ablation에서 관찰된 약 +1.0pp에 맞춘 사전등록 최소 효과다.

### Secondary metrics

- suite별 및 task별 episode success rate
- Dense와 Focus-50이 최종 Dense macro success rate에 처음 도달하는 checkpoint; checkpoint 간 보간 없이 최초 측정 checkpoint 사용, 미도달은 censored로 보고
- 고정 diagnostic batch의 flow-matching action loss; 별도 validation split을 사후 생성하지 않으며 일반화 지표로 해석하지 않음
- camera/layer/denoising-step별 selected spatial distribution
- selected top-k attention mass와 valid/effective token 수
- late expert attention FLOPs, peak allocated GPU memory, end-to-end action-chunk latency

## 효율 계측

동일 GPU, dependency environment, power/performance mode, batch size 1에서 측정한다.

- Latency: 50 warm-up chunks 뒤 200 action chunks, CUDA synchronization을 측정 경계에 두고 median과 p95 보고
- FLOPs: 동일 profiler scope로 조건별 100 action chunks
- Peak memory: 측정 전 peak statistics reset, 같은 input shape에서 max allocated memory 기록
- instrumentation이 output을 바꾸지 않음을 dense identity test로 확인

full prefix/cache를 보존하므로 전체-model FLOPs나 cache memory가 크게 줄 것이라고 가정하지 않는다. late expert attention 범위와 end-to-end 범위를 모두 보고한다.

## 통계 분석

Primary CI는 paired hierarchical bootstrap percentile CI로 계산한다.

1. RNG seed `20260804`, 10,000 bootstrap replicates를 사용한다.
2. 각 replicate에서 3 train seeds를 replacement sampling한다.
3. 각 sampled seed 안에서 suite별 10 task indices를 replacement sampling한다.
4. 동일하게 뽑힌 seed/task indices를 Dense와 Focus-50에 함께 적용한다.
5. 각 cell의 20-episode success rate로 40-task macro paired difference를 계산한다.
6. replicate 차이의 2.5/97.5 percentile을 95% CI로 보고한다.

Primary comparison이 하나이므로 multiplicity correction은 하지 않는다. Focus-25/75, checkpoint, suite/task 결과는 exploratory로 표시하고 같은 방식의 CI를 제시하되 primary 결론을 바꾸는 variant 선택에 쓰지 않는다.

3 paired train seeds가 모두 완료되지 않거나 pairing이 깨지면 primary 결과는 **inconclusive**다. 결측 run을 다른 seed로 교체하지 않는다.

## 반증 및 판정 기준

다음 중 하나면 primary 가설은 **unsupported**다.

- Focus-50 macro 개선이 +1.0pp 미만
- paired 95% CI가 0을 포함하거나 음수 영역에 걸침
- 3 seeds 중 2개 이상에서 개선이 양수가 아님
- success rate는 하락하고 효율만 개선됨

개선이 소수 task나 한 seed에 집중되면 전체 수치가 기준을 통과하더라도 불안정한 효과로 보고하고 일반화 주장을 제한한다. Focus-75만 개선되고 Focus-50이 실패하면 primary 가설은 unsupported다. Focus-25 실패와 Focus-50 성공은 과도 pruning의 정보 손실 sensitivity로 해석한다.

필수 run/episode가 예산 또는 환경 오류로 완료되지 않으면 **inconclusive**로 보고하며 supported/unsupported로 대체하지 않는다.

## 구현 검증 게이트

사용자 승인 뒤 `hypothesis_implementer`가 구현할 때 다음 최소 검사를 통과해야 한다.

1. camera별 `ceil` top-k, 독립 budget, spatial-order 복원
2. non-visual token과 fully masked camera 보존
3. keep ratio 1.0에서 기존 dense loss/action 수치 identity
4. 0–7과 8/10/12/14는 dense, 9/11/13/15만 sparse인 layer boundary
5. training direct-prefix와 inference cached-prefix 양 경로의 finite forward/backward/action
6. full cache length 불변과 기존 checkpoint strict load

검사 실패 상태에서는 본 실험을 실행하지 않는다.

## artifact와 불변 조건

- run별 commit hash, dirty status, resolved config, dependency versions, hardware, initial state hash를 저장한다.
- raw metrics와 episode 결과는 append-only 경로에 저장하고 기존 결과를 덮어쓰지 않는다.
- 결과를 보기 전 이 문서의 primary metric, seeds, budget, comparison, 통계, 반증 기준을 변경하지 않는다.
- 불가피한 변경은 별도 amendment에 시각과 이유를 기록하고 원 계획을 보존한다.
- raw results 완료 뒤 `paper_method_evaluator`, 그다음 `results_analyst`를 실행하며 최종 분석은 `06_analysis.md`에 저장한다.

## 승인 상태

**승인 대기.** 사용자 승인 전에는 `hypothesis_implementer`를 실행하지 않고 application source를 수정하지 않는다.
