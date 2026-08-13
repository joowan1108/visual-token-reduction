# Frozen-VLM Atomic SmolVLA 사전등록 실험 계획

## 0. 문서 상태

- 상태: 구현 전 동결 계획
- 근거 문서: `00_hypothesis.md`, `01_literature_review.md`, `02_implementation_map.md`
- 연구 질문: **동일한 frozen SmolVLA 표현과 동일한 SARM/LIBERO 학습 조건에서, six-skill SG-MoE action policy가 single-shared action policy보다 LIBERO-Long의 폐루프 composition 성공률을 높이는가?**
- 범위 밖: VLM fine-tuning, think-trace 학습, continual learning, unseen skill, unseen task/composition 일반화, SARM reward model 학습

이 문서는 raw result를 보기 전에 비교 조건, metric, seed, budget, 통계와 판정 기준을 고정한다. 변경이 필요하면 기존 문서를 수정하지 않고 날짜·이유·결과 열람 여부를 기록한 amendment를 추가한다.

## 1. 검증할 주장

### 1.1 주 가설

SARM의 ground-truth atomic skill로 action FFN을 전문화해 학습한 SG-MoE는 frozen VLM이 예측한 skill로 실행할 때, 동일하게 학습한 dense SmolVLA보다 LIBERO-Long task-balanced end-to-end success rate가 높다.

### 1.2 보존 조건

Long 성능의 향상이 Spatial/Object/Goal 개별 task 능력을 훼손하지 않아야 한다. 세 short-horizon suite를 합친 task-balanced success rate의 비열등성 margin은 **−3 percentage points**다.

### 1.3 해석 한계

SARM의 LIBERO-Long demonstrations를 train에 포함한다. 따라서 결과는 in-distribution closed-loop composition이며 novel composition generalization이 아니다.

## 2. 비교 조건

### A. Dense baseline

- `lerobot/smolvla_base`에서 동일 revision으로 시작한다.
- VLM 전체를 동결한다.
- 기존 single action-expert FFN을 유지한다.
- VLM 이외의 action-policy attention, state/action/time projections와 FFN은 학습한다.
- SG-MoE 조건과 같은 episode split, sampled anchor sequence, boundary mask, optimizer, step 수를 쓴다.
- 학습 중 skill label은 sampling과 boundary mask에만 사용하고 모델 입력에는 주지 않는다.
- 온라인 평가에는 skill planner가 필요하지 않으며 표준 dense action path를 실행한다.

### B. Six-skill SG-MoE

- A와 같은 base checkpoint와 frozen VLM을 쓴다.
- action expert transformer의 **모든 FFN layer**를 shared FFN + 6 skill FFN으로 바꾼다.
- skill 순서는 `pick=0`, `place=1`, `push=2`, `turn=3`, `open=4`, `close=5`로 고정한다.
- 각 layer는 독립 router를 가진다.
- 고정 embedding은 6차 scaled one-hot을 action hidden width까지 zero-pad한 buffer다. 대각 scale은 `[10, 28, 46, 64, 82, 100]`이다.
- router는 bias 없는 linear layer이고 6 logits를 만든다. 초기에는 입력 skill과 같은 index가 top-1이 되도록 대각 원소를 scale의 역수로 설정하고 나머지는 0으로 둔다.
- top-1 expert 확률을 `w`라 할 때 `output=(1-w)*shared(x)+w*skill_expert(x)`를 사용한다.
- 기존 pretrained dense FFN weights를 shared와 여섯 skill FFN 모두에 복사한다.
- router, shared/skill FFN을 포함한 VLM 이외 action policy 전체를 flow-matching loss로 공동 학습한다.
- router CE, load-balancing loss, language/text loss는 사용하지 않는다.
- 학습 routing은 SARM ground-truth skill, 온라인 routing은 frozen VLM prediction을 사용한다.

### 공정성 규칙

- A와 B의 VLM, trainable 범위, data exposure, optimization update 수와 evaluation manifest는 동일하다.
- parameter 수 차이는 제거하지 않고 total/trainable/active parameters, FLOPs, peak memory, action latency를 함께 보고한다.
- parameter-matched dense 모델이나 fixed-router 모델은 이번 주 실험에 추가하지 않는다. 따라서 결과는 specialization과 추가 capacity를 완전히 분리하지 못한다.

## 3. 고정 revision과 환경

구현 시작 시 다음 값을 실제 immutable SHA로 채운 뒤 바꾸지 않는다.

| 항목 | 동결 값 |
|---|---|
| LeRobot | 현재 local commit SHA + dirty diff hash |
| AtomicVLA reference | `zhanglk9/AtomicVLA` commit SHA |
| SmolVLA checkpoint | `lerobot/smolvla_base` revision SHA |
| SmolVLM | checkpoint config가 가리키는 exact repo/revision |
| SARM | `k1000dai/libero_subtask_sarm` revision SHA |
| LIBERO | 설치된 package/fork commit SHA |
| Python/PyTorch/CUDA/MuJoCo | resolved environment artifact |

학습은 BF16 mixed precision을 사용한다. 장치가 BF16을 지원하지 않는 run은 primary 결과에 섞지 않고 별도 protocol deviation으로 기록한다. 두 조건은 같은 GPU type에서 실행한다.

## 4. 데이터 준비

### 4.1 데이터 선택

주 데이터는 SARM 1,693 episodes, 273,465 frames, 40 LIBERO tasks다. `modified_libero_rlds`는 AtomicVLA-compatible semantic boundary를 재생성해야 하므로 사용하지 않는다.

### 4.2 52-to-6 mapping

52개 subtask 문자열을 versioned JSON에 모두 명시한다. unknown/default fallback은 없다. 다음 조건을 만족하지 않으면 full training을 시작하지 않는다.

- dataset vocabulary와 mapping key가 정확히 일치
- 52/52가 정확히 한 class에 매핑
- 여섯 class가 모두 하나 이상의 valid segment 보유
- `push` label은 독립 class이고 `place`로 매핑된 항목이 없음

### 4.3 자동 품질 검사와 제외

episode별 audit manifest를 만든다. 다음은 자동 제외한다.

- annotation 배열 길이 불일치
- segment start/end 범위 오류
- 시간 역전, overlap, zero/negative length
- 유효 frame의 미할당 또는 복수 할당
- vocabulary/mapping 불일치
- action/state의 NaN, Inf, 차원 불일치
- dataset provenance에서 uniform fallback으로 명시된 episode

provenance가 없는데 boundary 간격이 uniform하다는 이유만으로 자동 삭제하지 않는다. 해당 episode는 `suspect_uniform`으로 격리하고 주 학습에서 제외한다. motion과 semantic label의 모순은 자동 재라벨링하지 않고 `suspect_motion_label`로 제외한다. 모든 제외는 episode ID와 reason code를 보존한다.

filter 결과로 어떤 task 또는 skill의 train episode가 10개 미만이 되면 full training을 중단하고 data adequacy failure로 보고한다. 결과를 본 뒤 filter 기준을 완화하지 않는다.

### 4.4 split

- filter 후 episode 단위 **80% train / 10% validation / 10% offline test**
- `task_index`별 층화
- split seed `20260307`
- 같은 episode의 frame은 한 split에만 존재
- manifest에는 episode ID, task, suite, frame 수, segment/skill histogram을 저장

validation과 offline test는 자연 skill 분포를 유지한다. simulator evaluation은 40 task의 별도 closed-loop rollouts다.

### 4.5 action chunk와 boundary

- data rate: 10 Hz
- `chunk_size=10`: 1.0초 action 예측
- `n_action_steps=5`: 0.5초 실행 후 observation 갱신
- segment interval convention: `[start_frame, end_frame]` inclusive
- frame `t`의 skill은 `start <= t <= end`인 segment label
- action target indices: `t ... t+9`
- episode 끝 또는 `skill(t+j) != skill(t)`인 target은 invalid mask
- episode padding mask와 boundary mask를 OR
- masked flow MSE는 valid scalar action element 수로 재정규화
- valid action이 0인 anchor는 사전 index에서 제외

A와 B 모두 같은 boundary mask를 사용한다.

### 4.6 train sampling

train draw마다:

1. skill을 `P(skill)=1/6`으로 선택한다.
2. 선택 skill의 valid anchor frame 중 하나를 균등 복원 추출한다.

두 조건과 같은 training seed는 동일 global anchor sequence를 사용한다. DDP에서는 global draw sequence를 만든 후 rank별 shard한다. 매 10,000 draws의 class 비율이 각 `1/6 ± 1 percentage point`인지 assert한다.

## 5. 학습 프로토콜

### 5.1 seed와 총 budget

- training seeds: **42, 43, 44**
- conditions: A, B
- full runs: 2 conditions × 3 seeds = **6 runs**
- optimizer updates/run: **100,000**
- global batch size: **64**
- 총 budget: **600,000 optimizer updates**, 38.4M sampled anchors
- hyperparameter sweep 없음

OOM이면 gradient accumulation으로 global batch 64를 유지한다. batch를 줄이거나 seed/run을 생략하지 않는다.

### 5.2 optimizer와 schedule

- AdamW
- learning rate `1e-4`
- betas `(0.9, 0.95)`
- epsilon `1e-8`
- weight decay `1e-10`
- global gradient norm clip `10`
- linear warmup `1,000` updates
- cosine decay from update 1,000 to 100,000
- final learning rate `2.5e-6`
- EMA 사용 안 함

동일 seed에서 initialization, data order, flow noise와 time sampling seed를 일치시킨다. SG-MoE에만 존재하는 parameter의 initialization은 해당 seed로 결정한다.

### 5.3 checkpoint 선택

- checkpoint는 10,000 updates마다 append-only로 저장한다.
- 온라인 결과를 보지 않고 validation의 **six-skill macro masked flow-MSE**가 가장 낮은 checkpoint 하나를 선택한다.
- tie는 더 이른 checkpoint를 선택한다.
- NaN, Inf 또는 checkpoint corruption이 아니면 run을 재시작해 seed를 교체하지 않는다.
- final 100k checkpoint와 선택 checkpoint를 모두 보존한다.

### 5.4 필수 train logging

- global 및 skill별 masked flow-MSE
- valid action fraction과 boundary-masked fraction
- sampled skill histogram과 unique episode coverage
- gradient norm과 learning rate
- router layer별 selected expert confusion, gate probability, entropy
- shared/skill output norm과 expert별 gradient norm
- total/trainable/active parameter 수
- steps/sec, GPU memory, wall time
- VLM checksum before/after

## 6. Frozen-VLM planner

### 6.1 입력

SG-MoE 온라인 평가에서 같은 frozen SmolVLM checkpoint에 다음을 제공한다.

- 현재 main camera image
- 현재 wrist camera image
- 전체 task instruction
- 이전 skill 또는 첫 호출의 `none`
- 이전 판단 이후 실행한 action step 수
- 허용된 여섯 skill과 출력 schema

### 6.2 호출과 출력

- 최초 action 전 한 번 호출
- 이후 5 actions 실행 후마다 호출
- temperature `0`, sampling off
- 최대 출력 token `32`
- 출력은 정확히 다음 JSON schema만 허용

```json
{"decision":"continue|switch","skill":"pick|place|push|turn|open|close"}
```

`continue`는 이전 skill과 동일한 skill이어야 하고 `switch`는 다른 skill이어야 한다. first call은 `switch`만 허용한다.

### 6.3 실패 및 queue 규칙

- valid `switch`: 남은 action queue를 즉시 폐기하고 새 skill로 chunk를 예측
- valid `continue`: 현재 skill을 유지하고 예정대로 새 observation에서 chunk를 예측
- invalid JSON/enum/논리 모순: parse failure로 기록
- 이전 skill이 있으면 한 interval만 이전 skill 유지
- 연속 두 번 parse failure이면 episode failure 종료
- 첫 호출 parse failure이면 episode failure 종료
- re-prompt, keyword fallback, human correction 없음

prompt template와 tokenizer/model revision의 SHA를 artifact로 저장한다.

## 7. 평가 프로토콜

### 7.1 공통 rollout 수와 horizon

각 선택 checkpoint에 대해:

- Spatial 10 tasks × 50 rollouts
- Object 10 × 50
- Goal 10 × 50
- Long (`libero_10`) 10 × 50
- training seed당 2,000 rollouts
- condition당 6,000, 총 **12,000 rollouts**

max environment steps:

- Spatial: 280
- Object: 280
- Goal: 300
- Long: 520

두 조건은 같은 observation keys와 camera order, image transform, state/action normalization, relative action representation, simulator version을 사용한다.

### 7.2 exact initial-state pairing

평가 전에 `(suite, task_id, init_state_id, rollout_seed)` 50개씩을 `eval_manifest.json`으로 고정한다. model이 일찍 성공해 vector env가 auto-reset되더라도 다음 rollout의 init state가 달라지지 않도록 각 rollout마다 exact state를 지정하거나 env를 재생성한다.

실제로 적용된 initial state의 hash를 raw log에 저장한다. 비교 쌍의 hash가 다르면 해당 쌍은 paired analysis에서 제외하고 protocol deviation을 보고한다. seed만 같다는 이유로 paired라고 간주하지 않는다.

### 7.3 primary metric

**LIBERO-Long task-balanced end-to-end success rate**

각 task의 `successes/50`을 먼저 계산한 뒤 10 tasks를 동일 가중 평균하고, 세 training seed를 동일 가중 평균한다. primary effect는 `B - A`의 absolute percentage-point 차이다. B는 frozen-VLM routing으로 실행한다.

### 7.4 key secondary metric

Spatial/Object/Goal 30 tasks의 task-balanced end-to-end SR 차이 `B - A`. 비열등성 margin은 `-0.03`이다.

### 7.5 secondary/diagnostic metrics

- suite별 task-balanced SR와 40개 task별 `successes/50`
- Long ordered subgoals completed / total
- Long transition success와 최초 실패 subgoal
- success까지 걸린 environment steps
- planner six-way macro-F1, class별 precision/recall, confusion matrix
- continue/switch boundary precision/recall/F1, ±5 frame tolerance F1
- `push` recall과 `push -> place` confusion
- planner parse failure와 연속 failure episode 수
- held-out SARM natural-distribution skill별 masked flow-MSE 및 action L1
- SG-MoE router/expert usage와 collapse diagnostics
- inference latency p50/p95, active parameters, peak memory

planner metric은 held-out SARM frame/segment annotation을 reference로 계산한다. SARM label noise 때문에 이를 절대 ground truth 정확도라고 부르지 않는다.

온라인 GT skill oracle은 LIBERO state predicate로 재현 가능한 frame-level skill oracle이 없으므로 primary/secondary 평가에 포함하지 않는다. held-out demonstration GT routing은 offline action-policy upper-bound diagnostic으로만 보고한다.

## 8. 통계 분석

### 8.1 primary interval과 test

- A/B의 exact initial-state paired binary outcomes를 사용한다.
- training seed → task → paired initial state 순서를 보존하는 hierarchical paired bootstrap을 **10,000 replicates** 수행한다.
- primary effect `Long SR_B - Long SR_A`의 percentile 95% CI를 계산한다.
- two-sided alpha `0.05`.

### 8.2 short-horizon non-inferiority

같은 hierarchical paired bootstrap으로 30-task SR 차이의 95% CI를 구한다. lower bound가 `-0.03`보다 크면 비열등으로 판정한다.

### 8.3 보고와 다중 비교

- 각 seed/task의 raw `successes/50`
- seed별 suite SR, 3-seed 평균과 표준편차
- absolute difference와 95% CI
- task별 binary CI는 Wilson 95% interval
- task별 또는 skill별 유의성 주장을 할 때만 Holm correction
- primary Long comparison 한 개에는 추가 보정 없음
- initial-state pairing이 검증된 task별 탐색 분석에만 exact McNemar test 사용

rollout 50회를 training seed 50개처럼 취급하지 않는다. seed-level 결과와 방향을 모두 공개한다.

## 9. 판정과 반증 기준

### 지지됨

다음을 모두 만족할 때만 주 가설을 지지한다고 판정한다.

1. Long `B-A` 95% CI lower bound가 `0`보다 큼
2. Spatial/Object/Goal `B-A` 95% CI lower bound가 `-0.03`보다 큼
3. Long 효과 방향이 세 training seed 모두에서 양수
4. data/eval integrity gate와 frozen-VLM checksum 통과

### 반증됨

다음 중 하나이면 가설과 반대되는 증거로 판정한다.

- Long effect의 95% CI upper bound가 `0` 이하
- short-horizon effect의 95% CI upper bound가 `-0.03` 미만
- SG-MoE가 finite action을 안정적으로 생성하지 못해 사전 정의 rollout의 5% 이상이 numerical failure

### 불확실

나머지는 불확실로 보고한다. 특히 point estimate가 양수여도 CI가 0을 포함하거나, 한 seed만 전체 효과를 만들거나, planner failure로 action-policy 효과를 식별할 수 없으면 지지로 올리지 않는다.

frozen-VLM routing 실패와 SG-MoE action-policy 실패는 구분한다. offline GT routing에서 B가 A보다 낫지만 online B가 낫지 않으면 “SG-MoE action specialization은 유망하나 frozen planner를 포함한 주 가설은 지지되지 않음”으로 보고한다.

## 10. 구현 및 실행 gate

### Gate 1 — data

- revision pin과 52/52 mapping
- episode audit 완료 및 제외 reason 공개
- 80/10/10 split leakage 0
- skill/task별 usable counts 공개
- boundary convention unit test 통과

### Gate 2 — model

- atomic mode off에서 기존 SmolVLA test 무회귀
- dense checkpoint 승격 직후 모든 6 route의 output이 dense FFN과 tolerance 내 일치
- VLM optimizer parameter 0, backward gradient 0, train 전후 checksum 동일
- router/shared/선택 expert gradient finite nonzero
- 비선택 expert가 해당 sample에서 실행되지 않음
- save/resume 후 동일 route와 action

### Gate 3 — sampler/loss

- 고정 seed에서 sampled anchor sequence 재현
- 10k draws의 각 skill 비율 `1/6 ± 1%p`
- boundary 뒤 action 값을 바꿔도 loss 불변
- dense와 SG-MoE가 같은 anchor manifest 소비

### Gate 4 — planner/eval

- strict JSON parser test
- reset 시 previous skill과 action queue 초기화
- switch 시 남은 queue 즉시 폐기
- 각 suite 1 rollout smoke test
- A/B exact init-state hash 일치
- 50-rollout denominator와 max-step enforcement test

### Gate 5 — training smoke

- 각 condition seed 42로 20 optimizer steps
- loss/gradient finite
- expected trainable parameter allowlist 일치
- checkpoint save/reload 성공

Gate 실패는 full run을 시작하지 않는 구현 실패이며, 성능 기준으로 gate를 새로 만들지 않는다.

## 11. 결과 보존과 재현 산출물

각 run은 새 immutable directory에 다음을 보존한다.

- resolved config와 모든 revision/hash
- trainable parameter name list와 count
- data audit, split, anchor, evaluation manifests
- 52-to-6 mapping과 planner prompt/parser version
- train/validation raw metrics
- checkpoints와 selection record
- episode별 raw rollout outcome, init-state hash, route trace, subgoal trace
- bootstrap input과 replicate seed, summary output
- stdout/stderr와 environment inventory

raw result는 덮어쓰거나 삭제하지 않는다. 실패 run도 실패 원인과 마지막 valid artifact를 보존한다.

## 12. 실행 순서

1. revision과 환경을 pin한다.
2. data audit, 52-to-6 mapping, split/anchor manifests를 생성·검토한다.
3. dense/SG-MoE 공통 boundary mask와 sampler를 구현한다.
4. SG-MoE, checkpoint migration, VLM freeze assertions를 구현한다.
5. frozen planner와 fixed-init LIBERO evaluator를 구현한다.
6. Gate 1–5를 통과한다.
7. A/B × seeds 42/43/44의 6 runs를 완료한다.
8. validation rule로 checkpoint를 자동 선택한다.
9. 고정 manifest의 12,000 rollouts를 실행한다.
10. 사전 정의 script가 통계와 표를 생성한다.
11. `paper_method_evaluator`와 `results_analyst`가 raw result를 읽기 전용으로 독립 검토한다.

구현은 사용자가 명시적으로 시작을 요청할 때 AGENTS.md에 따라 단일 `hypothesis_implementer`가 소유한다. 그 전에는 application source를 변경하지 않는다.

