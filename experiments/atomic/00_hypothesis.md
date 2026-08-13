# Frozen-VLM SmolVLA with Atomic-Skill SG-MoE on LIBERO-10

- 상태: 가설 및 구현 방향 초안
- 작성일: 2026-08-12
- 대상 모델: `lerobot/smolvla_base`
- 주 평가 환경: LIBERO-10 (`libero_10`, 논문의 LIBERO-LONG에 대응)

## 1. 연구 질문

SmolVLA의 VLM 전체를 동결하고 action policy에만 5개 atomic skill(`pick`, `place`, `turn`, `open`, `close`)로 라우팅되는 Skill-Guided Mixture-of-Experts(SG-MoE)를 학습하면, 동일한 데이터와 학습 예산으로 action policy만 학습한 단일-expert SmolVLA보다 LIBERO-10의 long-horizon 성공률을 높일 수 있는가?

이 연구는 AtomicVLA 전체를 재현하지 않는다. 특히 `[think]`/`[act]` 토큰, task-chain 생성, skill 전환 판단을 VLM에 supervised fine-tuning하는 부분은 1차 실험 범위에서 제외한다. 대신 pretrained SmolVLM의 planning 능력을 동결한 채 사용하고, 학습 대상은 action policy의 shared expert, 5개 skill expert, skill router, state/action projection으로 제한한다.

## 2. 중심 가설

### H1: Atomic action specialization

동일한 frozen VLM 표현을 사용하는 조건에서, skill boundary를 넘지 않는 action chunk를 의미 있는 atomic skill 단위로 top-1 라우팅하면 단일 action expert에서 발생하는 skill 간 gradient interference가 줄어 LIBERO-10 평균 task success rate가 증가한다.

### H2: Frozen-VLM routing

SmolVLM을 추가 학습하지 않아도 현재 관측, 전체 task instruction, 초기 task plan, 직전 skill을 함께 주고 5개 허용 skill의 조건부 점수를 비교하면 현재 skill을 실용적으로 선택할 수 있다. 이 선택 결과로 SG-MoE를 라우팅했을 때 oracle skill label을 사용한 상한 성능과의 차이가 충분히 작을 것이다.

### H3: Long-horizon composition

`pick/place/turn/open/close` 전문가를 분리하면 각 skill의 실행 정밀도뿐 아니라 skill transition 이후의 오류 누적이 감소한다. 따라서 이득은 single-stage task보다 여러 skill을 순서대로 수행하는 LIBERO-10에서 더 크게 나타날 것이다.

## 3. 논문 이해와 본 실험의 차이

AtomicVLA는 VLM이 매 시점 `[think]` 또는 `[act]`를 예측한다. `[think]`에서는 task chain, 현재 진행 단계, atomic skill을 생성하고, `[act]`에서는 가장 최근 atomic skill로 action expert를 라우팅한다. SG-MoE는 shared expert와 skill-specific expert의 출력을 결합하며, 동일 skill stage의 모든 action token이 같은 expert를 사용한다.

본 실험은 다음과 같이 의도적으로 제한한다.

1. VLM, vision encoder, token embedding, LM head를 모두 동결한다.
2. `[think]`/`[act]` 전환 loss와 reasoning text loss를 사용하지 않는다.
3. 학습 시 dataset의 ground-truth atomic skill로 action chunk를 라우팅한다.
4. 평가 시에는 (a) oracle routing과 (b) frozen-VLM routing을 모두 측정한다.
5. oracle은 action-policy 가설의 상한을, frozen-VLM routing은 실제 배포 가능한 전체 시스템을 검증한다.

따라서 결과를 AtomicVLA 전체 재현으로 주장하지 않고 **AtomicVLA-style action-only SG-MoE for SmolVLA**로 명시한다.

## 4. 데이터셋 결정

### 결정: `k1000dai/libero_subtask_sarm`

학습 데이터는 [`k1000dai/libero_subtask_sarm`](https://huggingface.co/datasets/k1000dai/libero_subtask_sarm)을 사용하고 revision을 고정한다. 현재 공개 revision 기준으로 이 데이터셋은 LeRobot v3 형식의 1,693 episodes, 273,465 frames, 40 tasks, 10 Hz 데이터이며 다음 정보를 이미 제공한다.

- frame별 `subtask_index`
- 52개 자연어 subtask와 index의 대응표인 `meta/subtasks.parquet`
- episode별 dense subtask 이름과 start/end frame
- 두 카메라, 8차원 state, 7차원 action

### `openvla/modified_libero_rlds`를 주 데이터로 선택하지 않는 이유

[`openvla/modified_libero_rlds`](https://huggingface.co/datasets/openvla/modified_libero_rlds)는 OpenVLA용 RLDS 형식의 네 LIBERO suite를 제공하지만 atomic skill label과 skill boundary를 포함하지 않는다. AtomicVLA 공식 저장소에는 1,693 episodes에 대한 [`data_split_json/libero_lerobot.json`](https://github.com/zhanglk9/AtomicVLA/blob/main/data_split_json/libero_lerobot.json)이 있으나, 공개 학습 config는 로컬 `repo_id="libero_lerobot"`와 저장소에 없는 `output_report_test.json`을 가리키고, OpenVLA RLDS를 해당 episode ordering의 LeRobot 데이터로 변환·검증하는 완결된 경로를 제공하지 않는다. 이 상태에서 별도 변환을 작성하면 episode/frame 정렬 오류가 실험의 가장 큰 위험이 된다.

반면 SARM 데이터는 SmolVLA가 사용하는 LeRobot loader와 직접 호환되고 frame-level skill label이 이미 있으므로, 모델 가설을 데이터 변환 가설과 분리할 수 있다.

### 공식 annotation의 사용

공식 `libero_lerobot.json`은 학습의 단일 진실 원천으로 사용하지 않고 다음 사전 감사에 사용한다.

- episode 수, task instruction, frame 수가 SARM 데이터와 일치하는지 확인
- 두 annotation의 5-skill label agreement와 boundary IoU 측정
- disagreement가 큰 episode를 영상으로 표본 검사
- SARM이 uniform-time fallback을 사용한 episode를 별도 표기

SARM dataset card는 1,674 episodes에 VLM-localized boundary가 사용되었고 중복 subtask 또는 localization 실패와 관련된 약 48 cases에는 uniform split이 사용되었다고 기술한다. 그러나 두 수는 전체 1,693 episodes와 산술적으로 일치하지 않으므로 서로 배타적인 episode 수로 해석하지 않는다. 구현 전에 episode metadata에서 annotation source를 직접 집계한다. 실제 uniform-split episode를 식별할 수 있으면 기본 학습에는 포함하되, 이를 제외한 sensitivity run을 추가한다. 식별할 수 없으면 이 데이터 품질 한계를 명시하고 공식 AtomicVLA annotation과의 boundary agreement를 기준으로 sensitivity subset을 정의한다.

## 5. 52개 subtask를 5개 skill로 매핑

원래 자연어 subtask와 5-skill label을 모두 보존한다. 문자열을 소문자화하고 공백을 정규화한 뒤 다음 순서의 명시적 규칙을 적용한다.

| 원래 subtask의 동사 | atomic skill | 의미 |
|---|---|---|
| `pick up ...` | `pick` | 접근, grasp, lift |
| `place ...` / `put ...` | `place` | 운반, 정렬, release |
| `turn ...` | `turn` | stove knob 회전 |
| `open ...` | `open` | drawer/door 열기 |
| `close ...` | `close` | drawer/door 닫기 |
| `push the plate ...` | `place` | 5-skill 제약 아래 목표 위치로의 translational placement로 취급 |

`push`를 `place`에 넣는 것은 유일한 비동의어 예외다. 이 샘플을 제외한 sensitivity run으로 결과 의존성을 확인한다. 알려지지 않은 동사는 임의 class로 보내지 않고 preprocessing을 실패시킨다. 학습 전 다음 조건을 반드시 검증한다.

- 52개 subtask의 mapping coverage가 100%일 것
- 모든 frame에 정확히 하나의 atomic skill이 있을 것
- episode 내부 segment가 겹치거나 비지 않고 frame 범위를 덮을 것
- skill별 episode/frame 수와 class imbalance를 기록할 것
- 원본 `subtask_index`, 자연어 subtask, `atomic_skill_id`를 모두 보존할 것

## 6. 모델 설계

### 6.1 초기화와 동결 범위

- `lerobot/smolvla_base`에서 VLM과 pretrained action expert를 불러온다(`load_vlm_weights=True`).
- VLM의 vision encoder, connector, text transformer, embeddings, normalization, LM head 전체에 `requires_grad=False`를 적용하고 항상 eval mode를 유지한다.
- optimizer에는 action-policy parameter만 전달한다.
- 학습 전후 VLM state hash와 trainable-parameter manifest를 저장하여 VLM이 실제로 변하지 않았음을 검증한다.

### 6.2 SmolVLA SG-MoE

SmolVLA action expert의 attention/cross-attention 경로는 공유하고, 각 transformer layer의 FFN을 다음으로 교체한다.

- pretrained shared FFN 1개
- `pick/place/turn/open/close` 전용 FFN 5개
- 고정 atomic skill embedding과 학습 가능한 top-1 router
- shared FFN과 선택된 skill FFN의 gated weighted sum

이 구조는 action expert 전체를 여섯 벌 복제하지 않으면서도 AtomicVLA 공식 구현처럼 skill specialization을 FFN 수준에 적용한다. 모든 skill FFN은 pretrained action FFN weight에서 복사 초기화한다. 따라서 학습 시작 시 dense checkpoint의 동작을 최대한 보존하고, 이후 skill별로 분화할 수 있다.

한 action chunk의 모든 token과 모든 MoE layer에는 동일한 skill route를 사용한다. token별 routing은 금지한다. 공식 구현처럼 각 skill ID를 고정된 비학습 embedding에 대응시키고, 학습 가능한 linear router를 scaled-identity로 초기화해 top-1 expert와 shared/skill 결합 weight를 계산한다. 별도의 router classification loss나 load-balancing loss는 두지 않는다. Router는 flow-matching action loss의 gradient로 action expert와 함께 end-to-end 학습한다. Skill 불균형은 router loss를 추가하지 않고 skill-balanced batch sampling으로 처리한다.

여기서 **skill selector**와 **skill router**를 구분한다. VLM/planner는 관측으로부터 현재 atomic skill을 선택하며, SG-MoE router는 이미 선택된 atomic skill의 고정 embedding을 어떤 action expert와 어떤 결합 weight로 보낼지 결정한다. 학습 시 전자에는 dataset skill label을 사용하고, 후자만 action loss로 학습한다.

### 6.3 Skill boundary를 넘는 action chunk 방지

현재 frame의 skill만 보고 미래 50-frame action chunk를 그대로 구성하면 다음 skill의 action이 현재 expert loss에 섞인다. 이는 핵심 가설을 훼손하므로 dataset sampler는 action target을 현재 dense segment의 end frame에서 자르고, 나머지를 `action_is_pad`로 mask한다. 학습 loss와 normalization 통계 모두 이 mask를 존중해야 한다.

LIBERO는 10 Hz이고 AtomicVLA 공개 config는 action horizon 10을 사용하므로 첫 후보는 10-step chunk/10-step execution이다. 최종 horizon과 replanning 주기는 `03_experiment_plan.md`에서 compute/latency 측정 후 고정한다.

## 7. Frozen VLM을 이용한 skill 유지/전환

Think–Act classifier를 학습하지 않는다. 대신 동결된 SmolVLM의 기존 image-text generation head를 같은 weight로 사용한다.

1. task 시작 시 전체 instruction에서 허용된 ordered atomic plan을 한 번 생성한다.
2. action chunk를 요청할 때마다 현재 두 camera image, 전체 instruction, plan, 직전 skill, 최근 skill history를 입력한다.
3. 자유 형식 문장을 생성하지 않고 `pick/place/turn/open/close` 다섯 candidate의 conditional log-likelihood를 계산해 argmax를 선택한다.
4. 짧은 관측 잡음으로 expert가 진동하지 않도록 validation set에서 고정한 margin과 연속-confirmation hysteresis를 적용한다.
5. drop 또는 실행 실패 후 이전 skill로 돌아갈 수 있도록 후보를 현재/이전/다음 단계로 제한하되 recovery를 허용한다.

이 방식은 VLM parameter를 전혀 학습하지 않으면서도 “현재 skill 유지 또는 다음 skill 전환”을 관측에 기반해 다시 판단한다. 별도 VLM 복사본을 두지 않고 SmolVLA 내부 frozen VLM weight를 공유한다.

정책 학습 전에 held-out episodes에서 frozen-VLM router의 frame macro-F1, transition boundary F1(허용 오차 포함), skill별 confusion matrix를 측정한다. 이 검사는 VLM을 학습시키기 위한 것이 아니라 H2가 성립할 가능성을 사전 진단하기 위한 것이다.

## 8. 학습 및 평가 범위

### 주 실험

- 학습 데이터: `k1000dai/libero_subtask_sarm`의 전체 학습 풀(1,693 episodes, 40 LIBERO tasks)을 사용한다. 이 40 tasks는 `LIBERO-Spatial`, `LIBERO-Object`, `LIBERO-Goal`, `LIBERO-Long(LIBERO-10)` 각 10 tasks로 구성되므로 LIBERO-10도 학습 데이터에 포함된다.
- canonical skill/expert: SARM의 52개 자연어 subtask를 `{pick, place, push, turn, open, close}`로 mapping하고 정확히 6개의 독립 action expert를 둔다. `push` label과 push 동작 문구는 `place`로 합치지 않는다. 이 문서에서 5-skill/5-expert 또는 `push → place`라고 적힌 과거 설계는 모두 폐기한다.
- action-policy 학습 routing: 각 training chunk에는 SARM의 ground-truth subtask에서 얻은 6-way skill label을 fixed skill embedding으로 변환해 SG-MoE router에 입력한다. Router와 shared/skill-specific action expert는 flow-matching action loss로 함께 학습하며 별도의 router classification loss는 사용하지 않는다. Frozen VLM에는 gradient를 전달하지 않는다.
- frozen-VLM 추론 입력: planning query마다 현재 main-camera image, wrist-camera image, 전체 high-level task instruction, 직전 skill, 허용된 6개 skill의 이름과 의미를 입력한다. 출력은 `{decision: continue|switch, skill: pick|place|push|turn|open|close}`로 제한한다. Skill 문자열은 training과 동일한 fixed embedding으로 변환하고, 학습된 SG-MoE router가 top-1 skill expert와 결합 weight를 결정한다.
- AtomicVLA와의 차이: 원 논문의 학습된 VLM은 현재 multi-view observation과 전체 task instruction에서 `[think]`/`[act]`를 선택한다. `[think]`이면 전체 task chain, 현재 progress와 atomic skill `σ`를 생성하고, skill router는 raw image가 아니라 `σ`의 fixed embedding을 받아 top-1 expert를 선택한다. 본 실험은 VLM을 freeze하므로 이 학습된 Think-Act head를 재현한다고 주장하지 않으며, 위 prompt-based `continue|switch` 결정을 사용한다.
- temporal configuration: `chunk_size=10`, `n_action_steps=5`로 고정한다. SARM은 10 FPS이므로 action policy는 1초 길이의 chunk를 예측하되 0.5초만 실행하고 새 observation에서 다시 추론한다.
- boundary handling: anchor는 현재 SARM skill로 라우팅한다. 10-step target이 다음 skill 또는 episode 끝을 넘어가면 경계 이후 action position을 padding/loss mask 처리한다. 경계 때문에 anchor 전체를 버리지 않는다.
- 학습 sampling: episode split과 label quality filtering을 먼저 수행한다. 이후 valid action-chunk anchor를 `{pick, place, push, turn, open, close}`로 나누고, `P(skill)=1/6`로 skill을 고른 뒤 해당 skill의 anchor를 균등 복원추출한다. validation/test에는 oversampling이나 loss reweighting을 적용하지 않는다.
- short-horizon 실행 평가: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal의 30 tasks를 task당 50 rollouts로 평가한다. Task별 success rate를 측정해 각 task/context에서 action policy가 동작을 정확히 실행하는지 검증하며, 세 suite를 하나의 수치로만 합치지 않고 suite별·task별 결과와 95% confidence interval을 모두 보고한다. 이 30 tasks가 모두 단일 atomic skill이라고 가정하지 않는다.
- composition 평가: LIBERO-Long(LIBERO-10)의 10 long-horizon tasks를 task당 50 rollouts로 평가한다. 전체 task success rate, 순서대로 완료한 atomic subgoal 수, skill-transition 성공률을 측정한다. 이는 이미 학습된 atomic skill을 frozen-VLM routing으로 전환·조합하여 긴 task를 끝까지 수행할 수 있는지를 평가한다. SARM 학습 풀에 동일한 LIBERO-Long task demonstrations가 포함되므로 이 결과를 unseen-task 또는 novel-composition generalization으로 주장하지 않고, in-distribution closed-loop composition/execution 성능으로 해석한다.
- skill별 offline 평가: held-out natural-distribution split에서 skill 경계를 넘지 않는 chunk만 사용해 각 skill의 action error와 표본 수를 각각 보고하고, skill macro-average와 sample-weighted micro-average를 함께 보고한다.
- skill별 진단 평가: 주 결과인 task-level 평가는 LIBERO-Spatial/Object/Goal의 각 task success와 LIBERO-Long composition success로 구성한다. 추가 진단으로 각 subtask 진입 상태에서 ground-truth skill로 해당 expert를 context당 50회 실행하고 simulator의 subgoal predicate로 성공을 판정한다. skill별 성공률은 task/context별 성공률을 먼저 구한 뒤 macro-average하고 95% Wilson confidence interval을 보고한다. 전체 rollout에서는 `성공한 skill / 도달한 skill`도 기록해 앞 단계 실패를 뒤 skill의 실패로 세지 않는다.
- frozen-VLM 전체 시스템 평가는 expert 실행 성공률과 섞지 않고, 6-way skill 선택 정확도와 confusion matrix(`pick/place/push/turn/open/close`), `continue|switch` boundary F1, LIBERO-10 end-to-end success로 별도 보고한다. 특히 희소한 `push` recall과 push→place 오분류율을 독립적으로 보고한다.
- validation: task별 episode 단위 분할; 같은 episode의 frame이 train/validation에 동시에 들어가지 않음
- 평가: 표준 LIBERO-10 simulator initial states와 task success 판정
- 모든 비교 조건에서 VLM checkpoint, dataset, normalization, optimizer budget, action horizon, image inputs, evaluation initial states를 동일하게 유지

### 후속 compositional-transfer 실험

나머지 30개 LIBERO task로 5개 expert를 먼저 학습한 뒤 LIBERO-10에 적용하거나 동일 budget으로 fine-tuning한다. 이 결과는 주 실험과 섞지 않고 “short-skill pretraining → long-horizon composition”으로 별도 보고한다.

## 9. 필수 비교 조건

1. **Dense SmolVLA:** frozen VLM + 단일 action expert fine-tuning
2. **Parameter-matched dense:** MoE의 총 trainable parameter 증가 효과를 분리하기 위한 wider/deeper dense action expert
3. **SG-MoE + oracle routing:** ground-truth skill을 사용하는 action-policy 상한
4. **SG-MoE + frozen-VLM routing:** 실제 제안 시스템
5. **SG-MoE + shuffled skill labels:** semantic routing 자체의 효과 검증
6. **Shared-only ablation:** 동일 checkpoint에서 skill FFN을 비활성화
7. **No-boundary-mask ablation:** chunk contamination 방지의 효과 검증

## 10. 평가 지표

### Primary

- LIBERO-10 평균 task success rate

### Secondary

- task별 success rate
- completed atomic skills / required atomic skills
- skill transition 성공률과 transition 이후 실패율
- frozen-VLM skill-selector macro-F1 및 boundary F1
- oracle routing과 frozen routing의 success gap
- action flow-matching validation loss와 skill별 loss
- expert utilization, router confidence, route-switch 빈도
- trainable/total parameter 수, peak VRAM, training throughput, inference latency

최종 seed, rollout 수, 통계 검정, confidence interval, compute budget은 결과를 보기 전에 `03_experiment_plan.md`에서 동결한다. 예비 목표는 3 training seeds와 task당 seed별 50 simulator rollouts이다.

## 11. 사전 판정 원칙

H1은 SG-MoE + oracle routing이 Dense SmolVLA보다 LIBERO-10 평균 성공률에서 재현 가능한 개선을 보일 때 지지된다. H2는 frozen-VLM routing이 oracle routing의 이득 대부분을 유지할 때 지지된다.

- oracle SG-MoE가 dense baseline을 이기지 못하면 action specialization 가설은 기각한다.
- oracle은 이기지만 frozen-VLM routing이 이기지 못하면 H1은 지지되지만 H2는 기각한다. 이 경우 원인은 action expert가 아니라 frozen planner/routing이다.
- parameter-matched dense와 차이가 없으면 MoE의 이득을 semantic specialization으로 주장하지 않는다.
- shuffled label도 같은 수준이면 atomic skill 의미가 아니라 parameter 증가 또는 regularization 효과로 해석한다.
- annotation disagreement 또는 `push→place` 제외 여부에 따라 결론이 뒤집히면 결과를 inconclusive로 보고한다.

구체적인 최소 효과 크기와 통계적 falsification threshold는 raw 결과를 보기 전에 `03_experiment_plan.md`에서 고정한다.

## 12. 구현 완료 조건

- VLM parameter hash가 학습 전후 동일함
- optimizer에 VLM parameter가 0개임
- 52개 subtask mapping coverage 100%
- action target이 skill boundary를 넘지 않음
- 같은 chunk의 모든 token/layer가 동일 expert를 사용함
- oracle skill을 바꾸면 선택 expert가 결정적으로 바뀌는 unit test 통과
- shared/skill gate 합과 tensor shape/dtype 검증 통과
- dense checkpoint에서 복사 초기화 직후 출력이 허용 오차 안에서 보존됨
- train/eval episode 및 simulator seed 누수 검사 통과
- 모든 raw rollout 결과와 routing trace를 append-only로 저장

## 13. 주요 위험

1. **Frozen planner 한계:** SmolVLM이 manipulation progress를 충분히 인식하지 못할 수 있다. oracle/frozen 평가 분리로 원인을 식별한다.
2. **Annotation noise:** SARM boundary 일부는 VLM localization 또는 uniform split이다. 공식 JSON agreement와 sensitivity run으로 측정한다.
3. **Skill imbalance:** `pick/place`가 `turn/open/close`보다 많다. skill-balanced sampling과 skill별 metric을 사용한다.
4. **`push→place` 의미 손실:** 해당 샘플 제외 ablation을 수행한다.
5. **추가 parameter 효과:** parameter-matched dense와 shuffled-label 조건 없이는 SG-MoE 효과로 주장하지 않는다.
6. **전환 지연:** 긴 execution horizon은 VLM이 올바르게 판단해도 전환을 늦춘다. horizon과 hysteresis를 validation에서 고정하고 latency와 함께 보고한다.

## 14. 근거 자료

- [AtomicVLA paper](https://arxiv.org/pdf/2603.07648)
- [AtomicVLA official implementation](https://github.com/zhanglk9/AtomicVLA)
- [AtomicVLA official LIBERO annotations](https://github.com/zhanglk9/AtomicVLA/blob/main/data_split_json/libero_lerobot.json)
- [OpenVLA modified LIBERO RLDS](https://huggingface.co/datasets/openvla/modified_libero_rlds)
- [LIBERO SARM subtask dataset](https://huggingface.co/datasets/k1000dai/libero_subtask_sarm)
- [SmolVLA implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla)
> **확정 설계 (2026-08-12):** canonical atomic skill set은
> `{pick, place, push, turn, open, close}`이며, action policy에는 이에 대응하는
> **6개 독립 skill expert**를 둔다. `push`는 `place`와 병합하지 않는다.
> 모든 `push ...` SARM subtask는 독립 `push` skill/expert로만 라우팅한다.
>
> Push 데이터는 33 episodes, 4,990 action frames(10 FPS; 전체의 약 1.82%)다.
> 학습에서는 quality filtering 뒤의 valid action-chunk anchor를 skill별로 묶고,
> 먼저 6개 skill을 균등하게 뽑은 다음 해당 skill 안에서 anchor를 균등 복원추출한다.
> 즉 anchor `i`의 가중치는 `1 / n_skill(i)`이며, loss reweighting은 중복 적용하지 않는다.
> validation/test는 episode 단위로 training과 분리하고 원래 분포를 유지한다.
>
> 주 모델은 SmolVLA action decoder의 각 대상 FFN에 shared expert와
> `{pick, place, push, turn, open, close}`의 6개 skill-specific expert를 두는
> **Skill-Guided Mixture-of-Experts(SG-MoE)**다. 매 action token은 shared expert와
> 선택된 top-1 skill expert를 함께 통과하며, 두 출력을 결합해 action chunk를 예측한다.
> 주 가설은 frozen VLM을 유지한 SG-MoE가 single shared action policy보다
> LIBERO-Spatial/Object/Goal의 short-horizon task execution과 LIBERO-Long의
> atomic-skill composition 성공률을 높인다는 것이다.
