# Atomic SmolVLA 문헌 검토

## 1. 검토 목적

이 문서는 `00_hypothesis.md`의 가설을 구현 가능한 주장으로 좁히기 위해 다음 질문을 검토한다.

1. AtomicVLA의 SG-MoE는 무엇을 입력받고 무엇을 학습하는가?
2. SARM 데이터가 6개 atomic skill 학습과 LIBERO 평가에 적합한가?
3. VLM을 동결한 SmolVLA에 SG-MoE를 붙였을 때 무엇을 검증할 수 있는가?
4. 논문과 공개 코드에서 재현되지 않거나 불명확한 부분은 무엇인가?

검토 근거는 [AtomicVLA 논문](https://arxiv.org/pdf/2603.07648), [AtomicVLA 공식 구현](https://github.com/zhanglk9/AtomicVLA), [SmolVLA 논문](https://arxiv.org/pdf/2506.01844), [LeRobot 공식 SmolVLA 문서](https://huggingface.co/docs/lerobot/smolvla), [SARM 데이터 카드](https://huggingface.co/datasets/k1000dai/libero_subtask_sarm), [SARM 논문](https://arxiv.org/abs/2509.25358), [LIBERO 논문](https://arxiv.org/abs/2306.03310)과 [LIBERO 공식 구현](https://github.com/Lifelong-Robot-Learning/LIBERO)을 우선 사용한다.

## 2. 결론 요약

- AtomicVLA는 subtask마다 별도 모델을 학습하지 않는다. 한 데이터 스트림에서 각 프레임이 속한 atomic skill을 정하고, shared FFN과 선택된 skill FFN을 함께 실행하는 SG-MoE를 공동 학습한다.
- LIBERO의 공식 AtomicVLA는 `Pick`, `Place`, `Open`, `Close`, `Turn`의 5개 expert를 쓴다. 본 실험은 SARM에 명시적으로 존재하는 `Push`를 `Place`에 합치지 않고 여섯 번째 expert로 둔다.
- SG-MoE router의 입력은 raw image가 아니라 VLM이 산출한 현재 atomic skill의 고정 embedding이다. image와 instruction으로 현재 skill을 판단하는 부분과, 주어진 skill로 expert를 고르는 부분은 구분해야 한다.
- 논문은 희소 skill의 sampling frequency를 늘려 균형화했다고 설명하지만, 공개 `atomic_dataset.py`와 `data_loader.py`에는 정확한 확률이나 별도 weighted sampler가 드러나지 않는다. 재현 가능한 실험을 위해 본 연구는 `P(skill)=1/6`을 명시적으로 고정한다.
- SARM은 1,693 episodes, 273,465 frames, LIBERO 40 tasks를 담고 있으며 Spatial/Object/Goal/Long 각 10개 task를 모두 포함한다. 52개 문자열은 suite 이름이 아니라 네 suite 전체에서 수집한 subtask 표현의 합집합이다.
- SARM의 경계는 완전한 ground truth가 아니다. 1,674 episodes는 VLM-localized boundary를 사용하고, 중복 subtask나 localization 실패 사례에는 uniform temporal split이 섞여 있다. 자동 무결성 검사와 provenance 기반 제외 manifest가 필요하다.
- SmolVLA의 공식 설정은 VLM 전체를 동결하고 action expert만 학습하는 경로를 이미 제공한다. 다만 표준 SmolVLA inference는 skill text를 생성하지 않으므로, frozen VLM routing은 별도의 deterministic planning/generation 경로가 필요하다.
- Long demonstrations도 학습에 포함되므로 Long 결과는 새로운 task 조합에 대한 일반화가 아니다. 같은 benchmark distribution 안에서 여섯 atomic skill을 시간적으로 연결하는 폐루프 composition 능력을 측정한다.

## 3. AtomicVLA 방법

### 정량 근거와 반례

| 근거 | 발표 결과 | 본 실험에 주는 의미 |
|---|---|---|
| AtomicVLA Table 1 | LIBERO-Long: π0 85.2%, π0.5 92.4%, AtomicVLA 95.2%, AtomicVLA* 96.2% | semantic SG-MoE의 가장 가까운 직접 근거지만 backbone과 VLM 학습 조건이 다름 |
| AtomicVLA Table 5 | Long: non-MoE 85.2%, token MoE 88.6%, MoDE 89.5%, SG-MoE 95.2% | SG-MoE가 non-MoE보다 +10.0%p였으나 seed/CI와 capacity-matched 비교가 없음 |
| AtomicVLA Table 10 | π0 3.24B/71ms, K=5 SG-MoE 4.17B/92ms | specialization과 추가 parameter/latency가 confounded되므로 자원 지표를 함께 보고해야 함 |
| SmolVLA Table 2 | Spatial 90, Object 96, Goal 92, Long 71, 평균 87.3% | frozen-VLM action policy는 가능하지만 Long은 상대적으로 약함 |
| SmolVLA Table 10 | flow matching 80.25%, L1 regression 75.25% | 기존 flow-matching objective 유지의 근거 |
| SmolVLA Table 12 | chunk 1/10/30/50/100 중 10이 84.0%로 최고 | `chunk_size=10`의 직접 근거 |
| SmolVLA Table 13 | observation 갱신 전 실행 1/10/30/50 steps: 80.3/82.8/70.8/51.8% | `n_action_steps=5`는 합리적 중간값이지만 논문에서 직접 시험된 값은 아님 |
| LIBERO 원 논문 | architecture와 language embedding 효과가 suite에 따라 달랐고 단순 pretraining이 항상 이득이 아니었음 | modularity의 이득을 전제하지 말고 task별 결과와 불확실성을 보고해야 함 |
| [LIBERO-PRO](https://arxiv.org/abs/2510.03827) | 표준 LIBERO 90% 이상 모델이 perturbation에서 0%까지 하락 가능 | 표준 Long 성능을 novel generalization으로 확대 해석하면 안 됨 |

AtomicVLA 공식 `pi0_atomic.py`의 전체 objective는 action flow loss뿐 아니라 decision/reasoning text loss도 포함한다. “router 전용 CE가 없다”와 “전체 모델이 action loss만 쓴다”는 같은 말이 아니다. 본 실험은 text loss와 VLM update를 모두 제거하므로 AtomicVLA 재현이 아니라 **SG-MoE를 frozen SmolVLA에 이식한 새로운 ablation**이다.

SmolVLA 원 논문의 LIBERO 평가는 task당 10 rollout이고 AtomicVLA는 task당 50 rollout이다. 두 논문의 point estimate에는 training-seed 분산과 confidence interval이 충분히 제시되지 않으므로, 본 실험은 task당 50 paired rollout과 세 training seed의 계층적 불확실성을 직접 계산한다.

### 3.1 trajectory를 atomic segment로 바꾸는 방법

AtomicVLA는 전체 demonstration을 독립된 subtask dataset으로 쪼개 학습하는 방식이 아니다. 먼저 end-effector translation, rotation, gripper 신호의 주성분을 이용해 후보 경계를 만들고, video-language model로 의미적 경계와 skill 이름을 보정한다. 각 frame은 시간적으로 정렬된 segment와 현재 atomic skill label에 연결된다.

따라서 본 실험에서 SARM의 `subtask_index`를 사용하는 것은 AtomicVLA의 분할 단계를 이미 주어진 annotation으로 대체하는 것이다. 이 대체가 유효하려면 segment 순서, 범위, label mapping과 경계 품질을 먼저 검사해야 한다.

### 3.2 think/act와 skill 판단

AtomicVLA의 planning 경로는 현재 multi-camera observation과 전체 language instruction을 입력으로 받는다. `[think]` 단계는 task chain, 현재 진행 상황, 실행할 atomic skill을 갱신하고, `[act]` 단계는 가장 최근 skill을 조건으로 action을 생성한다.

여기에는 서로 다른 두 판단이 있다.

1. **VLM skill inference**: image, instruction, 진행 맥락으로 현재 skill 또는 continue/switch를 판단한다.
2. **SG-MoE expert routing**: 선택된 skill의 고정 embedding으로 대응 expert를 고른다.

본 가설은 첫 번째 판단을 학습하지 않는다. SmolVLA의 VLM은 동결하고 inference prompt만 사용한다. 반면 두 번째 경로의 router와 action expert는 flow-matching action loss로 학습한다.

### 3.3 SG-MoE

AtomicVLA의 SG-MoE는 action expert transformer의 FFN을 다음 구조로 바꾼다.

- 항상 활성화되는 shared expert
- atomic skill별 독립 SwiGLU FFN
- fixed skill embedding을 입력받는 router
- top-1 skill expert와 shared expert 출력의 결합

개념적으로 출력은 다음과 같다.

\[
h'=(1-w_k)F_{shared}(h)+w_kF_k(h), \qquad
k=\operatorname{argmax}(\operatorname{softmax}(R(e_s)))
\]

여기서 `e_s`는 skill `s`의 고정 embedding이다. 각 skill expert는 별도의 policy나 별도 transformer가 아니라 동일 action transformer layer 안의 독립 FFN이다. 본 실험도 같은 구조를 유지하되 expert 집합을 `pick/place/push/turn/open/close`로 확장한다.

논문과 공개 구현에서 router 전용 분류 loss는 확인되지 않는다. 따라서 별도 router cross-entropy를 추가하지 않고 action flow-matching loss가 router gate와 실행된 expert를 학습하게 한다. 단, top-1의 비미분 선택 때문에 초기 expert 정체성이 뒤섞이지 않도록 skill-to-expert 순서를 고정하고 초기화 회귀 검사를 둬야 한다.

### 3.4 skill imbalance 처리

AtomicVLA 논문이 보고한 LIBERO skill segment 수는 다음과 같이 크게 불균형하다.

| Skill | Segment 수 |
|---|---:|
| Pick | 2,462 |
| Place | 761 |
| Open | 201 |
| Close | 152 |
| Turn | 175 |

논문은 `Open`, `Close`, `Turn`의 sampling frequency를 높여 skill 분포를 균등화한다고 설명한다. 그러나 공개 loader는 일반적인 `DataLoader`/`DistributedSampler` 경로를 사용하며, 논문과 정확히 같은 sampling 비율·seed·replacement 알고리즘을 재현할 정보는 부족하다. 따라서 “공식 sampler의 정확한 재현”을 주장하지 않고, 다음의 명시적 대안을 사용한다.

\[
P(s)=1/6,\qquad P(i\mid s)=1/N_s
\]

즉 먼저 여섯 skill을 균등하게 고른 뒤 그 skill에 속한 유효 anchor frame을 복원 추출한다. validation과 offline test는 자연 분포를 유지한다.

## 4. 데이터셋 검토

### 4.1 `modified_libero_rlds`

[openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)는 LIBERO demonstration을 RLDS 형태로 제공해 일반 action imitation에는 적합하다. 그러나 AtomicVLA에 필요한 segment boundary, frame별 subtask index, AtomicVLA 형식의 semantic skill annotation을 완성하는 공식 변환 경로가 공개 저장소에 충분히 고정되어 있지 않다.

이 데이터를 쓰면 먼저 AtomicVLA의 motion decomposition과 semantic refinement를 재현해야 한다. 그것은 현재 가설인 “action-policy SG-MoE 효과”와 “annotation pipeline 품질”을 동시에 바꾸는 추가 변수가 된다. 따라서 주 학습 데이터로 채택하지 않는다.

### 4.2 `libero_subtask_sarm`

[k1000dai/libero_subtask_sarm](https://huggingface.co/datasets/k1000dai/libero_subtask_sarm)는 LeRobot 형태의 다음 정보를 제공한다.

- 1,693 episodes, 273,465 frames
- LIBERO 40 tasks
- `observation.images.image`, `observation.images.image2`
- 8차원 state, 7차원 action
- frame별 `subtask_index`
- 52개 natural-language subtask vocabulary
- 10 Hz trajectory

40 tasks는 [공식 LIBERO task map](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/libero_suite_task_map.py)의 `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` 각 10개다. [`temporal_proportions_dense.json`](https://huggingface.co/datasets/k1000dai/libero_subtask_sarm/blob/main/meta/temporal_proportions_dense.json)은 이 네 suite에서 나타난 52개 subtask 문자열의 dataset-wide 평균 시간 비율이다. 이 파일 자체가 suite membership을 뜻하지 않는다.

SARM 카드에 따르면 1,674 episodes는 Qwen3-VL-8B로 localization한 경계를 사용한다. 중복 subtask와 localization 실패 또는 개수 불일치 사례에는 uniform temporal split이 사용되었다. 카드에 제시된 “약 48 cases”는 episode 수와 단순 합산할 수 있는 상호 배타 집합으로 명확히 정의되어 있지 않으므로, 숫자만으로 clean subset을 추정해서는 안 된다.

### 4.3 데이터 품질 조건

학습 전에 episode 단위 manifest를 만들고 다음을 강제한다.

- 모든 segment가 episode 범위 안에 있고 시간순이며 겹치지 않는다.
- zero/negative-length segment가 없다.
- 모든 frame의 `subtask_index`가 유효한 vocabulary entry를 가리킨다.
- 52개 문자열 각각이 정확히 하나의 canonical skill에 매핑된다.
- uniform fallback provenance가 제공되면 해당 episode를 주 학습에서 제외한다.
- provenance가 없으면 uniform-like boundary를 자동 삭제하지 않고 `suspect`로 격리해 수량과 사례를 보고한다.
- motion과 label이 강하게 모순되는 구간은 자동 재라벨링하지 않고 검토 manifest에 남긴다.

잘못된 label을 motion heuristic 하나로 자동 삭제하면 slow manipulation, contact 유지, camera ambiguity를 오검출할 수 있다. 구조적 오류는 자동 제외하고 의미적 오류는 보수적으로 격리하는 것이 타당하다.

### 4.4 6개 skill taxonomy

52개 문자열은 versioned mapping artifact에서 다음 여섯 class로 완전 매핑한다.

| Canonical skill | 포함 의미 | 금지되는 병합 |
|---|---|---|
| `pick` | 물체 접근 후 grasp/들기 | `push` 포함 금지 |
| `place` | 든 물체를 목표 위치에 놓기 | `push` 포함 금지 |
| `push` | grasp 없이 접촉해 물체를 밀기 | `place`로 축약 금지 |
| `turn` | stove knob 등 회전 조작 | `open/close`와 혼합 금지 |
| `open` | drawer, cabinet, microwave 등을 열기 | `turn`과 혼합 금지 |
| `close` | drawer, cabinet, microwave 등을 닫기 | `turn`과 혼합 금지 |

알 수 없는 label의 fallback class는 두지 않는다. mapping coverage가 52/52가 아니면 학습을 시작하지 않는다.

## 5. SmolVLA 적용 가능성

SmolVLA는 multi-view image, sensorimotor state, instruction을 VLM으로 문맥화하고 더 작은 action expert transformer가 flow matching으로 action chunk를 생성한다. 공식 설정의 `train_expert_only=True`는 VLM 전체를 동결하는 본 가설과 일치한다.

다만 다음 차이가 있다.

- SmolVLA의 기본 `chunk_size`와 실행 horizon은 본 실험의 10/5와 다르므로 명시적으로 덮어써야 한다.
- 표준 forward는 action을 생성하지만 현재 skill text를 출력하지 않는다. 동일 frozen VLM에 대해 별도 planning prompt와 deterministic parser가 필요하다.
- SG-MoE는 VLM FFN이 아니라 `lm_expert`의 FFN에만 적용해야 한다.
- `eval()` 상태와 `requires_grad=False`를 모두 보장하고 optimizer parameter list에도 VLM이 없어야 한다.

SmolVLA의 VLM이 six-way routing을 잘할지는 사전 보장할 수 없다. SmolVLM2-500M-Video-Instruct의 일반 시각·언어 능력은 유용한 prior이지만, robot state progress와 미세한 manipulation boundary는 별도 문제다. 따라서 frozen-VLM routing 성능을 가정으로 숨기지 말고 six-way macro-F1, boundary F1, parse failure, push recall과 end-to-end Long success를 함께 측정해야 한다.

## 6. 평가 해석

### 6.1 short-horizon task 능력

`libero_spatial`, `libero_object`, `libero_goal`의 30개 task에서 각 task success rate를 측정한다. 이 결과는 SG-MoE가 개별 조작 능력을 유지하거나 개선하는지를 보여준다.

### 6.2 long-horizon composition

`libero_10`의 10개 task에서 end-to-end success와 완료된 ordered subgoal 비율을 측정한다. Long demonstration 자체가 학습에 포함되므로 이 결과의 정확한 해석은 “훈련 분포 안의 폐루프 skill composition”이다. held-out composition 또는 unseen task generalization이라고 부르지 않는다.

### 6.3 action policy와 VLM routing의 분리

- held-out demonstration의 ground-truth skill routing은 action SG-MoE가 올바른 skill을 받았을 때의 offline 상한을 측정한다.
- frozen-VLM routing은 실제 배포 가능한 전체 system의 병목을 측정한다.

온라인 simulator에서 ground-truth skill oracle을 주장하려면 LIBERO predicate로 재현 가능한 state oracle이 별도로 정의되어야 한다. 그런 oracle이 없으면 offline upper bound와 online frozen-VLM 결과를 혼동하지 않는다.

## 7. 재현 시 주의할 차이

| 항목 | 논문 | 공개 구현/본 실험 처리 |
|---|---|---|
| AtomicVLA 학습 스텝 | appendix에 100k로 기술 | 공개 LIBERO config는 140k로 보임; 본 실험은 별도 budget 고정 |
| skill balancing | 희소 skill oversampling | 정확한 sampler 미공개; 본 실험은 six-way uniform sampler |
| VLM 학습 | AtomicVLA 전체 설정과 공개 config에서 동결이 명확하지 않음 | SmolVLA VLM 완전 동결 |
| LIBERO experts | 5개 | `push`를 추가한 6개 |
| action horizon | AtomicVLA 계열 10 | chunk 10, 실제 실행 5 |
| annotation | 자체 motion+VLM pipeline | SARM `subtask_index`와 품질 manifest |

## 8. 문헌이 지지하는 가설과 지지하지 않는 주장

문헌은 “semantic skill로 action FFN을 전문화하면 shared dense expert보다 long-horizon composition이 좋아질 수 있다”는 가설을 지지한다. 또한 SmolVLA가 action expert만 학습할 수 있으므로 VLM 동결 실험은 기술적으로 타당하다.

반면 다음은 아직 증거가 아니라 검증 대상이다.

- frozen SmolVLM2-500M이 현재 skill과 boundary를 충분히 정확히 판단한다.
- SARM의 VLM-localized boundary가 AtomicVLA 자체 annotation과 동등하다.
- `push` 독립 expert가 단순히 표본을 더 쪼개는 손해보다 전문화 이득이 크다.
- Long success 개선이 parameter 증가가 아니라 skill routing에서 온다.

따라서 결론은 task success뿐 아니라 routing·skill별 action 성능·경계 품질을 함께 보고하고, 실패가 action expert와 frozen VLM 중 어디서 발생했는지 분리해야 한다.
