# Atomic SmolVLA 구현 지도

## 1. 구현 목표와 불변 조건

목표는 기존 SmolVLA의 VLM을 완전히 동결한 채 action expert의 FFN을 six-skill SG-MoE로 확장하고, SARM annotation으로 학습한 뒤 LIBERO 40 tasks에서 평가할 수 있는 완전한 경로를 만드는 것이다.

구현 전체에서 다음 조건을 유지한다.

- canonical skill 순서는 `pick=0`, `place=1`, `push=2`, `turn=3`, `open=4`, `close=5`로 고정한다.
- `push`를 `place` 또는 다른 expert로 fallback하지 않는다.
- VLM parameter는 `requires_grad=False`, evaluation mode이며 optimizer에도 포함되지 않는다.
- SG-MoE는 VLM이 아니라 SmolVLA action expert의 FFN에만 적용한다.
- 학습 때는 SARM ground-truth skill을 사용한다.
- router classification loss나 VLM fine-tuning loss를 추가하지 않는다.
- `chunk_size=10`, `n_action_steps=5`를 checkpoint와 runtime 양쪽에서 검증한다.
- skill boundary 뒤의 action target은 버리지 않고 loss mask로 제외한다.
- 기존 SmolVLA config/checkpoint/test 동작은 atomic mode가 꺼졌을 때 bitwise-compatible하도록 유지한다.

## 2. 기준 코드 경로

구현 전에 현재 checkout에서 심볼 이름을 다시 확인하되, 수정 지점은 다음 upstream 구조를 기준으로 한다.

| 영역 | 기준 파일 | 현재 책임 | Atomic 변경 |
|---|---|---|---|
| Policy config | `src/lerobot/policies/smolvla/configuration_smolvla.py` | chunk, action steps, freeze flags, optimizer 기본값 | atomic mode, 6 skills, router/FFN 설정과 config 검증 |
| Policy wrapper/loss | `src/lerobot/policies/smolvla/modeling_smolvla.py` | batch 전처리, flow-matching loss, action sampling | `atomic_skill_id`, boundary loss mask 전달; routing diagnostics 반환 |
| VLM + action expert | `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | VLM과 `lm_expert` 결합, transformer forward | action-expert FFN을 optional SG-MoE로 교체하고 skill condition 전파 |
| Policy factory | `src/lerobot/policies/factory.py` 및 config registry | policy 생성과 processor 연결 | atomic config가 별도 이름이면 lazy import/registry 연결 |
| Dataset | `src/lerobot/datasets/` | LeRobot episode-aware loading | 원본 loader 재작성 없이 wrapper/sampler가 skill과 mask를 추가 |
| Train pipeline | `src/lerobot/scripts/lerobot_train.py` 및 train config | sampler, optimizer, logging, checkpoint | skill-uniform sampler 연결, atomic metrics/logging |
| LIBERO env/eval | `src/lerobot/envs/libero.py`, `src/lerobot/scripts/lerobot_eval.py` | suite 생성, rollout, reset | 고정 init-state manifest, subgoal/routing log, frozen planner hook |
| Experiment assets | `experiments/atomic/` | hypothesis와 재현 자료 | mapping, split/filter/eval manifests, frozen configs, 실행 문서 |

애플리케이션 구현은 기존 SmolVLA 경로를 재사용한다. 별도 VLA 전체를 복제하지 않고 SG-MoE와 skill-conditioned data/eval 경로만 추가한다.

## 3. 데이터 계층

### 3.1 versioned skill mapping

`experiments/atomic/config/subtask_to_skill.json`에 SARM 52개 문자열을 전부 열거한다. keyword runtime mapping은 spelling 변화와 복합 명령을 조용히 오분류하므로 사용하지 않는다.

loader 시작 시 다음을 검사한다.

```text
dataset vocabulary == mapping keys
mapping values subset == {pick, place, push, turn, open, close}
all six skills have at least one valid segment
```

누락, 중복, 알 수 없는 label이 하나라도 있으면 즉시 실패한다. mapping 파일의 SHA-256을 모든 run metadata에 기록한다.

### 3.2 episode audit와 split manifest

원본 dataset을 수정하지 않고 다음 artifact를 생성한다.

- `data_audit.jsonl`: episode별 frame 수, segment, skill, provenance, 검사 결과, 제외 사유
- `split_manifest.json`: train/validation/test episode IDs
- `anchor_index.parquet` 또는 동등한 기존 datasets cache: 유효 anchor frame, skill ID, valid target length

검사 순서는 다음과 같다.

1. episode/frame/task index 무결성
2. subtask index 범위와 segment 시간 순서
3. zero-length, overlap, gap, out-of-range 검사
4. uniform-fallback provenance 검사
5. 52-to-6 mapping 검사
6. action/state finite 값과 차원 검사
7. motion-label 모순은 삭제하지 않고 `suspect` 표시

구조적으로 잘못된 episode와 확인된 uniform fallback은 주 학습에서 제외한다. 의미적으로 의심되는 사례는 별도 manifest로 남기고, 사람이 승인하지 않으면 제외한다. 모든 제외는 reason code와 원본 episode ID를 보존한다.

split은 필터 이후 episode 단위 80/10/10이며 task별로 층화한다. 같은 episode의 frame이 둘 이상의 split에 들어가지 않게 한다. dataset에 train split만 있다는 이유로 frame-random split을 만들지 않는다.

### 3.3 boundary-aware chunk

anchor frame `t`의 canonical skill `s_t`를 chunk 전체의 route로 사용한다. action target은 `[t, t+9]`이고 loss mask는 다음과 같다.

\[
m_j = \mathbf{1}[t+j < T_{episode}]\,\mathbf{1}[s_{t+j}=s_t],\quad j=0,\ldots,9
\]

즉 skill boundary나 episode 끝 이후 target은 loss에서 제외한다. anchor를 제거하지 않으므로 짧은 segment도 학습 기회를 가진다. mask된 평균은 유효 element 수로 나누고, 유효 action이 0개인 sample은 audit 단계에서 제거한다.

batch contract는 기존 key를 유지하면서 최소한 다음을 추가한다.

```text
atomic_skill_id: int64[B]
action_is_pad 또는 atomic_action_mask: bool[B, 10]
episode_index, frame_index, task_index: diagnostics용
```

### 3.4 skill-uniform sampler

train에서만 먼저 `skill ~ Uniform(0..5)`를 뽑고 해당 skill의 anchor를 균등 복원 추출한다. distributed training에서는 `(base_seed, epoch, global_draw_index)`로 동일 global sequence를 만든 뒤 rank별로 shard해 중복과 drift를 제어한다.

매 epoch 또는 고정 draw window마다 관측된 skill histogram을 기록하고 각 class 비율이 `1/6 ± 1%p`인지 검사한다. validation/test loader는 sampler와 loss weighting 없이 자연 분포를 유지한다.

## 4. SG-MoE 모델 계층

### 4.1 FFN 교체 범위

SmolVLA `lm_expert`의 각 transformer block에서 기존 Gemma/SwiGLU MLP 하나를 다음 module로 교체한다.

```text
SkillGatedFFN
├── shared_expert: 기존과 같은 SwiGLU MLP
├── skill_experts[6]: 각각 독립된 같은 크기 SwiGLU MLP
└── router: fixed skill embedding -> 6 logits -> softmax/top-1
```

VLM block의 FFN은 건드리지 않는다. 한 batch 안에서 skill이 달라도 expert별 index select/scatter로 처리하며, 선택되지 않은 다섯 expert를 실행하지 않는다.

### 4.2 router와 출력

fixed embedding table과 canonical expert index는 checkpoint schema에 저장한다. embedding은 학습하지 않는다. router는 학습 가능하며 action loss의 gradient만 받는다.

```text
logits = router(stop_gradient(skill_embedding[skill_id]))
prob = softmax(logits)
expert_id = argmax(prob)
output = (1 - prob[expert_id]) * shared(x) + prob[expert_id] * expert[expert_id](x)
```

학습 데이터의 `skill_id`가 route input이며 별도 CE target은 사용하지 않는다. expert 이름의 의미가 permutation으로 무너지는 것을 막기 위해 router를 canonical identity가 top-1이 되도록 초기화하고, 첫 forward에서 여섯 label이 서로 다른 expert를 선택하는지 assert한다. action loss는 gate confidence와 expert/shared parameter를 갱신한다.

router entropy, selected expert histogram, gate probability, expert별 gradient norm을 log해 expert collapse를 탐지한다. collapse를 보고 난 뒤 auxiliary loss를 추가하지 않는다. 이는 사전 등록된 가설을 바꾸기 때문이다.

### 4.3 pretrained initialization

기존 SmolVLA action FFN weights를 shared expert와 여섯 skill experts에 각각 복사한다. 모두 같은 함수로 시작하므로 SG-MoE를 켠 직후 어느 route에서도 원본 action FFN과 같은 출력이 나와야 한다. 그 뒤 action-policy parameter만 분화해 학습한다.

이 초기화에는 두 장점이 있다.

- 무작위 skill expert 때문에 pretrained action policy가 즉시 붕괴하지 않는다.
- step-0 output equivalence를 자동 회귀 테스트로 확인할 수 있다.

checkpoint loader는 기존 dense SmolVLA checkpoint를 위 방식으로 승격하고, 저장 시 taxonomy/version/router/experts를 함께 기록한다. 반대 방향의 atomic-to-dense 자동 축약은 정의하지 않는다.

### 4.4 loss

기존 SmolVLA flow-matching target과 MSE를 그대로 재사용하고 boundary mask만 적용한다.

\[
L=\frac{\sum_{b,j,d}m_{b,j}(u_{b,j,d}-v_{b,j,d})^2}
        {\sum_{b,j,d}m_{b,j}}
\]

추가 router loss, load-balancing loss, language-model loss를 두지 않는다. skill-uniform sampling이 class imbalance를 처리하고, top-1은 canonical skill 입력으로 제약된다.

### 4.5 freeze 보장

config 선언만 믿지 않고 다음 세 겹을 검사한다.

1. VLM parameter `requires_grad=False`
2. train 중 VLM은 계속 `eval()`이고 dropout이 꺼짐
3. optimizer parameter IDs와 VLM parameter IDs의 교집합이 공집합

한 번의 backward 뒤 VLM gradient가 모두 `None`, router/shared/선택 expert gradient가 finite nonzero인지 test한다. 선택되지 않은 expert에는 그 sample의 gradient가 없어야 한다.

## 5. Frozen-VLM skill planner

### 5.1 입력과 호출 주기

planner는 action conditioning과 같은 frozen VLM checkpoint를 사용한다. 매 control cycle이 아니라 실행한 5 actions마다 한 번 호출한다. 입력은 다음으로 고정한다.

- 현재 main camera image
- 현재 wrist camera image
- 전체 LIBERO instruction
- 이전 canonical skill
- 이전 판단 이후 실행한 step 수
- 허용 skill 목록 여섯 개

state vector는 별도 textual decoding을 만들지 않고 SmolVLA의 기존 multimodal input 경로가 지원하는 범위만 사용한다. AtomicVLA처럼 학습된 think trace를 가장하지 않는다.

### 5.2 출력 schema와 실패 처리

temperature 0의 deterministic generation으로 다음 한 줄만 요청한다.

```json
{"decision":"continue|switch","skill":"pick|place|push|turn|open|close"}
```

parser는 정확한 JSON과 enum만 허용한다. 실패 시 이전 skill을 한 planning interval만 유지하고 `parse_failure=1`을 기록한다. 첫 호출이 실패해 이전 skill이 없으면 안전하게 episode를 실패 종료한다. 자유 문자열 keyword fallback은 사용하지 않는다.

`continue`이면 이전 skill과 출력 skill이 일치해야 한다. 모순이면 parse failure로 처리한다. 모든 prompt, raw output, parsed output, latency, image frame index를 rollout artifact에 저장한다.

### 5.3 train/eval 차이

학습과 offline validation에서는 dataset의 ground-truth `atomic_skill_id`만 사용한다. VLM-generated skill로 action expert를 학습하지 않는다. 온라인 LIBERO 평가에서만 frozen planner가 route를 공급한다.

held-out dataset에는 같은 prompt를 실행해 ground-truth frame skill과 비교하고 다음을 측정한다.

- six-way macro-F1와 class별 precision/recall
- continue/switch boundary F1와 ±1 planning interval tolerance F1
- `push` recall, `push -> place` confusion
- parse failure rate

## 6. LIBERO 평가 계층

### 6.1 suite와 horizon

기존 LIBERO adapter를 재사용하고 horizon만 manifest에서 검증한다.

| Suite | Tasks | Episodes/task | Max env steps |
|---|---:|---:|---:|
| Spatial | 10 | 50 | 280 |
| Object | 10 | 50 | 280 |
| Goal | 10 | 50 | 300 |
| Long (`libero_10`) | 10 | 50 | 520 |

action chunk는 10개를 예측하지만 5개만 실행한 뒤 새 observation과 skill 판단으로 replanning한다.

### 6.2 초기 상태 고정

현재 LeRobot/LIBERO 조합에서는 termination 뒤 vector environment auto-reset이 다음 `init_state_id`를 소비할 수 있다. 성공 시점이 다른 model을 단순히 같은 seed와 batch 순서로 실행하면 이후 초기 상태가 어긋날 수 있다.

따라서 `eval_manifest.json`에 `(suite, task_id, init_state_id, rollout_seed)` 50개를 미리 고정한다. 각 rollout 시작 때 exact init state를 명시적으로 설정하거나, 그것을 보장할 수 없다면 env를 rollout마다 재생성한다. 실제 적용된 init-state hash를 log하고 두 조건의 hash가 다르면 paired 통계를 금지한다.

### 6.3 subgoal logging

LIBERO BDDL goal predicate를 읽기 전용으로 평가해 Long episode마다 ordered subgoal 완료 벡터와 최초 완료 step을 남긴다. 최종 environment success 외에 다음을 계산한다.

- 완료한 ordered subgoal 수 / 전체 subgoal 수
- 첫 실패 subgoal
- 각 skill 전환 전후의 predicate progress

task-specific hard-coded 성공 detector를 중복 구현하지 않고 LIBERO가 제공하는 predicate와 task success를 재사용한다.

## 7. 설정과 산출물

재현 가능한 full run은 최소한 다음을 저장한다.

```text
experiments/atomic/
├── config/
│   ├── subtask_to_skill.json
│   ├── train_dense.yaml
│   ├── train_sgmoe.yaml
│   └── eval.yaml
├── manifests/
│   ├── data_audit.jsonl
│   ├── split_manifest.json
│   └── eval_manifest.json
└── runs/<condition>/<seed>/
    ├── resolved_config.yaml
    ├── environment.json
    ├── metrics.jsonl
    ├── checkpoints/
    └── eval_rollouts.jsonl
```

각 run은 git commit, dirty diff hash, dataset revision, mapping hash, split/eval manifest hash, package lock hash, CUDA/PyTorch/device 정보를 기록한다. raw rollout과 raw metric은 덮어쓰지 않고 새 run ID에 저장한다.

## 8. 테스트 지도

### 8.1 unit tests

- 52개 label mapping completeness와 `push` 독립 ID
- segment audit의 overlap/gap/out-of-range/zero-length 검출
- boundary를 가로지르는 10-step mask
- six-way uniform sampler의 determinism과 분포
- SG-MoE shape/dtype/device, batch 내 mixed-skill routing
- dense checkpoint 승격 후 step-0 output equivalence
- VLM freeze와 optimizer exclusion
- router/shared/선택 expert gradient, 미선택 expert 비활성
- JSON parser의 valid/invalid/continue inconsistency 처리
- config/checkpoint round trip

### 8.2 integration tests

- synthetic batch 한 번의 train forward/backward/update
- 20-step tiny overfit에서 finite loss와 감소 경향
- DDP 두 rank sampler가 고정 global draw를 정확히 shard
- save/reload 뒤 동일 action과 동일 route
- LIBERO 각 suite 한 rollout smoke test
- 같은 eval manifest를 두 condition에 적용했을 때 init-state hash 일치
- 기존 non-atomic SmolVLA test suite 무회귀

## 9. 구현 순서와 완료 조건

1. SARM revision을 pin하고 52-to-6 mapping과 audit/split manifest를 생성한다.
2. boundary-aware sample contract와 six-way sampler를 구현한다.
3. optional SG-MoE FFN, dense checkpoint 승격, freeze assertions를 구현한다.
4. policy forward/loss/action sampling에 skill ID와 mask를 연결한다.
5. deterministic frozen-VLM planner와 offline routing evaluator를 연결한다.
6. fixed-init LIBERO runner와 subgoal logger를 연결한다.
7. unit/integration/smoke gate를 모두 통과한 뒤에만 full training을 시작한다.

full implementation 준비 완료의 기준은 다음과 같다.

- mapping coverage 52/52와 audit 결과가 review 가능하다.
- dense baseline과 SG-MoE가 같은 split, optimizer budget, eval manifest를 사용한다.
- VLM에 gradient와 optimizer state가 생기지 않는다.
- step-0 SG-MoE가 pretrained dense action expert와 일치한다.
- train sampler의 여섯 skill 빈도가 허용 오차 안이다.
- boundary mask와 exact init-state pairing test가 통과한다.
- raw 결과 경로가 append-only이고 재실행으로 기존 결과를 덮어쓰지 않는다.

## 10. AtomicVLA transition supervision 대응 (2026-08-20)

공식 AtomicVLA commit `c3583055`의 transition supervision은 class-weighted switch head가 아니라 경계 전후
deterministic `[think]` window다. 본 transition head는 reasoning mode를 생성하지 않고 현재 실행할 six-way skill을
직접 예측하며, SG-MoE action loss는 strict current-skill boundary mask를 사용한다. 따라서 official pre-boundary
next-skill target을 이 경로에 합치지 않는다.

고정 4배 switch CE는 제거하고 unreduced six-way CE에 `(1 - exp(-CE)) ** gamma`를 적용한다. 기본
`implicit_transition_focal_gamma=2.0`이며 `gamma=0`은 plain CE ablation이다. Sampling, target timing, unweighted
stay/switch metrics, transition-head-only gradient ownership은 바꾸지 않는다. Training log에는 focal objective와
plain CE를 분리해 남기고, natural held-out split의 stay accuracy와 switch precision/recall로 checkpoint를 비교한다.
