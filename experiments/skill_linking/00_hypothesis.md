# Experiment hypothesis: End-to-End Skill Linking for SmolVLA

## Claim

Global instruction, 현재 관측, 현재 skill state를 조건으로 SmolVLA action expert, skill embedding, transition head를 공동 fine-tuning하면 action-only SmolVLA보다 LIBERO-Long의 complete success rate가 높아질 것이다.

## Baseline

- 동일한 LIBERO-compatible SmolVLA checkpoint에서 시작한다.
- `sungkyunner/libero_10_subtask_semantic_clean`의 동일 episode와 동일 학습 budget을 사용한다.
- VLM/vision backbone은 동결하고 action expert는 fine-tuning한다.
- Skill embedding과 transition loss는 사용하지 않는다.

정확한 initialization checkpoint와 training budget은 `03_experiment_plan.md`에서 결과 관찰 전에 고정한다.

## Intervention

Baseline에 다음 두 모듈만 추가하고 action expert와 함께 공동 fine-tuning한다.

1. `START`와 `K`개 skill state를 나타내는 `nn.Embedding`
2. `{CONTINUE, skill_0, ..., skill_(K-1), DONE}`을 예측하는 단일 linear transition head

현재 skill embedding은 action suffix embedding에 더하고, transition head는 마지막 action-expert suffix hidden state를 pooling해 예측한다.

```text
L_total = L_flow_action + lambda_transition * L_transition_CE
```

Transition target은 dataset을 복제하지 않고 frame-level `subtask_index`에서 계산한다. 학습 index는 두 pool로 나눈다.

- **Atomic pool:** `H=10` window 안에서 `subtask_index`가 바뀌지 않는 원본 frame index
- **Transition-event pool:** episode 시작 1개, 실제 adjacent-subtask 경계 1개, episode 종료 1개

실제 adjacent-subtask 경계는 경계마다 정확히 하나의 pair sample만 만든다. 경계가 새 skill의 첫 frame `b`이면 pair sample의 시작은 `b-H`로 고정한다. 서로 다른 episode를 이어 붙이거나 별도 pair 영상/dataset을 저장하지 않는다.

```text
episode 시작     -> START에서 첫 skill
같은 skill 유지  -> CONTINUE
다른 skill 등장  -> 첫 next skill ID
episode 종료     -> DONE
```

## Dataset

- Repository: [`sungkyunner/libero_10_subtask_semantic_clean`](https://huggingface.co/datasets/sungkyunner/libero_10_subtask_semantic_clean), revision `d619815aeba9c06c70fc558838137dd57a651ce1`
- 495 episodes, 136,425 frames, 10 LIBERO-Long tasks, 10 Hz
- 두 개의 256×256 RGB camera, state 8D, action 7D
- 관측된 `subtask_index`는 `1..15`이고 `meta/subtasks.parquet`가 semantic 이름을 제공한다.
- 전체 941 contiguous atomic segments와 446 adjacent-subtask 경계가 있다.
- 446 episodes에는 경계가 하나씩 있고 49 episodes는 단일 subtask다.
- 446개 경계는 9개 directed semantic pair type으로 구성된다.

## Primary metric

- LIBERO-Long 10개 task의 macro-average complete episode success rate
- Primary comparison: predicted-transition intervention 대 action-only baseline
- 최소 의미 효과: `+5 percentage points`

## Secondary metrics

- Atomic/subtask macro success와 prefix success curve
- Transition macro-F1, boundary precision/recall, transition latency
- Episode당 false switch와 invalid transition 수
- Action loss, action latency, GPU memory

## Falsification criteria

다음 중 하나면 주 가설은 지지되지 않는다.

- Complete success 향상이 `+5%p` 미만이다.
- Paired 95% confidence interval이 0을 포함한다.
- Atomic macro success가 `2%p`보다 많이 하락한다.
- Transition 성능만 높고 complete success가 향상되지 않는다.
- 개선이 한 seed 또는 소수 task에만 집중된다.

## Scope

첫 실험에는 외부 planner, future-plan decoder, MoE, scene graph, phase camera masking, focus-token pruning을 포함하지 않는다. Skill linking이 단독으로 지지된 뒤에만 focus-token과 결합한다.

## Primary references

- [SmolVLA](https://arxiv.org/abs/2506.01844)
- [LoHoVLA](https://arxiv.org/abs/2506.00411)
- [AtomicVLA](https://arxiv.org/abs/2603.07648)
- [Compose by Focus](https://arxiv.org/abs/2509.16053)
- [Long-VLA](https://arxiv.org/abs/2508.19958)

## Status

사용자가 2026-08-08에 경계당 pair sample 하나의 설계와 P1 구현을 승인했다.
