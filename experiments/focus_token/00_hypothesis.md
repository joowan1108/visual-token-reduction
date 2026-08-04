# Experiment hypothesis

## Claim

SmolVLA action expert가 각 카메라의 모든 visual token을 계속 사용하는 대신, 충분한 초기 context aggregation 이후 현재 action query와 관련도가 높은 visual token만 카메라별로 선택해 사용하면 task-irrelevant background에 의한 attention dilution을 줄일 수 있다. 이에 따라 전체 visual token을 사용하는 baseline보다 LIBERO manipulation success rate가 높아지고, 더 적은 action-expert 연산으로 동일한 성능에 더 빠르게 도달할 것이다.

## Baseline

- Baseline configuration: 별도의 LIBERO fine-tuning이나 visual-token selection을 적용하지 않은 일반 SmolVLA이다. 
- Baseline checkpoint: Hugging Face에서 제공하는 pretrained `lerobot/smolvla_base`를 사용한다.
- Comparison model: 일반 SmolVLA를 LIBERO에서 10,000 steps 동안 fine-tune한 모델을 사용한다. 해당 checkpoint에도 task-relevant token selection은 적용하지 않으며 이미 학습이 진행된 상태이고 https://huggingface.co/sungkyunner/smolvla_libero_baseline 에 있다. 

## Intervention

Baseline에서 변경하는 핵심 독립변수는 action expert에 노출되는 visual K/V token의 task-relevant selection 여부이다.

- Action expert는 SmolVLA의 action expert 구조를 사용한다.
- Action expert의 초반 8개 layer(layer 0–7)는 모든 visual token을 사용하여 noisy-action latent가 language instruction, scene, robot state를 먼저 통합하도록 한다.
- 중후반 8개 layer(layer 8–15)에서는 현재 noisy-action query와 visual key 사이의 scaled QK compatibility를 계산한다.
- 각 카메라 안에서 token을 독립적으로 ranking하고 상위 `k`만 유지한다. 따라서 한 카메라의 token이 다른 카메라의 token budget을 모두 차지할 수 없다.
- Primary intervention은 카메라별 token의 50%를 유지하는 설정이다. 25%와 75% 유지 설정은 token budget sensitivity ablation으로 사용한다. 원래 카메라당 token 수를 `N`이라 하면 각각 `ceil(0.25N)`, `ceil(0.50N)`, `ceil(0.75N)`개를 유지한다.
- 선택된 token은 원래 spatial/positional order를 유지한다. Language token, state token, action token은 제거하지 않는다.
- VLM prefix와 KV cache는 항상 전체 visual token으로 생성한다. Token selection은 action expert가 prefix visual K/V를 조회하는 cross attention layer 구간에만 적용한다.
- Router parameter를 별도로 학습하는 방식이 아니라 현재 action-query/visual-key compatibility로 token을 선택하며, LIBERO demonstration의 flow-matching action loss로 end-to-end fine-tuning한다.

학습 setting은 다음으로 고정한다. 
initialization: SmolVLM2 pretrained VLM + newly initialized SmolVLA expert
dataset: HuggingFaceVLA/libero, all episodes
seed: 1000
batch_size: 4
optimizer: AdamW(lr=1e-4, betas=(0.9,0.95), eps=1e-8, wd=1e-10)
scheduler: cosine warmup 1K, decay 100K, final_lr=2.5e-6
augmentation: disabled
noise: Normal(0,1)
flow time: Beta(1.5,1.0) × 0.999 + 0.001
checkpoint step: 10,000
planned training: 100,000 steps

## Primary metric

- Metric: LIBERO Spatial, Object, Goal, Long의 40개 task에 대한 macro-average episode success rate (%)
- Expected direction: 50%-token intervention > dense-token baseline
- Minimum meaningful improvement: 3개 seed 평균 기준 success-rate 향상

## Secondary metrics

- Metric 1: Suite별·task별 success rate, 동일 success threshold에 도달하는 training step 수, validation flow-matching action loss
- Metric 2: 카메라별 selected-token spatial distribution, selected top-k attention mass, effective visual-token count, action-expert attention FLOPs, GPU peak memory, end-to-end action inference latency

## Evaluation setting

- Dataset/environment: Simulation은 `HuggingFaceVLA/libero`의 LIBERO Spatial, Object, Goal, Long을 사용하고 top/agent-view와 wrist camera를 입력한다. 
- Number of seeds: 3개 training seed. 각 seed 안에서 baseline과 모든 intervention이 동일한 dataset order, evaluation initial states, task seeds를 공유하도록 paired design을 사용한다.
- Training budget: 기본 SmolVLA 구조 baseline에서 시작해서 각 intervention을 seed당 추가 10,000 gradient steps 동안 학습한다. 총 학습량은 각 run 기준 10K steps이며, 비교 대상의 fine-tuning budget도 10K steps로 동일하다.
- Evaluation episodes/samples: 각 seed에서 4개 LIBERO suite × suite당 10개 task × task당 20 episodes, 총 800 episodes를 평가한다. 3개 seed 전체에서는 model variant당 2,400 episodes이다. Primary comparison에는 동일한 initial-state/task seed 목록을 사용한다.

Primary hypothesis test는 사전에 지정한 50%-token intervention과 dense baseline의 paired 차이로 수행한다. 25%와 75% 결과는 primary variant 선택에 사용하지 않고 sensitivity analysis로 보고한다. 평균 차이와 함께 task/seed 단위 bootstrap 95% confidence interval을 제시한다.

## Falsification criterion

다음 중 하나라도 만족하면 primary hypothesis가 지지되지 않는 것으로 판단한다.

- 50%-token intervention의 3-seed macro-average success-rate 향상이 baseline 대비 적다.
- Paired success-rate difference의 95% confidence interval이 0을 포함한다.
- 평균 향상이 특정 seed나 소수 task에만 집중되고 suite별 또는 seed별 비교에서 일관되게 재현되지 않는다.
- Success rate가 낮아지면서 latency·FLOPs 개선만 나타난다. 이 경우 효율성 trade-off는 있을 수 있지만 “task-relevant selection이 action accuracy를 향상한다”는 claim은 지지되지 않는다.

25% 설정만 실패하고 50% 설정이 기준을 만족하면 가설은 유지되지만, 25% token budget에서 task-critical visual information이 손실된 것으로 해석한다. 반대로 75%만 개선되고 50%가 실패하면 task-relevant selection보다는 단순히 더 많은 visual context를 보존한 효과일 가능성이 있으므로 primary hypothesis는 지지되지 않은 것으로 처리한다.

## Reference papers

- Paper URL/PDF:
  - FocusVLA: https://arxiv.org/abs/2603.28740 / https://arxiv.org/pdf/2603.28740
  - Compose by Focus: https://arxiv.org/html/2509.16053v2 / https://arxiv.org/pdf/2509.16053v2
  - SmolVLA: https://arxiv.org/abs/2506.01844 / https://arxiv.org/pdf/2506.01844
- Official implementation:
  - FocusVLA: arXiv 페이지에 공식 implementation repository가 공개되어 있지 않음(2026-08-03 확인)
  - Compose by Focus: https://github.com/han20192019/skill-composition-code
  - SmolVLA: https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla


  