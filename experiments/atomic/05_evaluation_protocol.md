# Atomic SmolVLA 방법론 평가

## 0. 상태와 판정

- 평가 시점: **pre-execution**
- 실행 상태: **blocked before Gate 1–5 completion**
- raw result: 없음; 생성·열람·수정하지 않음
- AtomicVLA 고정 참조: `zhanglk9/AtomicVLA@c3583055adde0a491a11ffe08c15ca6459a64254`
- 종합 판정: 공식 SG-MoE의 핵심 라우터/혼합 구조와 frozen-VLM online planner의 core runtime은
  정적으로 이식되었으나, 사전등록된 데이터 manifest·runtime smoke·exact-init 평가·재현 실행 경로가
  완성되지 않아 가설을 실행하거나 판정할 수 없다.

이 구현은 AtomicVLA exact reproduction이 아니다. 정확한 명칭은 사전등록과 동일하게
**AtomicVLA-style action-only SG-MoE adaptation for frozen SmolVLA**다.

## 1. 공식 구현과의 정렬

| 항목 | 고정 AtomicVLA 구현 | 현재 구현 | 판정 |
|---|---|---|---|
| router 소유권 | 최상위 Gemma module의 router 하나 | `SmolVLMWithExpertModel.atomic_router` 하나 | 정렬 |
| skill embedding | `linspace(10,100,n)` scaled one-hot, hidden width까지 zero-pad | 6개 skill에 같은 구성 | 정렬 |
| router 초기화 | identity kernel × `log(n-1)/55` | `n=6`에 같은 식 | 정렬 |
| route 재사용 | 동일 skill embedding을 action horizon에 반복하고 모든 scanned layer에 combine weight broadcast | sample별 route를 한 번 계산해 모든 action token/layer에 재사용 | 기능적 정렬 |
| mixture | `(1-w) * shared + w * top1_expert` | 같은 식 | 정렬 |
| router auxiliary loss | 실제 objective에 별도 router CE/load-balance loss 없음 | 없음 | 정렬 |
| expert 실행 | 모든 expert를 계산한 뒤 sparse combine | batch에서 선택된 expert만 계산 | deterministic FFN에서 기능적 등가인 효율 adaptation |
| 전체 objective | decision/reasoning text loss와 action loss | action flow-matching loss만 | 의도적 비재현 |
| VLM | 기본 LIBERO 경로에서 완전 동결이 아님 | `train_expert_only=True`로 동결 | 사전등록된 adaptation |
| taxonomy | `pick/place/open/close/turn`, unknown은 pick fallback | `pick/place/push/turn/open/close`, unknown 금지 | 사전등록된 adaptation |
| boundary | episode padding만 사용 | canonical skill 변경 뒤 target을 padding 처리 | 사전등록된 adaptation |
| expert 초기화 | 추가 expert를 별도 초기화 | dense FFN을 shared와 6 experts에 복사 | 사전등록된 adaptation |

PyTorch `Linear.weight`는 공식 Flax kernel의 전치 저장 형태이지만 계산식은 같다. 공식 구현의
token별 condition은 horizon 동안 동일하므로 현재 sample별 route와 의미가 같다.

## 2. 구현 경로 평가

### 정적으로 확인된 항목

- atomic mode off가 기본값이고, atomic SG-MoE는 `chunk_size=10`, `n_action_steps=5`, frozen VLM,
  non-compiled path를 요구한다.
- training batch의 anchor subtask를 strict six-way mapping으로 변환하고, canonical skill이 바뀐
  target을 기존 `action_is_pad`와 OR한다. masked MSE는 유효 action scalar 수로 재정규화된다.
- dense 조건도 `atomic_data_enabled=True`이면 같은 boundary mask와 balanced sampler를 사용할 수 있다.
- global router는 action expert에만 있고 VLM FFN은 교체하지 않는다.
- dense safetensor를 atomic checkpoint로 승격할 때 shared와 6 expert에 action FFN weight를 복사한다.
- synchronous `select_action`과 `predict_action_chunk`에서 외부 `atomic_skill_id`를 denoising path까지
  전달한다.
- planner mode는 내부 frozen SmolVLM weight와 processor를 공유해 action queue가 비는 5-action
  interval마다 strict JSON `{decision, skill}`을 생성한다. continue/switch 규칙, invalid-response
  유지/episode-failure 상태기계, switch queue 폐기와 batch-size-1 eval termination handoff가 구현되어 있다.
- mapping JSON은 52 entries이며 분포는 `pick=18`, `place=27`, `push=1`, `turn=1`, `open=3`,
  `close=2`다. unknown fallback은 없다.
- system Python `py_compile`은 atomic/planner/eval 관련 7개 source 파일과 focused test 2개에서 통과했다.

### 아직 입증되지 않은 항목

로컬 CUDA/PyTorch 환경 문제로 `tests/policies/smolvla/test_atomic_sgmoe.py`는 collection까지
도달하지 못했다. 따라서 다음은 코드 또는 테스트가 존재하더라도 runtime pass로 간주하지 않는다.

- dense-to-atomic step-zero numerical equivalence
- mixed-skill forward/backward와 router/shared/선택 expert gradient
- 미선택 expert 비실행
- boundary mask loss 불변성
- sampler determinism/resume/DDP sharding
- dense checkpoint promotion, atomic save/reload와 route/action 동일성
- VLM gradient 0, optimizer exclusion, train 전후 checksum
- non-atomic SmolVLA regression
- 실제 SmolVLM generation을 통한 planner smoke와 LIBERO episode-failure 통합 rollout

## 3. 사전등록 프로토콜 일치도

### 동결되어 있는 설계

- primary metric: LIBERO-Long task-balanced end-to-end success rate
- secondary non-inferiority: Spatial/Object/Goal task-balanced SR, margin `-0.03`
- training seeds: `42, 43, 44`
- 조건: frozen dense A와 frozen six-skill SG-MoE B
- budget: condition/seed당 100,000 updates, global batch 64, 총 6 runs
- rollout: task/seed당 50, 전체 12,000
- 통계: exact-init paired hierarchical bootstrap 10,000회, two-sided alpha 0.05
- falsification/inconclusive 기준과 raw artifact append-only 원칙

`03_amendment_01_atomicvla_alignment.md`는 결과 열람 전에 router 소유권과 초기화만 공식 코드에
맞게 수정했으며 metric, seed, budget, comparison, 통계와 반증 기준을 바꾸지 않았다. 이 amendment는
사전등록 무결성을 훼손하지 않는다.

### 실행 설정으로 아직 고정되지 않은 설계

문서의 수치는 구현 가능한 config/manifest/script로 동결되지 않았다. 특히 현재 SmolVLA 기본
`scheduler_decay_steps=30_000`은 계획의 100,000-step cosine decay와 다르다. 별도 frozen train config가
없으므로 CLI override 없이 실행하면 사전등록을 위반한다.

또한 dense A에서 `atomic_data_enabled=True`만 켜면 `train_expert_only=True`가 강제되지는 않는다.
기본값은 동결이지만, A/B의 frozen 범위와 다른 intervention flag를 동일하게 고정한 resolved config가
필수다.

## 4. 실행 차단 요소

다음이 해결되기 전에는 smoke training이나 full run을 시작하면 안 된다.

1. **Gate 1 — data 미구현:** dataset/revision pin, provenance 및 구조 감사, 제외 reason,
   task-stratified 80/10/10 split, leakage 검사, anchor manifest가 없다. 현재 sampler는 선택된 dataset의
   모든 frame을 바로 후보로 사용한다.
2. **Gate 2/3 — runtime 미검증:** CUDA 환경 때문에 focused pytest가 실행되지 않았다. VLM freeze,
   optimizer exclusion, checkpoint promotion/save-resume, boundary-loss invariance와 DDP sampler gate가
   아직 통과하지 않았다.
3. **Sampler protocol 미완성:** six-way epoch 총량은 균형화되지만, 사전등록한 매 10,000 draws의
   `1/6 ± 1%p` runtime assertion과 histogram logging은 없다.
4. **Gate 4 — planner core 구현, runtime 미검증:** strict JSON frozen-VLM planner,
   continue/switch 상태기계, queue reset/switch 처리와 batch-size-1 episode-failure handoff는 구현되었다.
   그러나 손상된 로컬 PyTorch/CUDA 환경 때문에 실제 SmolVLM generation 및 LIBERO smoke가 실행되지
   않았고, offline routing metric, fixed-init LIBERO manifest, applied-state hash와 ordered subgoal logger는
   아직 없다. 따라서 primary 조건 B의 사전등록 평가 실행은 아직 시작할 수 없다.
5. **Gate 5 — 실행/보존 미구현:** dense/SG-MoE frozen YAML, 20-step smoke, append-only run manager,
   checkpoint selection record와 environment/hash inventory가 없다.
6. **통계 실행 경로 미구현:** paired outcome validator, hierarchical bootstrap, Wilson interval,
   non-inferiority 및 사전 정의 판정 script가 없다.

## 5. 방법론 결론

현재 diff는 AtomicVLA의 핵심 SG-MoE 메커니즘과 frozen SmolVLM online planner를 SmolVLA에 적용하는
**core model/data/planner prototype**으로는 타당하다. 그러나 데이터 선정 manifest, planner runtime
smoke, exact-init rollout과 통계 판정까지의 사전등록 실험 파이프라인은 완성되지 않았다.

따라서 현 단계의 유일한 허용 결론은 다음과 같다.

> 구현 정렬은 정적으로 유망하지만 실행 검증이 차단되었고 결과가 없으므로, H1/H2/H3는 모두
> 평가 전이며 supported, refuted, inconclusive 중 어느 성능 판정도 내릴 수 없다.

## 6. Amendment 03 — visual encoder option 재평가

### 상태

- 평가 시점은 계속 **pre-execution**이며 raw result는 없다.
- sandbox helper 유실로 amendment/source를 로컬에서 재열람하거나 테스트하지 못했다. 다만 제공된
  exact diff를 정적으로 검토했다.
- `set_requires_grad()`는 `train_expert_only=True`에서 먼저 VLM 전체를 eval/frozen으로 만든 뒤
  `freeze_vision_encoder=False`일 때 vision model만 `requires_grad=True`로 되돌린다.
- `train(mode)`도 전체 VLM을 eval로 유지한 뒤 trainable vision model만 `train(mode)`로 되돌린다.
  따라서 text transformer, connector와 LM head는 frozen/eval이고 vision tower만 선택적으로 학습된다.
- planner config는 `freeze_vision_encoder=False`를 거부하므로 rollout 동안 shared vision weight는
  frozen/eval이다.
- metric, seed, budget, statistical test와 falsification criterion은 변경하지 않는 것으로 평가한다.

### 원 frozen-VLM 가설에 대한 영향

`freeze_vision_encoder=True`인 A/B 쌍은 기존 frozen-VLM primary 가설을 그대로 보존한다. 반대로
`freeze_vision_encoder=False`이면 observation representation이 action loss로 업데이트되므로, text와
connector가 동결되어도 전체 VLM이 동결된 실험은 아니다. 이 결과는
**frozen-language/action-trained-vision adaptation**으로만 해석해야 하며 기존 H1/H2의 primary 결과를
대체할 수 없다.

따라서 vision on/off를 단순한 구현 옵션으로 pooling하면 안 된다. 최소 조건 규칙은 다음과 같다.

1. primary A/B는 모두 `freeze_vision_encoder=True`로 유지한다.
2. trainable-vision dense/SG-MoE를 실행하려면 서로 matched인 별도 Aᵥ/Bᵥ condition으로 이름 붙인다.
3. A/B와 Aᵥ/Bᵥ의 rollout 또는 training seed를 합쳐 하나의 effect estimate를 만들지 않는다.
4. vision training의 효과는 `Aᵥ-A`, `Bᵥ-B` 또는 interaction
   `(Bᵥ-Aᵥ)-(B-A)`로 별도 탐색 보고한다.

사전등록 budget은 A/B × seeds 42/43/44의 6 runs로 이미 동결되어 있다. Aᵥ/Bᵥ까지 같은 세 seed로
추가하면 12 runs가 되므로 “budget 불변”과 양립하지 않는다. 결과 열람 전 amendment라도 추가
vision 조건을 실행하려면 별도 후속 budget과 exploratory status를 명시해야 한다. 6-run primary budget
안에서는 trainable-vision 조건을 실행하지 않는 것이 원 가설을 보존하는 유일한 해석이다.

### planner 공유의 해석

`freeze_vision_encoder=False`로 action training한 checkpoint의 vision encoder를 planner가 평가 시점에
고정해 공유하는 것은 parameter가 평가 중 변하지 않는다는 뜻일 뿐, **pretrained frozen-VLM planner**라는
뜻은 아니다. planner의 image feature는 이미 SARM action supervision과 training distribution에 적응했다.
따라서 이 조건의 routing metric이나 end-to-end success는 “frozen-at-inference planner with an
action-trained visual encoder”로 보고해야 한다.

이 공유는 같은 checkpoint를 사용한다는 점에서는 내부적으로 일관되지만 H2의 원인 분리를 약화한다.
planner 개선이 pretrained VLM의 zero-update planning 능력인지, action loss로 적응한 visual feature의
효과인지 구분할 수 없기 때문이다. primary H2 검증에는 pretrained vision encoder까지 동결된 planner를
사용하고, trainable-vision planner 결과는 별도 sensitivity로만 유지해야 한다.

planner 생성 시 `freeze_vision_encoder=True`를 강제하는 것만으로는 위 문제가 해소되지 않는다.
다음 두 사실을 artifact에 따로 기록해야 한다.

- training 동안 vision encoder가 업데이트되었는지 여부
- evaluation 동안 vision encoder가 동결되었는지 여부

### 구현 검증 gate 추가

실행 전 다음 정적/runtime 검사가 필요하다.

- `freeze_vision_encoder=True`: VLM 전체 parameter가 optimizer에서 제외되고 gradient가 모두 `None`
- `freeze_vision_encoder=False`: vision model만 trainable/optimizer 포함, text transformer·connector·LM
  head는 frozen/optimizer 제외
- `train()` 전환 뒤에도 위 allowlist가 유지되고 trainable vision의 module mode가 의도와 일치
- A/B resolved config의 vision setting과 initial vision checksum이 동일
- planner load 뒤 training-time vision provenance를 보존하고 evaluation 중 checksum이 불변
- pretrained-vision 조건과 action-trained-vision 조건의 metric·artifact directory를 분리하며 pooling 금지

Amendment 03은 CLI의 기존 의미를 바로잡는 source 변경으로는 합리적이며 제공된 diff에서 freeze/train
mode 계약도 일관된다. 그러나 trainable vision을 primary 비교에 포함하는 순간 연구 질문과 budget이
바뀐다. 사용자가 option on/off capability와 vision training 의도를 밝혔더라도 기존 primary를 명시적으로
교체하지는 않았으므로, 과학적으로 안전한 기본값은 frozen setting을 primary로 유지하고 trainable-vision
matched Aᵥ/Bᵥ를 별도 exploratory condition으로 두는 것이다.

그러므로 현재 방법론 판정은 이전과 같이 **pre-execution / blocked**다. full run 전에 사용자가
(a) 기존 6-run frozen primary만 실행할지, (b) 별도 budget으로 trainable-vision 6 runs를 추가할지,
(c) 사전등록 primary 자체를 교체할지를 명시적으로 결정해야 한다. (c)는 결과 열람 전이라도 가설과
budget을 바꾸는 새 amendment가 필요하다.
