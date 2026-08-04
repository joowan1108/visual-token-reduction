# 05. Paper-method evaluation: Focus Token Selection for SmolVLA

## 판정 요약

- **구현 fidelity:** 조건부 통과
- **로컬 runtime gate:** 통과 (`3 passed`)
- **Dense 대 Focus-50 LIBERO 성능:** remote GPU 미실행으로 pending/inconclusive
- **원격 학습 시작 가능 여부:** scheduler amendment 반영 후 가능

실제 LIBERO success rate, latency, FLOPs, GPU memory 결과는 생성·추정하지 않았다.

## 평가 범위

다음을 독립적으로 대조했다.

- `00_hypothesis.md`
- `01_literature_review.md`
- `02_implementation_map.md`
- 승인된 `03_experiment_plan.md`
- `04_change_log.md`
- 현재 SmolVLA source diff
- `tests/policies/smolvla/test_focus_token.py`

사용자는 실제 LIBERO 학습과 rollout을 remote Linux GPU server에서 수행한다고 지정했다. 현재 로컬 평가는 구현 fidelity와 synthetic attention 경로에 한정한다.

## 논문 방법 대비 구현 fidelity

| 요구사항 | 구현 상태 | 근거 |
|---|---|---|
| 초기 context aggregation dense | 충족 | layer 0–7은 selection branch 진입 안 함 |
| late cross-attention만 sparse | 충족 | 기본 interleave에서 9/11/13/15층만 branch 진입 |
| current noisy-action Q와 visual K의 scaled compatibility | 충족 | 실제 expert-projected Q/K, head/query 평균, `sqrt(head_dim)` scaling |
| camera별 독립 top-k | 충족 | camera span별 `ceil(ratio * N_valid)` |
| spatial order 복원 | 충족 | 선택 후 prefix position 기준 정렬 |
| non-visual token 보존 | 충족 | visual spans 밖 position은 retain |
| empty camera 보존 | 충족 | mask상 valid visual patch가 없으면 0개 선택 |
| full prefix/KV cache 유지 | 충족 | gathered view만 만들고 `DynamicCache`를 crop/mutate하지 않음 |
| 별도 router/trainable parameter 없음 | 충족 | config 값과 tensor 연산만 추가, state-dict key 추가 없음 |
| training direct-prefix와 inference cached-prefix | 충족 | 동일 selection helper 사용 |
| Dense default 완전 우회 | 충족 | `focus_token_keep_ratio=1.0`이면 helper 미호출 |

이 구현은 FocusVLA 전체 재현이 아니다. Cascaded attention, channel gate, raw shallow visual value 사용은 구현하지 않았으며, 이는 승인된 SmolVLA ablation 범위와 일치한다.

## 로컬 baseline/intervention 평가

실행 환경:

```text
Python 3.12.13
C:\Users\joowa\miniconda3\envs\test\python.exe
DEVICE=cpu
```

실행 명령:

```powershell
C:\Users\joowa\miniconda3\envs\test\python.exe `
  -m pytest -q -p no:cacheprovider `
  tests/policies/smolvla/test_focus_token.py
```

결과:

```text
3 passed
```

| 항목 | Dense | Focus-50 | 로컬 관찰 |
|---|---|---|---|
| `focus_token_keep_ratio` | 1.0 | 0.5 | config validation 통과 |
| early cross-attention test K/V length | 8 | 8 | 동일 |
| late 9/11/13/15 test K/V length | 8 | 4 | camera별 4→2, 두 camera 합계 8→4 |
| dense spans 유무 output | 동일 | N/A | `rtol=0`, `atol=0` identity |
| direct-prefix backward | finite | finite | gradient 확인 |
| cached-prefix backward/action | finite | finite | gradient와 output 확인 |
| full cache length | 8 | 8 | intervention 후에도 불변 |

추가 정적 검증:

- Python compilation: pass
- Ruff check: pass
- Ruff format check: pass
- `git diff --check`: pass

## 아직 검증하지 못한 항목

- 실제 pretrained SmolVLA full-model dense loss/action identity
- 기존 `lerobot/smolvla_base` strict checkpoint load
- 실제 LIBERO camera key/order와 runtime patch spans
- 3 train seeds의 paired initialization/data/noise hash
- LIBERO success rate와 bootstrap CI
- 실제 selected spatial distribution/top-k mass
- FLOPs, peak GPU memory, median/p95 latency
- 2K/4K/6K/8K/10K checkpoint의 convergence curve

따라서 synthetic test 통과를 task success 개선의 증거로 해석하면 안 된다.

## Preregistered scheduler blocker

승인 계획은 run당 10,000 steps, warmup 1,000 steps, cosine decay horizon 100,000 steps를 고정한다. 그러나 `src/lerobot/optim/schedulers.py::CosineDecayWithWarmupSchedulerConfig.build`는 다음 조건을 적용한다.

```text
num_training_steps < num_decay_steps
→ scale_factor = num_training_steps / num_decay_steps
→ actual_warmup_steps = int(num_warmup_steps * scale_factor)
→ actual_decay_steps = num_training_steps
```

따라서 현재 CLI에 `steps=10,000`, `warmup=1,000`, `decay=100,000`을 주면 실제 scheduler는 warmup 100, decay 10,000으로 바뀐다. 이는 승인된 `03_experiment_plan.md`와 다르다.

다음 중 하나를 결과 관찰 전에 amendment로 승인해야 한다.

1. LeRobot native auto-scaling을 받아들여 실제 warmup 100/decay 10K로 계획을 수정한다.
2. scheduler에 auto-scaling opt-out을 최소 추가해 10K run에서도 warmup 1K/decay horizon 100K를 보존한다.
3. run을 100K로 늘린다. 이는 승인된 training budget을 변경하므로 권장하지 않는다.

사용자는 LeRobot native auto-scaling을 유지하는 option 1을 결과 관찰 전에 승인했다. 자세한 내용은 `03_amendment_01_scheduler.md`에 고정했으며, 실제 scheduler는 warmup 100/decay 10K다.

## Remote GPU 실행 계약

### 조건 행렬

| Variant | Keep ratio | Seeds | Train checkpoints | Final evaluation |
|---|---:|---|---|---|
| Dense | 1.0 | 1000, 1001, 1002 | 2K/4K/6K/8K/10K | 모든 checkpoint |
| Focus-50 | 0.5 | 1000, 1001, 1002 | 2K/4K/6K/8K/10K | 모든 checkpoint |
| Focus-25 | 0.25 | 1000, 1001, 1002 | 10K | final only |
| Focus-75 | 0.75 | 1000, 1001, 1002 | 10K | final only |

모든 variant는 다음을 고정한다.

```yaml
policy.type: smolvla
policy.focus_token_start_layer: 8
policy.compile_model: false
policy.num_vlm_layers: 16
policy.num_expert_layers: -1
policy.attention_mode: cross_attn
policy.self_attn_every_n_layers: 2
batch_size: 4
steps: 10000
optimizer.lr: 1.0e-4
optimizer.betas: [0.9, 0.95]
optimizer.eps: 1.0e-8
optimizer.weight_decay: 1.0e-10
```

scheduler는 native auto-scaling을 적용해 actual warmup 100/decay 10K로 확정한다. resolved config와 auto-scaling log를 반드시 기록한다.

### 실행 전 필수 gate

1. Linux remote에서 `uv sync --locked --extra test --extra libero`
2. focused test `3 passed`
3. pretrained checkpoint strict-load smoke test
4. real-model Dense ratio 1.0 identity test
5. dataset camera key/order와 patch span 기록
6. 동일 seed의 Dense/Focus initial state hash 일치
7. scheduler resolved warmup/decay가 amendment와 일치

### Append-only raw artifact schema

권장 경로:

```text
experiments/focus_token/results/<run_id>/<variant>/seed_<seed>/
```

run별 필수 파일:

- `metadata.json`: git commit, dirty diff hash, hostname, GPU, CUDA, Python, dependency lock hash
- `resolved_config.yaml`: 모든 train/eval 설정과 실제 scheduler 값
- `initial_state.sha256`: paired initialization 검증
- `train_metrics.jsonl`: step, loss, learning rate, token diagnostics
- `eval_episodes.jsonl`: variant, train seed, checkpoint, suite, task, episode seed, success, failure reason
- `efficiency.json`: FLOPs scope, peak memory, warmup count, measured chunks, median/p95 latency
- `selection_metrics.jsonl`: camera, layer, denoising step, valid/selected token 수, top-k mass, spatial indices

기존 run directory를 재사용하거나 덮어쓰지 않는다. 실패/재시도 episode도 원 기록을 보존한다.

## 현재 결론

구현은 승인된 late action-aware Focus-50 ablation과 정합하고 focused runtime test를 통과했다. Scheduler 충돌은 native auto-scaling을 유지하는 amendment로 해소됐다. 실제 baseline/intervention task 성능은 아직 측정되지 않았으므로 현재 paper-method 결과는 **remote execution pending / inconclusive**이며 가설을 supported 또는 unsupported로 판정할 수 없다.
