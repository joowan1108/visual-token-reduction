# Experiment plan amendment 01: Native scheduler auto-scaling

## Decision

사용자 승인에 따라 LeRobot의 `CosineDecayWithWarmupSchedulerConfig` 자동 축소를 변경하거나 우회하지 않는다.

Remote run은 기존 설정값을 그대로 제공한다.

```yaml
steps: 10000
policy.scheduler_warmup_steps: 1000
policy.scheduler_decay_steps: 100000
policy.scheduler_decay_lr: 2.5e-6
```

LeRobot이 `num_training_steps < num_decay_steps` 조건에서 적용하는 resolved scheduler는 다음으로 고정한다.

```yaml
actual_warmup_steps: 100
actual_decay_steps: 10000
```

## Scope

- Dense, Focus-50, Focus-25, Focus-75의 모든 seed에 동일하게 적용한다.
- optimizer, peak/final learning rate, 10K training budget은 변경하지 않는다.
- scheduler source를 수정하지 않는다.
- resolved config와 auto-scaling log를 run artifact에 저장한다.
- primary metric, seeds, evaluation budget, bootstrap, falsification criteria는 변경하지 않는다.

## Timing and rationale

이 결정은 실제 LIBERO training/evaluation 결과를 관찰하기 전에 이루어졌다. 목적은 승인 계획과 현재 LeRobot native execution semantics의 불일치를 해소하고 remote run을 재현 가능하게 만드는 것이다.
