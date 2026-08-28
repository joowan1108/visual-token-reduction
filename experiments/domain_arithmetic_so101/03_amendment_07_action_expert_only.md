# Preregistered amendment 07: VLM-frozen expert/projection fine-tuning

## Status

Authorized by the user on 2026-08-28 before observing results from the amended run. Historical
full-scope training plans, checkpoints, merges, and rollout records remain unchanged and are not
reused as results of this amendment.

## Changed training scope

Both Experiment M one-trajectory fine-tunes now use:

```text
policy.train_expert_only=true
policy.freeze_vision_encoder=true
policy.train_state_proj=true
```

Under SmolVLA's built-in mask, `train_expert_only=true` freezes the entire VLM, including its vision
components with `freeze_vision_encoder=true`; the action expert and policy projection modules remain
trainable, and `train_state_proj` remains true. The source and target masks are identical. A fresh
`RUN_ROOT` is required.

The multi-task base, source episode `257`, target
`sungkyunner/record-test_20260826_191215@25589b8ebb14255c885edabb36168f5e36a6bafa`
episode `0`, optimizer settings, training budget, processor statistics, masked third-camera slot,
merge arithmetic, evaluation protocol, metrics, and falsification criteria remain unchanged.
