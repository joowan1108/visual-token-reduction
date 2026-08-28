# Preregistered amendment 05: replacement Experiment M target

## Status

Authorized on 2026-08-28 before any rollout outcome from this target was observed. Previous target
datasets, checkpoints, merges, and rollout artifacts remain unchanged and cannot be relabeled or
reused as results of this amendment.

## Immutable target

Experiment M now uses the public dataset:

```text
sungkyunner/record-test_20260826_191215
@25589b8ebb14255c885edabb36168f5e36a6bafa
episode 0
```

The pinned dataset contains one 300-frame episode at 10 FPS. Its exact task/interface contract is
the existing SO-101 pick-and-place contract: 640x480 left-wrist and top videos, six float32 joint
state/action values in the frozen order, and the unchanged task instruction.

## Execution constraint

This target requires a fresh `RUN_ROOT`; prior source, target, direct, or DArT artifacts are not
mixed into the amended run. Source/base pins, training hyperparameters, processor-statistics rules,
merge arithmetic, alpha, metrics, and evaluation protocol remain unchanged. Source and target
training and every rollout continue to set `policy.empty_cameras=1`, providing the three-camera
SmolVLA base a masked third slot while retaining the wrist/top camera mapping.
