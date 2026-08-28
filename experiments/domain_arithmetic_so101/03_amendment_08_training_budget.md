# Preregistered amendment 08: one-shot training budget

## Status

Authorized by the user on 2026-08-28 before observing results from the amended run. Historical
training configurations, checkpoints, merges, and rollout records remain unchanged and are not
reused as results of this amendment.

## Changed budget

Both source and target fine-tunes now use:

```text
optimizer updates: 1000
micro-batch size: 8
gradient accumulation steps: 8
effective batch size: 64
```

Checkpoint saving remains every 500 optimizer updates. A fresh `RUN_ROOT` is required.

The multi-task base, source episode `257`, target
`sungkyunner/record-test_20260826_191215@25589b8ebb14255c885edabb36168f5e36a6bafa`
episode `0`, expert/projection-only trainable mask, optimizer and scheduler settings, processor
statistics, masked third-camera slot, merge arithmetic, evaluation protocol, metrics, and
falsification criteria remain unchanged.
