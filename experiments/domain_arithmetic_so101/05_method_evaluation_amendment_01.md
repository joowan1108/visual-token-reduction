# Method evaluation: public target episode amendment

## Verdict

There is no remaining implementation blocker to starting the pinned source and target training.
No training, hardware evaluation, or experimental outcome has been produced.

The prepared target is atomically derived from public episode `0` only, validates its
329-frame/10-FPS/task/core-feature contract, and canonicalizes only the state and action gripper
columns with the frozen affine mapping. It preserves frame/video alignment through LeRobot's
dataset writer and records sorted per-file and aggregate SHA-256 provenance before training. This
prepared one-episode dataset is the sole target fine-tuning input.

Source and target optimization settings remain symmetric and both retain the base processor
statistics. Direct and DArT arithmetic, model and source/target revision pins, and Z/F/A/D policy
resolution are intact. Explicit and implicit Z resolution both pin the frozen base revision, and
formal evaluation records one local two-camera episode per unique trial ID.

## Remaining pre-hardware gates

- Load Z/F/A/D as normal SmolVLA artifacts and verify finite six-dimensional actions on one saved
  observation.
- Record processor and final artifact hashes and verify the base processors are identical for the
  arithmetic conditions.
- Verify before outcomes that the evaluation robot, gripper calibration, workspace, and camera
  viewpoints match the public demonstration rig closely enough for the amended claim.
- Freeze the randomized 96-trial manifest, then retain the rollout datasets/videos and append-only
  blinded scores required by `03_experiment_plan.md`.

These are preregistered execution gates, not implementation failures or scientific results.
