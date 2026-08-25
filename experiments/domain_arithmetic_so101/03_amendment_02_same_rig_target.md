# Amendment 02: same-rig real target demonstration

Date: 2026-08-25. Status: adopted after exploratory hardware smoke tests of the unchanged base and
the superseded Amendment-01 DArT artifact, but before training or merging the replacement target
and before any formal randomized evaluation. The smoke observations and logs are pilot-only and
are excluded from every confirmatory result.

## Replacement target

The target one-shot trajectory is replaced by episode `0` of
`sungkyunner/record-test_20260825_225339@97e2c1d4d49607210d1e63d46db2a43b530bdf89`.
This supersedes Amendment 01's target dataset and its interface conversion, but does not alter the
base model, source episode, optimization budget, merge hyperparameters, metrics, statistics, or
falsification rules.

The pinned dataset contains exactly one 300-frame, 30-second episode at 10 FPS. It uses the exact
task `Pick up the red block and place it on the blue dish.`, 640x480 `left_wrist` and `top` videos,
and six float32 state/action joints in the source order and degree convention. Both videos were
reviewed before replacement training and show a complete successful trajectory that deposits the
red block in the blue dish and leaves it there. The user identifies this as recorded on the actual
rollout rig.

The demonstrated initial object placement is the adaptation placement and is excluded from the 12
formal evaluation layouts. Camera mounts, physical camera identities, robot calibration, objects,
lighting, reset pose, and workspace must remain fixed through formal evaluation.

## Interface and preparation

The new episode already has the source task and current `[0, 100]` gripper convention; observed
gripper values are positive and within that range. No task rewrite, affine conversion, trimming,
augmentation, or frame rewriting is permitted. `prepare-target` validates the pinned metadata and
records immutable Hub/file provenance only; `train-target` reads episode `0` directly from the
pinned Hub revision with the unchanged wrist/top rename map and base processor statistics.

The AV1 videos are decoded with PyAV. The remote training container uses DataLoader workers `0` to
avoid its constrained `/dev/shm`; micro-batch `8`, accumulation `8`, and effective batch `64` remain
unchanged.

## Artifact separation and source reuse

All replacement target, direct, and DArT outputs use a fresh `RUN_ROOT`; Amendment-01 target and
merged artifacts are never overwritten or relabeled. The frozen source update may be reused only
if its base revision, optimizer/training configuration, final step, processor hashes, and model
hash match the preregistered source run. Otherwise source training is rerun. The merge records the
actual source path and hash.

## Runtime amendment before formal evaluation

Infrastructure smoke testing established that the physical cameras expose 640x480 at 30 FPS rather
than 10 FPS. They capture at 30 FPS while the policy observes and acts at the frozen 10 Hz loop.
All Z/F/A/D formal trials use RTC inference with execution horizon `10`, maximum guidance weight
`10`, `robot.max_relative_target=5`, and the same hardware/runtime configuration. This setting is
fixed for the replacement experiment and may not vary by condition or be tuned from outcomes.

The original 24 paired blocks, 96 valid trials, randomized manifest seed, blinded scoring, success
definition, primary metric, statistical test, and support thresholds remain unchanged. No pilot
rollout may be counted as a formal trial.
