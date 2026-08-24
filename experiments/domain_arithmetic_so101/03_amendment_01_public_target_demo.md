# Amendment 01: pinned public target-domain demonstration

Date: 2026-08-24. Status: adopted before running the affected training, merge, or real-world evaluation conditions; no outcomes are reported here.

## Change

The target one-shot trajectory is frozen to episode `0` of
`skkuprism/test_pick_red_place_blue_50epi_10fps@e19331e77f477a4be16f7c2884250ed6f491e048`.
It is a pre-existing public real-world demonstration with 329 frames over 32.9 seconds at 10 FPS,
showing a successful red-block-to-blue-dish trajectory from `left_wrist` and `top` views. Training
must select only `dataset.episodes=[0]`; the other episodes are out of scope.

This replaces only the plan's requirement to newly collect the target training episode. The source
checkpoint, source episode, optimizer budget, normalization, merge settings, metrics, statistical
tests, and falsification rules remain frozen.

## Declared deviations and claim limit

The source task string is `Pick up the red block and place it on the blue dish.` while the target
metadata string is `pick up the red block and place it on the blue dish`. The workflow does not
rewrite either dataset or introduce task canonicalization: those would add a new preprocessing
intervention. Consequently, the source/target update difference may include this capitalization and
punctuation change as well as the environmental shift.

Because the target episode was collected on a pre-existing candidate rig, it supports adaptation
only to that rig's camera placement, appearance, embodiment calibration, and dynamics. Formal
real-world evaluation is interpretable only on the matching physical setup (or a setup verified to
be equivalent before outcomes are observed). It does not establish adaptation to an arbitrary
SO-101 rig or to the current user's hardware merely because the robot and camera labels match.

This amendment supersedes only the task-string-equality clause of the coordinate-system invariant;
camera roles, state/action names and units, joint ordering, gripper convention, processor hashes,
and every other interface check remain mandatory.

The plan's assumed center adaptation placement is also superseded. The excluded adaptation layout
is the initial red-block placement visibly demonstrated in pinned target episode `0`, identified
relative to the fixed workspace and dish in that episode without asserting unobserved coordinates.
That demonstrated placement must be excluded from the 12 evaluation layouts whether or not it is
at the center of the physical workspace.

## Interface conversion correction

Metadata verification found that pinned episode `0` stores both `observation.state[-1]` and
`action[-1]` in the old normalized gripper convention, while the source/base interface and current
SO follower use gripper range `[0, 100]`. Before either fine-tune, the target episode is therefore
materialized once as a local one-episode dataset with the exact affine conversion
`new = (old + 100) / 2` applied to those two gripper columns. The five arm-joint columns are copied
unchanged. Its task is deterministically set to the frozen source prompt
`Pick up the red block and place it on the blue dish.`. Images are copied through the normal
LeRobot dataset/video APIs without augmentation.

This deterministic preparation supersedes the earlier decision to retain the target prompt and
gripper deviations. It restores those interface invariants rather than introducing a learned or
outcome-selected intervention. The raw public dataset remains unchanged, and the derived dataset
records its pinned source revision, episode, feature set, task replacement, and affine formulas.
The physical-rig claim limit above remains in force.

Before training, the preparation provenance also freezes SHA-256 for every derived dataset file
and a deterministic aggregate tree hash over sorted relative paths; the provenance file itself is
excluded to avoid self-reference.
