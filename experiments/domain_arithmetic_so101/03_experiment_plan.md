# Preregistered experiment plan: one-shot DArT for SO-101 sim-to-real transfer

## 1. Status and immutable references

- Status: implementation authorized by the user on 2026-08-24; no real-world outcomes observed.
- Hypothesis: `00_hypothesis.md`.
- Evidence review: `01_literature_review.md`.
- Implementation audit: `02_implementation_map.md`.
- LeRobot pre-implementation commit: `a677886392cdfb82af41584353a73376de737dc0` on `inference/smolvla-so101`.
- DArT reference: `snumprlab/dart@1b23c4f42f73168c78a20b353453145e74f64711`.
- Paper: arXiv `2607.00666v1`.
- Base model: `CoRL2026-CSI/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep-appendix@75d5905c5e27ba6f0a738cbcfcb167e7769dce0d`.
- Source data: `CoRL2026-CSI/IsaacLab-SO101-PickAndPlace-100epi-10fps-appendix@2b739e6be9b341e6359265ed99be81458ed4d879`, episode `0` only.

This document freezes the primary metric, conditions, seed, budget, statistics, and outcome rules before implementation and data collection. Any necessary change is an append-only amendment created before looking at affected outcomes. Raw demonstrations, rollouts, logs, and result rows are never overwritten.

## 2. Question and permitted claim

### Primary question

For the exact task `Pick up the red block and place it on the blue dish.`, does DArT adaptation using one simulation trajectory and one newly collected real SO-101 trajectory increase real-world binary task success over the unchanged simulation-trained SmolVLA checkpoint?

### Claim limit

The strongest permitted claim is same-task, one-environment, one-shot SO-101 sim-to-real adaptation for this checkpoint. The experiment cannot establish SmolVLA-wide, multi-task, cross-task, or general sim-to-real effectiveness.

## 3. Fixed data and interface

### Source trajectory

- Dataset/revision and episode are pinned above.
- Instruction: exact task string above.
- Cameras: left wrist and top, mapped to policy `camera1` and `camera2`.
- Rate: 10 FPS.
- State/action: six SO-101 joint positions/targets in dataset order and degrees.

### Target trajectory

- Exactly one successful real-world expert trajectory is used for training.
- Record at 10 FPS with the same instruction, joint ordering/units, 640×480 wrist/top camera roles, and no image augmentation.
- The adaptation placement is the center training placement and is excluded from evaluation layouts.
- Failed collection attempts are retained in an append-only attempt log but are not training demonstrations.
- After the first successful trajectory is accepted, no replacement, trimming, relabeling, or additional target trajectory is allowed.
- Record the resulting local dataset tree hash and, if uploaded, immutable Hub revision before training.

### Coordinate-system invariant

Both fine-tunes and every evaluated policy use the base checkpoint's saved normalizer and unnormalizer statistics. Per-trajectory normalization is forbidden. A mismatch in task string, camera roles, state/action names, degrees, gripper convention, or processor hashes invalidates the experiment rather than counting as a negative result.

## 4. Conditions

| ID | Policy | Purpose |
|---|---|---|
| Z | Immutable base checkpoint | Zero-shot sim-to-real baseline |
| F | Base fully fine-tuned on the one real trajectory | Ordinary one-shot fine-tuning baseline |
| A | `theta_0 + 0.8 * (Delta_tgt - Delta_src)` | Direct arithmetic ablation without subspace alignment |
| D | Rank-256 randomized-SVD DArT, `alpha=0.8` | Primary intervention |

All conditions use the base config and processor artifacts, same two physical cameras, task text, 10 FPS loop, 30-second horizon, degree-valued actions, and `robot.max_relative_target=5` safety limit.

No alpha, rank, checkpoint, task wording, camera transform, runtime, or safety-limit sweep is permitted using evaluation outcomes.

## 5. One-shot fine-tuning

Source and target runs independently start from the pinned base revision.

| Item | Frozen value |
|---|---|
| Training seed | `1000` |
| Optimizer | AdamW |
| Peak/constant LR | `5e-5` |
| Betas | `(0.9, 0.95)` |
| Epsilon | `1e-8` |
| Weight decay | `1e-10` |
| Gradient clip | `1.0` |
| Warmup | `0` |
| Optimizer updates | `1,000` per run |
| Micro-batch | `8` |
| Gradient accumulation | `8` |
| Effective batch | `64` on one process |
| Image augmentation | disabled |
| AMP | disabled |
| cuDNN | deterministic |
| VLM/action scope | `freeze_vision_encoder=false`, `train_expert_only=false`; identical built-in SmolVLA freeze mask |
| Checkpoint | final step `1,000` only |
| Normalization | pinned base processor statistics |

Total training budget is exactly `2,000` optimizer updates: one source run and one target run. Interrupted runs may resume exact state; they may not restart under a different seed or select an earlier checkpoint by outcome.

## 6. Merge

- DArT scaling coefficient: `alpha=0.8`.
- Randomized-SVD target rank: `256`.
- Randomized-SVD seed: `42`.
- Target singular energy cutoff for alignment score: `99.75%` within the available basis.
- Exact full SVD is used when `min(matrix_shape) <= 256`.
- Computation: float32 CPU, cast back to base dtype.
- Non-2-D tensors: direct arithmetic.
- The 4-D patch embedding is flattened `[out, -1]`, merged, then restored.
- Identical key sets and shapes are mandatory; unsupported tensors fail the merge.
- Base config and all processor files are copied unchanged into A and D.

The direct A condition uses the same source/target checkpoints and alpha but no SVD filtering or gamma scaling.

## 7. Implementation gates

Before hardware evaluation:

1. Unit tests cover literal 1-D and 2-D equations, zero updates, mismatch failures, native checkpoint output, processor preservation, and the training-statistics flag's default/opt-in behavior.
2. Tiny deterministic checkpoints reproduce a literal independent implementation.
3. A and D artifacts have the base key set/shapes/dtypes and identical processor-file SHA-256 hashes.
4. Z, F, A, and D load through the normal SmolVLA policy loader and produce finite six-dimensional action chunks from the same saved observation.
5. The real target dataset contains exactly one episode and its interface matches the source contract.

Failure before outcome collection is an implementation or environment block, not evidence for or against the hypothesis.

## 8. Real-world evaluation

### Layouts and pairing

- Define 12 fixed evaluation layouts before adaptation: a 3×4 grid of red-block start positions with the blue dish fixed.
- Exclude the center adaptation-demo placement.
- Run two repetitions per layout: `24` paired trial blocks.
- Every block evaluates Z, F, A, and D from the same recorded block/dish placement and robot reset pose.
- Randomize the four-condition order within each block using seed `20260824`; freeze the generated manifest before the first evaluation.
- Total budget: `24 × 4 = 96` valid robot trials.

### Success and failure

A trial succeeds only when, within 30 seconds and without human intervention, the red block is released fully supported by the blue dish and remains there for at least two seconds. Grasp without placement, contact without stable support, dropping, timeout, manual correction, or a safety stop is failure.

A trial is invalid only for a pre-action infrastructure failure such as camera/serial disconnect or failure to enter the initial pose. Log it and repeat the same condition/block once hardware is restored. Post-action disconnects and safety stops count as failures. If more than 10% of attempted trials are invalid, report protocol instability and treat the experiment as inconclusive.

The operator scoring success must use saved videos and be blind to condition labels. Store raw rollout datasets/videos and one append-only row per attempted trial containing manifest ID, anonymized condition code, timestamps, success, invalid reason, safety-stop flag, and artifact hashes.

## 9. Metrics and statistics

### Primary metric

Paired absolute success-rate difference:

```text
Delta_primary = mean(success_D - success_Z) over the 24 paired blocks
```

The minimum practically meaningful improvement is `+20 percentage points`.

### Confirmatory test

Use a one-sided exact McNemar test of D versus Z on the 24 paired binary outcomes, alternative `D > Z`, significance level `0.05`. This is the sole confirmatory comparison, so no multiplicity correction is applied.

Report the paired effect and a two-sided 95% paired bootstrap percentile interval using 10,000 block resamples and seed `20260824` as descriptive uncertainty.

### Secondary diagnostics

- F minus Z: whether ordinary one-shot fine-tuning helps.
- D minus F: whether aligned arithmetic improves over ordinary one-shot fine-tuning.
- A minus Z and D minus A: value of arithmetic and subspace alignment.
- Per-layout and repetition success tables, safety-stop rate, inference latency, and action rate.

Secondary comparisons are descriptive and cannot rescue a failed primary result.

## 10. Frozen outcome rules

### Supported

The primary hypothesis is supported only if both are true:

1. `Delta_primary >= +20 percentage points`;
2. one-sided exact McNemar `p < 0.05` for D versus Z.

### Unsupported

The hypothesis is unsupported if all protocol gates pass but either support criterion fails. A negative D-minus-F result must additionally be reported plainly: even if D beats Z, the experiment would not show that DArT is better than ordinary one-shot fine-tuning.

### Inconclusive or invalid

Report inconclusive rather than unsupported when fewer than 24 valid paired blocks are completed, invalid trials exceed 10%, scoring blindness is broken, or a hardware/environment change makes blocks incomparable. Report invalid if more than one real training demonstration is used, normalization/interface invariants fail, or conditions use different training budgets/settings.

No primary threshold, trial exclusion, checkpoint, alpha, rank, or metric may be changed after any condition outcome is observed.
