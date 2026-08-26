# Literature review: DArT for one-shot SO-101 sim-to-real adaptation

## Method

[Domain ARiThmetic (DArT)](https://arxiv.org/pdf/2607.00666) adapts a source-trained VLA by isolating the parameter direction associated with an environmental shift. It assumes task- and domain-specific changes produced by one-shot fine-tuning are approximately additive and separable in weight space.

Let `theta_0` be the immutable source policy, `D_m,src` one source-domain demonstration, and `D_m,tgt` one target-domain demonstration of the same task. Independently fine-tuning from `theta_0` gives `theta_m,src` and `theta_m,tgt`, with layer updates:

```text
Delta_m,d^(l) = theta_m,d^(l) - theta_0^(l), d in {src, tgt}
```

Direct domain arithmetic is:

```text
delta_tgt^(l) = Delta_m,tgt^(l) - Delta_m,src^(l)
```

For each 2-D weight, DArT refines this using the left singular bases of the source and target updates. The source-to-target alignment score is:

```text
gamma^(l) = ||U_tgt U_tgt^T Delta_src||_F / ||Delta_src||_F
```

It computes `C = U_tgt^T U_src`, scores each source basis by `e_j = ||C_:,j||_2^2`, and keeps the smallest high-overlap set whose cumulative energy reaches `gamma` times total overlap energy. With the retained source basis `U_src_aligned`:

```text
delta_tgt_refined^(l) = gamma^(l) *
    (Delta_tgt^(l) - U_src_aligned U_src_aligned^T Delta_src^(l))

theta_star = theta_0 + alpha * delta_tgt_refined
```

Non-2-D tensors use direct subtraction without SVD filtering or alignment scaling. The full method is specified in [paper Algorithm 1](https://arxiv.org/pdf/2607.00666#page=22). DArT adds no inference-time module: `theta_star` runs as an ordinary policy.

## What one-shot means

One-shot means one complete, successful target-domain expert trajectory for one task, not one frame or one gradient example. DArT additionally consumes one existing source-domain trajectory of the same task. The data budget is therefore one newly collected real trajectory plus one already available simulation trajectory.

The proposed experiment can use episode `0` from `CoRL2026-CSI/IsaacLab-SO101-PickAndPlace-100epi-10fps-appendix` as the fixed source trajectory and one newly recorded SO-101 trajectory with the identical instruction as the target trajectory.

## Paper protocol

- Both one-shot policies start from bitwise-identical base parameters.
- Both use AdamW, effective batch size `64`, peak learning rate `5e-5`, no warmup, and `1,000` optimizer steps.
- The source and target runs use the same trainable parameter mask and preprocessing.
- Source-policy normalization statistics remain fixed for fine-tuning and evaluation.
- The main viewpoint and real-world scaling coefficient is `alpha=0.8`.
- The paper's randomized SVD uses rank `256`; it reduces merge time from 15m35s to 6m33s with a 0.4 percentage-point average success reduction relative to full SVD ([Table 8](https://arxiv.org/pdf/2607.00666#page=23)).

The official implementation warns that its current full merge can use more than 100 GB RAM and supports OpenPI/JAX `pi0.5` checkpoints, not PyTorch SmolVLA checkpoints ([official repository](https://github.com/snumprlab/dart)).

## Evidence

The paper compares the unchanged base, ordinary target one-shot fine-tuning, FLA, RETAIN, and DArT. Its primary measure is binary task success; MimicGen additionally uses milestone progress.

| Setting | Base | One-shot FT | DArT |
|---|---:|---:|---:|
| `pi0.5` LIBERO viewpoint average | 54.5% | 31.5% | 79.1% |
| `pi0-FAST` LIBERO viewpoint average | 73.4% | 62.1% | 79.4% |
| `pi0.5` combined visual shifts | 60.5% | 29.8% | 75.0% |
| MimicGen cross-embodiment success | 62.0% | 56.4% | 69.4% |
| Real UR10e five-task average | 43.3% | 51.7% | 81.7% |

The real experiment used one target Stack Cube demonstration and 12 fixed object positions per task. The paper reports averages but no confidence intervals, hypothesis tests, or p-values. This SO-101 experiment therefore needs its own paired binary-success analysis.

## Assumptions

- The base policy already solves the adaptation task in simulation.
- A source trajectory matching the real task is available.
- Source and target share instruction, state/action dimensions and ordering, joint units, camera semantics, preprocessing, and normalization.
- Both fine-tunes start from the exact same base and use identical optimization settings.
- The target environment shift remains stable during adaptation and evaluation.
- A single successful real trajectory exposes enough of the target shift.
- `alpha` is fixed before final evaluation.

Exact task matching matters. The paper reports 80.8% for matching demonstrations on LIBERO Medium, 67.7–69.0% for similarity-retrieved tasks, and 57.7–62.7% for random source tasks ([Table 22](https://arxiv.org/pdf/2607.00666#page=37)).

## Applicability to the requested SmolVLA

The experiment is feasible as a native PyTorch/safetensors port, not as a direct invocation of the released OpenPI/JAX code.

Positive compatibility:

- SmolVLA is flow-matching based, like the paper's primary `pi0.5` model.
- The source checkpoint and dataset are public.
- Source and target both use a six-joint SO-101 interface.
- The source data has the exact instruction and top/wrist images at 10 FPS.

Material extrapolations and constraints:

1. The paper does not evaluate SmolVLA or sim-to-real transfer. Its real experiment shifts camera viewpoint between two real setups; this experiment also shifts appearance, calibration, latency, dynamics, and contact physics.
2. The requested checkpoint is task-specific. This tests same-task sim-to-real adaptation, not DArT's central cross-task capability-retention claim.
3. DArT uses full-model one-shot updates. The checkpoint defaults to a frozen vision encoder and expert-only training, so both fine-tunes must explicitly use the same full-model trainable mask.
4. Both fine-tunes and the merged output must reuse the base checkpoint's normalization processors. One-trajectory statistics must not replace them.
5. The checkpoint declares three visual slots, while the source dataset contains wrist and top images only. Source training, target training, and inference must use identical two-camera semantics and missing-camera behavior.
6. The real trajectory must preserve source joint ordering, degree-valued joint targets, gripper convention, 10 FPS rate, and exact instruction. Otherwise the intervention changes the interface rather than only the domain.

## Required comparisons

At minimum, evaluate:

1. unchanged base checkpoint;
2. ordinary target one-shot fine-tuning;
3. direct arithmetic without subspace alignment;
4. full DArT.

Without the target one-shot baseline, improvement over the base cannot show that domain arithmetic is better than ordinary one-shot adaptation. Results are unsupported if source/target interfaces cannot remain identical, more than one real demonstration is used, or DArT does not improve the preregistered primary success measure.

## Amendment 02 evidence: same-rig target demonstration

The replacement target is pinned to
[`sungkyunner/record-test_20260825_225339@97e2c1d4d49607210d1e63d46db2a43b530bdf89`](https://huggingface.co/datasets/sungkyunner/record-test_20260825_225339/tree/97e2c1d4d49607210d1e63d46db2a43b530bdf89).
Its immutable metadata contains exactly one 300-frame episode at 10 FPS with the exact source task,
six identically named degree-valued SO-101 joints, and 640x480 left-wrist/top videos. Manual review
of both videos before retraining confirmed one complete successful red-block-to-blue-dish trajectory.

This is closer to the paper's real-world protocol than the superseded other-rig demonstration:
DArT uses one target demonstration collected in the target environment and the corresponding
same-task source demonstration, with both updates independently initialized from the same base.
The paper's real experiment likewise collected the target demonstration and evaluated on the same
physical target setup. The evidence remains indirect because the paper evaluates Pi0.5/Pi0-FAST
and mostly real-to-real viewpoint shifts, not SmolVLA sim-to-real transfer.

The dataset already uses the current `[0, 100]` gripper convention and exact task string. The old
gripper affine conversion must not be applied. Existing base and superseded-DArT hardware runs were
observed before this replacement; they are disclosed infrastructure pilots only and cannot enter the
confirmatory analysis. All new target, direct, and DArT artifacts and formal outcomes must come from
a fresh amended run.

## Amendment 03 evidence: exact thin SVD

The paper's primary algorithm and released `domain_arithmetic/dart.py` use exact economy/thin SVD
(`full_matrices=False`) and retain all `min(m, n)` components before DArT's 99.75%-energy filter.
Appendix Table 8 reports 79.1% average success and 15m35s merge time for exact SVD versus 78.7%
and 6m33s for rank-256 randomized SVD on Pi0.5 LIBERO viewpoint shifts. The reported +0.4-point
average difference has no uncertainty analysis and is indirect for SmolVLA sim-to-real transfer.
Exact SVD therefore improves implementation fidelity but is not evidence that it will rescue the
observed failures of direct arithmetic or randomized-SVD DArT.
