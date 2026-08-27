# Preregistered amendment 04: Experiment M multi-task anchor

## Status

Authorized on 2026-08-27 before observing any Experiment M rollout outcomes. This is a separate
experiment from the specialized-anchor protocol; prior raw data and artifacts remain unchanged.

## Changed hypothesis condition

The immutable base becomes:

```text
Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod
@45f76f173c76c4e002131f8b48e345589a071d0f
```

The source fine-tune uses exactly episode `170` from:

```text
Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi_10fps
@09a0376348f60be89edcbc0eb76c3e26b5f3b094
```

The target candidate is episode `0` from:

```text
sungkyunner/record-test_20260826_210214
@295e6def6cb4df454f58894caea10c15446dc4e4
```

Training is forbidden until the source and target top-camera frames are confirmed to represent the
same approximate red-block start position. A failed visual-match gate requires an append-only
amendment and a fresh run root.

## Frozen training and merge

- Source and target independently start from the pinned Experiment M base.
- Exactly one trajectory is used per fine-tune.
- Seed `1000`; 1,000 optimizer updates each; AdamW LR `5e-5`; batch `8`; accumulation `8`;
  workers `0`; AMP and image augmentation disabled.
- Both use the base processor statistics, exact task text, two camera roles, six degree-valued joint
  positions, and 10 FPS.
- Direct and exact-thin-SVD DArT use `alpha=0.8` and add their domain vector back to this same base.
- No prior source, target, direct, or DArT checkpoint may be reused.

## Conditions and evaluation

The four conditions remain Z (unchanged base), F (target one-shot fine-tune), A (direct arithmetic),
and D (DArT), now all anchored to Experiment M. The existing 24 paired blocks, binary success
definition, one-sided exact McNemar test, `+20` percentage-point practical threshold, seeds, safety
rules, and falsification criteria remain unchanged.

The hypothesis is unsupported if protocol gates pass but D versus Z fails either preregistered
support criterion. It is invalid if source/target layouts or interfaces do not match, either
fine-tune uses more than one episode or different optimization, or any artifact from the prior base
is mixed into this run.

