# Method evaluation amendment 04: Experiment M

## Verdict

The Experiment M training and merge topology matches DArT: one pinned multi-task `theta_0`, two
independent and symmetric one-shot fine-tunes, updates computed against that same `theta_0`, and the
refined domain vector added back to it. Exact thin SVD and base processor preservation remain
unchanged.

The initial audit found two high-severity workflow gaps, both fixed before commit:

1. Target training now validates that `target_provenance.json` exactly matches the current target
   repo, immutable revision, episode, matched source identity, and confirmation flag.
2. `prepare-target` now requires `VISUAL_MATCH_CONFIRMED=1` and records the matched source dataset,
   revision, episode, and confirmation in provenance.

The operator must still perform the actual visual comparison before setting that flag. The full
24-block randomized evaluation manifest, append-only raw attempt table, blinded scoring, paired
McNemar test, and bootstrap interval remain hardware-evaluation work; no confirmatory claim is
permitted without them.

## Confirmed implementation properties

- Base: `Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod@45f76f...`.
- Source: exactly episode `170` from the pinned 3,300-episode IsaacLab dataset.
- Target: exactly one immutable selected episode, defaulting to the preregistered real dataset
  episode `0`.
- Both runs: seed 1000, 1,000 updates, batch 8, accumulation 8, workers 0, identical optimizer and
  trainable mask, no AMP/augmentation, and preserved base processors.
- Merge: same-run source/target checkpoints only; direct and DArT both use and return to the same
  Experiment M base.

## Validation

```text
bash syntax: passed
run.sh check: passed
Ruff: passed
git diff --check: passed
focused pytest: environment-blocked before collection
```

The pytest block is `ImportError: libcublasLt.so.12` from the local PyTorch installation, not a test
failure. The remote CUDA server must run the focused test before training.

## Remaining semantic limits

- This is a PyTorch SmolVLA port of an OpenPI/JAX method.
- Direct IsaacLab-to-real transfer is a broader shift than the paper's controlled viewpoint tests.
- Source packed-file hashes may cover colocated episodes; immutable revision plus episode remains
  the authoritative selection.
- Hardware evaluation still requires processor/model integrity checks and the frozen paired-trial
  protocol before interpreting success rates.

