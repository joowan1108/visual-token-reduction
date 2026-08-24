# Method evaluation: SO-101 one-shot DArT

## Verdict

The implementation is suitable for producing the four preregistered policies. Hardware evaluation
must wait until the real demonstration exists and every pre-hardware gate in `03_experiment_plan.md`
has passed. No scientific result exists yet.

## Review findings

### Resolved during review — recorded evaluation trials

The initial workflow used the non-recording `base` rollout strategy, which could not provide videos
for blinded scoring. The final workflow adds an `evaluate` command using the existing `episodic`
strategy. It records one local-only 30-second video episode under a required unique, anonymized
`TRIAL_ID`; the original `rollout` command remains an optional non-recording smoke test.

### Required pre-hardware validation

Before collecting outcomes, archive:

- the target dataset episode count, task, FPS, camera names and roles, joint names and order, units,
  and tree hash;
- Z/F/A/D model key, shape, and dtype checks;
- base/A/D processor SHA-256 equality;
- one saved-observation forward smoke test showing finite six-dimensional action chunks for all four
  policies;
- final checkpoint and merge artifact hashes.

These checks require the real demonstration and trained artifacts, so they cannot run as part of this
code-only review. Failure of any gate invalidates the experiment rather than supporting or refuting
the hypothesis.

### Documented numerical deviation

The released JAX implementation adds `1e-6` to its singular-energy and source-norm denominators. This
port handles exact zero updates separately and otherwise follows the paper equations without epsilon.
The difference should be negligible for ordinary nonzero updates but can matter for extremely small
tensors. This is a declared port deviation.

### Local checkpoint provenance

Merge metadata records resolved paths and Hub revisions but does not hash local source and target
weights. Compute and archive SHA-256 hashes before merging and include them in each append-only trial
row, as required by the frozen plan.

## Confirmed fidelity

- Direct arithmetic implements `theta_0 + alpha * (tau_tgt - tau_src)`.
- DArT uses the target left-singular subspace, 99.75% target-energy cutoff, source alignment ratio,
  greedy source-basis selection with tied energies retained, projected source subtraction, and
  `alpha * gamma` scaling.
- Rank-1 tensors use direct arithmetic; rank-2 tensors use DArT; rank-4 convolution weights are
  flattened to `[out, -1]` and restored.
- The merge requires identical floating key sets, shapes, and dtypes, computes in float32, restores
  base dtypes, and emits a normal LeRobot checkpoint with unchanged base processor artifacts.
- `preserve_pretrained_processor_stats` defaults to false and prevents one-trajectory statistics from
  replacing base statistics only when explicitly enabled.
- Source and target runs independently start from the pinned base and symmetrically use AdamW, 1,000
  updates, effective batch 64, seed 1000, no warmup, no augmentation, no AMP, and the same trainable
  parameter mask.
- Source episode 0 and exactly one target episode are selected. Camera roles, instruction, FPS,
  degree-valued control, duration, and relative-target safety limit are consistent.
- Unit coverage includes direct and literal DArT equations, zero updates, mismatches, native output,
  processor preservation, flag behavior, and the deterministic randomized-SVD branch.

## Claim limits

- SmolVLA retains its built-in frozen final VLM/head parameters despite the two full-fine-tuning
  overrides; this is an acknowledged architecture-specific deviation from the paper.
- Rank-256 randomized SVD approximates the released exact-SVD algorithm.
- The paper evaluates Pi-family policies, not SmolVLA.
- The study supports only same-task, one-checkpoint, one-environment SO-101 sim-to-real claims.
- With 24 paired blocks, the exact McNemar design is valid but has limited power. Failure to meet the
  frozen criteria is unsupported, not proof that DArT generally does not work.

## Test environment

Static syntax, Ruff, shell parsing, and diff checks pass. Focused pytest could not collect in this
mounted workspace because the installed PyTorch environment cannot load `libcublasLt.so.12`; this is
an environment failure, not a test result. No hardware or model forward test was run.
