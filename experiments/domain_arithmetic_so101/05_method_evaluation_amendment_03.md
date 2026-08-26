# Amendment 03 method evaluation: exact-SVD sensitivity

## Verdict

The implementation is method-compatible with the released DArT exact-SVD arithmetic. It uses exact
economy SVD, not `full_matrices=True`: every `min(m, n)` singular component is computed before
DArT's intended 99.75%-energy and greedy-overlap filtering.

## Exact reproduction and adaptation

Reproduced from the official implementation:

- `torch.linalg.svd(..., full_matrices=False)` corresponds to the official
  `jnp.linalg.svd(..., full_matrices=False)`;
- all thin-spectrum components are retained;
- target singular-energy filtering, alignment ratio, source-basis overlap selection, projected
  source subtraction, and `alpha * gamma` scaling are unchanged;
- rank-1 tensors use direct arithmetic.

Necessary SmolVLA adaptations remain unchanged: PyTorch/safetensors loading, per-tensor float32 CPU
arithmetic, rank-4 patch-embedding flatten/restore, base processor preservation, strict key/shape/
dtype checks, and native LeRobot checkpoint output. Explicit zero handling and gamma clamping replace
the official implementation's `1e-6` denominator guards; this is an existing numerical port
deviation, not an exact-SVD change.

## Matched protocol

The exact condition reuses the frozen base, source episode-0 fine-tune, and one-shot real target
fine-tune. No dataset split, preprocessing, normalization, training seed, optimizer step, checkpoint,
alpha, or hardware setting changes. Source and target checkpoints are passed through
`SOURCE_CHECKPOINT` and `TARGET_CHECKPOINT`; only one fresh merge is run.

The preregistered rank-256 DArT condition remains the confirmatory intervention. Exact SVD is a
post-hoc sensitivity condition. Any rollout must be reported descriptively against the unchanged
base, target fine-tune, direct arithmetic, and preserved rank-256 artifact under matched layouts.
The original binary success metric, paired rate differences, bootstrap interval, and trial-level
failure logging remain applicable, but exact-SVD results cannot enter the original McNemar claim or
replace the preregistered D condition.

The merge has no random seed and runs once. Training seed 1000 remains attached to the reused
checkpoints. Computational budget is one CPU float32 merge, stopping on checkpoint mismatch,
unsupported tensor shape, existing output path, SVD failure, or insufficient RAM. Actual peak RAM
and runtime must be recorded with `/usr/bin/time -v`; the documented 16 GB minimum has not yet been
validated on the complete 0.5B checkpoint.

## Provenance

`dart_merge.json` records the base/source/target resolved paths or revisions, SHA-256 hashes, alpha,
energy cutoff, tensor counts, zero-update count, and:

- `svd.implementation = torch.linalg.svd`
- `svd.full_matrices = false`
- `svd.retained_components = all`

A fresh `RUN_ROOT` prevents overwriting the prior randomized-SVD and direct artifacts.

## Verification

Passed: shell syntax and workflow condition checks, Python bytecode compilation, focused Ruff checks,
and diff whitespace validation.

Blocked: focused pytest collection failed before any test body ran because local PyTorch cannot load
`libcublasLt.so.12`. No complete-checkpoint exact merge, model forward pass, hardware rollout, or
matched baseline/intervention evaluation was run.

No machine-readable experimental results were produced under `results/raw/`. Therefore this review
supports implementation fidelity only, not effectiveness.
