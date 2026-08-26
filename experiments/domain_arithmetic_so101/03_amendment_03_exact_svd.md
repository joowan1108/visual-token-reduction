# Amendment 03: exploratory exact-SVD DArT

Date: 2026-08-26. Status: adopted after observing exploratory failures from direct arithmetic and
rank-256 DArT. This is an outcome-informed implementation-fidelity sensitivity analysis and cannot
replace or be reported as the untouched preregistered D condition.

The base, source update, target update, one-shot data, alpha `0.8`, 99.75% energy cutoff, training
budgets, processor statistics, task, cameras, robot safety settings, and rollout success definition
remain unchanged. Training is not repeated: exact-SVD DArT reuses the already frozen base/source/
target checkpoints and performs one new merge. Prior direct and randomized-SVD artifacts and all
pilot observations are preserved.

The only intervention is replacing rank-256 randomized SVD with exact economy SVD via
`torch.linalg.svd(..., full_matrices=False)`, retaining all `min(m, n)` components. The merge RNG
and rank are removed because the exact decomposition does not use them. The new artifact must use
a fresh output path/run root and record `svd.implementation=torch.linalg.svd`,
`svd.full_matrices=false`, plus input model hashes.

Any exact-SVD rollout is exploratory. Evidence against exact SVD is a valid, interface-checked
artifact that still cannot complete the task under the same hardware setup. A positive result must
be reported as post-hoc sensitivity evidence and requires a new prospective evaluation plan before
confirmatory claims.
