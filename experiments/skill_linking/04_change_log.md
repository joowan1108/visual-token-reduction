# P1 implementation change log

## Scope

Implemented the approved P1 intervention without creating a copied pair dataset and without changing LIBERO evaluation success logic, the shared Euler integrator, RTC, or focus-token code.

## Data path

- `SmolVLAConfig` now exposes opt-in skill-linking fields and requests the inclusive `H+1` `subtask_index` window only when enabled.
- `SkillLinkingSampler` builds index views over the original train episodes:
  - every atomic start whose `H+1` semantic labels are constant;
  - exactly one `b-H` start per adjacent semantic boundary `b`;
  - one START and one DONE event per eligible episode.
- Atomic and event pools alternate 50:50. The shorter pool cycles during sampling; persisted pair records are not duplicated.
- B0 sets `skill_linking_sampler_enabled=true` while keeping model linking disabled; P1 enables model linking. This gives both conditions the same stage-stratified sample order without exposing stage tensors to B0.
- The sampler remains deterministic in `(seed, epoch)` and uses the existing epoch/offset resume contract.
- Transition class weights are computed once from the complete train split before policy construction, frozen into the resolved config, and follow the preregistered inverse-square-root, mean-normalized, `[0.25, 4.0]` clipped rule. Reserved class `0` has weight `0`.

## Model path

- Added a zero-initialized expert-width skill embedding with rows `skill_0..skill_15, START`.
- Added an expert-width transition head with classes `skill_0..skill_15, CONTINUE, DONE`.
- The current-skill embedding is added after `action_time_mlp_out` and broadcast across the action suffix.
- The transition head pools post-transformer suffix states over the first `H` action tokens.
- Training uses `flow_loss + 0.1 * weighted_transition_CE`; the disabled path retains the original flow-only return behavior.
- Inference keeps transition logits only in the denoising-call closure, uses the final Euler denoise call, and applies its pending argmax immediately before the next chunk after exactly `H` queued actions are consumed.
- `CONTINUE` and `DONE` preserve the current skill; `DONE` does not terminate the environment.
- Reserved class `0` also preserves the current skill. The last raw transition prediction is retained for DONE/invalid diagnostics.
- Enabled RTC and direct `predict_action_chunk` are rejected.

## Checkpoints

- Disabled loading delegates to the existing loader.
- Enabled bootstrap loading permits an old checkpoint to miss only the new embedding/head tensors.
- Unexpected keys, shared-key shape errors, non-skill missing keys, and partial skill-linking checkpoints are rejected.
- The embedding starts at zero so loading an action-only checkpoint does not perturb the initial action suffix.

## Verification

- Added `tests/policies/smolvla/test_skill_transition.py` for config/delta windows, one-boundary sampling, START/CONTINUE/switch/DONE targets, deterministic 50:50 order and resume, class counts/weights, BF16 embedding, causal batched queue timing, disabled behavior, and checkpoint whitelist behavior.
- Write-free Python syntax compilation passed for all six modified source/test files.
- `git diff --check` passed for the scoped implementation and artifacts.
- A dependency-light smoke check executed the actual sampler implementation and passed candidate, class-count, 50:50 order, and resume assertions.
- Full pytest collection could not run in this local WSL environment because its project venv lacks pytest and contains an incomplete CUDA-linked PyTorch installation. This is an environment limitation, not a passing runtime result; the focused pytest file must be run in the GPU container before Gate B or training.
