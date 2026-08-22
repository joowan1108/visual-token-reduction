# Atomic SmolVLA implementation change log

## Revisions

- Local LeRobot base commit: `b20db9b3ef1c5830a2abdb24a0c2b82d771aa095` plus the pre-existing dirty worktree and this intervention
- AtomicVLA reference: `zhanglk9/AtomicVLA@c3583055adde0a491a11ffe08c15ca6459a64254`
- SARM metadata snapshot inspected: `8ec70343c56430f5dbae09af6b073d879207fe7c`
- Results inspected: none

## Upstream comparison

| Official path | Compared behavior | LeRobot treatment |
|---|---|---|
| `src/openpi/models/gemmoe.py` | `Module` owns one router; combine weights are computed once and broadcast through scanned layers | one `AtomicSkillRouter` on `SmolVLMWithExpertModel`, with one route tuple reused at every action FFN |
| `src/openpi/models/gemmoe.py` | top-1 probability weights the selected skill expert; shared weight is its complement | exact same mixing equation |
| `src/openpi/models/gemmoe.py` | all experts execute and stack before sparse weighting | only experts selected by a batch item execute; intentional numerically equivalent efficiency deviation |
| `src/openpi/models/pi0_atomic.py` | `sigma_emb` is scaled one-hot `linspace(10,100,n)` and repeated over the action horizon | same six-row fixed buffer; one per-sample route naturally applies to every token |
| `src/openpi/models/pi0_atomic.py` | router identity scale is `log(n-1)/55` for target weight 0.5 | same formula with `n=6` |
| `src/openpi/models/pi0_atomic.py` | decision, reasoning-text, and action losses | action flow loss only; VLM remains frozen as preregistered |
| `src/openpi/policies/atomic_dataset.py` | ordinary episode padding, no semantic boundary loss mask | SARM skill changes are ORed into existing LeRobot `action_is_pad` |
| `src/openpi/models/tokenizer.py` | five LIBERO skills and unknown-to-pick fallback | strict six-skill SARM mapping; unknown labels fail |

Additional upstream experts appear separately initialized. This intervention copies the dense SmolVLA FFN
into shared and all six skill experts so dense checkpoint promotion is step-zero equivalent, as required by
the frozen plan.

## Changed files

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
  - optional atomic data/SG-MoE flags, fixed 10/5 temporal validation, mapping path/list contract
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
  - one official-style global router and shared-plus-six-skill FFNs on action layers only; deterministic
    generation through the same internal frozen VLM and processor
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - skill propagation through training and denoising, boundary mask reuse, dense safetensor promotion,
    strict planner parser/state/failure history, five-action replanning, and synchronous skill propagation
- `src/lerobot/scripts/lerobot_eval.py`
  - turns `AtomicPlannerEpisodeFailure` into a batch-size-1 unsuccessful rollout instead of aborting evaluation
- `src/lerobot/policies/pretrained.py`
  - dependency-neutral `RolloutEpisodeFailure` contract used by generic evaluation without importing SmolVLA
- `src/lerobot/datasets/sampler.py`
  - deterministic six-way balanced replacement sampler using the existing epoch/resume protocol
- `src/lerobot/scripts/lerobot_train.py`
  - exact SARM vocabulary/order validation, mapping resolution, common A/B atomic sampler hookup
- `experiments/atomic/config/subtask_to_skill.json`
  - all 52 SARM strings mapped to strict `pick/place/push/turn/open/close`
- `tests/policies/smolvla/test_atomic_sgmoe.py`
  - router formula/taxonomy, dense equivalence, selected-expert gradients, canonical boundary masking,
    sampler balance/determinism/resume, strict planner parsing/state/failure, and online skill propagation
- `tests/scripts/test_lerobot_eval_atomic_failure.py`
  - policy-requested termination returns unsuccessful `done` and `policy_failure` rollout tensors
- `experiments/atomic/03_amendment_01_atomicvla_alignment.md`
  - pre-result correction of the stale per-layer/inverse-scale router specification
- `experiments/atomic/03_amendment_02_frozen_planner_runtime.md`
- `experiments/atomic/03_amendment_03_visual_encoder_option.md`
  - records the pre-result contract for matched optional visual-encoder training and frozen planner eval
  - sequential online runtime, strict parsing, queue-switch, failure, and oracle-separation contract

The worktree already contained unrelated SmolVLA focus-token, skill-linking, phase-masking, and RTC edits.
They were preserved; the aggregate Git diff is not solely this intervention.

## Commands and verification

Executed:

```bash
git -C /tmp/atomicvla-c3583055 checkout --detach c3583055adde0a491a11ffe08c15ca6459a64254
uv run ruff format <atomic changed Python files>
/usr/bin/python3 -m py_compile <atomic changed Python files>
uv run pytest tests/policies/smolvla/test_atomic_sgmoe.py -q
```

The final direct Ruff import/error checks and system-Python `py_compile` passed. The focused pytest did not collect: the local
CUDA PyTorch installation first lacked `libcurand.so.10`. Reinstalling the affected locked NVIDIA wheels
then failed while copying cuDNN with `Cannot allocate memory (os error 12)`, leaving pytest blocked by the
environment rather than a test assertion.
`tests/scripts/test_lerobot_eval_atomic_failure.py` was added after that failure and could not be executed
for the same environment reason.

## Assumptions and deferred work

- The observed SARM `subtask_index` vocabulary is contiguous `0..51`; both IDs and the ordered names in
  `meta/subtasks.parquet` must exactly match the frozen JSON before a run starts.
- Dataset quality audit/exclusion, task-stratified 80/10/10 persisted manifests,
  exact-init LIBERO manifest/runner, ordered predicate logger, append-only run manager, full 20-step smoke, six
  100k trainings, and 12,000 rollouts are intentionally not claimed complete.
- The frozen planner loop and episode-failure handoff are implemented, but a real SmolVLM generation smoke
  and LIBERO rollout remain gated by the unrepaired local PyTorch/CUDA environment.
- No training, simulator rollout, checkpoint write, or raw-result mutation was performed.

## Optional visual-encoder training amendment

- `SmolVLAConfig.freeze_vision_encoder` is reused as the single CLI switch; no atomic-only flag was added.
- With `train_expert_only=True`, setting the switch to `False` now unfreezes exactly `vision_model` after the
  rest of the VLM is frozen and restores its train/eval mode independently. The language/text VLM, connector,
  and LM head retain the existing frozen contract.
- The existing optimizer parameter path returns policy parameters, so the newly trainable vision parameters
  are included and receive updates when gradients exist; frozen parameters continue to have no gradients.
- Planner mode rejects `freeze_vision_encoder=False`, preserving fully frozen shared-VLM generation.
- Dense A and SG-MoE B can use the same switch value. The default remains frozen, and no run or raw result was
  produced while making this change.

## GT-routed closed-loop atomic-skill success metrics (2026-08-18)

Results inspected: none. No metric, dataset, seed, rollout budget, or evaluation criterion was changed.

### Changed files

- `src/lerobot/envs/libero.py`
  - Added privileged atomic-attempt snapshots with stable `skill + target/goal` identities.
  - Completion uses grasp state for `pick`, LIBERO `on`/`in` predicates for `place`/`push`, articulated-region
    state for required intermediate `open`, and goal predicates for direct `open`/`close`/`turn`.
  - Preserved `atomic_oracle_skill()` as the existing routing API over the new snapshot.
- `src/lerobot/scripts/lerobot_eval.py`
  - Under the existing `atomic_gt_routing` flag only, records one event per contiguous attempt activation,
    closes superseded active attempts as failures unless their simulator condition completed, and persists
    per-episode events.
  - Adds count-based `attempts`, `successes`, and `success_rate` under each task, task group, and overall
    `per_skill` entry in `eval_info.json`; existing task metrics and planner timelines remain unchanged.
  - Logs one compact overall skill table at evaluation end.
- `tests/scripts/test_lerobot_eval_atomic_failure.py`
  - Added focused coverage for concrete-condition completion, frame de-duplication, `place -> pick` failure,
    `pick -> place` success, and task-group/overall count aggregation.

`experiments/atomic/eval_gt_routed_action_loss.py` was not edited.

### Configuration

- Reused `EvalPipelineConfig.atomic_gt_routing`; no new flag or dependency was added.
- Metric fields are absent when GT routing is disabled.

### Commands and exact results

```bash
uv run ruff check src/lerobot/envs/libero.py src/lerobot/scripts/lerobot_eval.py \
  tests/scripts/test_lerobot_eval_atomic_failure.py
# All checks passed!

uv run pytest tests/scripts/test_lerobot_eval_atomic_failure.py -vv --tb=short
# 0 tests collected: tests/conftest.py could not import torch because libcublasLt.so.12 was unavailable.
```

The targeted pytest process was stopped after the environment import failure; no broad tests were run.

### Assumptions and deviations

- A new attempt starts only when its stable identity becomes active before an action; a post-terminal skill is
  not counted if no action ran under it. Re-entering the same identity after another active skill is a new attempt.
- Per-skill group/overall rates are micro-aggregated from event counts, not inferred from timeline transitions.
- This is a privileged-state GT-routing diagnostic, not AtomicVLA's learned think/act method and not a new
  preregistered primary or secondary outcome. No experimental result is interpreted here.

## Natural-distribution action sampler (2026-08-19)

Results inspected: none. No metric, dataset, seed, loss mask, or evaluation criterion changed.

### Changed files and configuration

- `src/lerobot/datasets/sampler.py`: with the existing `classifier_event_sampling=False`, selected episode
  frames now use the base sampler's deterministic shuffled traversal exactly once per epoch; episode filtering,
  absolute-to-relative mapping, and resume offsets are preserved. `classifier_event_sampling=True` retains its
  existing 75:25 stay/event replacement sampling. Non-atomic training remains behind existing configuration.
- `tests/policies/smolvla/test_atomic_sgmoe.py`: replaced the balanced-action test with imbalanced natural-count,
  exact coverage, deterministic-order, filtered/index-mapped episode, and exact resume-suffix coverage. The
  classifier sampler test is unchanged.

### Commands and exact results

```bash
uv run pytest tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_sampler_shuffles_each_selected_frame_once_and_resumes_exactly \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_classifier_sampler_draws_current_boundaries_75_25 -q
# Collection failed before tests ran: ImportError: libcublasLt.so.12 was unavailable.

uv run ruff check src/lerobot/datasets/sampler.py tests/policies/smolvla/test_atomic_sgmoe.py
# All checks passed!

/usr/bin/python3 -m py_compile src/lerobot/datasets/sampler.py tests/policies/smolvla/test_atomic_sgmoe.py
# Passed.
```

### Assumptions and deviations

- This pre-result user-directed change supersedes the balanced replacement sampling in frozen plan section 4.6
  and its equal-frequency Gate 3 assertion for SG-MoE/action training only. Skill-boundary masking remains a
  separate unchanged experiment variable.
- No dependency repair, training, evaluation, checkpoint write, or raw-result mutation was performed.

## Opt-in implicit FAST-KI context (2026-08-19)

Results inspected for this intervention: none. No existing metric, dataset, seed, rollout manifest,
statistical test, or evaluation criterion was changed.

### Changed files

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
  - Added disabled-by-default IAR/FAST configuration and validation for frozen VLM/cache/atomic SG-MoE use.
  - Independent-review correction: rejects scratch random frozen teachers and selected layers that collide
    after negative-index normalization.
  - User-directed follow-up: requires `train_state_proj=true` for the implicit condition.
  - Added positive `implicit_transition_loss_weight=0.1`, rejected the competing frozen atomic planner, and
    changed implicit subtask anchors to `[-2H, -H, 0..chunk-1]`.
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
  - Added independent per-selected-layer IAR queries and Q/K/V projections over detached raw cache tensors,
    with fixed mean aggregation.
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - Added the isolated gradient paths: FAST reuses frozen SmolVLM autoregression and LM head; flow receives
    detached projected context and no raw VLM K/V or hidden memory; inference omits FAST generation.
  - Allowed complete legacy atomic checkpoints to initialize the new IAR/token-path parameters while rejecting
    partial implicit checkpoints.
  - Independent-review correction: repeats the pretrained-teacher and normalized-layer checks at runtime.
  - User-directed follow-up: excludes state from the IAR VLM prefix/cache, reuses the existing `state_proj` to
    append one state context token to projected IAR tokens, trains the fused context with FAST, and detaches the
    same fused context before flow and during inference.
  - Added a dedicated six-class implicit transition head over detached fused context and two masked fixed-router
    skill embeddings. Training CE reaches only this head; synchronous batched inference keeps and resets two
    executed-skill history slots and routes each replanned chunk with the predicted skill.
  - Legacy implicit checkpoints may omit the complete new transition head; partial transition-head checkpoints
    remain invalid.
- `src/lerobot/policies/smolvla/processor_smolvla.py`
  - Added the opt-in SmolVLA FAST action processor, applying the canonical atomic boundary mask before
    tokenization and reusing SmolVLM vocabulary slots.
  - Updated implicit label slicing to preserve two history anchors while applying FAST/flow masking only to the
    unchanged action horizon.
- `src/lerobot/processor/tokenizer_processor.py`
  - Extended the existing FAST tokenizer helper to accept a contiguous action padding suffix and tokenize only
    the valid action prefix.
- `src/lerobot/scripts/lerobot_train.py`
  - Builds the opt-in processor from the resolved policy config when starting from the older base processor;
    resume continues to load the checkpoint processor.
- `tests/policies/smolvla/test_atomic_sgmoe.py`
  - Added one focused contract test for layer independence, shape/no-target-leakage, shared boundary masking,
    disabled defaults, and separate FAST/flow backward gradients.
  - Independent-review correction: the flow backward check calls production `_forward_implicit_action` and
    verifies scratch-teacher rejection, normalized layer uniqueness, and that inference has no FAST-loss call.
  - User-directed follow-up: production FAST/flow helpers verify FAST gradients reach IAR, `fast_context_proj`,
    and `state_proj`, while flow gradients reach only SG-MoE; the test also checks state-free IAR prefixes/cache.
  - Added focused transition checks for six-way output, zero-vector missing history, exactly two valid histories,
    episode-safe anchors, batched executed-history append/reset, fixed router embeddings, and head-only CE grads.
- `experiments/atomic/03_amendment_05_implicit_fast_ki.md`
  - Froze configuration, data masking, objective, gradient ownership, inference behavior, and interpretation.

Only the files above and this change log were normalized/edited; unrelated worktree changes were preserved.
No dependency or vocabulary change was made.

### Configuration

```text
implicit_fast_ki_enabled=false                 # default; preserves baseline
implicit_iar_layers=[-4,-3,-2,-1]
implicit_iar_num_queries=4
implicit_fast_loss_weight=0.1
implicit_transition_loss_weight=0.1
implicit_transition_focal_gamma=2.0
implicit_fast_max_action_tokens=256
implicit_fast_skip_tokens=128
implicit_fast_action_tokenizer_name=lerobot/fast-action-tokenizer
```

Enabling the flag requires `atomic_data_enabled=true`, `atomic_sgmoe_enabled=true`,
`train_expert_only=true`, `freeze_vision_encoder=true`, `use_cache=true`, and `compile_model=false`.
It also requires `load_vlm_weights=true` or a policy loaded through `pretrained_path`; implicit mode forces
`train_state_proj=true` so FAST can learn the reused proprioceptive state projection.
The transition head is always active with implicit FAST-KI and replaces the frozen atomic planner for this
condition.
The existing atomic `chunk_size=10`, `n_action_steps`, mapping, and boundary contract are unchanged.

### Natural-transition focal-loss amendment (2026-08-20)

- The previously implemented fixed 4x switch multiplier is superseded and removed. Official AtomicVLA commit
  `c3583055` instead uses deterministic `[think]` windows over the first 11 episode frames and six frames on
  each side of a skill boundary; it does not provide a switch class weight.
- Added finite non-negative `implicit_transition_focal_gamma=2.0`. The focal objective enters both scalar and
  per-sample total losses; `gamma=0` exactly recovers plain CE.
- `implicit_transition_ce` logs unweighted CE while `implicit_transition_loss` logs the actual focal objective.
  Sampling, targets/history, unweighted metrics, detach paths, and gradient ownership are unchanged.
- Held-out evaluation now aggregates and logs the existing stay/switch counts for implicit FAST-KI as well as
  the legacy atomic classifier; metric definitions and training behavior are unchanged.
- Scoped Ruff, `py_compile`, and semantic diff checks passed; focused pytest remained blocked by the local
  PyTorch import failure for missing `libcublasLt.so.12`.

### Commands and exact results

```bash
uv run ruff format src/lerobot/policies/smolvla/configuration_smolvla.py \
  src/lerobot/policies/smolvla/smolvlm_with_expert.py \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  src/lerobot/policies/smolvla/processor_smolvla.py \
  src/lerobot/processor/tokenizer_processor.py src/lerobot/scripts/lerobot_train.py \
  tests/policies/smolvla/test_atomic_sgmoe.py
# Completed; only listed modified Python files were formatted.

uv run ruff check --ignore SIM102,E402 <the Python files listed above>
# All checks passed. SIM102 and E402 are pre-existing findings in untouched code regions.

/usr/bin/python3 -m py_compile <the Python files listed above>
# Passed.

uv run pytest \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_implicit_fast_ki_layer_independence_gradient_isolation_and_no_leakage \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_config_keeps_dense_defaults_and_freezes_temporal_contract \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_ffn_starts_dense_equivalent_and_only_runs_selected_experts \
  -q --tb=short
# Collection failed before tests ran: ImportError: libcublasLt.so.12 was unavailable.
```

The system Python has no `torch`, so no alternate runtime test was claimed. No dependency repair, training,
evaluation, checkpoint write, or raw-result mutation was performed.

Independent-review correction validation:

```bash
uv run ruff check --ignore SIM102,E402 <the modified Python files>
# All checks passed.

/usr/bin/python3 -m py_compile <the modified Python files>
# Passed.

git diff --check --ignore-space-at-eol -- <the intervention files>
# Passed.

timeout 25s uv run pytest \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_implicit_fast_ki_layer_independence_gradient_isolation_and_no_leakage \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_config_keeps_dense_defaults_and_freezes_temporal_contract \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_ffn_starts_dense_equivalent_and_only_runs_selected_experts \
  -q --tb=short
# Timed out before pytest emitted collection output (exit 124).

timeout 15s uv run python -c 'import torch; print(torch.__version__)'
# Failed during torch import: ImportError: libcublasLt.so.12: cannot open shared object file.
```

User-directed state-fusion follow-up validation:

```bash
.venv/bin/ruff check --ignore SIM102,E402 <the modified Python files>
# All checks passed.

/usr/bin/python3 -m py_compile <the modified Python files>
# Passed.

git diff --check --ignore-space-at-eol -- <the intervention files>
# Passed.

timeout 25s uv run pytest \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_implicit_fast_ki_layer_independence_gradient_isolation_and_no_leakage \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_config_keeps_dense_defaults_and_freezes_temporal_contract \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_ffn_starts_dense_equivalent_and_only_runs_selected_experts \
  -q --tb=short
# Collection failed before tests ran (exit 4):
# ImportError: libcublasLt.so.12: cannot open shared object file: No such file or directory
```

Dedicated implicit transition-head validation:

```bash
.venv/bin/ruff check --ignore SIM102,E402 \
  src/lerobot/policies/smolvla/configuration_smolvla.py \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  src/lerobot/policies/smolvla/processor_smolvla.py \
  tests/policies/smolvla/test_atomic_sgmoe.py
# All checks passed.

/usr/bin/python3 -m py_compile <the modified Python files>
# Passed.

timeout 25s uv run pytest \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_implicit_fast_ki_layer_independence_gradient_isolation_and_no_leakage \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_implicit_transition_history_anchors_are_episode_safe_and_reset_per_batch \
  tests/policies/smolvla/test_atomic_sgmoe.py::test_atomic_config_keeps_dense_defaults_and_freezes_temporal_contract \
  -q --tb=short
# Timed out before pytest emitted collection output (exit 124).

timeout 15s uv run python -c 'import torch; print(torch.__version__)'
# Failed during torch import:
# ImportError: libcublasLt.so.12: cannot open shared object file: No such file or directory
```

### Assumptions and deviations

- FAST action tokenization is variable-horizon over the valid contiguous chunk prefix because compressed FAST
  tokens do not retain a one-token-per-timestep mask. This preserves the frozen boundary contract without
  leaking masked future targets.
- The existing reverse-vocabulary mapping is applied to the SmolVLM tokenizer; no new token IDs or embedding
  rows are created.
- The frozen SmolVLM transformer and LM head are reused as the autoregressive token path. This is the minimal
  KI-style reusable path and intentionally does not add or claim an exact separate decoder reproduction.
- IAR emits four expert-width context tokens from an image-language-only cache. A single FAST-trained projection
  maps them to SmolVLM width and concatenates the existing projected-state token; flow consumes the detached
  fused result, so flow cannot update IAR or either projection.
- This implementation does not authorize or add training/evaluation runs. Any future results belong to a
  separately named opt-in condition and must not replace or be pooled with the original preregistered A/B result.

## Atomic temporal anchor thinning (2026-08-20)

Added opt-in `atomic_anchor_stride` (default `1`). In natural atomic sampling, values above one retain every
no-history anchor in the first `n_action_steps` episode frames and every anchor whose mapped skill differs from
the skill at `t - n_action_steps`; remaining stays use segment-relative stride offsets. The requested condition
original thinning condition used stride `5`, transition horizon `5`, and `chunk_size=20`. Stride `1` preserves
the existing all-frame `chunk_size=10` baseline, while classifier 75:25 sampling ignores stride and horizon.

### Transition-history alignment correction

The first version retained exact mapped boundaries, but the transition target compares `skill(t)` with
`skill(t - n_action_steps)`. The sampler now receives that horizon, preserves every matching no-history and
switch-window anchor within each episode, and thins only model-defined stays. Tests cover exact counts,
same-skill raw-label changes, cross-episode isolation, deterministic resume, and unchanged classifier sampling.
Ruff, `py_compile`, and diff checks passed; focused pytest timed out before local CUDA collection.

## AtomicVLA-aligned unmasked chunks (2026-08-21)

Results inspected: none. This user-approved correction supersedes atomic skill-boundary action masking and the
earlier 5-step/20-frame thinning condition. Metrics, datasets, seeds, and evaluation criteria are unchanged.

### Changed files and configuration

- `src/lerobot/policies/smolvla/processor_smolvla.py`: FAST tokenization now combines only true subtask/episode
  padding with existing action padding; skill changes no longer enter its mask.
- `src/lerobot/policies/smolvla/modeling_smolvla.py`: atomic flow loss now uses only episode/action padding while
  retaining the anchor skill for SG-MoE routing.
- `src/lerobot/policies/smolvla/configuration_smolvla.py`: atomic thinning supports `chunk_size=10` and
  `n_action_steps=10`; the global SmolVLA defaults remain unchanged.
- `tests/policies/smolvla/test_atomic_sgmoe.py`: added focused pick/place/pick FAST and flow-padding regressions,
  `t-10` history/target coverage, and deterministically shuffled stride-5 sampler coverage. Existing config
  propagation already passes `n_action_steps` as the sampler transition horizon.

### Commands and results

```bash
.venv/bin/ruff format <the four modified Python files>
# 4 files left unchanged.
.venv/bin/ruff check --ignore SIM102,E402 <the four modified Python files>
# All checks passed!
/usr/bin/python3 -m py_compile <the four modified Python files>
# Passed.
timeout 45s uv run pytest <the six focused atomic tests> -q --tb=short
# Stopped during pytest plugin metadata discovery before collection (exit 130); no tests ran.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 25s uv run pytest <the same six tests> -q --tb=short
# Timed out before emitting collection output (exit 124); no tests ran.
git diff --check --ignore-space-at-eol -- <the five intervention files>
# Passed.
```

Only boundary-derived padding and the superseded temporal condition changed. FAST still rejects malformed
non-suffix episode padding, classifier 75:25 behavior is unchanged, and stride `1` retains its exact baseline
sampling path. This is an AtomicVLA-aligned adaptation, not a claim of reproducing the paper's think windows.
No training/evaluation run, dependency change, result interpretation, commit, or push was performed.

## Implicit FAST-KI diagnostics and GT routing override (2026-08-22)

Results inspected: none. No metric definition, dataset revision, split, seed, rollout budget, or evaluation
criterion was changed. Baseline and normal implicit predicted-routing behavior remain the default.

### Changed files

- `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - Added explicit `return_loss_components=true` plumbing for `reduction="none"`; it exposes detached pure
    flow, FAST auxiliary, transition focal auxiliary, and plain transition CE tensors without changing scalar
    training loss/logging.
  - Permits batch-size-1 in-range GT skill IDs in implicit mode, keeps IAR/projected KV/state context and flow
    denoising active, bypasses only transition argmax, appends the executed GT skill to two-slot history, and
    records `source="gt_oracle"`. Predicted routing is unchanged when no override is supplied.
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
  - Added opt-in, detached IAR capture for selected configured layers/queries before layer averaging. Capture is
    bounded by an explicit batch limit and must be popped before another batch.
- `src/lerobot/policies/smolvla/attention_analysis.py`
  - Added padding-excluding image/language/other attention mass, normalized entropy, normalized JSD query/layer
    diversity with pair counts, and separate IAR/state context norm/variance reducers.
- `experiments/atomic/eval_gt_routed_action_loss.py`
  - Reports per-skill mean/median/p95 from pure flow-matching loss only. Clearly named FAST and transition
    auxiliaries are separate when present; legacy SG-MoE falls back to its already-pure unreduced loss.
- `experiments/atomic/eval_iar_diagnostics.py`
  - Added held-out seeded-subset checkpoint evaluation with the required policy/max-samples/batch/workers/seed/
    output CLI, concise tables, JSON, all-present-skill profiles/differences, and explicit absent groups.
  - Raw token tensors are consumed one batch at a time and are not stored; optional NPZ output was omitted.
- `tests/policies/smolvla/test_atomic_sgmoe.py`
  - Added focused small-tensor coverage for bounded selected IAR capture/metrics, GT override validation/history/
    timeline, and pure-flow versus auxiliary loss components.

No dependency, checkpoint schema, or persistent configuration field was added. IAR capture is enabled only by
the diagnostic script's runtime call. `lerobot-eval --atomic_gt_routing=true` retains its existing validation
that `eval.batch_size=1` and `env.max_parallel_tasks=1`.

### Commands and exact results

```bash
uv run ruff format <six modified Python files>
# 5 files reformatted; 1 unchanged.

uv run ruff check --ignore SIM102 <six modified Python files>
# All checks passed. SIM102 is ignored because two pre-existing unrelated findings remain in modeling_smolvla.py.

uv run python -m py_compile <six modified Python files>
# Passed.

uv run pytest -q tests/policies/smolvla/test_atomic_sgmoe.py \
  -k 'iar_capture or implicit_gt_override or unreduced_loss_components'
# Collection failed before tests ran: ImportError for libcublasLt.so.12.

LD_LIBRARY_PATH=<all bundled .venv NVIDIA library directories> uv run python -c 'import torch'
# Exited 135 during torch import; the local CUDA/PyTorch environment remains unusable.
```

### Assumptions and deviations

- Token-type skill differences use normalized JSD over each skill's mean `[image, language, other]` attention
  profile; query and layer diversity use normalized JSD over full valid prefix-token distributions. Pair counts
  are persisted and absent skills, including `pick`/`place`/`open` when absent, are marked rather than imputed.
- State remains the separate `state_proj(state)` context concatenated after IAR. It is never labeled as IAR
  token attention and is reported in a separate JSON section/table line.
- GT closed-loop routing is a privileged oracle diagnostic, not AtomicVLA's learned think/act policy. Transition
  logits are still computed for diagnostics, but only their argmax is bypassed.
- No checkpoint evaluation, LIBERO rollout, training, raw-result write, result interpretation, commit, or push
  was performed.

### Evaluator corrections (2026-08-22)

Results inspected: none. These corrections change diagnostic implementation/provenance labels only; no
dataset, split fraction, seed default, metric threshold, or evaluation criterion was changed.

- `src/lerobot/policies/smolvla/attention_analysis.py` now returns bounded per-sample normalized
  layer-by-query-by-token attention signatures, plus visual-only and language-only signatures. Modality masses
  remain separate summaries.
- `experiments/atomic/eval_iar_diagnostics.py` pads only the token axis across batches, then flattens the aligned
  layer/query/token signature. Skill comparisons are normalized JSD between group-mean full signatures and
  report `left_count`, `right_count`, and `comparison="group_mean_vs_group_mean"`; the misleading Cartesian
  `sample_pair_count` was removed. Visual-only and language-only group-mean JSD are also reported.
- `experiments/atomic/eval_gt_routed_action_loss.py` wraps every repeat forward in `torch.random.fork_rng` and
  resets CPU/active-CUDA RNG from `(seed + 1000003 * batch_index + repeat_index) % (2**63 - 1)`. This makes
  internally sampled flow noise/time checkpoint-independent without changing DataLoader ordering.
- Both evaluators identify their source as `dataset_factory_validation_split`, produced by
  `make_train_eval_datasets(DatasetConfig(eval_split=0.1))`, and explicitly state that it is validation rather
  than the preregistered offline test. Existing `eval_split` and result fields remain for compatibility; no
  nonexistent 80/10/10 manifest is claimed.
- `tests/policies/smolvla/test_atomic_sgmoe.py` adds small-tensor position-sensitive signature coverage and
  static evaluator contract checks.

`src/lerobot/policies/smolvla/smolvlm_with_expert.py` was not touched, preserving its original mixed line endings.
Only targeted Ruff and `py_compile` checks were authorized for this correction.

```bash
uv run ruff check src/lerobot/policies/smolvla/attention_analysis.py \
  experiments/atomic/eval_gt_routed_action_loss.py \
  experiments/atomic/eval_iar_diagnostics.py tests/policies/smolvla/test_atomic_sgmoe.py
# All checks passed!

uv run python -m py_compile <the same four Python files>
# Passed.
```

No pytest, training, checkpoint evaluation, dependency change, commit, or push was performed.
