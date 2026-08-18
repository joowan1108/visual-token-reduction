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
