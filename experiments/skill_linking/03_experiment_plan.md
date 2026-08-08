# Preregistered experiment plan: End-to-End Skill Linking for SmolVLA

## 1. Status and decision boundary

- Status: **approved by user on 2026-08-08; P1 implementation authorized**
- Hypothesis artifact: `00_hypothesis.md`
- Evidence review: `01_literature_review.md`
- Implementation audit: `02_implementation_map.md`
- Local code reference: commit `ab162194c074f2b9c4ab9782250de172273653e5`
- Dataset: `sungkyunner/libero_10_subtask_semantic_clean`, revision `d619815aeba9c06c70fc558838137dd57a651ce1`
- Initialization: `lerobot/smolvla_libero`, revision `a567cc17bd9be971c5822d0cc9f9a77231bfbf24`

This document freezes the experiment before application source changes or outcome observation. After approval, only `hypothesis_implementer` may modify source. Hyperparameters, the primary metric, the primary checkpoint, and the support criteria below may not be changed after a primary result is observed.

## 2. Exact question and claim scope

### Primary question

On the ten seen LIBERO-10 tasks, does jointly fine-tuning a shared SmolVLA action expert with a semantic current-subtask embedding and a next-subtask transition objective improve complete episode success over an action-only continuation from the same checkpoint?

### Permitted claim

The strongest possible claim is limited to **semantic subtask linking on seen LIBERO-10 trajectories and nine observed directed pair types**. It cannot establish unseen skill composition or general-purpose planning.

### Fixed intervention

The intervention adds only:

1. `nn.Embedding(17, expert_hidden_size)` for skills `0..15` and `START=16`;
2. one linear transition head with classes `skill_0..skill_15`, `CONTINUE=16`, and `DONE=17`; unused label `0` remains reserved for checkpoint-stable indexing.

### Fixed one-pair-per-boundary sampling

- Clean dataset totals: 495 episodes, 136,425 frames, 941 contiguous atomic segments.
- Adjacent semantic transitions: 446 instances across 9 directed pair types.
- 446 episodes have one adjacent transition; 49 episodes have none.
- For each boundary `b`, create exactly one pair index `t=b-H`; do not create multiple offsets.
- Pair samples are generated as an index view at training time and are never saved as copied clips or a second dataset.
- Atomic indices are all valid starts whose `H+1` label window stays inside one segment.
- START and DONE each contribute one event index per episode.
- The deterministic training sampler draws atomic and event pools 50:50.

The skill embedding, transition head, and existing action expert are jointly trained. The VLM/vision backbone remains frozen. No planner, MoE, scene graph, phase mask, focus-token pruning, scheduled sampling, confidence threshold, or transition-confirmation rule is allowed.

## 3. Pre-training data gate

The approved implementer must run this audit before editing model behavior or launching training and save it as `data_audit.json` plus `data_audit.md`.

### Loader and schema checks

1. Load the pinned dataset revision through the repository's normal `LeRobotDataset` path.
2. Confirm 495 episodes, 10 tasks, and a raw integer `subtask_index` feature plus semantic metadata.
3. Scan the complete dataset, not a batch, and require observed IDs to be exactly `1..15`.
4. Confirm temporal lookup returns `H+1` stage values and an aligned `subtask_index_is_pad` mask.
5. Require 941 atomic segments, 446 boundaries, 9 directed pairs, minimum pre-boundary length at least `H`, and exactly one generated pair index per boundary.

Failure of any check is **blocked**, not a negative scientific result. No vocabulary may be inferred from one batch or repaired with a handwritten mapping.

### Episode split

Use the existing deterministic split with `dataset.eval_split=0.1`: for each global task, the final `ceil(0.1 * task_episode_count)` episodes are held out. The clean revision yields 445 train and 50 held-out episodes. Training, class counts, and class weights use only the train split.

### Deterministic horizon rule

`chunk_size` remains 50. The execution/transition horizon `H=n_action_steps` is selected before training by this fixed rule:

1. Audit `H=10`.
2. Among valid windows whose `t+H` frame is not padded, compute the fraction containing more than one stage boundary in `(t, t+H]`.
3. Use `H=10` if that fraction is at most 1%.
4. Otherwise audit and use `H=5` if its fraction is at most 1%.
5. If neither candidate passes, stop as **blocked**. Do not add boundary-offset prediction or per-token skill conditioning.

The chosen `H` is written to `data_audit.json` and then shared by every condition.

### Fixed target

For stage window `s[t:t+H]` and pad mask `p[t:t+H]`:

```text
current_skill = START     if frame_index[t] == 0
current_skill = s[t]      otherwise

next_state = DONE         if p[t+H] is true
next_state = s[t+H]       otherwise

target = next_state       if current_skill == START
target = CONTINUE         if next_state == current_skill
target = next_state       otherwise
```

This target is the state needed at the next policy invocation. It is not the first boundary inside the 50-action training chunk. Flow loss remains active over all valid action targets.

### Class weights

Let `n_c` be the target count for class `c` in the train split. Every observed class must have at least one target. Reserved absent skill class `0` receives weight `0`. Define other weights as:

```text
raw_w_c = 1 / sqrt(n_c)
w_c = raw_w_c / mean(raw_w)
w_c = clip(w_c, 0.25, 4.0)
```

Store the 18 resulting values in the resolved policy config. They are never recomputed per batch or tuned from evaluation outcomes.

## 4. Comparison conditions

| ID | Train-time current stage | Transition loss | Evaluation state | Use |
|---|---|---:|---|---|
| B0 | none | no | none | Primary action-only baseline |
| O1 | ground truth | weight `0` | ground truth, offline only | Tests whether stage conditioning carries action-relevant information |
| P1 | ground truth | weighted CE, `lambda=0.1` | predicted, one argmax per chunk | Primary intervention |
| P1-oracle | P1 checkpoint | already trained | ground truth, offline only | Measures predicted-state exposure gap without another training run |

B0 and P1 both use the approved 50:50 `SkillLinkingSampler` with identical `H`, seed, and episode split. B0 sets only `skill_linking_sampler_enabled=true`; its model remains action-only and does not receive temporal `subtask_index` tensors. P1 sets `skill_linking_enabled=true`, which also selects that sampler and adds the registered intervention.

`O1` is a 10K-step diagnostic run for seed 1000 only. `P1-oracle` reuses P1 weights and is not a separate training condition.

LIBERO rollout observations do not expose ground-truth `subtask_index` after the policy diverges from a demonstration. Therefore O1 and P1-oracle are not assigned simulator complete-success scores. Fabricating an online oracle from task progress would introduce an unreviewed planner/completion detector.

The primary causal comparison is B0 versus P1. They must use the same pinned initialization, split, sample order, random seed, optimizer, scheduler, action/noise generation, batch size, number of updates, preprocessing, and evaluation initial states.

## 5. Model and inference contract

- Add the current-stage embedding after `action_time_mlp_out`, broadcast over the 50 suffix tokens.
- Pool post-final-RMSNorm suffix hidden states over tokens `[0:H]` and apply the linear head after a float32 upcast.
- Train with `L_total = L_flow + 0.1 * L_transition_CE`.
- At inference, use transition logits from the last Euler denoise call (`t=1/num_steps`).
- Apply the previous chunk's pending transition immediately before generating the next chunk, after exactly `H` actions have been consumed.
- `CONTINUE` preserves the current stage; a skill class replaces it; `DONE` is diagnostic and never terminates the environment.
- Support synchronous `select_action` only. Skill linking with RTC or direct/async `predict_action_chunk` must fail clearly.
- With linking disabled, existing SmolVLA behavior must remain unchanged.
- Old-checkpoint loading may miss only the new embedding/head keys; all other missing, unexpected, or shape-mismatched keys are fatal.

## 6. Fixed training protocol

### Shared settings

| Item | Frozen value |
|---|---|
| Dataset | `sungkyunner/libero_10_subtask_semantic_clean@d619815aeba9c06c70fc558838137dd57a651ce1` |
| Initialization | `lerobot/smolvla_libero@a567cc17bd9be971c5822d0cc9f9a77231bfbf24` |
| Train/held-out split | deterministic per-task 90/10 episode split |
| Training seeds | `1000`, `2000`, `3000` |
| Main updates | 50,000 per condition and seed |
| Per-GPU batch | 4 on one GPU |
| Effective batch | 4; no gradient accumulation |
| DataLoader workers | 4, sufficient container shared memory required |
| Image transforms | disabled |
| VLM | frozen (`train_expert_only=true`) |
| Optimizer | existing SmolVLA AdamW preset: LR `1e-4`, betas `(0.9, 0.95)`, eps `1e-8`, weight decay `1e-10`, grad clip `10` |
| Scheduler | warmup 1,000; decay through step 50,000 to `2.5e-6` |
| AMP | disabled |
| cuDNN | deterministic |
| `chunk_size` | 50 |
| `n_action_steps` | audit-selected `H`, shared by all conditions |
| Transition weight | fixed `0.1` for P1 |
| Checkpoints | 10K, 20K, 30K, 40K, 50K; 50K is primary |
| Checkpoint selection | final 50K only; no best-checkpoint selection |

The primary compute budget is six 50K runs: B0 and P1 for three seeds, totaling 300K optimizer updates. O1 adds one 10K diagnostic run. Interrupted runs resume from their latest exact training state; they are not restarted with a new seed.

### Pairing controls

- For each seed, B0 and P1 use identical episode split and sampler order.
- Record resolved config, git commit, dataset/model revisions, package lock hash, CUDA/PyTorch versions, GPU model, and environment variables.
- B0 and P1 must run with the same number of GPUs and batch size. A run with changed world size or batch size is invalid for the paired comparison.

## 7. Gates before full training

### Gate A: implementation verification

All minimum tests in `02_implementation_map.md` must pass, including disabled-path identity, temporal targets, mixed dtype, pending-state timing, batched reset, checkpoint whitelist, and exact enabled round-trip. The CPU suite must not download model or dataset assets.

### Gate B: small-subset overfit

Use at most 32 train episodes and seed 1000. This is a diagnostic run and is excluded from primary results.

- finite joint loss and gradients for action expert, embedding, and head;
- transition event macro-F1 at least `0.95` at the sampled training timestep;
- transition event macro-F1 at least `0.90` at fixed `t=1/num_steps`;
- final-step F1 no more than `0.05` below fixed-`t=0.5` F1;
- exact state switch after `H` deque pops.

If the final-denoise criteria fail, stop as **implementation/method blocked**. Do not silently add a second planning forward.

### Gate C: oracle utility at 10K

Train paired B0 and O1 diagnostics for seed 1000 to 10K updates. Evaluate held-out demonstrations with identical fixed noise/timestep draws. O1 must reduce macro-averaged per-stage flow MSE by at least 2% relative to B0:

```text
(MSE_B0 - MSE_O1) / MSE_B0 >= 0.02
```

If this gate fails, the current numeric stage input has not shown action-relevant utility. Stop and report the hypothesis **unsupported at the oracle-conditioning gate**; do not add a more complex transition mechanism.

Gate decisions use only the thresholds above. They do not tune `H`, `lambda`, architecture, or the 50K primary checkpoint.

## 8. Evaluation protocol

### Simulator complete success

- Environment suite: `libero_10`, all ten tasks.
- Evaluate only B0 and P1 50K checkpoints for all three training seeds.
- 20 episodes per task and training seed, 200 episodes per checkpoint, 1,200 episodes total.
- Use the same 20 LIBERO initial states and evaluation seed `0` for each paired B0/P1 checkpoint.
- Use synchronous `select_action`, no RTC, no test-time augmentation, and no checkpoint ensembling.
- Environment success/timeout is authoritative; predicted `DONE` never ends an episode.

### Offline held-out diagnostics

Use the fixed 50 held-out episodes. Evaluation sampling order, flow timestep/noise draws, and windows are paired between conditions.

Report:

1. transition macro-F1 over the 16 observed event classes (`skill_1..15`, `DONE`); reserved absent `skill_0` is excluded;
2. `CONTINUE` precision, recall, and F1 separately;
3. event boundary precision/recall at the next-invocation horizon;
4. false switches and missed switches per 100 evaluated chunks;
5. autoregressive predicted-current-stage accuracy for P1;
6. P1 versus P1-oracle flow MSE exposure gap;
7. macro per-stage flow MSE for B0, O1, P1, and P1-oracle where applicable;
8. action inference latency, peak GPU memory, and action throughput.

Transition latency, prefix success, and atomic success are not reported unless a separately reviewed evaluator can derive them from authoritative environment state. The current dataset/environment pair does not expose online ground-truth stage boundaries, so video inspection or numeric IDs must not be mislabeled as atomic success.

## 9. Primary metric and statistical test

### Primary metric

For each checkpoint, compute complete episode success for each of ten LIBERO-10 tasks, then macro-average the ten task rates. The primary effect is:

```text
Delta = macro_success(P1_50K) - macro_success(B0_50K)
```

The minimum scientifically meaningful effect is `+5.0 percentage points`.

### Paired hierarchical bootstrap

Use 10,000 bootstrap replicates and RNG seed `20260807`.

For each replicate:

1. sample the three training-seed pairs with replacement;
2. within each sampled seed, sample ten task pairs with replacement;
3. within each sampled seed/task pair, sample the 20 paired episode outcomes with replacement;
4. compute P1 minus B0 macro success.

Report the observed effect and two-sided percentile 95% confidence interval. This is the sole confirmatory comparison, so no multiple-comparison correction is applied. Per-task, checkpoint-trajectory, oracle, and transition results are secondary or diagnostic.

## 10. Frozen outcome rules

### Supported

The primary hypothesis is supported only if all are true:

1. observed `Delta >= +5.0%p`;
2. the paired bootstrap 95% CI lower bound is greater than zero;
3. at least two of three training seeds have positive B0-to-P1 macro-success differences;
4. all paired protocol and data gates pass.

The stronger mechanism statement, “predicted stage linking caused the gain,” additionally requires event macro-F1 at least `0.90` and no greater than 5% relative degradation in P1's held-out macro per-stage flow MSE versus B0.

### Unsupported

The primary hypothesis is unsupported if a valid complete experiment fails any of support conditions 1–3. It is also unsupported at the oracle gate if O1 fails the preregistered 2% held-out improvement. High transition F1 without complete-success improvement is explicitly unsupported.

### Inconclusive or blocked

- **Blocked:** loader/schema/horizon/checkpoint compatibility or implementation gates fail before valid training.
- **Inconclusive:** a required seed, paired evaluation, or raw episode record is missing or corrupted and cannot be resumed exactly.
- Infrastructure failures, OOM, missing LIBERO assets, or interrupted jobs are not counted as task failures.
- No missing run may be replaced by a different seed, checkpoint, batch size, or initialization.

Because authoritative online subtask success is unavailable, the `00_hypothesis.md` atomic-regression clause is operationalized here as held-out macro per-stage flow-MSE degradation. No atomic-success claim will be made from this proxy.

## 11. Required artifacts and sequence

```text
experiments/skill_linking/
  00_hypothesis.md
  01_literature_review.md
  02_implementation_map.md
  03_experiment_plan.md
  data_audit.json
  data_audit.md
  04_change_log.md
  05_method_evaluation.md
  raw/<run_id>/
    resolved_config.json
    environment.json
    train.log
    checkpoints/
    heldout_metrics.json
    eval_info.json
    episode_results.jsonl
  06_analysis.md
```

Execution remains sequential:

1. user approves this plan;
2. `hypothesis_implementer` performs the audit, source implementation, and minimum tests;
3. the orchestrator reviews the diff and tests;
4. `paper_method_evaluator` runs/checks B0 and P1 under this protocol and writes `05_method_evaluation.md`;
5. only after all raw results are complete may `results_analyst` write `06_analysis.md`.

Raw results are append-only and are never overwritten.

## 12. Approval record

The user explicitly approved the one-pair-per-boundary amendment and authorized P1 source implementation and verification on 2026-08-08. No additional approval prompt is required for that scope.
