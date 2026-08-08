# Implementation map: SmolVLA Skill Linking

## Audit scope and fixed references

- Local baseline: `ab162194c074f2b9c4ab9782250de172273653e5`
- Dataset: [`sungkyunner/libero_10_subtask_semantic_clean`](https://huggingface.co/datasets/sungkyunner/libero_10_subtask_semantic_clean), revision `d619815aeba9c06c70fc558838137dd57a651ce1`
- Dataset schema: v3.0, 495 episodes, 136,425 frames, 10 Hz, `task_index=0..9`, observed `subtask_index=1..15`, semantic mapping in `meta/subtasks.parquet`
- SmolVLA reference: [paper](https://arxiv.org/abs/2506.01844), [official LeRobot guide](https://huggingface.co/docs/lerobot/smolvla)
- Subtask format reference: [official LeRobot subtask guide](https://huggingface.co/docs/lerobot/en/dataset_subtask)
- RTC reference: [Physical Intelligence implementation](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)

Local LeRobot and SmolVLA code is Apache-2.0. The RTC reference is MIT. This implementation only follows public API/algorithm behavior and does not copy RTC source. The dataset card says Apache-2.0; reports must also preserve attribution to the original LIBERO data.

## Reader conclusion

The intervention is implementable with the existing dataset reader and SmolVLA action path, but the first version must use the following narrower contract.

1. `subtask_index` is a named semantic subtask on the ten seen LIBERO-10 tasks; the experiment does not claim unseen composition.
2. `chunk_size=50` stays unchanged for checkpoint compatibility; `n_action_steps=H` is the execution and transition horizon.
3. The transition target is the state required at the **next model invocation** (`t+H`), not the first boundary inside the chunk.
4. Transition logits use the post-final-norm action-expert hidden states for the first `H` suffix tokens.
5. A pending transition is applied once, immediately before the next chunk is generated. Two-call confirmation is removed because it delays a switch by another chunk.
6. The first implementation supports synchronous `select_action` evaluation only. RTC/async `predict_action_chunk` is rejected while skill linking is enabled.
7. Loading an old checkpoint may miss only the new embedding/head parameters; every other missing, unexpected, or shape-mismatched key is an error.
8. Pair data is an index view, not a copied dataset: one fixed sample starts at `b-H` for every adjacent-subtask boundary `b`.

## Fixed atomic/pair index contract

The cleaned revision has 941 contiguous atomic segments and 446 adjacent-subtask boundaries. For `H=10`, every pre-boundary segment has at least 74 frames, so all 446 fixed starts `b-H` are valid. There are 127,015 valid atomic starts whose inclusive `H+1` label window remains inside one segment.

Training constructs these pools once from the train split:

```text
atomic_indices = every t where s[t:t+H] is one skill and t is not an episode endpoint
pair_indices   = exactly one t=b-H for each observed adjacent-skill boundary b
start_indices  = exactly episode start t=0 per episode
done_indices   = exactly t=episode_end-H per episode
event_indices  = start_indices + pair_indices + done_indices
```

The sampler draws atomic and event examples 50:50, deterministically from `(seed, epoch)`. Cycling the event pool across optimizer steps is sampling with replacement; it does not create duplicate pair records. Pair windows always come from one original episode, and no video/action/state is concatenated or persisted separately.

This tests within-task latent stage linking. Without subtask names or a verified global ID ontology, it cannot support a claim that the same semantic atomic skill is shared across LIBERO tasks.

## Exact current execution flow

### Dataset and temporal expansion

```text
lerobot-train
  -> make_train_eval_datasets / make_dataset
  -> resolve_delta_timestamps
  -> LeRobotDataset(..., delta_timestamps=...)
  -> DatasetReader._get_query_indices
  -> clamp query to episode and emit <key>_is_pad
  -> DatasetReader._query_hf_dataset
  -> DatasetReader.get_item
  -> DataLoader + policy preprocessor
```

| Location | Current behavior | Decision |
|---|---|---|
| `src/lerobot/datasets/factory.py:34-66` | Expands only reward, action, and `observation.*` features | Add `subtask_index: [0, ..., H]` only when linking is enabled and the feature exists. |
| `src/lerobot/datasets/factory.py:85-102,178-205` | Passes resolved timestamps to full/train/eval datasets | Reuse unchanged. Train and held-out eval receive identical windows. |
| `src/lerobot/datasets/dataset_reader.py:215-232` | Clamps each requested index inside its episode and emits a Boolean pad mask | Reuse unchanged; `subtask_index_is_pad[:, H]` identifies `DONE`. |
| `src/lerobot/datasets/dataset_reader.py:253-268` | Stacks any non-video temporal feature | Reuse unchanged; no `DatasetReader` special case is needed. |
| `src/lerobot/datasets/dataset_reader.py:312-353` | Merges temporal values and padding, then resolves only the global task string | Reuse unchanged. The raw numeric subtask window remains in the batch. |
| `src/lerobot/processor/device_processor.py:84-120,122-169` | Moves all complementary tensors to the policy device and preserves integer dtype | Reuse unchanged; convert IDs explicitly to `torch.long` in the target helper. |

### SmolVLA training

```text
SmolVLAPolicy.forward                         # modeling_smolvla.py:277-332
  -> prepare images/state/language/actions
  -> VLAFlowMatching.forward                  # :697-733
     -> sample float32 noise and beta time
     -> x_t = t * noise + (1-t) * actions
     -> embed_prefix(images, language, state) # :557-652
     -> embed_suffix(x_t, time)               # :654-695
     -> SmolVLMWithExpertModel.forward
     -> final RMSNorm                         # smolvlm_with_expert.py:955-963
     -> suffix_out.float()
     -> action_out_proj
     -> per-element flow MSE
  -> mask episode-padding action loss
  -> reduce valid action loss
```

The shared flow helper samples float32 noise/time in `src/lerobot/policies/common/flow_matching.py:35-58`. The target velocity is `u_t = noise - actions` at `modeling_smolvla.py:707-709`.

### SmolVLA synchronous inference

```text
SmolVLAPolicy.select_action                   # modeling_smolvla.py:243-272
  -> if action queue is empty
     -> apply transition predicted by previous chunk
     -> VLAFlowMatching.sample_actions        # :735-786
        -> prefix prefill and KV cache
        -> euler_integrate                     # common/flow_matching.py:61-122
           -> num_steps denoise calls, t=1 ... 1/num_steps
           -> x_t <- x_t - v_t / num_steps
        -> action chunk and final-step transition logits
     -> enqueue first n_action_steps actions
  -> pop one action
```

`lerobot-eval` calls `policy.reset()` and then `policy.select_action()` at `src/lerobot/scripts/lerobot_eval.py:219-220,291-296`, so this is the primary experiment path.

### RTC and direct chunk inference

- `select_action()` explicitly rejects RTC at `modeling_smolvla.py:254-256`.
- `predict_action_chunk()` returns a whole tensor and does not own queue-consumption timing (`:231-241`).
- The RTC engine invokes it asynchronously and merges overlapping chunks at `src/lerobot/rollout/inference/rtc.py:288-343`.
- RTC guidance modifies every denoising call through `common/flow_matching.py:105-117` and `policies/rtc/modeling_rtc.py:117-249`.

The policy cannot know when an asynchronously merged chunk has actually reached its transition horizon. Supporting state switches there requires an engine-level acknowledgement protocol. That is outside the minimal intervention, so config validation must reject `skill_linking_enabled=True` together with enabled RTC, and direct/async `predict_action_chunk()` must fail clearly rather than silently keep `START` forever.

## Resolved data and label contract

Let `H = n_action_steps`, `s[t:t+H]` be the expanded subtask window, and `p[t:t+H]` its pad mask.

```text
current_skill = START                     if frame_index[t] == 0
current_skill = s[t]                      otherwise

next_state = DONE                         if p[t+H] is true
next_state = s[t+H]                       otherwise

target = next_state                       if current_skill == START
target = CONTINUE                         if next_state == current_skill
target = next_state                       otherwise
```

Class layout is fixed and checkpoint-stable.

```text
skill embedding rows: 0..15, START=16
transition classes:   skill_0..skill_15, CONTINUE=16, DONE=17
```

Do not infer vocabulary size from one batch. Before training, scan the train split once and fail unless all values are integers in `1..15`. ID `0` remains a reserved checkpoint-compatible row/class with zero training weight. Also record counts by `(task_index, subtask_index)`, transition edge, class, and boundary pool.

### Semantic vocabulary scope

The selected revision includes the official sibling mapping as `meta/subtasks.parquet`. Reports resolve IDs through metadata rather than a source-code lookup table. Global instruction remains in the VLM prefix. Claims remain limited to the ten seen tasks and nine observed directed pair types.

Task-scoped IDs such as `task_index * 16 + subtask_index` are intentionally not used: `task_index` is present in offline training batches but not in standard LIBERO inference observations. Adding a brittle task-string lookup table would expand the implementation without resolving the missing ontology.

## Chunk-boundary decision

Keep `chunk_size=50` and set `n_action_steps=H=10` for the pilot, subject to the pre-run boundary audit in `03_experiment_plan.md`.

- `chunk_size` controls the 50 action targets and suffix tokens (`configuration_smolvla.py:29,180-181`).
- `n_action_steps` controls how many actions are queued before the model is called again (`modeling_smolvla.py:170-174,262-269`).
- Reducing only `n_action_steps` is supported and does not change checkpoint tensor shapes.
- The current skill means "skill at chunk start," not "every future action belongs to this skill." Therefore action loss remains valid across a boundary and is not masked.
- The transition target is `s[t+H]`, exactly the skill state needed when the next chunk is generated. Using the first changed skill would apply it too late whenever a boundary occurs before `t+H`.

If more than one boundary occurs in over 1% of valid `H=10` windows, the orchestrator must reduce `H` before any result is observed. Do not add boundary-offset prediction or per-token skill embeddings in the first implementation.

## Model adaptation

### Configuration: `configuration_smolvla.py:26-147`

Add only:

```text
skill_linking_enabled: bool = False
skill_linking_num_skills: int = 16
skill_transition_loss_weight: float = 0.1
skill_transition_class_weights: list[float] | None = None
skill_linking_sampler_enabled: bool = False
```

Validation requires `num_skills > 0`, exactly `num_skills + 2` positive finite class weights when supplied, `n_action_steps > 0`, and no enabled RTC. `skill_transition_threshold` and `skill_transition_confirmations` are removed from the first implementation: argmax over a class that already includes `CONTINUE` is the aligned one-decision-per-chunk rule. Keep `compile_model=False` while linking is enabled because the enabled path returns auxiliary logits and captures the last denoise result.

`skill_linking_sampler_enabled` separates stage-stratified index selection from model conditioning. B0 enables only this sampler flag; P1 enables `skill_linking_enabled`. Both therefore receive the same seed/H/sample order while B0 receives no `subtask_index` temporal tensor and has no embedding/head/loss.

Expose a `subtask_delta_indices` property returning `range(n_action_steps + 1)` when enabled and `None` otherwise; `resolve_delta_timestamps()` consumes it only for the exact `subtask_index` feature.

### Modules: `VLAFlowMatching.__init__` at `modeling_smolvla.py:492-543`

When enabled, create:

```text
skill_embedding = nn.Embedding(num_skills + 1, expert_hidden_size)
transition_head = nn.Linear(expert_hidden_size, num_skills + 2)
```

- `expert_hidden_size` at `smolvlm_with_expert.py:299` is the only width source; do not use the 960-wide VLM text size.
- Zero-initialize the skill embedding so an old action checkpoint starts with unchanged action inputs.
- Initialize the small transition head normally; do not alter existing action/VLM parameters.
- The new modules remain trainable together with the action expert. `train_expert_only=True` freezes the VLM at `smolvlm_with_expert.py:450-458` but does not freeze modules owned by `VLAFlowMatching`.

### Skill conditioning: `embed_suffix` at `modeling_smolvla.py:654-695`

Pass `current_skill` explicitly and add its embedding after `action_time_mlp_out` (`:678-680`), broadcast over 50 suffix tokens. Cast the embedding to `action_time_emb.device` and `action_time_emb.dtype` before addition.

Disabled mode must call the current two-argument path unchanged. Do not add a planner token, second transformer, or VLM-width projection.

### Transition hidden state

Use the post-final-RMSNorm `suffix_out` returned by `SmolVLMWithExpertModel.forward` (`smolvlm_with_expert.py:955-963`). In both train and inference:

```text
transition_features = suffix_out[:, :n_action_steps].float().mean(dim=1)
transition_logits = transition_head(transition_features)
```

This is preferable to the final token or all 50 tokens because it represents exactly the action prefix that will execute before the pending state is applied. Apply the head after the existing float32 upcast at training `modeling_smolvla.py:728-731` and inference `:827-830`; this avoids BF16/Float linear mismatches.

Training obtains this hidden state from the existing single flow forward at random beta-sampled time. Inference uses the hidden state from the **last denoise call**, which occurs at `t=1/num_steps` before the final Euler update, not from a nonexistent `t=0` forward. In `sample_actions()`, use a closure-local variable to capture the latest logits while `euler_integrate()` still receives velocity tensors only. Do not mutate a model-global `_last_logits` field and do not modify the shared Euler helper.

The random-time versus final-denoise distribution gap is not resolved by an official reference implementation. The overfit gate must report transition CE/F1 at fixed `t in {1.0, 0.5, 1/num_steps}`. If final-step F1 is materially worse, stop before full training; a second planning forward is a follow-up, not a silent addition.

### Loss: `SmolVLAPolicy.forward` at `modeling_smolvla.py:277-332`

Keep the existing valid-action reduction unchanged, then add:

```text
L_total = L_flow + skill_transition_loss_weight * weighted_cross_entropy(logits, target)
```

Compute targets in a small pure helper from `frame_index`, `subtask_index`, and `subtask_index_is_pad`. Validate rank, batch length, ID range, and `H+1` window length. Class weights are computed once during the preregistered data audit and stored in config; do not estimate them per batch.

For `reduction="none"`, add per-sample transition CE before returning so RA-BC/sample weighting semantics remain valid. Log flow and transition losses separately without changing the existing `loss` key.

## Causal state and pending timing

`SmolVLAPolicy.reset()` currently clears only the action deque (`modeling_smolvla.py:170-174`). Enabled mode additionally sets `current_skill`, `pending_transition`, and diagnostic flags to `None`; tensors are lazily allocated as `[batch_size]` on the first call so vectorized LIBERO evaluation maintains one state per environment.

Synchronous queue semantics are:

1. On the first empty queue, initialize every environment to `START`.
2. Before generating a later chunk, apply the pending class from the previous generation per environment.
3. `CONTINUE` keeps the current skill.
4. A skill class replaces it.
5. `DONE` is recorded for diagnostics but does not terminate or change the LIBERO environment; success/timeout remains authoritative.
6. Generate actions and one transition prediction from the same forward.
7. Store that argmax as pending and enqueue exactly the first `H` actions.

There is no two-call hysteresis. At chunk-level inference, two confirmations would use the stale skill for an additional `H` or `2H` actions. Calibration thresholds may be added only after reliability curves show that argmax causes false switches.

## Checkpoint initialization and round-trip

Current loading behavior is:

- CLI `--policy.path` loads checkpoint config plus policy overrides at `src/lerobot/configs/train.py:157-179`.
- `make_policy()` passes that overridden config into `from_pretrained()` at `src/lerobot/policies/factory.py:334-340`.
- `PreTrainedPolicy.from_pretrained()` constructs the new model and loads safetensors with default `strict=False` at `src/lerobot/policies/pretrained.py:168-235`.
- The training seed is set before dataset/model construction at `src/lerobot/scripts/lerobot_train.py:282-299`, so new-module initialization is reproducible.

Default `strict=False` is too permissive for this experiment. Add a SmolVLA-local `_load_as_safetensor()` override following the existing narrow override pattern in `src/lerobot/policies/vla_jepa/modeling_vla_jepa.py:469-504`:

- With linking disabled, delegate unchanged to the parent loader.
- With linking enabled and an old checkpoint, allow missing keys only under `model.skill_embedding.` and `model.transition_head.`.
- Reject all unexpected keys, missing keys outside those prefixes, and every shared-key shape mismatch.
- With a new linking checkpoint, require no missing/unexpected keys.
- Log the two intentionally initialized modules and their initialization rule.

Bootstrap training uses `strict=False` with the whitelist above; `strict=True` correctly rejects the old checkpoint. Save/load of a linking checkpoint must be exact because config, embedding, head, preprocessors, and normalization stats are stored together.

## Concept-to-code mapping

| Concept | Reference symbol/file | Baseline target | Required adaptation | Validation |
|---|---|---|---|---|
| Global visual-language context | SmolVLA action expert; official guide | `VLAFlowMatching.embed_prefix`, `modeling_smolvla.py:557-652` | None | Disabled fixed-seed identity |
| Current latent skill | No official SmolVLA equivalent | `embed_suffix`, `:654-695` | Add zero-initialized expert-width embedding after action-time MLP | Different IDs get gradient; zero init preserves initial actions |
| Action flow objective | `x_t=t*noise+(1-t)*action`, velocity regression | `VLAFlowMatching.forward`, `:697-733` | Preserve; add auxiliary output only in enabled path | Baseline flow loss identity and finite joint backward |
| Next-invocation transition | Proposed intervention; no selected official implementation | post-norm suffix hidden at `smolvlm_with_expert.py:955-963` | Pool first `H`, linear `K+2` head | Label fixtures and transition overfit F1 |
| Temporal label/padding | LeRobot delta timestamps | `factory.py:34-66`, `dataset_reader.py:215-268` | Add one generic numeric window | Exact values and `DONE` at episode end |
| Euler denoising | openpi-derived shared helper | `common/flow_matching.py:61-122` | Capture last logits in local closure; helper unchanged | Last captured timestep equals `1/num_steps` |
| Chunk execution | SmolVLA action queue | `SmolVLAPolicy.select_action`, `:243-272` | Apply previous pending state before refill | Switch occurs after exactly `H` pops |
| RTC | PI official RTC implementation | `RTCProcessor`, rollout RTC engine | Reject in first version | Config and `predict_action_chunk` failure tests |
| Old checkpoint bootstrap | LeRobot `from_pretrained(strict=False)` | `pretrained.py:168-235` | SmolVLA-local missing-key whitelist | Old baseline load, injected bad key failure, exact round-trip |

## Minimal source changes

1. `src/lerobot/policies/smolvla/configuration_smolvla.py`
2. `src/lerobot/datasets/factory.py`
3. `src/lerobot/datasets/sampler.py`
4. `src/lerobot/scripts/lerobot_train.py`
5. `src/lerobot/policies/smolvla/modeling_smolvla.py`
6. `tests/policies/smolvla/test_skill_transition.py`

Do not modify `dataset_reader.py`, `dataset_metadata.py`, `smolvlm_with_expert.py`, the shared flow-matching helper, RTC engine, pre/postprocessors, focus-token code, or evaluator success logic. Add no dependency.

## Minimum validation suite

1. Config default disabled; invalid vocabulary/weights/RTC/compile combinations fail.
2. Delta expansion requests exactly `H+1` subtask frames and the generic reader returns aligned pad masks.
3. The index builder emits one and only one `b-H` sample per boundary; no pair crosses an episode; atomic/event sampling is deterministic and 50:50.
4. Pure target helper covers `START`, `CONTINUE`, next skill, `DONE`, malformed windows, and out-of-range IDs.
5. Boundary inside a 50-step action target does not mask flow loss; target is the state at `t+H`.
6. Joint backward gives finite gradients to action expert, skill embedding, and transition head.
7. Skill embedding and transition input obey expert width, device, FP32, BF16/autocast paths.
8. Train logits pool only suffix tokens `[0:H]`; inference captures only the last denoise call.
9. Batched state updates independently per environment and `reset()` clears queue/current/pending/DONE diagnostics.
10. Pending transition changes the conditioning state after exactly `H` deque pops; no extra confirmation delay.
11. Enabled `predict_action_chunk`/RTC fails clearly; normal `lerobot-eval` `select_action` works.
12. Old checkpoint load permits only the two new module prefixes; corrupted/mismatched checkpoints fail.
13. Enabled checkpoint save/load is bit-exact; disabled fixed-seed loss/action output stays unchanged.

Reuse the lightweight fake-layer/mixed-dtype style in `tests/policies/smolvla/test_focus_token.py:107-259` and queue/API expectations in `tests/policies/smolvla/test_smolvla_rtc.py:90-244`. Do not make the CPU unit suite download SmolVLA or LIBERO assets.

## Remaining blockers for `03_experiment_plan.md`

1. **Scientific interpretation:** numeric IDs have no name ontology. The pilot can establish latent stage linking, not shared semantic atomic-skill composition.
2. **Data audit:** class/edge counts and multi-boundary rate for `H=10` must be measured before freezing class weights and horizon.
3. **Initialization:** choose one exact baseline checkpoint whose SmolVLA architecture, processors, and LIBERO normalization statistics match this dataset.
4. **Hidden-state robustness:** fixed-timestep overfit diagnostics must show that random-time training transfers to the final denoise hidden state.
5. **Scope:** RTC/async inference remains unsupported until queue-consumption acknowledgements are designed and evaluated separately.

No application source should be modified until these choices are frozen in an approved `03_experiment_plan.md`.
