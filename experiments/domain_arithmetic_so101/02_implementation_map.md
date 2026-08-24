# Implementation map: DArT × SmolVLA × SO-101

Reference: [`snumprlab/dart@1b23c4f`](https://github.com/snumprlab/dart/tree/1b23c4f42f73168c78a20b353453145e74f64711), 2026-08-03, Apache-2.0.

## Compatibility verdict

The released implementation cannot directly consume the requested SmolVLA checkpoint. It loads named OpenPI configs into JAX/Flax parameter pytrees, applies Pi/PaLI-Gemma-specific reshaping, and saves Orbax checkpoints. SmolVLA is a LeRobot PyTorch model stored in `model.safetensors`.

Relevant upstream paths are [`domain_arithmetic/dart.py`](https://github.com/snumprlab/dart/blob/1b23c4f42f73168c78a20b353453145e74f64711/domain_arithmetic/dart.py), [`base_merge.py`](https://github.com/snumprlab/dart/blob/1b23c4f42f73168c78a20b353453145e74f64711/domain_arithmetic/base_merge.py), and [`utils.py`](https://github.com/snumprlab/dart/blob/1b23c4f42f73168c78a20b353453145e74f64711/domain_arithmetic/utils.py).

## Required artifacts

1. Immutable base `theta_0`: `CoRL2026-CSI/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep-appendix`.
2. Source one-shot `theta_m,src`: the base independently fine-tuned on source dataset episode `0`.
3. Target one-shot `theta_m,tgt`: the base independently fine-tuned on exactly one successful real SO-101 episode.
4. DArT `theta_star`: the base plus the aligned source/target domain vector.

Source and target episodes must use the exact task text `Pick up the red block and place it on the blue dish.`, 10 FPS, six degree-valued joint positions/targets in the same order, and wrist/top camera roles.

## Existing LeRobot paths to reuse

- Episode selection: `DatasetConfig.episodes` in `src/lerobot/configs/default.py`.
- Demonstration recording: existing `lerobot-record` with `dataset.num_episodes=1`.
- Fine-tuning: existing `lerobot-train`; no new trainer.
- Checkpoint format: `PreTrainedPolicy.save_pretrained` / `from_pretrained` in `src/lerobot/policies/pretrained.py`.
- Hardware inference: existing `lerobot-rollout` and `examples/smolvla/run_so101_pick_place.sh`.
- Missing `camera3`: existing SmolVLA `prepare_images` processes present image keys and tolerates the absent slot.

Use this image mapping for both one-shot datasets:

```json
{
  "observation.images.left_wrist": "observation.images.camera1",
  "observation.images.top": "observation.images.camera2"
}
```

## One shared training change

`src/lerobot/scripts/lerobot_train.py` currently replaces pretrained processor statistics with the fine-tuning dataset's statistics. Using separate one-trajectory statistics would put the source and target updates in different normalization coordinate systems.

Add a default-false `preserve_pretrained_processor_stats` field to `TrainPipelineConfig`, and when it is true do not override the loaded normalizer or unnormalizer statistics. Both one-shot runs must enable it. All existing training remains unchanged by default.

The paper uses full-model fine-tuning. Both SmolVLA runs therefore override:

```text
policy.freeze_vision_encoder=false
policy.train_expert_only=false
```

SmolVLA still freezes a small built-in set of final VLM/head parameters through its existing `set_requires_grad`; keep this symmetric and report it as an architecture-specific deviation rather than adding another unfreeze path.

## Experiment-local merger

Implement `experiments/domain_arithmetic_so101/dart_merge.py` using only installed `torch`, `safetensors`, and `huggingface_hub` dependencies.

Inputs may be local checkpoint directories or Hub IDs. The merger resolves each to a single `model.safetensors`, then:

- requires identical key sets, tensor shapes, and floating types;
- computes in float32 on CPU;
- treats 2-D linear weights as `[out, in]`;
- flattens the single 4-D patch-embedding convolution to `[out, -1]` and restores its shape;
- uses direct arithmetic for 1-D tensors;
- copies the base exactly for zero source-and-target updates;
- casts output to the base dtype;
- raises on unsupported shapes or mismatches instead of silently copying partial results.

Use rank-256 randomized SVD by default for practical memory/runtime. If `rank >= min(shape)`, use exact full SVD. The published checkpoint contains roughly 500 tensors (305 rank-2, 194 rank-1, one rank-4), so a per-tensor port avoids upstream's whole-model JAX/Orbax machinery.

## Native output

Write a standard LeRobot directory:

```text
OUTPUT/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_5_normalizer_processor.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── dart_merge.json
```

Copy config and processor artifacts from the base unchanged. Only model weights are merged. `dart_merge.json` records resolved revisions/paths, alpha, rank, seed, tensor counts, and zero-update count. The result then loads through the normal policy loader and uses the existing SO-101 rollout without inference changes.

## Fine-tuning contract

Both runs start from the pinned base revision and use:

- AdamW;
- effective batch size `64`;
- learning rate `5e-5`;
- no warmup and constant learning rate;
- `1,000` optimizer updates;
- identical seed, optimizer, parameter mask, preprocessing, and disabled image augmentation;
- base processor normalization statistics;
- no EMA/PEFT.

The source run selects simulation episode `0`. The target run selects the sole real episode. The final checkpoint from each run is the arithmetic input.

## Minimum tests

One focused test file must cover:

1. exact 1-D arithmetic;
2. a small 2-D matrix against a literal paper-equation implementation;
3. zero updates return the base bitwise;
4. mismatched keys and shapes raise;
5. tiny synthetic safetensors checkpoints produce a loadable output and preserve base processor files;
6. the new training flag preserves checkpoint stats while its default retains current behavior.

No robot test is required for the merger. Hardware behavior is assessed only by the preregistered real-world protocol.

## Claim limits

- This is a native SmolVLA port, not direct execution of released DArT.
- This tests same-task sim-to-real improvement, not cross-task transfer.
- The paper validates Pi0.5 and Pi0-FAST, not SmolVLA.
- Using only the real demo without the matching source fine-tune is ordinary one-shot fine-tuning, not DArT.
- Fixed base normalization and identical source/target interfaces are required for causal attribution.
