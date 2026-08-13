# Pre-result amendment 03: optional visual-encoder training

## Timing and scope

This amendment was written before any atomic training, simulator rollout, or raw-result inspection. It does
not change the preregistered metrics, seeds, update budget, comparison conditions, statistical tests, or
falsification criteria.

## Runtime contract

The existing `SmolVLAConfig.freeze_vision_encoder` switch controls whether the visual encoder is trained; no
atomic-only duplicate flag is introduced. When `train_expert_only=True` and
`freeze_vision_encoder=False`, the language/text VLM, connector, and language-model head remain frozen while
exactly `vision_model` and the existing action-expert training set are trainable. The vision encoder follows
the policy's train/eval mode, and its `requires_grad=True` parameters are exposed through the existing
optimizer parameter path.

Dense A and SG-MoE B must use the same `freeze_vision_encoder` value in any comparison. The default remains
`True`, preserving the original frozen-VLM condition. Optional visual-encoder training is valid only for
training with `atomic_planner_enabled=False`. Online planner evaluation requires
`freeze_vision_encoder=True`, because generation must share a fully frozen SmolVLM. A trained checkpoint may
therefore be evaluated by loading it with the planner-enabled evaluation override and the vision encoder
frozen.

## Reporting

Every resolved training and evaluation config must record `freeze_vision_encoder`. Results from frozen and
trainable visual-encoder settings must not be pooled or treated as the same comparison condition.
