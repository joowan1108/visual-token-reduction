# Amendment 04: supervised atomic skill classifier

- Date: 2026-08-14
- Results inspected before amendment: yes
- Trigger: the zero-shot generative SmolVLM planner produced invalid free text and then a repeated incorrect
  `turn` route on the LIBERO-10 task-0 smoke rollout.

The generative planner is replaced for amended evaluations by a supervised six-class head. The head consumes
the frozen action policy's current visual and task-token representations, robot state, and previous skill
(`none` at episode start), and predicts `pick|place|push|turn|open|close`. It is trained with the existing SARM
ground-truth atomic skill at the anchor frame. The action policy and frozen backbone are loaded from the completed
SG-MoE checkpoint and remain frozen; only the classifier and previous-skill embedding are optimized.

This is a post-result protocol amendment. Classifier-routed results must be reported separately from the
preregistered zero-shot generative-planner result and cannot replace it. Primary metrics, task manifests, seeds,
and action-policy checkpoints are unchanged.

Held-out classifier evaluation excludes episode starts from transition metrics and reports stay accuracy plus
binary switch precision and recall. A predicted skill different from the previous skill is a predicted switch.
