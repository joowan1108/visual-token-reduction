# Literature-review amendment 04: multi-task anchor (Experiment M)

## Verdict

Experiment M uses `Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod` as the immutable
source-domain policy `theta_0`. This is closer to DArT's stated setup than the prior
pick-and-place-specialized anchor because DArT assumes a source-trained multi-task policy, then
independently fine-tunes copies of that same policy on matched source- and target-domain
demonstrations of one adaptation task.

This is a protocol-fidelity claim, not a performance claim. The DArT paper does not compare
multi-task and task-specialized anchors, evaluates OpenPI policies rather than SmolVLA, and does
not test IsaacLab-to-real SO-101 transfer.

## Evidence

- Paper: [Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts](https://arxiv.org/html/2607.00666), arXiv `2607.00666v1`.
- Official implementation: [`snumprlab/dart@1b23c4f`](https://github.com/snumprlab/dart/tree/1b23c4f42f73168c78a20b353453145e74f64711).
- The paper defines one immutable source-trained multi-task policy `theta_0`, with independent
  updates `Delta_src = theta_src - theta_0` and `Delta_tgt = theta_tgt - theta_0`, and returns
  `theta_star = theta_0 + alpha * refined_domain_delta`.
- The official implementation likewise computes both updates against `w_base` and adds the
  result back to `w_base`.
- The Experiment M checkpoint is a full-model, non-LoRA SmolVLA checkpoint trained at 10 FPS on
  3,300 IsaacLab SO-101 episodes across 11 task families. Its task table contains the exact
  instruction `Pick up the red block and place it on the blue dish.`.

## Constraints

- Source and target demonstrations must match task, approximate object position, camera roles,
  FPS, action/state convention, and trajectory semantics.
- Both fine-tunes must start from the exact same pinned `theta_0` and use identical optimization.
- Both fine-tunes and merged outputs must preserve `theta_0` processor statistics.
- Sim-to-real changes more than viewpoint alone, so a negative result does not refute DArT under
  the paper's controlled shift settings.

