# Amendment 01 — AtomicVLA router alignment

- Date: 2026-08-13
- Timing: before training, evaluation, or result inspection
- Reference: `zhanglk9/AtomicVLA@c3583055adde0a491a11ffe08c15ca6459a64254`
- Scope: model initialization only; metrics, seeds, budget, splits, comparison conditions, and falsification criteria are unchanged

## Correction

Section 2.B of `03_experiment_plan.md` specified one router per layer and a per-row inverse-scale
initialization. The pinned official implementation instead owns one router in
`src/openpi/models/gemmoe.py::Module`, computes its combine weights once, and broadcasts those
weights through every scanned transformer layer.

The implemented six-skill condition therefore uses the official behavior:

1. One global router is shared by every action-expert FFN layer.
2. Fixed skill embeddings are scaled one-hot rows with scales `linspace(10, 100, 6)`, zero-padded
   to the action hidden width.
3. The bias-free router kernel is initialized to a padded identity multiplied by
   `log(6 - 1) / 55`, following `src/openpi/models/pi0_atomic.py` with target weight 0.5 and mean
   skill scale 55.
4. A skill embedding is selected once per sample; the resulting expert ID and gate are reused for
   every action token and every action layer.
5. If the selected expert probability is `w`, the FFN output is
   `(1 - w) * shared(x) + w * selected_expert(x)`.
6. No router classification or load-balancing loss is added.

LeRobot executes only experts selected by at least one batch item rather than stacking all expert
outputs as upstream does. This is an efficiency-only deviation with the same top-1 mixture result.

## Preserved intentional deviations

- Six strict SARM skills include `push`; upstream LIBERO tokenization defines five and falls back
  unknown labels to pick.
- The SmolVLM is frozen and the objective is action flow matching only; upstream AtomicVLA also
  trains decision/reasoning text losses and its default LIBERO setup does not freeze the VLM.
- Targets after a SARM skill boundary are added to `action_is_pad`; upstream does not add this
  semantic boundary mask.
- Shared and skill FFNs are copied from the dense SmolVLA FFN for step-zero equivalence; upstream
  creates additional expert parameters separately.

These deviations were preregistered properties of the frozen-VLM SmolVLA intervention and do not
change the statistical protocol.
