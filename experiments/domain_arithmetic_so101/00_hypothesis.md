# Hypothesis: one-shot DArT adaptation for SO-101 sim-to-real transfer

## Question

Does Domain Arithmetic (DArT) adaptation from one real-world SO-101 pick-and-place demonstration improve the real-world success rate of `CoRL2026-CSI/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep-appendix` over the unchanged simulation-trained checkpoint?

## Hypothesis

Applying the DArT one-shot adaptation procedure to one real-world demonstration of the checkpoint's original task—"Pick up the red block and place it on the blue dish."—will increase binary task success under a fixed real-world evaluation protocol compared with the unadapted checkpoint.

## Null hypothesis

The adapted policy's real-world binary task success rate is no higher than the unadapted checkpoint's success rate under the same conditions.

## Scope

- Robot: calibrated SO-101 follower.
- Source policy: the specified simulation-trained SmolVLA checkpoint.
- Adaptation data: exactly one successful real-world demonstration.
- Shift: simulation to the user's fixed real-world camera, lighting, background, object, dish, and robot setup.
- Primary outcome: per-trial binary pick-and-place success.
- Safety failures and invalid trials must be recorded, not silently retried.

## Falsification

The hypothesis is unsupported if the preregistered comparison does not show the required improvement in primary success rate, or if adaptation cannot be applied without changing the policy/task interface or using more than one real-world demonstration.
