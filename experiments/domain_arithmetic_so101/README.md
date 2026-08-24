# Run the SO-101 one-shot DArT experiment

The target is pinned to episode `0` only of
`skkuprism/test_pick_red_place_blue_50epi_10fps@e19331e77f477a4be16f7c2884250ed6f491e048`.
It is a pre-existing 329-frame, 32.9-second successful trajectory at 10 FPS with `left_wrist` and
`top` views. See `03_amendment_01_public_target_demo.md` for the resulting claim limit and the
frozen interface correction.

## Train and merge

Use a fresh `RUN_ROOT`: training and merging intentionally refuse to overwrite prior artifacts.
The single command prepares the pinned public episode, runs the simulation and target episode-0
fine-tunes, then produces the direct and DArT merges in order.

```bash
uv sync --locked --extra smolvla --extra feetech
export RUN_ROOT=experiments/domain_arithmetic_so101/artifacts/public_ep0_run1  # must not exist yet
./experiments/domain_arithmetic_so101/run.sh adapt
```

To run stages separately, use `prepare-target`, `train-source`, `train-target`, then `merge` with
the same `RUN_ROOT`. Preparation creates `target_canonical_ep0/`, refuses an existing output, keeps
only state, action, wrist/top views, and converts only the two gripper columns with
`new=(old+100)/2`; it also sets the exact source prompt and records provenance in
`target_preparation.json`, including per-file SHA-256 values and a deterministic tree hash. This is
deterministic interface conversion, not augmentation.

Both fine-tunes use the base processor statistics, 1,000 updates, effective batch 64, constant LR
`5e-5`, seed 1000, and only episode 0. `train-target` consumes only the prepared local dataset. The
optional `record` command is not part of this pinned run; it always writes to a separate local
dataset and refuses the public target repo ID.

## Connected-hardware rollout

Rollout requires a calibrated SO-101 follower and two connected USB cameras. Defaults are follower
`/dev/ttyACM0`, wrist camera index `0`, and top camera index `2`; override `ROBOT_PORT`,
`WRIST_CAMERA`, or `TOP_CAMERA` as needed. The physical cameras must provide the same wrist/top
semantics at 640x480 and 10 FPS.

```bash
CONDITION=Z ./experiments/domain_arithmetic_so101/run.sh rollout  # unchanged base
CONDITION=F ./experiments/domain_arithmetic_so101/run.sh rollout  # target one-shot fine-tune
CONDITION=A ./experiments/domain_arithmetic_so101/run.sh rollout  # direct arithmetic
CONDITION=D ./experiments/domain_arithmetic_so101/run.sh rollout  # DArT
```

These are hardware smoke tests, not evidence. Since the public target demo came from a candidate
rig, formal evaluation only measures the intended shift on the matching physical setup, or one
verified equivalent before outcomes are observed. Matching camera labels alone is insufficient.

## Recorded formal evaluation

`evaluate` runs one safety-limited 30-second rollout and saves its two camera videos. For blinded
evaluation, resolve the randomized condition code to a policy path privately and expose only the
anonymized manifest ID to the scorer/operator:

```bash
POLICY_PATH="$(./experiments/domain_arithmetic_so101/run.sh condition-path D)" \
TRIAL_ID=block01_code2 \
  ./experiments/domain_arithmetic_so101/run.sh evaluate
```

Replace `D` through the private condition-code map; do not encode Z/F/A/D in `TRIAL_ID`. Never reuse
a trial ID or tune from outcomes. Follow `03_experiment_plan.md` and its amendment for the frozen
layouts, 96 trials, blinded scoring, artifact hashes, and append-only result log.
