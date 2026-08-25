# Run the SO-101 one-shot DArT experiment

The target is pinned to episode `0` only of
`sungkyunner/record-test_20260825_225339@97e2c1d4d49607210d1e63d46db2a43b530bdf89`.
It is the actual rollout rig's 300-frame, 30-second successful trajectory at 10 FPS with
`left_wrist` and `top` views. See `03_amendment_02_same_rig_target.md` for the frozen replacement.

## Train and merge

Use a fresh `RUN_ROOT`: training and merging intentionally refuse to overwrite prior artifacts.
The single command validates the pinned Hub episode, runs the simulation and target episode-0
fine-tunes, then produces the direct and DArT merges in order.

```bash
uv sync --locked --extra training --extra smolvla --extra feetech
export RUN_ROOT=experiments/domain_arithmetic_so101/artifacts/public_ep0_run1  # must not exist yet
./experiments/domain_arithmetic_so101/run.sh adapt
```

To run stages separately, use `prepare-target`, `train-source`, `train-target`, then `merge` with
the same `RUN_ROOT`. Preparation downloads the original pinned files to the Hugging Face cache,
validates the exact task/interface, and writes only `target_provenance.json` with per-file SHA-256
values and a deterministic tree hash. It performs no task, gripper, frame, or video rewrite.

Both fine-tunes use the base processor statistics, 1,000 updates, batch 8 with accumulation 8
(effective batch 64), PyAV, zero DataLoader workers, constant LR `5e-5`, seed 1000, and only
episode 0. `train-target` reads the pinned Hub dataset directly. The
optional `record` command is not part of this pinned run; it always writes to a separate local
dataset and refuses the public target repo ID.

To reuse a previously verified source checkpoint, keep the replacement outputs in a fresh
`RUN_ROOT` and pass its local checkpoint directory only to merge:

```bash
export RUN_ROOT=experiments/domain_arithmetic_so101/artifacts/same_rig_ep0_run1
export SOURCE_CHECKPOINT=/absolute/path/to/source_finetune/checkpoints/last/pretrained_model
./experiments/domain_arithmetic_so101/run.sh prepare-target
./experiments/domain_arithmetic_so101/run.sh train-target
./experiments/domain_arithmetic_so101/run.sh merge
```

On the remote GPU server, update the experiment branch and run the replacement target as follows:

```bash
cd /visual-token-reduction
git fetch origin
git switch inference/smolvla-so101
git pull --ff-only origin inference/smolvla-so101
uv sync --locked --extra training --extra smolvla --extra feetech

export RUN_ROOT="$PWD/experiments/domain_arithmetic_so101/artifacts/same_rig_ep0_run1"  # new path
export SOURCE_CHECKPOINT=/absolute/path/to/verified/source_finetune/checkpoints/last/pretrained_model
test ! -e "$RUN_ROOT"
test -f "$SOURCE_CHECKPOINT/model.safetensors"
./experiments/domain_arithmetic_so101/run.sh check
./experiments/domain_arithmetic_so101/run.sh prepare-target
./experiments/domain_arithmetic_so101/run.sh train-target
./experiments/domain_arithmetic_so101/run.sh merge
```

Unset `SOURCE_CHECKPOINT` and insert `train-source` before `train-target` when the prior source
checkpoint has not been verified against the frozen base, optimizer, step, processor, and model
hashes.

## Connected-hardware rollout

Rollout requires a calibrated SO-101 follower and two connected USB cameras. Defaults are follower
`/dev/ttyACM0`, wrist camera index `0`, and top camera index `2`; override `ROBOT_PORT`,
`WRIST_CAMERA`, or `TOP_CAMERA` as needed. The physical cameras capture at 640x480 and 30 FPS while
the policy loop runs at 10 Hz. RTC is fixed by default to execution horizon 10 and maximum guidance
weight 10. `CAMERA_FPS`, `POLICY_FPS`, `INFERENCE_TYPE`, `RTC_EXECUTION_HORIZON`,
`RTC_MAX_GUIDANCE_WEIGHT`, and `MAX_RELATIVE_TARGET` remain explicit hardware calibration/runtime
knobs; do not vary them between formal conditions.

```bash
CONDITION=Z ./experiments/domain_arithmetic_so101/run.sh rollout  # unchanged base
CONDITION=F ./experiments/domain_arithmetic_so101/run.sh rollout  # target one-shot fine-tune
CONDITION=A ./experiments/domain_arithmetic_so101/run.sh rollout  # direct arithmetic
CONDITION=D ./experiments/domain_arithmetic_so101/run.sh rollout  # DArT
```

These are hardware smoke tests, not evidence. Formal evaluation must keep the demonstrated rig,
camera mounts, calibration, objects, lighting, reset pose, and runtime configuration fixed.

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
