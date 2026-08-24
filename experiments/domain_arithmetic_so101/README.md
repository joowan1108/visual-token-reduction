# Run the preregistered SO-101 DArT experiment

Install the existing extras, then run each stage once and in order:

```bash
uv sync --locked --extra smolvla --extra feetech
LEADER_PORT=/dev/ttyACM1 ROBOT_PORT=/dev/ttyACM0 ./experiments/domain_arithmetic_so101/run.sh record
./experiments/domain_arithmetic_so101/run.sh train-source
./experiments/domain_arithmetic_so101/run.sh train-target
./experiments/domain_arithmetic_so101/run.sh merge
```

The recorder accepts exactly one episode. Use Left Arrow to reject a failed attempt before accepting the first successful demonstration, and append every failed attempt/reason to a separate log; never replace the accepted episode. Verify its two camera roles, six degree-valued joints, exact task text, and 10 FPS metadata before training.

Use `rollout` only for a non-recording hardware smoke test. For each frozen trial, use
`evaluate`; it records exactly one 30-second episode and its camera videos for blinded scoring:

```bash
./experiments/domain_arithmetic_so101/run.sh rollout  # optional Z smoke test

# Run the policy assigned to one row of the frozen randomized manifest. The trial ID
# must contain only the manifest ID and anonymized condition code, not Z/F/A/D.
TRIAL_ID=block01_code2 POLICY_PATH=experiments/domain_arithmetic_so101/artifacts/dart \
  ./experiments/domain_arithmetic_so101/run.sh evaluate
```

Set `POLICY_PATH` from the private condition-code map to the base checkpoint (Z), target fine-tune
(F), direct merge (A), or DArT merge (D). Each `TRIAL_ID` creates a new local dataset under
`artifacts/evaluation/`; never reuse one. Do not tune from outcomes. Follow `03_experiment_plan.md`
for the 96-row frozen manifest, layouts, blinded scoring, hashes, and append-only result log.
