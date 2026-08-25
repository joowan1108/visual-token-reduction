# Change log: Amendment 02 implementation

Date: 2026-08-26

## Changed files

- `run.sh`: pins the same-rig target revision and episode 0, trains it directly from the Hub,
  forces PyAV and zero DataLoader workers, preserves batch 8 / accumulation 8 / 1,000 steps,
  refuses existing outputs, and accepts a local `SOURCE_CHECKPOINT` override for merge.
- `prepare_target_dataset.py`: replaces target rewriting with exact metadata/interface validation
  and an immutable-file SHA-256 provenance manifest. It performs no task, joint, gripper, frame,
  image, or video conversion.
- `dart_merge.py`: records SHA-256 for each resolved input model, including a reused source model,
  in `dart_merge.json`.
- `README.md`: documents the replacement target, fresh-run/source-reuse workflow, remote GPU
  commands, and frozen hardware runtime.
- `examples/smolvla/run_so101_pick_place.sh`: defaults physical capture to 30 FPS, policy control
  to 10 Hz, RTC execution horizon to 10, and RTC maximum guidance weight to 10 while retaining
  environment overrides and trailing CLI arguments.
- `tests/experiments/test_domain_arithmetic_so101.py`: replaces obsolete gripper/image conversion
  tests with target-contract, direct-Hub training command, source override, and RTC/runtime tests.

## Frozen configuration

- Target: `sungkyunner/record-test_20260825_225339`
- Revision: `97e2c1d4d49607210d1e63d46db2a43b530bdf89`
- Episode / frames / dataset FPS: `0` / `300` / `10`
- Task: `Pick up the red block and place it on the blue dish.`
- Cameras: `left_wrist -> camera1`, `top -> camera2`; Hub videos decoded with PyAV
- Training: 1,000 optimizer steps, micro-batch 8, accumulation 8, effective batch 64,
  DataLoader workers 0, unchanged optimizer/scheduler/seeds/base processor statistics
- Runtime: camera capture 30 FPS, policy loop 10 Hz, RTC horizon 10, RTC guidance 10,
  relative target safety limit 5

## Commands and verification

```bash
bash -n experiments/domain_arithmetic_so101/run.sh examples/smolvla/run_so101_pick_place.sh
RUN_ROOT=/tmp/domain-arithmetic-check-$$ experiments/domain_arithmetic_so101/run.sh check
uv run pytest tests/experiments/test_domain_arithmetic_so101.py -q --tb=short
```

The remote environment must include the `training` extra; `smolvla` and `feetech` alone do not
install datasets, PyAV, or the training dependencies. Use `uv sync --locked --extra training
--extra smolvla --extra feetech`.

The shell syntax check and workflow check passed (`workflow condition paths OK`). The focused
pytest collection was blocked by the local CUDA environment before tests ran:
`ImportError: libcublasLt.so.12: cannot open shared object file`. No hardware or experimental
rollout was run and no results were interpreted.

## Assumptions and deviations

- The prior source checkpoint is reused only after external verification against the frozen base,
  optimizer/training configuration, final step, processor hashes, and model hash; otherwise
  `train-source` is rerun in the fresh run root.
- `prepare-target` downloads the exact pinned Hub files into the normal revision-safe Hugging Face
  cache so it can hash them, but creates no rewritten dataset.
- This remains the preregistered native SmolVLA port of DArT rather than the paper's Pi0.5 or
  Pi0-FAST implementation. No paper metric, seed, budget, comparison, or falsification rule changed.
- Merge metadata records the resolved input path/revision and model SHA-256. External verification
  of the remaining source run configuration and processor hashes is still required before reuse.
