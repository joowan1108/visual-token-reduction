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

---

# Change log: Amendment 03 exact-SVD sensitivity

Date: 2026-08-26

## Changed files

- `dart_merge.py`: replaces rank-256 randomized SVD with exact thin SVD through
  `torch.linalg.svd(..., full_matrices=False)`, retains every `min(m, n)` component, and removes
  merge rank/seed parameters and metadata.
- `run.sh`: removes obsolete merge rank/seed flags and accepts a verified `TARGET_CHECKPOINT`
  override alongside `SOURCE_CHECKPOINT` for a merge-only fresh output root.
- `README.md`: documents the merge-only source/target reuse workflow and system-RAM expectation.
- `tests/experiments/test_domain_arithmetic_so101.py`: checks the full thin spectrum, exact-SVD
  provenance, removed rank/seed flags, and both checkpoint overrides.

## Configuration and commands

- SVD: `torch.linalg.svd`, `full_matrices=false`, all thin-spectrum components retained.
- Arithmetic: float32 CPU, `alpha=0.8`, target energy cutoff `0.9975`; unchanged 1-D direct
  arithmetic, 4-D flatten/restore, zero-update skip, processor copy, and input SHA-256 metadata.
- Training dataset, training seed `1000`, budgets, metrics, runtime, and evaluation criteria are
  unchanged. No training is required for this amendment.

```bash
bash -n experiments/domain_arithmetic_so101/run.sh
RUN_ROOT=/tmp/domain-arithmetic-check-$$ experiments/domain_arithmetic_so101/run.sh check
uv run pytest tests/experiments/test_domain_arithmetic_so101.py -q --tb=short
```

Shell syntax, workflow check, and focused Ruff checks passed. Pytest was blocked during conftest
import because the local PyTorch installation cannot load `libcublasLt.so.12`; no test body ran.

## Assumptions and deviations

- Source and target overrides must be the already verified Amendment 02 checkpoints. Exact-SVD
  outputs use a fresh `RUN_ROOT`; prior randomized-SVD and direct artifacts are never overwritten.
- This retains the SmolVLA per-tensor safetensors implementation rather than the paper's JAX/Pi
  checkpoint loader. It is an outcome-informed exploratory sensitivity analysis as declared in
  `03_amendment_03_exact_svd.md`, not a replacement for the preregistered D condition.
- No experimental outcomes were run or interpreted during implementation.

---

# Change log: Amendment 04 Experiment M

Date: 2026-08-27

## Changed files

- `run.sh`: pins the multi-task Experiment M base and episode-170 source, makes the target Hub
  coordinates configurable with preregistered defaults, rejects mutable target revisions, requires
  visual-match confirmation before target preparation or either fine-tune, binds target training to
  matching provenance coordinates, and prevents checkpoint reuse across runs.
- `prepare_target_dataset.py`: accepts target repo/revision/episode arguments, validates one
  nonempty 10-FPS episode against the exact task/joint/camera contract without requiring a video
  codec, writes immutable selected-content and matched-source confirmation provenance without
  rewriting data, and verifies that provenance before target training.
- `tests/experiments/test_domain_arithmetic_so101.py`: covers Experiment M pins, independent
  same-base source/target commands, episode selection, target configurability/revision rejection,
  codec-independent validation, auditable visual confirmation, provenance binding, content hashing,
  and the unchanged Experiment M merge anchor.

## Frozen configuration and commands

```text
Base: Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod
      @45f76f173c76c4e002131f8b48e345589a071d0f
Source: Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi_10fps
        @09a0376348f60be89edcbc0eb76c3e26b5f3b094, episode 170
Target default: sungkyunner/record-test_20260826_210214
                @295e6def6cb4df454f58894caea10c15446dc4e4, episode 0
Training: seed 1000, 1,000 updates each, batch 8, accumulation 8, workers 0, LR 5e-5
Merge: direct and exact-thin-SVD DArT, alpha 0.8, same Experiment M base anchor
```

```bash
bash -n experiments/domain_arithmetic_so101/run.sh
RUN_ROOT=/tmp/domain-arithmetic-m-check-$$ experiments/domain_arithmetic_so101/run.sh check
uv run ruff check experiments/domain_arithmetic_so101/prepare_target_dataset.py \
  tests/experiments/test_domain_arithmetic_so101.py
uv run pytest tests/experiments/test_domain_arithmetic_so101.py -q --tb=short
```

Shell syntax, workflow, and focused Ruff checks passed. Pytest collection was blocked by the local
CUDA installation before tests ran: `ImportError: libcublasLt.so.12: cannot open shared object
file`. No training, merge, hardware rollout, or result interpretation was performed.

## Assumptions and deviations

- The operator must visually confirm that source episode 170 and the selected target episode show
  the same approximate red-block start position, then explicitly set `VISUAL_MATCH_CONFIRMED=1`;
  target preparation records that confirmation and target training rejects mismatched provenance.
- Target overrides remain valid only with an immutable 40-character lowercase Hub commit SHA and a
  fresh `RUN_ROOT`; all Experiment M checkpoints are produced inside that run.
- DArT arithmetic, exact-SVD behavior, alpha, metrics, seeds, budgets, and evaluation criteria were
  not changed. This remains a native PyTorch/SmolVLA port rather than the paper's OpenPI/JAX code.
