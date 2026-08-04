# 04. Implementation change log: Focus Token Selection for SmolVLA

## Scope

Implemented only the preregistered late action-aware visual-token selection. The dense baseline remains the default, and no dependency, dataset, optimizer, scheduler, flow-matching, Euler integration, evaluator, or raw-result path was changed.

## Changed files

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
  - Added `focus_token_keep_ratio` with dense default `1.0`.
  - Added `focus_token_start_layer` with default `8`.
  - Validates ratio/layer range and rejects sparse use with a non-preregistered layer layout or `compile_model=True`.
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - Records camera patch spans while constructing the existing prefix.
  - Passes spans through training direct-prefix and inference cached-prefix paths.
  - Does not change image embeddings, prefix order, loss, noise/time sampling, or integration.
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
  - Uses the existing projected expert query/key tensors to compute scaled QK scores.
  - Applies independent camera budgets with `max(1, ceil(ratio * N_valid))`, selects only valid patches, and restores original prefix order.
  - Retains every nonvisual position and selects zero patches from a fully masked camera.
  - Gathers K, V, and mask views only at sparse late cross-attention; it does not crop or mutate the full `DynamicCache`.
  - Dense ratio `1.0` bypasses selection completely.
- `tests/policies/smolvla/test_focus_token.py`
  - Adds focused config, camera-budget/order/mask, layer-boundary, dense-identity, direct-prefix backward, cached-prefix action/backward, and cache-length checks.

## Effective configuration

Primary intervention:

```yaml
policy.focus_token_keep_ratio: 0.5
policy.focus_token_start_layer: 8
policy.compile_model: false
policy.num_vlm_layers: 16
policy.num_expert_layers: -1
policy.attention_mode: cross_attn
policy.self_attn_every_n_layers: 2
```

Baseline changes only `policy.focus_token_keep_ratio` to `1.0`. Sensitivity conditions use `0.25` and `0.75`; all other fields remain paired and unchanged.

With the fixed 16-layer interleaving, only cross-attention layers 9, 11, 13, and 15 enter the sparse branch. Layers 0-7 remain dense, and layers 8, 10, 12, and 14 retain their existing self-attention path.

## Verification commands and results

Passed:

```powershell
$env:PYTHONUTF8='1'; python -S -m py_compile src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py
git diff --check
ruff format src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py
ruff check src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py
```

Results: Python syntax compilation passed, diff whitespace check passed, four files were formatted, and Ruff reported `All checks passed!`.

Attempted focused runtime test:

```powershell
$env:PYTHONUTF8='1'; pytest -q tests/policies/smolvla/test_focus_token.py
```

Collection was blocked before the test module ran:

```text
ImportError while loading conftest
ModuleNotFoundError: No module named 'draccus'
```

The repository `uv` command was not on `PATH`. The discovered executable at `C:\Users\joowa\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts\uv.exe` initially failed to access its default cache with `os error 5`. A writable temporary cache could be initialized, but only Python 3.10 was installed and the available system/conda environments contained neither `torch` nor `draccus`. No dependency installation or long network sync was performed.

Therefore the focused runtime tests are present but not executed in this environment. Experimental training/evaluation must not start until they pass in the repository's locked Python 3.12+ test environment.

## Primary review addendum

The primary orchestrator independently reviewed the source diff and reran:

```powershell
python -S -m py_compile <the four changed Python files>
ruff check <the four changed Python files>
ruff format --check <the four changed Python files>
git diff --check
```

All static checks passed, and no source/test file outside the approved three SmolVLA modules plus `test_focus_token.py` was changed.

An attempt to create the locked Python 3.12 environment with `uv python install 3.12` used a writable temporary cache but failed after three network retries because the GitHub download tunnel was refused (`os error 10061`). Consequently the runtime gate remains incomplete and `paper_method_evaluator` has not been started.

### Package-install retry

After explicit user authorization to install packages, the primary orchestrator retried the locked setup with:

```powershell
uv python install 3.12
uv sync --locked --extra test
```

The Python download again failed after three retries with the same refused network tunnel (`os error 10061`), so dependency sync could not begin. Local Conda contains Python 3.11 and pytest but not Torch or Draccus; no usable local wheel cache, Docker environment, or accessible WSL environment was found.

The preregistered evaluator also requires Linux: `pyproject.toml` installs `hf-libero` only when `sys_platform == 'linux'`. The current host is Windows. Both the runtime-test gate and the LIBERO evaluation environment therefore remain unavailable, and `paper_method_evaluator` was not started.

## Assumptions

- Layer indices are zero-based, as frozen in `03_experiment_plan.md`.
- Prefix patch spans are contiguous per camera because `embed_prefix` appends each camera embedding as one tensor.
- Validity is derived from the existing attention mask; ragged per-sample selections are padded only inside the gathered view and the padding columns are masked false.
- Existing checkpoint strict loading is preserved because no parameter, buffer, or state-dict key was added or renamed.

## Paper adaptation and deviations

- This is the preregistered SmolVLA ablation, not a full FocusVLA reproduction.
- No learned router, channel gate, cascaded selector, or new trainable parameter was added.
- Selection is recomputed from the current noisy-action query at every target layer and denoising step.
- The VLM prefix and full cache remain dense; only the expert's late cross-attention lookup view is reduced.
- Runtime test execution is the only implementation-validation deviation, caused by missing local dependencies. Metrics, seeds, datasets, budgets, and evaluation criteria were not changed.

## Runtime gate completion (2026-08-04)

The earlier runtime-blocked notes above record the initial environment state and are superseded by this addendum. The user provided C:\Users\joowa\miniconda3\envs\test\python.exe (Python 3.12.13 with Torch, Draccus, and pytest), so the focused runtime gate was completed without installing or changing dependencies.

The runtime retry changed only tests/policies/smolvla/test_focus_token.py:

- The sparse-layout validation case now supplies a valid start layer so it tests the intended 16-layer layout constraint.
- The fake expert K/V projections now accept the flattened VLM KV width (2), matching the real cross-attention projections; the expert query projection continues to accept expert hidden width (4).
- The cached-path assertion reads the sole expert output at index 0, matching the existing cached forward contract used by the model-level fallback.

No application source changed during this retry. Inspection confirmed that the production expert K/V projection input is config.text_config.num_key_value_heads multiplied by config.text_config.head_dim, so the implementation shape is correct and the original fixture was not.

Commands and results:

    C:\Users\joowa\miniconda3\envs\test\python.exe -m pytest -q -p no:cacheprovider tests/policies/smolvla/test_focus_token.py
    # 3 passed in 1.35s

    C:\Users\joowa\miniconda3\envs\test\python.exe -m py_compile src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py
    # passed

    C:\Users\joowa\AppData\Local\Programs\Python\Python310\Scripts\ruff.exe check src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py
    # All checks passed!

    C:\Users\joowa\AppData\Local\Programs\Python\Python310\Scripts\ruff.exe format --check src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py
    # 4 files already formatted

    git diff --check
    # passed

The Conda environment does not contain the Ruff module, so the already-installed standalone Ruff executable performed the read-only lint and format checks. This does not change the experiment dependency environment. There is no paper-method deviation from this retry, and metrics, datasets, seeds, budgets, and evaluation criteria remain unchanged.
