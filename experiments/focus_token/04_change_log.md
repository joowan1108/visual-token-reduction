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

## Inference diagnostics and visualization addendum (2026-08-05)

### Scope and changed files

- `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - Starts one diagnostics context per inference action-chunk call and clears it after Euler integration.
  - Passes the exact post-resize/pad, SigLIP-range `[-1, 1]` camera tensors already consumed by the model to the opt-in diagnostics saver. Inference tensors are not modified.
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
  - Saves one RGB PNG per call, batch item, and camera, and adds `call_index` plus the matching relative `image_path` to every emitted JSONL record.
  - Validates RGB NCHW shape, a shared batch size, finite values, and the `[-1, 1]` model-input range before converting with `round((x + 1) * 127.5)` to uint8.
  - Serializes the full camera-span softmax as `attention_distribution`; invalid positions are zero. `topk_attention_mass` is computed by summing that serialized distribution at `selected_indices`.
  - Refuses to overwrite an existing diagnostic PNG. Dense/default mode does not create diagnostic images.
- `experiments/focus_token/visualize_focus_tokens.py`
  - Adds a stdlib plus Pillow CLI that reads the JSONL-relative source images, validates a non-empty square patch grid (`64` tokens resolve to `8x8`), draws a transparent attention heatmap, and outlines selected patches.
  - Refuses to overwrite an existing overlay.
- `tests/policies/smolvla/test_focus_token.py`
  - Adds focused checks for the full distribution, partially and fully invalid positions, top-k mass equality, opt-in behavior, call/image alignment, and `[-1, 1]` to uint8 conversion.

No configuration field, dependency, checkpoint weight, trainable parameter, training path, metric, dataset, seed, budget, or evaluation criterion changed. `policy.focus_token_diagnostics_path` remains opt-in with default `None`; setting it to a JSONL path enables these evaluation diagnostics for sparse Focus variants.

Visualization command:

```powershell
C:\Users\joowa\miniconda3\envs\test\python.exe experiments/focus_token/visualize_focus_tokens.py <diagnostics.jsonl> --output-dir <overlay-directory>
```

### Verification commands and results

```powershell
C:\Users\joowa\miniconda3\envs\test\python.exe -m pytest -q -p no:cacheprovider tests/policies/smolvla/test_focus_token.py
# 4 passed in 5.88s

C:\Users\joowa\AppData\Local\Programs\Python\Python310\Scripts\ruff.exe check src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py experiments/focus_token/visualize_focus_tokens.py
# All checks passed!

C:\Users\joowa\AppData\Local\Programs\Python\Python310\Scripts\ruff.exe format --check src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py experiments/focus_token/visualize_focus_tokens.py
# 4 files already formatted

$env:PYTHONDONTWRITEBYTECODE='1'
C:\Users\joowa\miniconda3\envs\test\python.exe experiments/focus_token/visualize_focus_tokens.py --help
# passed

git diff --check
# passed (only the existing Windows LF-to-CRLF warning was emitted)
```

### Assumptions and paper deviations

- Camera order is the existing `images` list order, which is also the order used to build the visual-token spans; `camera` therefore aligns the saved PNG with the corresponding distribution.
- Each camera span is one spatial patch grid. The visualization fails explicitly for non-square counts rather than guessing a layout.
- The PNG is a faithful display conversion of the tensor presented to the vision encoder, including resize padding; PNG encoding does not feed back into inference.
- The overlay is post-hoc diagnostics only. It is not part of FocusVLA or the preregistered intervention and does not change selection, training, evaluation, or statistical analysis.

## Exact expert-attention heatmaps addendum (2026-08-05)

### Changed files and configuration

- src/lerobot/policies/smolvla/smolvlm_with_expert.py
  - Diagnostics now capture the post-mask, post-softmax, post-dtype-cast probabilities used by expert cross-attention.
  - Probabilities are averaged over expert heads and action query tokens, then scattered through the gathered-to-original prefix map into action_visual_attention_mass and a 64-value action_visual_attention_distribution per camera. Sparse unselected locations are zero.
  - Opt-in diagnostics now also work with focus_token_keep_ratio=1.0; normal dense and sparse paths remain unchanged when focus_token_diagnostics_path=None.
- tests/policies/smolvla/test_focus_token.py
  - Covers sparse original-index remapping, 64-value exact-attention output, zero unselected locations, camera-mass conservation, and dense diagnostics.
- experiments/focus_token/visualize_focus_tokens.py
  - Preserves the original selector-score render_overlays API and default CLI behavior; --composite renders smoothly interpolated inferno heatmaps of actual expert attention in a 4-layer by 2-camera layout.
- experiments/focus_token/run_attention_heatmaps.sh
  - Runs the Dense Hub checkpoint and latest local Focus50 seed1000 checkpoint 050000 with identical LIBERO suite/task/episode settings and refuses to reuse an output path.
  - Runs each suite in a separate evaluator process and diagnostics JSONL, so each variant produces three first-call composites per suite (12 total) for paired environment seeds 0, 1, and 2.

No dependency, trainable parameter, checkpoint key, training behavior, metric, dataset, seed, primary budget, or evaluation criterion changed. Diagnostics remain disabled by default.

One-line remote invocation:

    bash experiments/focus_token/run_attention_heatmaps.sh

The runner freezes the requested diagnostic comparison settings: Dense sungkyunner/smolvla_libero_baseline; Focus50 seed1000 checkpoint 050000; suites libero_10, libero_object, libero_goal, and libero_spatial; task id 0; n_episodes=batch_size=3; seed 0; LIBERO stored initial states; no recording. FOCUS_CHECKPOINT, FOCUS_SEARCH_ROOT, and OUT only locate inputs/outputs and do not change evaluation semantics.

### Verification

    python -S -m py_compile src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py experiments/focus_token/visualize_focus_tokens.py
    # passed

    ruff format --check src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py experiments/focus_token/visualize_focus_tokens.py
    # 3 files already formatted

    ruff check src/lerobot/policies/smolvla/smolvlm_with_expert.py tests/policies/smolvla/test_focus_token.py experiments/focus_token/visualize_focus_tokens.py
    # All checks passed!

    git diff --check
    # passed; only existing LF-to-CRLF warnings were emitted
Two CPU pytest attempts produced no output and stalled during Python/Torch startup; they were terminated without a test result. No GPU or LIBERO command was run. WSL bash -n was unavailable on this Windows host (E_ACCESSDENIED), so the Linux runner was not executed locally.

### Assumptions and paper deviations

- SmolVLA supplies 64 visual patch tokens per camera; diagnostics intentionally serialize a fixed 8x8 grid.
- Head/query aggregation is the arithmetic mean, so camera masses remain fractions of total attention and the per-camera distribution sums to its reported mass.
- Identical evaluator seed, stored LIBERO initial states, task id, episode count, and batch size reproduce the same initial observations for the two separately seeded evaluator processes.
- These heatmaps and the small four-suite diagnostic rollout are post-hoc inspection only, not a new intervention or a replacement for the preregistered evaluation. There is no paper-method change.

## Cascaded Focus Attention amendment implementation (2026-08-07)

Implemented the user-approved `03_amendment_02_cascaded_focus.md` as a separate opt-in path.

- `configuration_smolvla.py` adds `focus_cascaded_attention=False` and
  `focus_channel_gate=False`. Gate without cascaded attention is rejected. Existing Dense and legacy
  Focus configs retain their exact defaults and module layout.
- `modeling_smolvla.py` passes the two options to the expert model; preprocessing, flow matching,
  denoising, cache construction, and checkpoint defaults are unchanged.
- `smolvlm_with_expert.py` splits each opt-in expert cross-attention into independently normalized
  nonvisual-condition and visual branches. The visual branch uses head-averaged scaled QK logits and
  an exact per-action-query global Top-K mask across all cameras. An optional sigmoid element-wise
  gate modulates visual output before a trainable fusion projection combines both branches.
- The initial implementation intentionally retains dense visual K/V tensors and applies the Top-K as
  an attention mask. This matches the amendment's accuracy-first screening scope and makes no FLOP
  reduction claim.
- Diagnostics record per-patch query selection frequency, retained selector mass, actual visual
  attention, and gate statistics without changing inference tensors.
- `visualize_focus_tokens.py --composite` retains its CLI and now creates one reference-style image
  per batch/layer: two camera columns and three rows containing original observations, blockwise
  Top-K selections, and bicubic-smoothed action-to-visual heatmaps with attention-mass labels.
- `test_focus_token.py` adds config validation, exact per-query global budgets with different query
  selections, branch masks, gate range/shape, gate/fusion gradients, and the new composite layout.

Verification in the current WSL session:

```text
python3 -m py_compile <five changed Python files>
# passed

git diff --check -- <five changed Python files>
# passed
```

`uv` is not installed in this WSL environment, Linux Python lacks Torch/Pillow, and WSL-to-Windows
process interop failed with `UtilBindVsockAnyPort: socket failed 1`; therefore pytest and Ruff could
not run here. No raw result, dependency, or unrelated untracked file was modified.

## Cascaded Focus evaluator follow-up (2026-08-07)

- Replaced the single linear channel gate with the approved minimal
  `Linear -> SiLU -> Linear -> sigmoid` path. The final bias is 2.0 and its weights use a small
  initialization, so gates start near open without being constant.
- Cascaded JSONL now marks `action_visual_attention_scope="visual_branch"` and records the distinct
  `visual_branch_camera_attention_share`. Dense and legacy Focus records use `total_prefix` and
  `total_prefix_camera_attention_mass`. The legacy `action_visual_attention_mass` remains for old
  readers, but composite labels state the correct semantics.
- Eager attention now explicitly zeros masked post-softmax probabilities, including fully invalid
  queries. Tests cover exact per-query ceil budgets with different valid counts, masked probability
  zero, fully invalid zero output, condition-padding invariance, and both gate-layer gradients.
- The requested 3-row by 2-camera composite remains. Its two camera heatmaps share one scale within
  each batch/layer image; the label explicitly calls this camera-relative, so it is not evidence for
  cross-layer or cross-variant intensity differences.

## Global Focus budget and probability-map amendment (2026-08-11)

- Legacy Focus now applies one `ceil(keep_ratio * total_valid_visual_tokens)` Top-K budget across all
  cameras. It no longer reserves 50% independently for every camera; nonvisual tokens remain dense.
- Attention maps now come from the actual post-mask multi-head softmax probabilities. Head and action
  query dimensions are averaged, selected layers are averaged, and pruned patches are restored as zero
  at their original image-grid positions.
- Raw maps, model-input images, and task descriptions are saved as `call_*_flow_*_{cross,self}.npz`.
  The `visualize_smolvla_attention` command renders task-labeled, shared-scale jet overlays.
- This is a post-result intervention amendment. Existing per-camera Focus50 results remain results of
  the old method and must not be relabeled as global-budget results.
