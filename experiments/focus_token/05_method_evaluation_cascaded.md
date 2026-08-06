# Cascaded Focus paper-method evaluation

## Verdict

- **Static method fidelity:** supported.
- **Runtime and experiment readiness:** conditional; CUDA, checkpoint, optimizer, and pytest execution remain unverified in the current WSL environment.
- **Performance claim:** not evaluated. No new raw LIBERO result was produced.

## Verified implementation properties

- Expert cross-attention uses independently normalized condition and visual branches.
- Every action query receives a global visual Top-K mask over all valid camera patches, with `ceil(ratio * N_valid)` tokens and zero tokens for fully invalid queries.
- The visual branch uses an action-conditioned `Linear -> SiLU -> Linear -> sigmoid` element-wise gate.
- Condition and gated visual outputs pass through trainable fusion before the existing residual/FFN path.
- Masked attention probabilities and fully masked query outputs are explicitly zero.
- Dense and legacy Focus defaults stay disabled and continue through the historical path without new trainable parameter keys.
- Diagnostics distinguish Cascaded `visual_branch_camera_attention_share` from Dense/legacy `total_prefix_camera_attention_mass`; these quantities must not be compared as though they share one denominator.
- Composite visualization uses two camera columns and three rows: original observation, Top-K selection, and smooth action-to-visual heatmap. The two cameras in one layer use a shared color scale.

## Verification evidence

Static review found no blocking correctness issue after the follow-up fixes. The changed Python files passed `py_compile`; the focused diff passed `git diff --check`. Tests were added for configuration validation, query-specific Top-K, per-query valid-token budgets, condition/visual masks, gate and fusion gradients, fully masked probabilities/output, condition padding, diagnostic semantics, and composite layout.

Runtime pytest and Ruff could not run because the current WSL environment has no `uv`, project virtual environment, Torch, Pillow, or pytest, and Windows executable interop failed. The following remain mandatory before training:

1. load real Dense and legacy Focus checkpoints;
2. run Full-model CUDA forward/backward;
3. verify AMP mask/softmax stability;
4. perform one optimizer update and confirm gate/fusion parameter changes;
5. save and reload a Full checkpoint;
6. verify Top-K counts on a real LIBERO batch;
7. run `tests/policies/smolvla/test_focus_token.py`.

The initial Top-K implementation uses a dense per-query attention mask. It is suitable for accuracy evaluation but does not yet establish actual FLOP reduction.
