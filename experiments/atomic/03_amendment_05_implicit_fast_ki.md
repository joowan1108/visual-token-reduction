# Amendment 05 — opt-in implicit FAST-KI action context

- Date: 2026-08-19
- Timing: before training, evaluation, or raw-result inspection for this intervention
- Scope: an additional opt-in SmolVLA/SG-MoE condition; the existing dense/SG-MoE conditions, metrics,
  datasets, seeds, rollout manifests, statistical tests, and falsification criteria are unchanged

## Frozen configuration

The intervention is disabled by default with `implicit_fast_ki_enabled=False`. Enabling it requires the
existing atomic data and SG-MoE path, a fully frozen SmolVLM/vision encoder, KV caching, and the non-compiled
path. It also requires either pretrained SmolVLM initialization (`load_vlm_weights=True`) or a policy loaded
through the repository's `pretrained_path` checkpoint path; a fully scratch random frozen teacher is invalid.
The existing `state_proj` remains trainable and `train_state_proj=True` is required. The frozen defaults are
selected VLM layers `[-4, -3, -2, -1]`, four learnable queries per layer,
FAST loss weight `0.1`, at most 256 action tokens, 128 skipped vocabulary slots, and
`lerobot/fast-action-tokenizer`. The dedicated transition CE weight is `0.1`, and the implicit transition
switch-sample weight is `4.0`.

Each selected layer independently owns `Q_i`, `Wq_i`, `Wk_i`, and `Wv_i`, initialized from a small normal
distribution (`std=0.02`). Its detached raw SmolVLM K/V is projected by query attention; projected layer
contexts are aggregated by an unweighted arithmetic mean. No learned layer gate or separate decoder is added.
Selected layer indices must remain unique after negative indices are normalized against the actual VLM layer
count.

## Data and objectives

FAST and flow use the same normalized 10-step action chunk and the same episode/atomic-boundary padding mask.
The valid contiguous action prefix is passed to the existing FAST tokenizer before slot-reversed SmolVLM token
mapping, so masked future actions cannot enter FAST targets. No vocabulary expansion is performed.

Training uses

`loss = masked_flow_loss + implicit_fast_loss_weight * masked_FAST_next_token_loss +
implicit_transition_loss_weight * atomic_transition_CE`.

The transition CE remains unreduced until each sample is classified as stay or switch. A sample receives the
`implicit_transition_switch_weight` only when its latest history slot is valid and its target differs from that
latest executed skill; stay and episode-start/no-history samples retain weight 1. Metrics remain unweighted.
Natural-distribution labels contain approximately 4% switches, so the frozen 4x multiplier gives switches an
effective loss prevalence of about 14% (`4 * .04 / (.96 + 4 * .04)`). This is intentionally less aggressive
than the prior 75:25 replacement sampler, which harmed transition precision, and does not alter sampling.

FAST teacher forcing reuses the frozen SmolVLM autoregressive transformer and LM head. It trains IAR and the
single expert-to-VLM context projection plus the existing state projection, but not SmolVLM, the vision encoder,
or SG-MoE. IAR's VLM prefix/cache contains image and language only. The separately projected raw-state token is
concatenated with projected IAR tokens for FAST teacher forcing. Flow receives that same fused context only after
detachment; it trains the existing atomic SG-MoE and action path, but not SmolVLM, IAR, layer aggregation, the
FAST context projection, or state projection. Continuous/noisy actions are never inputs to IAR or FAST context
construction.

The dedicated transition head predicts exactly one of the six existing `ATOMIC_SKILLS`; it has no BOS or DONE
class. Its detached inputs are the projected IAR context, the existing projected-state token, and two explicitly
masked history slots. Valid history IDs index the fixed SG-MoE router skill-embedding buffer directly; no history
embedding table is learned. With no valid history, both skill vectors are zero and image/language context plus
state alone determine the prediction. Transition CE updates only the transition head.

Training history anchors are `[-2 * n_action_steps, -n_action_steps]` relative to the current planning anchor,
followed by the unchanged `0..chunk_size-1` action window. Dataset episode padding supplies explicit history-valid
masks, preventing labels from crossing episode boundaries. The current skill at offset zero remains the six-way
transition target and the flow routing target.

Inference performs one image-language VLM prefix prefill, then fuses `detached KV -> IAR -> projected context`
with `state_proj(current state)` before detaching the combined context for SG-MoE at every denoising step. FAST
token generation and decoding are skipped.
Inference stores two executed skill IDs per synchronous batch environment. Each queue refill predicts and routes
the action chunk with one six-way skill, then appends that actually selected skill. Policy reset clears every
environment history; missing slots remain invalid rather than receiving a learned start vector.

## Interpretation boundary

This is a KI-style auxiliary adaptation, not an exact reproduction of a separate KI decoder. It intentionally
reuses the existing frozen SmolVLM transformer/head and existing FAST slot mapping. Because this is opt-in and
adds no preregistered run budget, any execution must be named and reported separately from the original A/B
comparison rather than pooled into its primary estimate.
