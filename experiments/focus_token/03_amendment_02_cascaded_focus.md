# Amendment 02: Cascaded Focus Attention

## Approval and scope

The user approved this amendment before implementation and before observing results from the new method. It preserves the historical intervention and primary metric in `03_experiment_plan.md`; the new method is reported separately.

The intervention is a SmolVLA-compatible **Cascaded + per-query Top-K + element-wise gate** expert cross-attention path. Dense and legacy Focus behavior remain unchanged by default.

## Frozen architecture

At expert cross-attention layers, action hidden states independently attend to non-visual prefix conditions and visual patches with separate softmax normalization. Existing action self-attention supplies the action branch. Condition and visual outputs are fused before the existing residual/FFN path.

For the visual branch:

- relevance is scaled action-query/visual-key dot product;
- heads are aggregated, while each action query gets its own Top-K mask;
- valid patches from all cameras share one global budget;
- `K = max(1, ceil(focus_token_keep_ratio * N_valid_visual))` per query;
- selected patches retain positional identity;
- `sigmoid(MLP(action_hidden))` gates the visual output element-wise before fusion;
- no sparse CUDA kernel, extra vision backbone, shallow-value path, or new dependency is included.

| Variant | Cascaded | Keep ratio | Gate |
|---|---:|---:|---:|
| Dense | off | 1.0 | off |
| Cascaded | on | 1.0 | off |
| Cascaded+TopK | on | 0.5 | off |
| Full | on | 0.5 | on |

The new primary comparison is **Full versus Dense**. Component ablations cannot replace Full after observing results.

## Frozen training and evaluation

- Matched initialization, dataset, preprocessing, optimizer, scheduler, action chunking, denoising, effective batch size, and paired evaluation states.
- Final training: 100,000 steps; seeds `1000`, `1001`, `1002`; checkpoints every 10,000 steps.
- Primary evaluation: all 40 LIBERO tasks, 20 paired episodes per task and seed.
- A one-seed 10K screening run may detect crashes or collapse only; it cannot change the Full architecture or primary conclusion.
- Primary metric remains 40-task macro episode success.
- Support requires Full−Dense of at least `+1.0pp`, a paired hierarchical confidence interval excluding zero, improvement in at least two seeds, and fewer effective visual tokens.
- A LIBERO-Goal regression of at least `3.0pp` is reported as instability even if the macro criterion passes.
- Secondary diagnostics: loss, selector entropy and retained mass, camera allocation, layerwise visual attention, gate statistics, peak memory, and latency. The initial dense per-query mask makes no FLOP-reduction claim.

## Implementation gates

1. New options disabled preserve Dense output and existing checkpoint loading.
2. Condition and visual branches normalize independently and respect their masks.
3. Different action queries can select different patches with the exact global Top-K count; masked patches receive zero attention.
4. Gate shape matches action hidden states, lies in `[0, 1]`, and receives gradients.
5. Fusion, selection Q/K projections, and gate receive gradients.
6. Diagnostics render paired originals, Top-K selection, and actual action-to-visual heatmaps without changing inference tensors.

Failure of the primary rule, gate collapse, Goal regression, or concentration without accuracy improvement must be reported directly. Earlier raw results remain append-only and are never overwritten.
