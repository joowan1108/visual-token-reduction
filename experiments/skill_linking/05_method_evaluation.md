# P1 method evaluation

## Verdict

**PASS WITH REQUIRED REMOTE TEST**

The read-only method review confirmed that the implementation now matches the preregistered P1 method:

- B0 and P1 can use the same stage-stratified sample order while only P1 receives stage conditioning and transition loss.
- Every semantic boundary contributes one `b-H` pair index; no copied pair dataset is created.
- START, CONTINUE, semantic switch, DONE, and reserved class `0` have aligned target/state behavior.
- Transition logits come from the final Euler denoise call and are applied only after `H` actions are consumed.
- Full-train class weights are frozen before policy construction; subset diagnostics must supply those frozen values.
- Old-checkpoint bootstrap is restricted to the new embedding/head keys.

The evaluator's initial blockers were corrected and the missing sampler regression test was added. Local syntax, diff, and dependency-light sampler smoke checks pass. The remaining requirement is to execute the focused pytest suite in the GPU container with its complete locked environment:

```bash
uv run --locked --extra test --extra libero pytest -q -p no:cacheprovider tests/policies/smolvla/test_skill_transition.py
```

Do not begin Gate B or training unless that command passes.
