# Amendment 02 — Frozen planner runtime contract

- Date: 2026-08-13
- Timing: before training, evaluation, or result inspection
- Scope: runtime mechanics only; metrics, seeds, budget, prompts, failure rules, and comparison conditions are unchanged

The frozen-planner condition is evaluated as sequential batch-size-1 rollouts. This makes a planner parse
failure terminate exactly its own episode and preserves the preregistered exact initial-state pairing. It
does not change the 50 rollout seeds or task denominators.

At the initial action and after every five executed actions, `SmolVLAPolicy.select_action()` uses the
existing internal frozen SmolVLM and processor to generate at most 32 new tokens deterministically from:

- current main and wrist images;
- the full task instruction;
- previous skill (`none` initially);
- actions executed since the previous decision;
- the strict two-field JSON schema and six allowed skills.

The complete decoded suffix must parse as one JSON object. First-call `continue`, mismatched `continue`,
same-skill `switch`, extra/duplicate keys, invalid enums, and trailing text are parse failures. A valid
`switch` clears any remaining action queue. One later parse failure retains the prior skill for one
interval; a first-call failure or two consecutive later failures ends the rollout with success=false.
No re-prompt or keyword fallback is used. Prompt, raw suffix, parsed decision/skill, parse status, and
latency are retained in `policy.atomic_planner_history` for the rollout logger.

Manual `atomic_skill_id` injection remains available only when the online planner is disabled (offline GT
routing diagnostics). Passing it while the online planner is enabled is an error, preventing accidental
oracle routing in the primary condition.
