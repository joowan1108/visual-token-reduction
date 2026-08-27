# Implementation-map amendment 04: Experiment M

## Pinned inputs

```text
theta_0:
Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod
@45f76f173c76c4e002131f8b48e345589a071d0f

theta_0 training dataset:
Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi_10fps
@09a0376348f60be89edcbc0eb76c3e26b5f3b094

source adaptation episode: 170
source pick target (dataset robot frame): approximately (0.322549, 0.079013, 0.200000)
```

Episode `170` is the preregistered candidate for the fixed right-side placement. Before training,
its rendered layout must be visually matched to target episode `0`; a mismatch blocks this run and
requires an append-only amendment selecting a different source episode.

## Minimal code changes

1. `run.sh` pins the Experiment M base and 3,300-episode source dataset, selects exactly one source
   episode, and requires an immutable target repo/revision/episode for target-related commands.
2. `prepare_target_dataset.py` receives target coordinates as CLI arguments, validates exactly one
   nonempty 10-FPS episode and the SO-101 interface, and records the selected content hash.
3. Focused tests assert the pins, independent same-base fine-tunes, one source/target episode,
   immutable provenance, and unchanged merge anchor.

No `dart_merge.py` change is required. It already computes both updates around `BASE`, applies exact
thin-SVD DArT, adds the result to `BASE`, and copies `BASE` config/processors.

All checkpoints produced with the previous specialized base are incompatible with Experiment M and
must not be reused.

