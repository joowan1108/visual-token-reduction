# Skill-linking data audit

- Dataset: `sungkyunner/libero_10_subtask_semantic_clean@d619815aeba9c06c70fc558838137dd57a651ce1`
- Horizon: `H=10`
- Full data: 495 episodes, 136,425 frames, 941 contiguous atomic segments
- Atomic candidate starts: 127,015
- Adjacent-subtask boundaries: 446 instances and 9 directed pair types
- Pair rule: exactly one start `b-H` for every boundary `b`
- Minimum pre-boundary segment: 74 frames, so all 446 pair starts are valid
- Split: 445 train episodes / 50 held-out episodes
- Train pools: 114,352 atomic starts / 401 pair starts
- Held-out pools: 12,663 atomic starts / 45 pair starts

No clips are concatenated or copied. Pair samples are index views into one original episode. START and DONE each add one event index per selected episode; these are tracked separately from the 446 semantic pairs.

| Pair | Full count | Train | Held out |
|---|---:|---:|---:|
| `5 -> 4` | 47 | 42 | 5 |
| `6 -> 3` | 49 | 44 | 5 |
| `6 -> 7` | 50 | 45 | 5 |
| `8 -> 2` | 50 | 45 | 5 |
| `9 -> 12` | 50 | 45 | 5 |
| `9 -> 13` | 50 | 45 | 5 |
| `12 -> 11` | 50 | 45 | 5 |
| `14 -> 1` | 50 | 45 | 5 |
| `15 -> 4` | 50 | 45 | 5 |
