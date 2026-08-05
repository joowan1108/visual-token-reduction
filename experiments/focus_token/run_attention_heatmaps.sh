#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
command -v lerobot-eval >/dev/null || { echo 'activate the LIBERO environment first' >&2; exit 2; }

if [[ -z "${FOCUS_CHECKPOINT:-}" ]]; then
  MODEL_FILE="$(find "${FOCUS_SEARCH_ROOT:-.}" -type f -path '*focus50*seed_1000*/checkpoints/050000/pretrained_model/model.safetensors' -print0 2>/dev/null | xargs -0 -r ls -1dt | head -n 1)"
  FOCUS_CHECKPOINT="${MODEL_FILE%/model.safetensors}"
fi
[[ -n "$FOCUS_CHECKPOINT" && -f "$FOCUS_CHECKPOINT/model.safetensors" ]] || {
  echo 'Set FOCUS_CHECKPOINT to the latest Focus50 seed1000 checkpoint 050000 pretrained_model directory.' >&2
  exit 2
}

OUT="${OUT:-experiments/focus_token/results/attention_heatmaps_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

for VARIANT in dense focus50; do
  if [[ "$VARIANT" == dense ]]; then
    POLICY='sungkyunner/smolvla_libero_baseline'
    RATIO=1.0
  else
    POLICY="$FOCUS_CHECKPOINT"
    RATIO=0.5
  fi
  mkdir "$OUT/$VARIANT"
  for SUITE in libero_10 libero_object libero_goal libero_spatial; do
    RUN="$OUT/$VARIANT/$SUITE"
    mkdir "$RUN"
    ARGS=(
      --policy.path="$POLICY"
      --policy.device=cuda
      --policy.use_amp=false
      --policy.compile_model=false
      --policy.focus_token_keep_ratio="$RATIO"
      --policy.focus_token_diagnostics_path="$RUN/attention.jsonl"
      --env.type=libero
      --env.task="$SUITE"
      --env.task_ids='[0]'
      --env.control_mode=relative
      --env.init_states=true
      --env.hard_reset=true
      --env.max_parallel_tasks=1
      --eval.n_episodes=3
      --eval.batch_size=3
      --eval.recording=false
      --seed=0
      --output_dir="$RUN/eval"
      --job_name="attention_heatmap_${VARIANT}_${SUITE}"
    )
    lerobot-eval "${ARGS[@]}" 2>&1 | tee "$RUN/eval.log"
    python experiments/focus_token/visualize_focus_tokens.py "$RUN/attention.jsonl" --output-dir "$RUN/composites" --composite
  done
done

echo "Dense and Focus50 composites: $OUT"
