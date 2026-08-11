#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
command -v lerobot-eval >/dev/null || { echo 'activate the LIBERO environment first' >&2; exit 2; }
command -v python >/dev/null || { echo 'python is required' >&2; exit 2; }

FOCUS_CHECKPOINT="${FOCUS_CHECKPOINT:-experiments/focus_token/results/20260804T133556Z_focus50_seed_1000_100k/focus50/seed_1000/train/checkpoints/100000/pretrained_model}"
CASCADED_CHECKPOINT="${CASCADED_CHECKPOINT:-experiments/focus_token/results/20260807T034113Z_cascaded_focus50_seed_1000_100k/train/checkpoints/100000/pretrained_model}"
for CHECKPOINT in "$FOCUS_CHECKPOINT" "$CASCADED_CHECKPOINT"; do
  [[ -f "$CHECKPOINT/model.safetensors" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
done

OUT="${OUT:-experiments/focus_token/results/focus_comparison_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

SUITES=(libero_10 libero_object libero_goal libero_spatial)
VARIANTS=(vanilla focus50 cascaded50)
FULL_EVAL_EPISODES="${FULL_EVAL_EPISODES:-10}"
HEATMAP_EPISODES="${HEATMAP_EPISODES:-3}"

set_variant() {
  case "$1" in
    vanilla) POLICY='sungkyunner/smolvla_libero_baseline'; RATIO=1.0; CASCADED=false; GATE=false ;;
    focus50) POLICY="$FOCUS_CHECKPOINT"; RATIO=0.5; CASCADED=false; GATE=false ;;
    cascaded50) POLICY="$CASCADED_CHECKPOINT"; RATIO=0.5; CASCADED=true; GATE=true ;;
  esac
}

for VARIANT in "${VARIANTS[@]}"; do
  set_variant "$VARIANT"
  mkdir -p "$OUT/full_eval/$VARIANT"
  for SUITE in "${SUITES[@]}"; do
    RUN="$OUT/full_eval/$VARIANT/$SUITE"
    mkdir "$RUN"
    lerobot-eval \
      --policy.path="$POLICY" \
      --policy.device=cuda \
      --policy.use_amp=false \
      --policy.compile_model=false \
      --policy.focus_token_keep_ratio="$RATIO" \
      --policy.focus_cascaded_attention="$CASCADED" \
      --policy.focus_channel_gate="$GATE" \
      --env.type=libero \
      --env.task="$SUITE" \
      --env.control_mode=relative \
      --env.init_states=true \
      --env.hard_reset=true \
      --env.max_parallel_tasks=1 \
      --eval.n_episodes="$FULL_EVAL_EPISODES" \
      --eval.batch_size="$FULL_EVAL_EPISODES" \
      --eval.recording=false \
      --seed=0 \
      --output_dir="$RUN/eval" \
      --job_name="full_eval_${VARIANT}_${SUITE}" \
      2>&1 | tee "$RUN/eval.log"
  done
done

for VARIANT in "${VARIANTS[@]}"; do
  set_variant "$VARIANT"
  for SUITE in libero_10 libero_object libero_goal libero_spatial; do
    for TASK_ID in {0..9}; do
      RUN="$OUT/heatmaps/$VARIANT/$SUITE/task_$TASK_ID"
      mkdir -p "$RUN"
      TASK_INSTRUCTION="$(python -c 'import sys; from libero.libero import benchmark; suite = benchmark.get_benchmark_dict()[sys.argv[1]](); print(suite.get_task(int(sys.argv[2])).language)' "$SUITE" "$TASK_ID" 2>/dev/null | tail -n 1)"
      [[ -n "$TASK_INSTRUCTION" ]] || { echo "Missing instruction for $SUITE task $TASK_ID" >&2; exit 2; }
      lerobot-eval \
        --policy.path="$POLICY" \
        --policy.device=cuda \
        --policy.use_amp=false \
        --policy.compile_model=false \
        --policy.focus_token_keep_ratio="$RATIO" \
        --policy.focus_cascaded_attention="$CASCADED" \
        --policy.focus_channel_gate="$GATE" \
        --policy.focus_token_diagnostics_path="$RUN/attention.jsonl" \
        --policy.attention_map=true \
        --policy.attention_map_output_dir="$RUN/attention_maps_rank0" \
        --policy.attention_map_layers='[-4,-3,-2,-1]' \
        --policy.attention_map_flow_steps='[-1]' \
        --env.type=libero \
        --env.task="$SUITE" \
        --env.task_ids="[$TASK_ID]" \
        --env.control_mode=relative \
        --env.init_states=true \
        --env.hard_reset=true \
        --env.max_parallel_tasks=1 \
        --eval.n_episodes="$HEATMAP_EPISODES" \
        --eval.batch_size="$HEATMAP_EPISODES" \
        --eval.recording=false \
        --seed=0 \
        --output_dir="$RUN/eval" \
        --job_name="attention_heatmap_${VARIANT}_${SUITE}_${TASK_ID}" \
        2>&1 | tee "$RUN/eval.log"
      python experiments/focus_token/visualize_focus_tokens.py \
        "$RUN/attention.jsonl" \
        --output-dir "$RUN/composites" \
        --variant "$VARIANT" \
        --suite "$SUITE" \
        --task-id "$TASK_ID" \
        --task-instruction "$TASK_INSTRUCTION" \
        --composite
      visualize_smolvla_attention \
        "$RUN/attention_maps_rank0" \
        --output-dir "$RUN/attention_overlays"
    done
  done
done

echo "Full evaluation and attention heatmaps: $OUT"
