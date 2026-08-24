#!/usr/bin/env bash
set -euo pipefail

BASE="CoRL2026-CSI/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep-appendix"
BASE_REV="75d5905c5e27ba6f0a738cbcfcb167e7769dce0d"
SOURCE_DATASET="CoRL2026-CSI/IsaacLab-SO101-PickAndPlace-100epi-10fps-appendix"
SOURCE_REV="2b739e6be9b341e6359265ed99be81458ed4d879"
TARGET_REPO_ID="skkuprism/test_pick_red_place_blue_50epi_10fps"
TARGET_REV="e19331e77f477a4be16f7c2884250ed6f491e048"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$HERE/artifacts}"
SOURCE_OUTPUT="$RUN_ROOT/source_finetune"
TARGET_OUTPUT="$RUN_ROOT/target_finetune"
PREPARED_TARGET_REPO="local/domain-arithmetic-so101-target-canonical-ep0"
PREPARED_TARGET_ROOT="$RUN_ROOT/target_canonical_ep0"
RENAME_MAP='{"observation.images.left_wrist":"observation.images.camera1","observation.images.top":"observation.images.camera2"}'

condition_path() {
  case "$1" in
    Z) echo "$BASE" ;;
    F) echo "$TARGET_OUTPUT/checkpoints/last/pretrained_model" ;;
    A) echo "$RUN_ROOT/direct" ;;
    D) echo "$RUN_ROOT/dart" ;;
    *) echo "condition must be one of Z, F, A, or D" >&2; return 2 ;;
  esac
}

selected_policy() {
  if [[ -n "${POLICY_PATH:-}" ]]; then
    echo "$POLICY_PATH"
  else
    condition_path "${CONDITION:-Z}"
  fi
}

selected_revision() {
  if [[ -n "${POLICY_PATH:-}" ]]; then
    if [[ -n "${POLICY_REVISION:-}" ]]; then
      echo "$POLICY_REVISION"
    elif [[ "$POLICY_PATH" == "$BASE" ]]; then
      echo "$BASE_REV"
    fi
  elif [[ "${CONDITION:-Z}" == Z ]]; then
    echo "${POLICY_REVISION:-$BASE_REV}"
  fi
}

train_common=(
  --policy.path="$BASE"
  --policy.pretrained_revision="$BASE_REV"
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.use_amp=false
  --policy.push_to_hub=false
  --preserve_pretrained_processor_stats=true
  --rename_map="$RENAME_MAP"
  --use_policy_training_preset=false
  --optimizer.type=adamw
  --optimizer.lr=5e-5
  --optimizer.betas='[0.9, 0.95]'
  --optimizer.eps=1e-8
  --optimizer.weight_decay=1e-10
  --optimizer.grad_clip_norm=1.0
  --scheduler.type=constant_with_warmup
  --scheduler.num_warmup_steps=0
  --steps=1000
  --batch_size=8
  --accelerator.gradient_accumulation.steps=8
  --seed=1000
  --cudnn_deterministic=true
  --dataset.image_transforms.enable=false
  --save_freq=0
  --env_eval_freq=0
  --wandb.enable=false
)

case "${1:-}" in
  record)
    RECORD_REPO_ID="${RECORD_REPO_ID:-local/domain-arithmetic-so101-target-recording}"
    RECORD_DATASET_ROOT="${RECORD_DATASET_ROOT:-$RUN_ROOT/recorded_target_dataset}"
    if [[ "$RECORD_REPO_ID" == "$TARGET_REPO_ID" ]]; then
      echo "record refuses to write to the pinned public target repository" >&2
      exit 2
    fi
    exec uv run lerobot-record \
      --robot.type=so101_follower --robot.port="${ROBOT_PORT:-/dev/ttyACM0}" --robot.id="${ROBOT_ID:-so101_follower}" \
      --robot.use_degrees=true \
      --robot.cameras="{left_wrist: {type: opencv, index_or_path: ${WRIST_CAMERA:-0}, width: 640, height: 480, fps: 10}, top: {type: opencv, index_or_path: ${TOP_CAMERA:-2}, width: 640, height: 480, fps: 10}}" \
      --teleop.type=so101_leader --teleop.port="${LEADER_PORT:-/dev/ttyACM1}" --teleop.id="${LEADER_ID:-so101_leader}" \
      --dataset.repo_id="$RECORD_REPO_ID" --dataset.root="$RECORD_DATASET_ROOT" \
      --dataset.no_stamp=true --dataset.push_to_hub=false --dataset.num_episodes=1 \
      --dataset.single_task="Pick up the red block and place it on the blue dish." \
      --dataset.fps=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10 --display_data=true
    ;;
  train-source)
    exec uv run lerobot-train "${train_common[@]}" \
      --dataset.repo_id="$SOURCE_DATASET" --dataset.revision="$SOURCE_REV" --dataset.episodes='[0]' \
      --output_dir="$SOURCE_OUTPUT" --job_name=dart_so101_source
    ;;
  prepare-target)
    exec uv run "$HERE/prepare_target_dataset.py" --output="$PREPARED_TARGET_ROOT"
    ;;
  train-target)
    [[ -d "$PREPARED_TARGET_ROOT" ]] || { echo "run prepare-target first" >&2; exit 2; }
    exec uv run lerobot-train "${train_common[@]}" \
      --dataset.repo_id="$PREPARED_TARGET_REPO" --dataset.root="$PREPARED_TARGET_ROOT" --dataset.episodes='[0]' \
      --output_dir="$TARGET_OUTPUT" --job_name=dart_so101_target
    ;;
  merge)
    uv run "$HERE/dart_merge.py" --base="$BASE" --base-revision="$BASE_REV" \
      --source="$SOURCE_OUTPUT/checkpoints/last/pretrained_model" \
      --target="$TARGET_OUTPUT/checkpoints/last/pretrained_model" \
      --output="$RUN_ROOT/direct" --method=direct --alpha=0.8 --rank=256 --seed=42
    exec uv run "$HERE/dart_merge.py" --base="$BASE" --base-revision="$BASE_REV" \
      --source="$SOURCE_OUTPUT/checkpoints/last/pretrained_model" \
      --target="$TARGET_OUTPUT/checkpoints/last/pretrained_model" \
      --output="$RUN_ROOT/dart" --method=dart --alpha=0.8 --rank=256 --seed=42
    ;;
  adapt)
    for output in "$PREPARED_TARGET_ROOT" "$SOURCE_OUTPUT" "$TARGET_OUTPUT" "$RUN_ROOT/direct" "$RUN_ROOT/dart"; do
      [[ ! -e "$output" ]] || { echo "refusing to reuse $output; set a fresh RUN_ROOT" >&2; exit 2; }
    done
    "$HERE/run.sh" prepare-target
    "$HERE/run.sh" train-source
    "$HERE/run.sh" train-target
    exec "$HERE/run.sh" merge
    ;;
  condition-path)
    condition_path "${2:-}"
    ;;
  rollout)
    policy_path="$(selected_policy)"
    policy_revision="$(selected_revision)"
    POLICY_PATH="$policy_path" POLICY_REVISION="$policy_revision" \
      exec "$HERE/../../examples/smolvla/run_so101_pick_place.sh"
    ;;
  evaluate)
    : "${TRIAL_ID:?Set TRIAL_ID to the frozen manifest ID and anonymized condition code.}"
    EVAL_REPO_ID="${EVAL_REPO_ID:-local/rollout_dart_so101_${TRIAL_ID}}"
    EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT:-$RUN_ROOT/evaluation/$TRIAL_ID}"
    policy_path="$(selected_policy)"
    policy_revision="$(selected_revision)"
    ROLLOUT_STRATEGY=episodic POLICY_PATH="$policy_path" POLICY_REVISION="$policy_revision" \
      exec "$HERE/../../examples/smolvla/run_so101_pick_place.sh" \
      --dataset.repo_id="$EVAL_REPO_ID" --dataset.root="$EVAL_DATASET_ROOT" \
      --dataset.no_stamp=true --dataset.push_to_hub=false --dataset.num_episodes=1 \
      --dataset.single_task="Pick up the red block and place it on the blue dish." \
      --dataset.fps=10 --dataset.episode_time_s=30 --dataset.reset_time_s=0
    ;;
  check)
    [[ "$TARGET_REPO_ID" == "skkuprism/test_pick_red_place_blue_50epi_10fps" ]]
    [[ "$TARGET_REV" == "e19331e77f477a4be16f7c2884250ed6f491e048" ]]
    [[ "$PREPARED_TARGET_REPO" == "local/domain-arithmetic-so101-target-canonical-ep0" ]]
    [[ "$PREPARED_TARGET_ROOT" == "$RUN_ROOT/target_canonical_ep0" ]]
    [[ "$(condition_path Z)" == "$BASE" ]]
    [[ "$(condition_path F)" == "$TARGET_OUTPUT/checkpoints/last/pretrained_model" ]]
    [[ "$(condition_path A)" == "$RUN_ROOT/direct" ]]
    [[ "$(condition_path D)" == "$RUN_ROOT/dart" ]]
    [[ "$(CONDITION=Z POLICY_PATH= POLICY_REVISION= selected_revision)" == "$BASE_REV" ]]
    [[ -z "$(CONDITION=F POLICY_PATH= POLICY_REVISION= selected_revision)" ]]
    [[ "$(POLICY_PATH="$BASE" POLICY_REVISION= selected_revision)" == "$BASE_REV" ]]
    [[ -z "$(POLICY_PATH="$TARGET_OUTPUT/checkpoints/last/pretrained_model" POLICY_REVISION= selected_revision)" ]]
    [[ "$(POLICY_PATH=custom POLICY_REVISION=custom-rev selected_revision)" == custom-rev ]]
    if condition_path invalid >/dev/null 2>&1; then
      echo "invalid condition unexpectedly succeeded" >&2
      exit 1
    fi
    echo "workflow condition paths OK"
    ;;
  *)
    echo "usage: $0 {adapt|prepare-target|train-source|train-target|merge|rollout|evaluate|record|condition-path|check}" >&2
    exit 2
    ;;
esac
