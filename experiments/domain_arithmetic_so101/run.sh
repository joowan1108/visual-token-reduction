#!/usr/bin/env bash
set -euo pipefail

BASE="Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod"
BASE_REV="45f76f173c76c4e002131f8b48e345589a071d0f"
SOURCE_DATASET="Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi_10fps"
SOURCE_REV="09a0376348f60be89edcbc0eb76c3e26b5f3b094"
SOURCE_EPISODE=170
DEFAULT_TARGET_REPO_ID="sungkyunner/record-test_20260826_210214"
DEFAULT_TARGET_REV="295e6def6cb4df454f58894caea10c15446dc4e4"
DEFAULT_TARGET_EPISODE=0
TARGET_REPO_ID="${TARGET_REPO_ID:-$DEFAULT_TARGET_REPO_ID}"
TARGET_REV="${TARGET_REV:-$DEFAULT_TARGET_REV}"
TARGET_EPISODE="${TARGET_EPISODE:-$DEFAULT_TARGET_EPISODE}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$HERE/artifacts}"
SOURCE_OUTPUT="$RUN_ROOT/source_finetune"
TARGET_OUTPUT="$RUN_ROOT/target_finetune"
TARGET_PROVENANCE="$RUN_ROOT/target_provenance.json"
SOURCE_CHECKPOINT="$SOURCE_OUTPUT/checkpoints/last/pretrained_model"
TARGET_CHECKPOINT="$TARGET_OUTPUT/checkpoints/last/pretrained_model"
RENAME_MAP='{"observation.images.left_wrist":"observation.images.camera1","observation.images.top":"observation.images.camera2"}'

require_absent() {
  [[ ! -e "$1" ]] || { echo "refusing to overwrite $1; set a fresh RUN_ROOT" >&2; exit 2; }
}

require_checkpoint() {
  [[ -f "$1/model.safetensors" ]] || { echo "missing checkpoint model.safetensors at $1" >&2; exit 2; }
}

require_target_coordinates() {
  [[ "$TARGET_REV" =~ ^[0-9a-f]{40}$ ]] || {
    echo "TARGET_REV must be an immutable 40-character lowercase commit SHA" >&2
    exit 2
  }
  [[ "$TARGET_EPISODE" =~ ^[0-9]+$ ]] || { echo "TARGET_EPISODE must be a nonnegative integer" >&2; exit 2; }
}

require_visual_match() {
  [[ "${VISUAL_MATCH_CONFIRMED:-}" == 1 ]] || {
    echo "visually match source episode 170 and the target layout, then set VISUAL_MATCH_CONFIRMED=1" >&2
    exit 2
  }
}

condition_path() {
  case "$1" in
    Z) echo "$BASE" ;;
    F) echo "$TARGET_CHECKPOINT" ;;
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
  --policy.empty_cameras=1
  --policy.freeze_vision_encoder=true
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
  --steps=2000
  --batch_size=4
  --num_workers=0
  --accelerator.gradient_accumulation.steps=16
  --seed=1000
  --cudnn_deterministic=true
  --dataset.image_transforms.enable=false
  --dataset.video_backend=pyav
  --save_freq=500
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
      --robot.cameras="{left_wrist: {type: opencv, index_or_path: ${WRIST_CAMERA:-0}, width: 640, height: 480, fps: ${CAMERA_FPS:-30}}, top: {type: opencv, index_or_path: ${TOP_CAMERA:-2}, width: 640, height: 480, fps: ${CAMERA_FPS:-30}}}" \
      --teleop.type=so101_leader --teleop.port="${LEADER_PORT:-/dev/ttyACM1}" --teleop.id="${LEADER_ID:-so101_leader}" \
      --dataset.repo_id="$RECORD_REPO_ID" --dataset.root="$RECORD_DATASET_ROOT" \
      --dataset.no_stamp=true --dataset.push_to_hub=false --dataset.num_episodes=1 \
      --dataset.single_task="Pick up the red block and place it on the blue dish." \
      --dataset.fps=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10 --display_data=true
    ;;
  train-source)
    require_visual_match
    require_absent "$SOURCE_OUTPUT"
    exec uv run lerobot-train "${train_common[@]}" \
      --dataset.repo_id="$SOURCE_DATASET" --dataset.revision="$SOURCE_REV" --dataset.episodes="[$SOURCE_EPISODE]" \
      --output_dir="$SOURCE_OUTPUT" --job_name=dart_so101_source
    ;;
  prepare-target)
    require_target_coordinates
    require_visual_match
    require_absent "$TARGET_PROVENANCE"
    exec uv run "$HERE/prepare_target_dataset.py" \
      --repo-id="$TARGET_REPO_ID" --revision="$TARGET_REV" --episode="$TARGET_EPISODE" \
      --visual-match-confirmed --output="$TARGET_PROVENANCE"
    ;;
  train-target)
    require_target_coordinates
    require_visual_match
    [[ -f "$TARGET_PROVENANCE" ]] || { echo "run prepare-target first" >&2; exit 2; }
    uv run "$HERE/prepare_target_dataset.py" \
      --repo-id="$TARGET_REPO_ID" --revision="$TARGET_REV" --episode="$TARGET_EPISODE" \
      --verify-provenance="$TARGET_PROVENANCE"
    require_absent "$TARGET_OUTPUT"
    exec uv run lerobot-train "${train_common[@]}" \
      --dataset.repo_id="$TARGET_REPO_ID" --dataset.revision="$TARGET_REV" \
      --dataset.episodes="[$TARGET_EPISODE]" \
      --output_dir="$TARGET_OUTPUT" --job_name=dart_so101_target
    ;;
  merge)
    require_checkpoint "$SOURCE_CHECKPOINT"
    require_checkpoint "$TARGET_CHECKPOINT"
    require_absent "$RUN_ROOT/direct"
    require_absent "$RUN_ROOT/dart"
    uv run "$HERE/dart_merge.py" --base="$BASE" --base-revision="$BASE_REV" \
      --source="$SOURCE_CHECKPOINT" \
      --target="$TARGET_CHECKPOINT" \
      --output="$RUN_ROOT/direct" --method=direct --alpha=0.8
    exec uv run "$HERE/dart_merge.py" --base="$BASE" --base-revision="$BASE_REV" \
      --source="$SOURCE_CHECKPOINT" \
      --target="$TARGET_CHECKPOINT" \
      --output="$RUN_ROOT/dart" --method=dart --alpha=0.8
    ;;
  adapt)
    for output in "$SOURCE_OUTPUT" "$TARGET_PROVENANCE" "$TARGET_OUTPUT" "$RUN_ROOT/direct" "$RUN_ROOT/dart"; do
      require_absent "$output"
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
    [[ "$BASE" == "Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod" ]]
    [[ "$BASE_REV" == "45f76f173c76c4e002131f8b48e345589a071d0f" ]]
    [[ "$SOURCE_DATASET" == "Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi_10fps" ]]
    [[ "$SOURCE_REV" == "09a0376348f60be89edcbc0eb76c3e26b5f3b094" ]]
    [[ "$SOURCE_EPISODE" == 170 ]]
    [[ "$DEFAULT_TARGET_REPO_ID" == "sungkyunner/record-test_20260826_210214" ]]
    [[ "$DEFAULT_TARGET_REV" == "295e6def6cb4df454f58894caea10c15446dc4e4" ]]
    [[ "$DEFAULT_TARGET_EPISODE" == 0 ]]
    require_target_coordinates
    [[ "$TARGET_PROVENANCE" == "$RUN_ROOT/target_provenance.json" ]]
    [[ " ${train_common[*]} " == *" --policy.empty_cameras=1 "* ]]
    [[ " ${train_common[*]} " == *" --policy.freeze_vision_encoder=true "* ]]
    [[ " ${train_common[*]} " == *" --policy.train_expert_only=false "* ]]
    [[ " ${train_common[*]} " == *" --steps=2000 "* ]]
    [[ " ${train_common[*]} " == *" --save_freq=500 "* ]]
    [[ " ${train_common[*]} " == *" --scheduler.type=constant_with_warmup "* ]]
    [[ " ${train_common[*]} " == *" --scheduler.num_warmup_steps=0 "* ]]
    [[ " ${train_common[*]} " == *" --batch_size=4 "* ]]
    [[ " ${train_common[*]} " == *" --num_workers=0 "* ]]
    [[ " ${train_common[*]} " == *" --accelerator.gradient_accumulation.steps=16 "* ]]
    [[ " ${train_common[*]} " == *" --dataset.video_backend=pyav "* ]]
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
