#!/usr/bin/env bash
set -euo pipefail

# Setup: uv sync --locked --extra smolvla --extra feetech
# Run:   ROBOT_PORT=/dev/ttyACM0 WRIST_CAMERA=0 TOP_CAMERA=2 ./examples/smolvla/run_so101_pick_place.sh
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
WRIST_CAMERA="${WRIST_CAMERA:-0}"
TOP_CAMERA="${TOP_CAMERA:-2}"
CAMERA_FPS="${CAMERA_FPS:-30}"
POLICY_FPS="${POLICY_FPS:-10}"
DURATION="${DURATION:-30}"
DEVICE="${DEVICE:-cuda}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-5}"
POLICY_PATH="${POLICY_PATH:-CoRL2026-CSI/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep-appendix}"
POLICY_REVISION="${POLICY_REVISION:-}"
ROLLOUT_STRATEGY="${ROLLOUT_STRATEGY:-base}"
INFERENCE_TYPE="${INFERENCE_TYPE:-rtc}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10}"

policy_args=(--policy.path="$POLICY_PATH")
[[ -z "$POLICY_REVISION" ]] || policy_args+=(--policy.pretrained_revision="$POLICY_REVISION")
inference_args=(--inference.type="$INFERENCE_TYPE")
if [[ "$INFERENCE_TYPE" == rtc ]]; then
  inference_args+=(
    --inference.rtc.execution_horizon="$RTC_EXECUTION_HORIZON"
    --inference.rtc.max_guidance_weight="$RTC_MAX_GUIDANCE_WEIGHT"
  )
fi

exec uv run lerobot-rollout \
  --strategy.type="$ROLLOUT_STRATEGY" \
  "${inference_args[@]}" \
  "${policy_args[@]}" \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  --robot.use_degrees=true \
  --robot.cameras="{camera1: {type: opencv, index_or_path: $WRIST_CAMERA, width: 640, height: 480, fps: $CAMERA_FPS}, camera2: {type: opencv, index_or_path: $TOP_CAMERA, width: 640, height: 480, fps: $CAMERA_FPS}}" \
  --task="Pick up the red block and place it on the blue dish." \
  --fps="$POLICY_FPS" \
  --duration="$DURATION" \
  --device="$DEVICE" \
  "$@"
