#!/usr/bin/env bash
set -euo pipefail

# Setup: uv sync --locked --extra smolvla --extra feetech
# Run:   ROBOT_PORT=/dev/ttyACM0 WRIST_CAMERA=0 TOP_CAMERA=2 ./examples/smolvla/run_so101_pick_place.sh
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
WRIST_CAMERA="${WRIST_CAMERA:-0}"
TOP_CAMERA="${TOP_CAMERA:-2}"
DURATION="${DURATION:-30}"
DEVICE="${DEVICE:-cuda}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-5}"

exec uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=CoRL2026-CSI/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep-appendix \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  --robot.use_degrees=true \
  --robot.cameras="{camera1: {type: opencv, index_or_path: $WRIST_CAMERA, width: 640, height: 480, fps: 10}, camera2: {type: opencv, index_or_path: $TOP_CAMERA, width: 640, height: 480, fps: 10}}" \
  --task="Pick up the red block and place it on the blue dish." \
  --fps=10 \
  --duration="$DURATION" \
  --device="$DEVICE" \
  "$@"
