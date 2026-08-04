#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-1000}"
RESULTS_DIR="experiments/focus_token/results"
RUN_ROOT="${RUN_ROOT:-${RESULTS_DIR}/$(date -u +%Y%m%dT%H%M%SZ)_seed_${SEED}}"
DATASET_REV="${DATASET_REV:-}"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }
command -v lerobot-train >/dev/null || { echo "activate the conda environment first" >&2; exit 2; }
command -v lerobot-eval >/dev/null || { echo "activate the conda environment first" >&2; exit 2; }

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "$DATASET_REV" ]]; then
  DATASET_REV="$(python - <<'PY'
from huggingface_hub import HfApi

print(HfApi().dataset_info("HuggingFaceVLA/libero").sha)
PY
)"
fi

mkdir -p "$RESULTS_DIR"
if [[ ! -e "$RUN_ROOT" ]]; then
  mkdir "$RUN_ROOT"
  {
    date -u
    git rev-parse HEAD
    git status --short
    printf 'dataset_revision=%s\nseed=%s\n' "$DATASET_REV" "$SEED"
    nvidia-smi
    python --version
    python -m pip freeze
  } >"$RUN_ROOT/environment.txt"
  git diff --binary >"$RUN_ROOT/worktree.patch"
fi

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA GPU is required"
print(f"GPU: {torch.cuda.get_device_name()}")
PY
python -m pytest -q -p no:cacheprovider tests/policies/smolvla/test_focus_token.py

declare -A RATIOS=(
  [dense]=1.0
  [focus50]=0.5
  [focus25]=0.25
  [focus75]=0.75
)

for VARIANT in dense focus50 focus25 focus75; do
  OUT="$RUN_ROOT/$VARIANT/seed_$SEED/train"
  FINAL="$OUT/checkpoints/010000/pretrained_model/model.safetensors"

  if [[ -f "$FINAL" ]]; then
    echo "Skipping completed training: $VARIANT"
    continue
  fi
  [[ ! -e "$OUT" ]] || { echo "Refusing to overwrite incomplete run: $OUT" >&2; exit 1; }
  mkdir -p "$OUT"

  lerobot-train \
    --policy.type=smolvla \
    --policy.load_vlm_weights=true \
    --policy.focus_token_keep_ratio="${RATIOS[$VARIANT]}" \
    --policy.focus_token_start_layer=8 \
    --policy.compile_model=false \
    --policy.num_vlm_layers=16 \
    --policy.num_expert_layers=-1 \
    --policy.attention_mode=cross_attn \
    --policy.self_attn_every_n_layers=2 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.scheduler_decay_lr=2.5e-6 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.push_to_hub=false \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.revision="$DATASET_REV" \
    --dataset.image_transforms.enable=false \
    --seed="$SEED" \
    --cudnn_deterministic=true \
    --batch_size=4 \
    --steps=10000 \
    --save_checkpoint=true \
    --save_freq=2000 \
    --log_freq=100 \
    --eval_steps=0 \
    --env_eval_freq=0 \
    --wandb.enable=false \
    --output_dir="$OUT" \
    --job_name="focus_token_${VARIANT}_${SEED}" \
    2>&1 | tee "$OUT/train.log"
done

for VARIANT in dense focus50 focus25 focus75; do
  if [[ "$VARIANT" == dense || "$VARIANT" == focus50 ]]; then
    CHECKPOINTS=(002000 004000 006000 008000 010000)
  else
    CHECKPOINTS=(010000)
  fi

  for STEP in "${CHECKPOINTS[@]}"; do
    POLICY_PATH="$RUN_ROOT/$VARIANT/seed_$SEED/train/checkpoints/$STEP/pretrained_model"
    EVAL_OUT="$RUN_ROOT/$VARIANT/seed_$SEED/eval/$STEP"

    [[ -f "$POLICY_PATH/model.safetensors" ]] || { echo "Missing checkpoint: $POLICY_PATH" >&2; exit 1; }
    if [[ -f "$EVAL_OUT/eval_info.json" ]]; then
      echo "Skipping completed evaluation: $VARIANT $STEP"
      continue
    fi
    [[ ! -e "$EVAL_OUT" ]] || { echo "Refusing to overwrite incomplete eval: $EVAL_OUT" >&2; exit 1; }
    mkdir -p "$EVAL_OUT"

    lerobot-eval \
      --policy.path="$POLICY_PATH" \
      --policy.device=cuda \
      --policy.use_amp=false \
      --env.type=libero \
      --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
      --env.control_mode=relative \
      --env.init_states=true \
      --env.hard_reset=true \
      --env.max_parallel_tasks=1 \
      --eval.n_episodes=20 \
      --eval.batch_size="${EVAL_BATCH_SIZE:-20}" \
      --eval.recording=false \
      --seed=0 \
      --output_dir="$EVAL_OUT" \
      --job_name="focus_token_${VARIANT}_${SEED}_${STEP}" \
      2>&1 | tee "$EVAL_OUT/eval.log"
  done
done

python - "$RUN_ROOT" "$SEED" <<'PY' | tee "$RUN_ROOT/success_summary.tsv"
import json
import sys
from pathlib import Path

root, seed = Path(sys.argv[1]), sys.argv[2]
results = {}
print("variant\tpc_success\tn_episodes")
for variant in ("dense", "focus50", "focus25", "focus75"):
    path = root / variant / f"seed_{seed}" / "eval" / "010000" / "eval_info.json"
    info = json.loads(path.read_text())
    results[variant] = float(info["overall"]["pc_success"])
    print(variant, results[variant], info["overall"]["n_episodes"], sep="\t")
print(f"focus50_minus_dense_pp\t{results['focus50'] - results['dense']:+.3f}")
PY

echo "Complete: $RUN_ROOT"
