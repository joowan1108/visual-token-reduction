#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from lerobot.configs import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import ATOMIC_SKILLS
from lerobot.scripts.lerobot_train import _resolve_atomic_subtask_mapping
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.random_utils import set_seed


def summarize(losses: dict[int, list[float]]) -> dict:
    per_skill = {}
    all_losses = []
    for skill_id, skill in enumerate(ATOMIC_SKILLS):
        values = torch.tensor(losses.get(skill_id, []), dtype=torch.float64)
        all_losses.extend(values.tolist())
        per_skill[skill] = (
            {"count": 0, "mean": None, "std": None, "median": None, "p95": None}
            if values.numel() == 0
            else {
                "count": values.numel(),
                "mean": values.mean().item(),
                "std": values.std(unbiased=False).item(),
                "median": values.median().item(),
                "p95": values.quantile(0.95).item(),
            }
        )
    if not all_losses:
        raise ValueError("No held-out losses were collected.")
    means = [stats["mean"] for stats in per_skill.values() if stats["mean"] is not None]
    return {
        "per_skill": per_skill,
        "macro_mean": sum(means) / len(means),
        "micro_mean": sum(all_losses) / len(all_losses),
        "total_samples": len(all_losses),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure held-out GT-routed atomic action loss.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-repo-id", default="k1000dai/libero_subtask_sarm")
    parser.add_argument("--dataset-revision", default="8ec70343c56430f5dbae09af6b073d879207fe7c")
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0, help="0 evaluates the full split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="outputs/eval/atomic_sgmoe_gt_action_loss.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.noise_repeats <= 0 or args.max_batches < 0:
        raise ValueError("batch_size and noise_repeats must be positive; max_batches must be non-negative.")

    set_seed(args.seed)
    policy_path = Path(args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    if not getattr(policy_cfg, "atomic_data_enabled", False):
        raise ValueError("The checkpoint must use atomic GT labels.")
    if getattr(policy_cfg, "atomic_classifier_enabled", False):
        raise ValueError("Use the SG-MoE action checkpoint from before classifier-only training.")
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        revision=args.dataset_revision,
        eval_split=args.eval_split,
    )
    split_cfg = SimpleNamespace(
        dataset=dataset_cfg,
        trainable_config=policy_cfg,
        tolerance_s=1e-4,
        num_workers=args.num_workers,
    )
    train_dataset, eval_dataset = make_train_eval_datasets(split_cfg)
    if eval_dataset is None:
        raise ValueError("eval_split must create a held-out dataset.")
    _resolve_atomic_subtask_mapping(train_dataset, policy_cfg)

    policy = make_policy(cfg=policy_cfg, ds_meta=eval_dataset.meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
        collate_fn=lerobot_collate_fn if eval_dataset.meta.has_language_columns else None,
    )

    losses: dict[int, list[float]] = {skill_id: [] for skill_id in range(len(ATOMIC_SKILLS))}
    total_batches = len(dataloader) if args.max_batches == 0 else min(len(dataloader), args.max_batches)
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(dataloader, total=total_batches, desc="GT-routed eval")):
            if args.max_batches and batch_index >= args.max_batches:
                break
            for camera_key in eval_dataset.meta.camera_keys:
                if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].float() / 255.0
            batch = preprocessor(batch)
            skill_ids, _ = policy._atomic_batch_contract(batch)
            sample_loss = torch.zeros(skill_ids.shape[0], device=skill_ids.device)
            for _ in range(args.noise_repeats):
                repeat_loss, _ = policy.forward(batch, reduction="none")
                sample_loss += repeat_loss.float() / args.noise_repeats
            for skill_id, loss in zip(skill_ids.tolist(), sample_loss.tolist(), strict=True):
                losses[skill_id].append(loss)

    result = {
        "policy_path": str(policy_path),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_revision": args.dataset_revision,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "noise_repeats": args.noise_repeats,
        **summarize(losses),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"{'skill':<8} {'samples':>9} {'mean':>12} {'median':>12} {'p95':>12}")
    for skill, stats in result["per_skill"].items():
        if stats["mean"] is None:
            print(f"{skill:<8} {0:>9}")
        else:
            print(
                f"{skill:<8} {stats['count']:>9} {stats['mean']:>12.6f} "
                f"{stats['median']:>12.6f} {stats['p95']:>12.6f}"
            )
    print(f"macro_mean={result['macro_mean']:.6f} micro_mean={result['micro_mean']:.6f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
