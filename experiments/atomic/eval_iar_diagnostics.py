#!/usr/bin/env python

"""Summarize implicit FAST-KI IAR attention on a dataset-factory validation subset.

Diversity is normalized Jensen-Shannon divergence: 0 means identical attention and 1 means
maximally different attention. Full raw token attention is consumed one batch at a time; optional
heatmaps retain only bounded visual-token slices.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.configs import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.attention_analysis import iar_capture_metrics, normalized_jsd
from lerobot.policies.smolvla.modeling_smolvla import ATOMIC_SKILLS
from lerobot.scripts.lerobot_train import _resolve_atomic_subtask_mapping
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.random_utils import set_seed


def _stats(values: torch.Tensor) -> dict:
    values = values.double().flatten()
    if not values.numel():
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "median": values.median().item(),
        "p95": values.quantile(0.95).item(),
    }


def _concatenate_metrics(name: str, values: list[torch.Tensor]) -> torch.Tensor:
    if not name.endswith("_signature"):
        return torch.cat(values)
    max_tokens = max(value.shape[-1] for value in values)
    padded = [F.pad(value, (0, max_tokens - value.shape[-1])) for value in values]
    return torch.cat(padded).flatten(1)


def _group_mean_jsd(
    signatures: torch.Tensor, left_selected: torch.Tensor, right_selected: torch.Tensor
) -> float | None:
    left = signatures[left_selected].mean(dim=0)
    right = signatures[right_selected].mean(dim=0)
    if left.sum() == 0 or right.sum() == 0:
        return None
    return normalized_jsd(left, right).item()


def _save_iar_heatmaps(
    output_dir: Path,
    capture: dict,
    images: list[torch.Tensor],
    camera_keys: list[str],
    batch: dict,
    skill_ids: torch.Tensor,
    dataset_indices: list[int],
    sample_offset: int,
) -> list[Path]:
    """Save raw IAR visual slices in the existing attention-overlay NPZ format."""
    attention = capture["attention"].detach().float().cpu()
    spans = capture["visual_token_spans"]
    if attention.ndim != 4:
        raise ValueError("IAR attention must be [batch, layer, query, token].")
    if len(images) != len(spans) or len(camera_keys) != len(spans):
        raise ValueError("IAR images, camera keys, and visual-token spans must align.")
    token_counts = {end - start for start, end in spans}
    if len(token_counts) != 1:
        raise ValueError("IAR heatmap cameras must have the same visual-token count.")
    token_count = token_counts.pop()
    grid_size = math.isqrt(token_count)
    if grid_size * grid_size != token_count:
        raise ValueError(f"Expected a square visual-token grid, got {token_count} values.")

    sample_count = len(dataset_indices)
    stacked_images = torch.stack(images, dim=1)
    if not 0 < sample_count <= attention.shape[0] or stacked_images.shape[0] != attention.shape[0]:
        raise ValueError("Requested IAR heatmap samples must fit the captured image batch.")
    image_array = (
        ((stacked_images[:sample_count].detach().float().cpu() + 1) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 1, 3, 4, 2)
        .numpy()
    )

    tasks = batch.get("task", [""] * attention.shape[0])
    tasks = [tasks] if isinstance(tasks, str) else list(tasks)
    if len(tasks) < sample_count or not all(isinstance(task, str) for task in tasks[:sample_count]):
        raise ValueError("Task descriptions must be one string per IAR heatmap sample.")

    def batch_ids(key: str) -> list[int]:
        values = batch.get(key)
        if values is None:
            return [-1] * sample_count
        values = torch.as_tensor(values).detach().cpu().reshape(attention.shape[0], -1)
        return [int(value) for value in values[:sample_count, 0]]

    episode_indices = batch_ids("episode_index")
    frame_indices = batch_ids("frame_index")
    skill_ids = skill_ids.detach().cpu().reshape(-1)[:sample_count]
    layers = tuple(capture["layers"])
    queries = tuple(capture["queries"])
    if attention.shape[1:3] != (len(layers), len(queries)):
        raise ValueError("IAR attention must align with captured layers and queries.")

    outputs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch_index in range(sample_count):
        skill_id = int(skill_ids[batch_index])
        skill = ATOMIC_SKILLS[skill_id]
        identity = (
            f"sample_{sample_offset + batch_index:04d}_index_{dataset_indices[batch_index]}_"
            f"episode_{episode_indices[batch_index]}_frame_{frame_indices[batch_index]}"
        )
        for layer_index, layer in enumerate(layers):
            for query_index, query in enumerate(queries):
                output = output_dir / f"{identity}_skill_{skill}_iar_layer_{layer}_query_{query}.npz"
                if output.exists():
                    raise FileExistsError(f"Refusing to overwrite IAR heatmap: {output}")
                token_attention = attention[batch_index, layer_index, query_index]
                maps = np.stack([token_attention[start:end].numpy() for start, end in spans])[None]
                np.savez_compressed(
                    output,
                    images=image_array[batch_index : batch_index + 1],
                    task_descriptions=np.asarray([tasks[batch_index]], dtype=np.str_),
                    attention_maps=maps,
                    attention_mass=maps.sum(axis=-1),
                    layers=np.asarray([layer], dtype=np.int64),
                    queries=np.asarray([query], dtype=np.int64),
                    flow_step=np.asarray(-1, dtype=np.int64),
                    flow_step_semantics=np.asarray("not_applicable"),
                    attention_kind=np.asarray("iar_searchable_query"),
                    attention_scope=np.asarray("full_prefix_visual_slice"),
                    attention_source=np.asarray("implicit_action_reasoner"),
                    camera_keys=np.asarray(camera_keys, dtype=np.str_),
                    sample_identities=np.asarray([identity], dtype=np.str_),
                    dataset_indices=np.asarray([dataset_indices[batch_index]], dtype=np.int64),
                    episode_indices=np.asarray([episode_indices[batch_index]], dtype=np.int64),
                    frame_indices=np.asarray([frame_indices[batch_index]], dtype=np.int64),
                    skill_ids=np.asarray([skill_id], dtype=np.int64),
                    skill_names=np.asarray([skill], dtype=np.str_),
                )
                outputs.append(output)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset-factory validation-split implicit IAR attention diagnostics."
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/eval/atomic_iar_diagnostics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", help="Actual selected VLM layer indices.")
    parser.add_argument("--queries", type=int, nargs="+", help="Zero-based IAR query indices.")
    parser.add_argument("--heatmap-samples", type=int, default=0)
    parser.add_argument("--heatmap-dir", default="outputs/eval/atomic_iar_heatmaps")
    parser.add_argument("--dataset-repo-id", default="k1000dai/libero_subtask_sarm")
    parser.add_argument("--dataset-revision", default="8ec70343c56430f5dbae09af6b073d879207fe7c")
    parser.add_argument("--eval-split", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("max_samples and batch_size must be positive; workers must be non-negative.")
    if not 0 <= args.heatmap_samples <= args.max_samples:
        raise ValueError("heatmap_samples must be non-negative and no greater than max_samples.")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite IAR diagnostics: {output}")

    set_seed(args.seed)
    policy_path = Path(args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    if not getattr(policy_cfg, "implicit_fast_ki_enabled", False):
        raise ValueError("The checkpoint must enable implicit FAST-KI.")
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
        num_workers=args.workers,
    )
    train_dataset, eval_dataset = make_train_eval_datasets(split_cfg)
    if eval_dataset is None:
        raise ValueError("eval_split must create a dataset-factory validation split.")
    _resolve_atomic_subtask_mapping(train_dataset, policy_cfg)

    generator = torch.Generator().manual_seed(args.seed)
    sample_count = min(args.max_samples, len(eval_dataset))
    indices = torch.randperm(len(eval_dataset), generator=generator)[:sample_count].tolist()
    subset = torch.utils.data.Subset(eval_dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lerobot_collate_fn if eval_dataset.meta.has_language_columns else None,
    )

    policy = make_policy(cfg=policy_cfg, ds_meta=eval_dataset.meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    reasoner = policy.model.implicit_action_reasoner
    reasoner.enable_diagnostics(args.layers, args.queries, max_batches=1)

    collected: dict[str, list[torch.Tensor]] = {}
    skill_ids_all = []
    heatmap_samples_saved = 0
    heatmap_files_saved = 0
    layers = queries = None
    try:
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="IAR diagnostics"):
                for camera_key in eval_dataset.meta.camera_keys:
                    if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                        batch[camera_key] = batch[camera_key].float() / 255.0
                batch = preprocessor(batch)
                skill_ids, _ = policy._atomic_batch_contract(batch)
                heatmap_batch_size = min(args.heatmap_samples - heatmap_samples_saved, skill_ids.shape[0])
                heatmap_images = camera_keys = None
                if heatmap_batch_size:
                    heatmap_images, _ = policy.prepare_images(batch)
                    present_keys = [key for key in policy.config.image_features if key in batch]
                    missing_keys = [key for key in policy.config.image_features if key not in batch]
                    camera_keys = present_keys + missing_keys[: len(heatmap_images) - len(present_keys)]
                state_context = policy.model.state_proj(policy.prepare_state(batch))
                policy.forward(batch, reduction="none")
                captures = reasoner.pop_diagnostics()
                if len(captures) != 1:
                    raise RuntimeError("Expected exactly one IAR capture per policy forward.")
                capture = captures[0]
                layers, queries = capture["layers"], capture["queries"]
                if heatmap_batch_size:
                    outputs = _save_iar_heatmaps(
                        Path(args.heatmap_dir),
                        capture,
                        heatmap_images,
                        camera_keys,
                        batch,
                        skill_ids,
                        indices[heatmap_samples_saved : heatmap_samples_saved + heatmap_batch_size],
                        heatmap_samples_saved,
                    )
                    heatmap_samples_saved += heatmap_batch_size
                    heatmap_files_saved += len(outputs)
                metrics = iar_capture_metrics(capture, state_context)
                for name, values in metrics.items():
                    collected.setdefault(name, []).append(values)
                skill_ids_all.append(skill_ids.detach().cpu())
    finally:
        reasoner.disable_diagnostics()

    metrics = {name: _concatenate_metrics(name, values) for name, values in collected.items()}
    skill_ids = torch.cat(skill_ids_all)
    layer_query = []
    for layer_index, layer in enumerate(layers):
        for query_index, query in enumerate(queries):
            layer_query.append(
                {
                    "layer": layer,
                    "query": query,
                    "image_mass": _stats(metrics["image_mass"][:, layer_index, query_index]),
                    "language_mass": _stats(metrics["language_mass"][:, layer_index, query_index]),
                    "other_mass": _stats(metrics["other_mass"][:, layer_index, query_index]),
                    "normalized_entropy": _stats(metrics["normalized_entropy"][:, layer_index, query_index]),
                    "context_norm": _stats(metrics["layer_context_norm"][:, layer_index, query_index]),
                    "context_variance": _stats(
                        metrics["layer_context_variance"][:, layer_index, query_index]
                    ),
                }
            )

    per_skill = {}
    present = []
    for skill_id, skill in enumerate(ATOMIC_SKILLS):
        selected = skill_ids == skill_id
        if not selected.any():
            per_skill[skill] = {"status": "absent", "count": 0}
            continue
        present.append(skill_id)
        per_skill[skill] = {
            "status": "present",
            "count": int(selected.sum()),
            "image_mass": _stats(metrics["image_mass"][selected]),
            "language_mass": _stats(metrics["language_mass"][selected]),
            "normalized_entropy": _stats(metrics["normalized_entropy"][selected]),
        }

    skill_differences = []
    for left_offset, left_id in enumerate(present):
        for right_id in present[left_offset + 1 :]:
            left_selected = skill_ids == left_id
            right_selected = skill_ids == right_id
            skill_differences.append(
                {
                    "left": ATOMIC_SKILLS[left_id],
                    "right": ATOMIC_SKILLS[right_id],
                    "left_count": int(left_selected.sum()),
                    "right_count": int(right_selected.sum()),
                    "comparison": "group_mean_vs_group_mean",
                    "signature": "layer_query_token_positions",
                    "normalized_jsd": _group_mean_jsd(
                        metrics["attention_signature"], left_selected, right_selected
                    ),
                    "visual_only_normalized_jsd": _group_mean_jsd(
                        metrics["visual_attention_signature"], left_selected, right_selected
                    ),
                    "language_only_normalized_jsd": _group_mean_jsd(
                        metrics["language_attention_signature"], left_selected, right_selected
                    ),
                }
            )

    result = {
        "policy_path": str(policy_path),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_revision": args.dataset_revision,
        "eval_split": args.eval_split,
        "split_provenance": {
            "name": "dataset_factory_validation_split",
            "source": "make_train_eval_datasets(DatasetConfig(eval_split=...))",
            "is_validation": True,
            "is_preregistered_offline_test": False,
        },
        "seed": args.seed,
        "sample_count": sample_count,
        "layers": list(layers),
        "queries": list(queries),
        "raw_attention_stored": False,
        "diversity_metric": "normalized Jensen-Shannon divergence in [0,1]",
        "query_diversity": _stats(metrics["query_diversity"]),
        "layer_diversity": _stats(metrics["layer_diversity"]),
        "layer_query": layer_query,
        "iar_averaged_context": {
            "norm": _stats(metrics["context_norm"]),
            "variance": _stats(metrics["context_variance"]),
        },
        "state_context_separate": {
            "description": "state_proj(state), concatenated after IAR; not token attention",
            "norm": _stats(metrics["state_context_norm"]),
            "variance": _stats(metrics["state_context_variance"]),
        },
        "per_skill": per_skill,
        "skill_attention_differences": skill_differences,
    }
    if args.heatmap_samples:
        result["iar_heatmaps"] = {
            "sample_count": heatmap_samples_saved,
            "file_count": heatmap_files_saved,
            "directory": str(Path(args.heatmap_dir)),
            "attention_kind": "iar_searchable_query",
            "scope": "full_prefix_visual_slice",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"{'layer':>5} {'query':>5} {'image':>9} {'language':>9} {'entropy':>9} {'ctx_norm':>9}")
    for row in layer_query:
        print(
            f"{row['layer']:>5} {row['query']:>5} {row['image_mass']['mean']:>9.4f} "
            f"{row['language_mass']['mean']:>9.4f} {row['normalized_entropy']['mean']:>9.4f} "
            f"{row['context_norm']['mean']:>9.4f}"
        )
    print(
        f"query_jsd={result['query_diversity']['mean']} (pairs={result['query_diversity']['count']}) "
        f"layer_jsd={result['layer_diversity']['mean']} (pairs={result['layer_diversity']['count']})"
    )
    print(
        f"state_context_norm={result['state_context_separate']['norm']['mean']} "
        "(separate concatenated state context)"
    )
    print(f"{'skill':<8} {'count':>7} {'image':>9} {'language':>9} {'entropy':>9}")
    for skill, summary in per_skill.items():
        if summary["status"] == "absent":
            print(f"{skill:<8} {0:>7} {'absent':>9}")
        else:
            print(
                f"{skill:<8} {summary['count']:>7} {summary['image_mass']['mean']:>9.4f} "
                f"{summary['language_mass']['mean']:>9.4f} "
                f"{summary['normalized_entropy']['mean']:>9.4f}"
            )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
