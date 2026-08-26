#!/usr/bin/env python
"""Merge one-shot source/target SmolVLA checkpoints with direct arithmetic or DArT."""

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_FILE = "model.safetensors"
BASE_ARTIFACTS = (
    "config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
ENERGY_CUTOFF = 0.9975


def _resolve(identifier: str, revision: str | None, *, base: bool = False) -> tuple[Path, dict]:
    local = Path(identifier)
    if local.exists():
        if revision is not None:
            raise ValueError(f"A revision cannot be used with local checkpoint {identifier!r}.")
        root = local.resolve()
        resolved_revision = None
    else:
        root = Path(
            snapshot_download(
                repo_id=identifier,
                revision=revision,
                allow_patterns=[MODEL_FILE, *BASE_ARTIFACTS] if base else [MODEL_FILE],
            )
        )
        parts = root.parts
        resolved_revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else revision

    model = root / MODEL_FILE
    if not root.is_dir() or not model.is_file():
        raise FileNotFoundError(f"Expected one {MODEL_FILE} at {root}.")
    with model.open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return root, {
        "input": identifier,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "resolved_path": str(root),
        "model_sha256": model_sha256,
    }


def _left_svd(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    return u, s


def _project(matrix: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return basis @ (basis.T @ matrix)


def dart_delta(
    source_update: torch.Tensor,
    target_update: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Apply Algorithm 1 to one `[out, in]` float32 update matrix."""
    if source_update.ndim != 2 or target_update.shape != source_update.shape:
        raise ValueError("DArT expects equal 2-D source and target updates.")

    u_target, s_target = _left_svd(target_update)
    u_source, _ = _left_svd(source_update)

    singular_energy = s_target.square()
    if singular_energy.sum() > 0:
        cutoff = int(
            torch.searchsorted(
                singular_energy.cumsum(0) / singular_energy.sum(),
                torch.tensor(ENERGY_CUTOFF),
            ).item()
        )
        target_signal_basis = u_target[:, : cutoff + 1]
    else:
        target_signal_basis = u_target[:, :0]

    source_norm = torch.linalg.vector_norm(source_update)
    gamma = (
        torch.linalg.vector_norm(_project(source_update, target_signal_basis)) / source_norm
        if source_norm > 0
        else source_norm
    ).clamp(0, 1)

    overlap_energy = (u_target.T @ u_source).square().sum(dim=0)
    if gamma > 0 and overlap_energy.sum() > 0:
        sorted_energy = overlap_energy.sort(descending=True).values
        cutoff = int(torch.searchsorted(sorted_energy.cumsum(0), gamma * overlap_energy.sum()).item())
        threshold = sorted_energy[min(cutoff, len(sorted_energy) - 1)]
        common_source_basis = u_source[:, overlap_energy >= threshold]
    else:
        common_source_basis = u_source[:, :0]

    return alpha * gamma * (target_update - _project(source_update, common_source_basis))


def merge_tensor(
    base: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    method: str,
    alpha: float,
) -> tuple[torch.Tensor, bool]:
    source_update = source.float() - base.float()
    target_update = target.float() - base.float()
    if not torch.count_nonzero(source_update) and not torch.count_nonzero(target_update):
        return base.clone(), True

    if method == "direct" or base.ndim == 1:
        delta = alpha * (target_update - source_update)
    else:
        shape = base.shape
        delta = dart_delta(
            source_update.reshape(shape[0], -1),
            target_update.reshape(shape[0], -1),
            alpha=alpha,
        ).reshape(shape)
    return (base.float() + delta).to(base.dtype).contiguous(), False


def merge_checkpoints(
    base: str,
    source: str,
    target: str,
    output: str | Path,
    *,
    method: str = "dart",
    alpha: float = 0.8,
    base_revision: str | None = None,
    source_revision: str | None = None,
    target_revision: str | None = None,
) -> dict:
    if method not in {"direct", "dart"}:
        raise ValueError(f"Unknown method {method!r}.")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    base_root, base_info = _resolve(base, base_revision, base=True)
    source_root, source_info = _resolve(source, source_revision)
    target_root, target_info = _resolve(target, target_revision)
    for name in BASE_ARTIFACTS:
        if not (base_root / name).is_file():
            raise FileNotFoundError(f"Base checkpoint is missing required artifact {name!r}.")

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output {output}.")

    merged: dict[str, torch.Tensor] = {}
    counts = {"rank_1": 0, "rank_2": 0, "rank_4": 0}
    zero_updates = 0
    with (
        safe_open(base_root / MODEL_FILE, framework="pt", device="cpu") as base_file,
        safe_open(source_root / MODEL_FILE, framework="pt", device="cpu") as source_file,
        safe_open(target_root / MODEL_FILE, framework="pt", device="cpu") as target_file,
    ):
        base_keys = set(base_file.keys())
        if set(source_file.keys()) != base_keys or set(target_file.keys()) != base_keys:
            raise ValueError("Base, source, and target checkpoints must have identical key sets.")

        for key in sorted(base_keys):
            tensors = (base_file.get_tensor(key), source_file.get_tensor(key), target_file.get_tensor(key))
            if tensors[0].shape != tensors[1].shape or tensors[0].shape != tensors[2].shape:
                raise ValueError(f"Shape mismatch for {key!r}: {[tuple(t.shape) for t in tensors]}.")
            if not all(t.is_floating_point() for t in tensors):
                raise TypeError(f"Non-floating tensor {key!r} is unsupported.")
            if tensors[0].dtype != tensors[1].dtype or tensors[0].dtype != tensors[2].dtype:
                raise TypeError(f"Dtype mismatch for {key!r}: {[t.dtype for t in tensors]}.")
            if tensors[0].ndim not in (1, 2, 4):
                raise ValueError(f"Unsupported rank {tensors[0].ndim} for {key!r}.")
            counts[f"rank_{tensors[0].ndim}"] += 1
            merged[key], unchanged = merge_tensor(
                *tensors,
                method=method,
                alpha=alpha,
            )
            zero_updates += unchanged

    metadata = {
        "method": method,
        "alpha": alpha,
        "energy_cutoff": ENERGY_CUTOFF,
        "svd": {
            "implementation": "torch.linalg.svd",
            "full_matrices": False,
            "retained_components": "all",
        }
        if method == "dart"
        else None,
        "inputs": {"base": base_info, "source": source_info, "target": target_info},
        "tensor_count": len(merged),
        "tensor_counts_by_rank": counts,
        "zero_update_count": zero_updates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        save_file(merged, temp / MODEL_FILE, metadata={"format": "pt"})
        for name in BASE_ARTIFACTS:
            shutil.copy2(base_root / name, temp / name)
        (temp / "dart_merge.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temp.rename(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("direct", "dart"), default="dart")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--base-revision")
    parser.add_argument("--source-revision")
    parser.add_argument("--target-revision")
    args = parser.parse_args()
    merge_checkpoints(**vars(args))


if __name__ == "__main__":
    main()
