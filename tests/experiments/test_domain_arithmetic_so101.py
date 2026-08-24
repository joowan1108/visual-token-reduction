import os
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments.domain_arithmetic_so101.dart_merge import (
    BASE_ARTIFACTS,
    ENERGY_CUTOFF,
    _left_svd,
    dart_delta,
    merge_checkpoints,
    merge_tensor,
)
from experiments.domain_arithmetic_so101.prepare_target_dataset import (
    canonicalize_joint_vector,
    dataset_content_manifest,
    image_for_writer,
)
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import _use_dataset_processor_stats


def test_workflow_condition_paths(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "experiments/domain_arithmetic_so101/run.sh"
    result = subprocess.run(
        [script, "check"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "RUN_ROOT": str(tmp_path / "run")},
    )
    assert result.stdout.strip() == "workflow condition paths OK"


def test_old_gripper_convention_is_canonicalized_without_touching_arm_joints() -> None:
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, -40.0])
    converted = canonicalize_joint_vector(vector)

    assert torch.equal(converted[:5], vector[:5])
    assert converted[-1].item() == 30.0
    assert canonicalize_joint_vector(vector.index_put((torch.tensor([5]),), torch.tensor([-100.0])))[-1] == 0
    assert canonicalize_joint_vector(vector.index_put((torch.tensor([5]),), torch.tensor([100.0])))[-1] == 100


def test_decoded_chw_image_is_prepared_for_hwc_writer() -> None:
    image = torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5)
    prepared = image_for_writer(image, {"shape": (4, 5, 3)})
    assert prepared.shape == (4, 5, 3)
    assert prepared.is_contiguous()
    assert torch.equal(prepared.permute(2, 0, 1), image)


def test_target_content_manifest_is_sorted_and_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "z").write_bytes(b"z")
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "target_preparation.json").write_bytes(b"ignored")
    manifest = dataset_content_manifest(tmp_path)

    assert [file["path"] for file in manifest["files"]] == ["a", "z"]
    assert len(manifest["tree_sha256"]) == 64


def _checkpoint(path: Path, tensors: dict[str, torch.Tensor], artifacts: bool = False) -> Path:
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    if artifacts:
        for index, name in enumerate(BASE_ARTIFACTS):
            (path / name).write_bytes(f"base artifact {index}".encode())
    return path


def _literal_dart(source: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    u_target, s_target, _ = torch.linalg.svd(target, full_matrices=False)
    u_source, _, _ = torch.linalg.svd(source, full_matrices=False)
    energy = s_target.square()
    cutoff = int(torch.searchsorted(energy.cumsum(0) / energy.sum(), torch.tensor(ENERGY_CUTOFF)))
    target_basis = u_target[:, : cutoff + 1]
    gamma = (torch.linalg.vector_norm(target_basis @ (target_basis.T @ source)) / source.norm()).clamp(0, 1)
    overlap = (u_target.T @ u_source).square().sum(0)
    sorted_overlap = overlap.sort(descending=True).values
    cutoff = int(torch.searchsorted(sorted_overlap.cumsum(0), gamma * overlap.sum()))
    threshold = sorted_overlap[min(cutoff, len(sorted_overlap) - 1)]
    source_basis = u_source[:, overlap >= threshold]
    return alpha * gamma * (target - source_basis @ (source_basis.T @ source))


def test_tensor_arithmetic_and_literal_dart() -> None:
    base = torch.tensor([1.0, 2.0])
    source = torch.tensor([2.0, 4.0])
    target = torch.tensor([4.0, 8.0])
    actual, _ = merge_tensor(base, source, target, method="dart", alpha=0.5, rank=256, seed=42)
    torch.testing.assert_close(actual, torch.tensor([2.0, 4.0]), rtol=0, atol=0)

    source_update = torch.tensor([[2.0, 0.2], [0.1, 1.0], [0.4, -0.3]])
    target_update = torch.tensor([[1.2, 0.7], [0.5, 1.8], [-0.2, 0.9]])
    torch.testing.assert_close(
        dart_delta(source_update, target_update, alpha=0.8, rank=256, seed=42),
        _literal_dart(source_update, target_update, 0.8),
    )


def test_zero_update_is_bitwise_base() -> None:
    base = torch.tensor([[1.0, -2.0]], dtype=torch.float16)
    merged, unchanged = merge_tensor(base, base, base, method="dart", alpha=0.8, rank=256, seed=42)
    assert unchanged
    assert torch.equal(merged, base)


def test_randomized_svd_is_seeded_and_finite() -> None:
    matrix = torch.randn(30, 25, generator=torch.Generator().manual_seed(7))
    first_u, first_s = _left_svd(matrix, rank=4, seed=42)
    second_u, second_s = _left_svd(matrix, rank=4, seed=42)

    assert first_u.shape == (30, 4)
    assert first_s.shape == (4,)
    assert torch.isfinite(first_u).all() and torch.isfinite(first_s).all()
    torch.testing.assert_close(first_u, second_u, rtol=0, atol=0)
    torch.testing.assert_close(first_s, second_s, rtol=0, atol=0)
    torch.testing.assert_close(first_u.T @ first_u, torch.eye(4), rtol=1e-5, atol=1e-5)


def test_checkpoint_mismatches_raise(tmp_path: Path) -> None:
    base = _checkpoint(tmp_path / "base", {"weight": torch.ones(2, 2)}, artifacts=True)
    source = _checkpoint(tmp_path / "source", {"other": torch.ones(2, 2)})
    target = _checkpoint(tmp_path / "target", {"weight": torch.ones(2, 2)})
    with pytest.raises(ValueError, match="key sets"):
        merge_checkpoints(str(base), str(source), str(target), tmp_path / "keys")

    source = _checkpoint(tmp_path / "source_shape", {"weight": torch.ones(2, 3)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        merge_checkpoints(str(base), str(source), str(target), tmp_path / "shapes")


def test_native_output_preserves_base_processors(tmp_path: Path) -> None:
    tensors = {"weight": torch.eye(2), "bias": torch.ones(2)}
    base = _checkpoint(tmp_path / "base", tensors, artifacts=True)
    source = _checkpoint(tmp_path / "source", {key: value + 1 for key, value in tensors.items()})
    target = _checkpoint(tmp_path / "target", {key: value + 2 for key, value in tensors.items()})
    output = tmp_path / "merged"
    metadata = merge_checkpoints(str(base), str(source), str(target), output, method="direct")

    assert set(load_file(output / "model.safetensors")) == set(tensors)
    assert metadata["tensor_count"] == len(tensors)
    assert (output / "dart_merge.json").is_file()
    for name in BASE_ARTIFACTS:
        assert (output / name).read_bytes() == (base / name).read_bytes()


def test_pretrained_processor_stats_flag_defaults_to_current_behavior() -> None:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/dataset"))
    assert cfg.preserve_pretrained_processor_stats is False
    assert _use_dataset_processor_stats(cfg, Path("checkpoint"))

    cfg.preserve_pretrained_processor_stats = True
    assert not _use_dataset_processor_stats(cfg, Path("checkpoint"))
    assert _use_dataset_processor_stats(cfg, None)
