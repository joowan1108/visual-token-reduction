import hashlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments.domain_arithmetic_so101 import prepare_target_dataset
from experiments.domain_arithmetic_so101.dart_merge import (
    BASE_ARTIFACTS,
    ENERGY_CUTOFF,
    _left_svd,
    dart_delta,
    merge_checkpoints,
    merge_tensor,
)
from experiments.domain_arithmetic_so101.prepare_target_dataset import (
    dataset_content_manifest,
    validate_target_contract,
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


def test_same_rig_target_contract_needs_no_interface_conversion() -> None:
    joint_names = list(prepare_target_dataset.JOINT_NAMES)
    joint_feature = {"dtype": "float32", "shape": [6], "names": joint_names}
    camera_feature = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "info": {"video.fps": 10, "video.codec": "av1"},
    }
    meta = SimpleNamespace(
        tasks=SimpleNamespace(index=[prepare_target_dataset.SOURCE_TASK]),
        features={
            "observation.state": joint_feature,
            "action": joint_feature,
            "observation.images.left_wrist": camera_feature,
            "observation.images.top": camera_feature,
        },
        stats={
            "observation.state": {"min": [0, 0, 0, 0, 0, 0.3], "max": [0, 0, 0, 0, 0, 36.3]},
            "action": {"min": [0, 0, 0, 0, 0, 1.2], "max": [0, 0, 0, 0, 0, 36.8]},
        },
    )

    class Target:
        fps = 10
        num_episodes = 1

        def __init__(self) -> None:
            self.meta = meta

        def __len__(self) -> int:
            return 300

    validate_target_contract(Target())
    assert prepare_target_dataset.SOURCE_REPO == "sungkyunner/record-test_20260825_225339"
    assert prepare_target_dataset.SOURCE_REVISION == "97e2c1d4d49607210d1e63d46db2a43b530bdf89"
    assert not hasattr(prepare_target_dataset, "canonicalize_joint_vector")


def test_target_content_manifest_is_sorted(tmp_path: Path) -> None:
    (tmp_path / "z").write_bytes(b"z")
    (tmp_path / "a").write_bytes(b"a")
    manifest = dataset_content_manifest(tmp_path)

    assert [file["path"] for file in manifest["files"]] == ["a", "z"]
    assert len(manifest["tree_sha256"]) == 64


def _capture_uv(tmp_path: Path) -> tuple[Path, Path]:
    capture = tmp_path / "uv-args"
    binary = tmp_path / "uv"
    binary.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CAPTURE"\n', encoding="utf-8")
    binary.chmod(0o755)
    return binary, capture


def test_target_training_uses_pinned_hub_episode_and_frozen_loader_settings(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "experiments/domain_arithmetic_so101/run.sh"
    _, capture = _capture_uv(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "target_provenance.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(
        [script, "train-target"],
        check=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE": str(capture),
            "RUN_ROOT": str(run_root),
        },
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert "--dataset.repo_id=sungkyunner/record-test_20260825_225339" in args
    assert "--dataset.revision=97e2c1d4d49607210d1e63d46db2a43b530bdf89" in args
    assert "--dataset.episodes=[0]" in args
    assert "--dataset.video_backend=pyav" in args
    assert "--num_workers=0" in args
    assert "--batch_size=8" in args
    assert "--accelerator.gradient_accumulation.steps=8" in args


def test_merge_accepts_verified_checkpoint_overrides_and_exact_svd(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "experiments/domain_arithmetic_so101/run.sh"
    _, capture = _capture_uv(tmp_path)
    run_root = tmp_path / "run"
    source = tmp_path / "reused-source"
    target = tmp_path / "reused-target"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    (source / "model.safetensors").touch()
    (target / "model.safetensors").touch()
    subprocess.run(
        [script, "merge"],
        check=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE": str(capture),
            "RUN_ROOT": str(run_root),
            "SOURCE_CHECKPOINT": str(source),
            "TARGET_CHECKPOINT": str(target),
        },
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert args.count(f"--source={source}") == 2
    assert args.count(f"--target={target}") == 2
    assert not any(arg.startswith(("--rank=", "--seed=")) for arg in args)


def test_rollout_defaults_to_same_rig_rtc_and_forwards_extra_args(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "examples/smolvla/run_so101_pick_place.sh"
    _, capture = _capture_uv(tmp_path)
    subprocess.run(
        [script, "--test-extra=value"],
        check=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE": str(capture),
            "WRIST_CAMERA": "12",
            "TOP_CAMERA": "10",
        },
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert "--inference.type=rtc" in args
    assert "--inference.rtc.execution_horizon=10" in args
    assert "--inference.rtc.max_guidance_weight=10" in args
    assert "--fps=10" in args
    assert "fps: 30" in next(arg for arg in args if arg.startswith("--robot.cameras="))
    assert "index_or_path: 12" in next(arg for arg in args if arg.startswith("--robot.cameras="))
    assert "index_or_path: 10" in next(arg for arg in args if arg.startswith("--robot.cameras="))
    assert args[-1] == "--test-extra=value"


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
    actual, _ = merge_tensor(base, source, target, method="dart", alpha=0.5)
    torch.testing.assert_close(actual, torch.tensor([2.0, 4.0]), rtol=0, atol=0)

    source_update = torch.tensor([[2.0, 0.2], [0.1, 1.0], [0.4, -0.3]])
    target_update = torch.tensor([[1.2, 0.7], [0.5, 1.8], [-0.2, 0.9]])
    torch.testing.assert_close(
        dart_delta(source_update, target_update, alpha=0.8),
        _literal_dart(source_update, target_update, 0.8),
    )


def test_zero_update_is_bitwise_base() -> None:
    base = torch.tensor([[1.0, -2.0]], dtype=torch.float16)
    merged, unchanged = merge_tensor(base, base, base, method="dart", alpha=0.8)
    assert unchanged
    assert torch.equal(merged, base)


def test_exact_svd_retains_the_full_thin_spectrum() -> None:
    matrix = torch.randn(30, 25, generator=torch.Generator().manual_seed(7))
    actual_u, actual_s = _left_svd(matrix)
    expected_u, expected_s, _ = torch.linalg.svd(matrix, full_matrices=False)

    assert actual_u.shape == (30, 25)
    assert actual_s.shape == (25,)
    assert torch.isfinite(actual_u).all() and torch.isfinite(actual_s).all()
    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
    torch.testing.assert_close(actual_u.T @ actual_u, torch.eye(25), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_u @ actual_u.T, expected_u @ expected_u.T, rtol=1e-5, atol=1e-5)


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
    metadata = merge_checkpoints(str(base), str(source), str(target), output, method="dart")

    assert set(load_file(output / "model.safetensors")) == set(tensors)
    assert metadata["tensor_count"] == len(tensors)
    assert metadata["svd"] == {
        "implementation": "torch.linalg.svd",
        "full_matrices": False,
        "retained_components": "all",
    }
    assert "rank" not in metadata and "seed" not in metadata
    with (source / "model.safetensors").open("rb") as stream:
        assert metadata["inputs"]["source"]["model_sha256"] == hashlib.file_digest(
            stream, "sha256"
        ).hexdigest()
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
