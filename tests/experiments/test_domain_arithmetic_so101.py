import hashlib
import json
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
    validate_target_provenance,
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


def _target_fixture(frame_count: int = 300) -> object:
    joint_names = list(prepare_target_dataset.JOINT_NAMES)
    joint_feature = {"dtype": "float32", "shape": [6], "names": joint_names}
    camera_feature = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "info": {"video.fps": 10, "video.codec": "any-supported-codec"},
    }
    meta = SimpleNamespace(
        tasks=SimpleNamespace(index=[prepare_target_dataset.TARGET_TASK]),
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
            self.reader = SimpleNamespace(get_episodes_file_paths=lambda: ["selected.bin"])

        def __len__(self) -> int:
            return frame_count

    return Target()


def test_experiment_m_target_contract_needs_no_interface_or_codec_conversion() -> None:
    validate_target_contract(_target_fixture())
    assert prepare_target_dataset.DEFAULT_TARGET_REPO == "sungkyunner/record-test_20260826_210214"
    assert (
        prepare_target_dataset.DEFAULT_TARGET_REVISION
        == "295e6def6cb4df454f58894caea10c15446dc4e4"
    )
    assert not hasattr(prepare_target_dataset, "canonicalize_joint_vector")


def test_target_contract_rejects_an_empty_selected_episode() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        validate_target_contract(_target_fixture(frame_count=0))


def test_target_content_manifest_is_sorted(tmp_path: Path) -> None:
    (tmp_path / "z").write_bytes(b"z")
    (tmp_path / "a").write_bytes(b"a")
    manifest = dataset_content_manifest(tmp_path)

    assert [file["path"] for file in manifest["files"]] == ["a", "z"]
    assert len(manifest["tree_sha256"]) == 64


def test_target_provenance_records_selected_coordinates_and_hash(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "selected.bin").write_bytes(b"selected episode")
    target = _target_fixture(frame_count=17)
    target.root = dataset_root
    monkeypatch.setattr(prepare_target_dataset, "LeRobotDataset", lambda *args, **kwargs: target)
    output = tmp_path / "provenance.json"

    prepare_target_dataset.prepare_target_dataset(
        output,
        "owner/target",
        "a" * 40,
        2,
        True,
    )

    provenance = json.loads(output.read_text(encoding="utf-8"))
    assert provenance["target_repo"] == "owner/target"
    assert provenance["target_revision"] == "a" * 40
    assert provenance["target_episode"] == 2
    assert provenance["visual_match_confirmed"] is True
    assert (
        provenance["matched_source_dataset"]
        == "Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi_10fps"
    )
    assert provenance["matched_source_revision"] == "09a0376348f60be89edcbc0eb76c3e26b5f3b094"
    assert provenance["matched_source_episode"] == 170
    assert provenance["selected_frames"] == 17
    assert provenance["selected_content_sha256"] == provenance["content_manifest"]["tree_sha256"]
    validate_target_provenance(output, "owner/target", "a" * 40, 2)

    provenance["target_episode"] = 3
    output.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match this run"):
        validate_target_provenance(output, "owner/target", "a" * 40, 2)


def test_target_preparation_requires_visual_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be confirmed"):
        prepare_target_dataset.prepare_target_dataset(
            tmp_path / "provenance.json",
            "owner/target",
            "a" * 40,
            2,
            False,
        )


def _capture_uv(tmp_path: Path) -> tuple[Path, Path]:
    capture = tmp_path / "uv-args"
    binary = tmp_path / "uv"
    binary.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CAPTURE"\n', encoding="utf-8")
    binary.chmod(0o755)
    return binary, capture


def test_source_training_uses_experiment_m_anchor_and_episode_170(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "experiments/domain_arithmetic_so101/run.sh"
    _, capture = _capture_uv(tmp_path)
    subprocess.run(
        [script, "train-source"],
        check=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE": str(capture),
            "RUN_ROOT": str(tmp_path / "run"),
            "VISUAL_MATCH_CONFIRMED": "1",
        },
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert "--policy.path=Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod" in args
    assert "--policy.pretrained_revision=45f76f173c76c4e002131f8b48e345589a071d0f" in args
    assert "--dataset.repo_id=Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi_10fps" in args
    assert "--dataset.revision=09a0376348f60be89edcbc0eb76c3e26b5f3b094" in args
    assert "--dataset.episodes=[170]" in args


def test_target_training_uses_same_anchor_and_configurable_immutable_episode(tmp_path: Path) -> None:
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
            "TARGET_REPO_ID": "owner/real-target",
            "TARGET_REV": "a" * 40,
            "TARGET_EPISODE": "2",
            "VISUAL_MATCH_CONFIRMED": "1",
        },
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert "--policy.path=Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod" in args
    assert "--policy.pretrained_revision=45f76f173c76c4e002131f8b48e345589a071d0f" in args
    assert "--dataset.repo_id=owner/real-target" in args
    assert f"--dataset.revision={'a' * 40}" in args
    assert "--dataset.episodes=[2]" in args
    assert f"--verify-provenance={run_root / 'target_provenance.json'}" in args
    assert "--dataset.video_backend=pyav" in args
    assert "--num_workers=0" in args
    assert "--batch_size=8" in args
    assert "--accelerator.gradient_accumulation.steps=8" in args


@pytest.mark.parametrize("command", ["prepare-target", "train-target"])
def test_target_commands_reject_mutable_revisions(tmp_path: Path, command: str) -> None:
    script = Path(__file__).parents[2] / "experiments/domain_arithmetic_so101/run.sh"
    result = subprocess.run(
        [script, command],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RUN_ROOT": str(tmp_path / "run"),
            "TARGET_REV": "main",
            "VISUAL_MATCH_CONFIRMED": "1",
        },
    )
    assert result.returncode == 2
    assert "immutable 40-character" in result.stderr


def test_merge_uses_only_experiment_m_run_checkpoints_and_anchor(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "experiments/domain_arithmetic_so101/run.sh"
    _, capture = _capture_uv(tmp_path)
    run_root = tmp_path / "run"
    source = run_root / "source_finetune/checkpoints/last/pretrained_model"
    target = run_root / "target_finetune/checkpoints/last/pretrained_model"
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
            "SOURCE_CHECKPOINT": str(tmp_path / "old-source"),
            "TARGET_CHECKPOINT": str(tmp_path / "old-target"),
        },
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert args.count(f"--source={source}") == 2
    assert args.count(f"--target={target}") == 2
    assert args.count("--base=Cache-SCA/smolVLA-IsaacLab-Multi-Task-8epoch-mod") == 2
    assert args.count("--base-revision=45f76f173c76c4e002131f8b48e345589a071d0f") == 2
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
