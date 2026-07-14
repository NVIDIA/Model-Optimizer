# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic tests for the full-Qwen PDD smoke evidence contract."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HARNESS = _REPO_ROOT / "tests" / "gpu" / "torch" / "fastgen" / "pdd_qwen_operability_smoke.py"
_SPEC = importlib.util.spec_from_file_location("pdd_qwen_operability_smoke", _HARNESS)
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def _stage_result(stage: str) -> dict:
    step = 1 if stage == "train-one" else 2
    sample_ids = [f"synthetic-pdd-smoke-step-{step}-rank-{rank}" for rank in range(2)]
    learning_rate = 2.0e-5
    return {
        "schema_version": 1,
        "record_type": "pdd_qwen_smoke_stage",
        "stage": stage,
        "pid": 100 + step,
        "world_size": 2,
        "model": {
            "id": "Qwen/Qwen-Image",
            "revision": "75e0b4be04f60ec59a75f475837eced720f823b6",
            "dtype": "bfloat16",
        },
        "pdd": {
            "grid_size": 128,
            "grid_max_t": 0.999,
            "flow_shift": 5.0,
            "block_size_min": 4,
            "block_size_max": 64,
            "teacher_integrator": "euler",
            "guidance_scale": 4.0,
            "guidance_rescale": 1.0,
            "guidance_eps": 1e-5,
        },
        "source": {"commit": "1" * 40, "dirty": False},
        "config_sha256": "2" * 64,
        "automodel": {
            "distribution": "nemo_automodel",
            "version": "0.5.0",
            "package_tree_sha256": smoke._AUTOMODEL_TREE_SHA256,
            "wheel_sha256": smoke._AUTOMODEL_WHEEL_SHA256,
            "runtime_versions": {"diffusers": "0.38.0"},
        },
        "gpu": {
            "names": ["GPU", "GPU"],
            "total_memory_bytes": [80_000_000_000, 80_000_000_000],
            "host_available_bytes": [500_000_000_000, 500_000_000_000],
            "allocated_before_step_bytes": [30_000_000_000, 30_000_000_000],
            "peak_memory_bytes": [40_000_000_000, 40_000_000_000],
            "student_parameter_bytes": 40_000_000_000,
            "teacher_parameter_bytes": 40_000_000_000,
            "step_seconds": 12.5,
        },
        "pair": {"n": 0 if step == 1 else 124, "k": 63 if step == 1 else 127},
        "sample_ids": sample_ids,
        "diagnostics": {
            "completed_step": step,
            "loss": 0.5,
            "grad_norm": 1.25,
            "student_adamw_nominal_update_ratio": 1e-4,
            "pdd_projection_update_ratio": 2e-4,
            "learning_rate": learning_rate,
            "student_velocity_rms": 0.75,
            "teacher_velocity_rms": 0.8,
            "student_teacher_velocity_rms_ratio": 0.9375,
            "reconstructed_state_rms": 1.1,
        },
        "teacher_calls_per_rank": [2, 2],
        "checkpoint": {
            "path": f"checkpoints/step_{step:08d}",
            "manifest_sha256": "5" * 64,
            "completed_steps": step,
            "parent_checkpoint": None if step == 1 else "step_00000001",
        },
        "resume": None
        if step == 1
        else {
            "selected_checkpoint": "step_00000001",
            "completed_steps": 1,
            "parent_checkpoint": None,
            "first_sample_ids": sample_ids,
            "learning_rate": learning_rate,
        },
    }


def test_gpu_harness_uses_shared_unambiguous_ordered_id_hash() -> None:
    assert smoke._ordered_id_sha256(("a", "b")) == (
        "8cf774af4e8509811c2d4bc2adec6b852e4c614f9d8d833924502ead7c0689d7"
    )
    assert smoke._ordered_id_sha256(("a\nb",)) == (
        "41e07cc133e8a85fc4a08e60a38c223f3c24dbca80312d106f251e533254eedf"
    )
    source = _HARNESS.read_text()
    assert "modelopt-pdd-ordered-{split}-ids-v1" not in source
    assert 'digest.update(b"\\n")' not in source


def test_gpu_harness_checkpoint_identity_uses_exact_shared_hashes() -> None:
    from modelopt.torch.fastgen import PDDLayerSpec, PDDMetadata

    train_ids = smoke._training_sample_ids(2)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2.0e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    setup = SimpleNamespace(
        metadata=PDDMetadata(
            grid_size=128,
            grid_max_t=0.999,
            flow_shift=5.0,
            block_size_min=4,
            block_size_max=64,
            inference_blocks=(32, 32, 32, 32),
            teacher_integrator="euler",
            layer_spec=PDDLayerSpec(
                projection_path="transformer.proj_out",
                head_layout="channel_major",
            ),
            projection_in_features=3072,
            projection_out_features=64,
            projection_bias=True,
        ),
        automodel_snapshot={
            "distribution": "nemo_automodel",
            "version": "0.5.0",
            "package_tree_sha256": "a" * 64,
            "wheel_sha256": "b" * 64,
            "runtime_versions": {"diffusers": "0.38.0"},
        },
        optimizer=optimizer,
    )
    config = SimpleNamespace(
        model_id="Qwen/Qwen-Image",
        model_revision="75e0b4be04f60ec59a75f475837eced720f823b6",
        pdd=SimpleNamespace(guidance_scale=4.0),
        guidance=SimpleNamespace(rescale=1.0, eps=1e-5),
        training=SimpleNamespace(
            seed=17,
            validation_seed=29,
            validation_every_steps=1000,
            max_grad_norm=1.0,
            zero_grad_warmup_steps=0,
        ),
        parallel=SimpleNamespace(activation_checkpointing=False),
    )
    sampler = SimpleNamespace(
        dataset=SimpleNamespace(metadata=[{"sample_id": sample_id} for sample_id in train_ids])
    )
    raw = {"pdd": {"grid_size": 128, "block_size_max": 64}}

    identity = smoke._identity(
        setup=setup,
        training=SimpleNamespace(scheduler=scheduler),
        config=config,
        sampler=sampler,
        raw=raw,
    )

    assert identity["data"] == {
        "ordered_train_id_sha256": (
            "4df732c492d043d5b0ea3549bcc80dbb847369021d0d3f242cd60386d1e94313"
        ),
        "ordered_heldout_id_sha256": (
            "38486bc077b6bd9b06a82167399e720f6c8dc70329dd0ce5fa23a92e4f30c198"
        ),
        "dataset_snapshot_sha256": smoke._canonical_sha256(
            {"domain": "modelopt-pdd-synthetic-smoke-v1", "config": raw["pdd"]}
        ),
        "local_batch_size": 1,
        "grad_accumulation_steps": 1,
    }


def _automodel_snapshot_fixture() -> tuple[dict, dict]:
    records = []
    tree = hashlib.sha256()
    for index in range(smoke._AUTOMODEL_PACKAGE_FILE_COUNT):
        path = f"nemo_automodel/file_{index:03d}.py"
        digest = hashlib.sha256(f"file-{index}".encode()).hexdigest()
        size = index + 1
        tree.update(path.encode())
        tree.update(b"\0")
        tree.update(digest.encode())
        tree.update(b"\0")
        tree.update(str(size).encode())
        tree.update(b"\n")
        records.append({"path": path, "sha256": digest, "size": size})
    automodel = {
        "distribution": "nemo_automodel",
        "version": "0.5.0",
        "package_tree_sha256": tree.hexdigest(),
        "wheel_sha256": smoke._AUTOMODEL_WHEEL_SHA256,
        "runtime_versions": {"diffusers": "0.38.0"},
    }
    snapshot = {
        **automodel,
        "files": records,
        "import_origin": "/opt/pdd/site-packages/nemo_automodel/__init__.py",
        "package_file_count": smoke._AUTOMODEL_PACKAGE_FILE_COUNT,
        "release_commit": smoke._AUTOMODEL_RELEASE_COMMIT,
        "release_tag": smoke._AUTOMODEL_RELEASE_TAG,
        "root": "/opt/pdd/site-packages",
        "wheel": smoke._AUTOMODEL_WHEEL,
    }
    return snapshot, automodel


def _checkpoint_identity(stage: dict) -> dict:
    return {
        "schema_version": 1,
        "model": stage["model"],
        "pdd_metadata": {
            "schema_version": 1,
            "grid_size": 128,
            "grid_max_t": 0.999,
            "flow_shift": 5.0,
            "block_size_min": 4,
            "block_size_max": 64,
            "inference_blocks": [32, 32, 32, 32],
            "teacher_integrator": "euler",
            "layer_spec": {
                "projection_path": "transformer.proj_out",
                "head_layout": "channel_major",
                "output_channels": None,
            },
            "base_projection": {"in_features": 3072, "out_features": 64, "bias": True},
        },
        "guidance": {"scale": 4.0, "rescale": 1.0, "eps": 1e-5},
        "automodel": stage["automodel"],
        "data": {},
        "topology": {"world_size": 2, "pure_data_parallel": True},
        "training": {},
        "optimizer": {},
        "scheduler": {},
    }


def _bundle_link_fixture() -> tuple[dict, dict, dict, dict, dict, dict]:
    snapshot, automodel = _automodel_snapshot_fixture()
    stage1 = _stage_result("train-one")
    stage2 = _stage_result("resume-one")
    stage1["automodel"] = copy.deepcopy(automodel)
    stage2["automodel"] = copy.deepcopy(automodel)
    identity = _checkpoint_identity(stage1)
    manifest1 = {"identity": copy.deepcopy(identity)}
    manifest2 = {"identity": copy.deepcopy(identity)}
    export = {
        "identity": copy.deepcopy(identity),
        "modelopt_source": copy.deepcopy(stage2["source"]),
        "source_checkpoint": {
            "name": "step_00000002",
            "manifest_sha256": stage2["checkpoint"]["manifest_sha256"],
            "completed_steps": 2,
        },
    }
    return stage1, stage2, manifest1, manifest2, export, snapshot


def test_training_stage_contract_accepts_only_exact_canonical_chain() -> None:
    stage1 = _stage_result("train-one")
    stage2 = _stage_result("resume-one")

    smoke.validate_stage_result(stage1, stage="train-one")
    smoke.validate_stage_result(stage2, stage="resume-one")

    wrong_pair = copy.deepcopy(stage2)
    wrong_pair["pair"] = {"n": 120, "k": 127}
    with pytest.raises(ValueError, match="support pair"):
        smoke.validate_stage_result(wrong_pair, stage="resume-one")

    zero_update = copy.deepcopy(stage1)
    zero_update["diagnostics"]["pdd_projection_update_ratio"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        smoke.validate_stage_result(zero_update, stage="train-one")

    stale_resume = copy.deepcopy(stage2)
    stale_resume["resume"]["selected_checkpoint"] = "step_00000000"
    with pytest.raises(ValueError, match="resume evidence"):
        smoke.validate_stage_result(stale_resume, stage="resume-one")


def test_inference_contract_authenticates_exact_pdd4_counters_and_png(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nnonempty-hashed-test-fixture")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    result = {
        "schema_version": 1,
        "record_type": "pdd_inference",
        "condition": "pdd_4",
        "schedule": "pdd-4",
        "blocks": [32, 32, 32, 32],
        "height": 1024,
        "width": 1024,
        "scheduler_steps": 4,
        "actual_transformer_invocations": 4,
        "batch_normalized_transformer_evaluations": 4,
        "latency_seconds": 1.5,
        "output": {"path": "image.png", "sha256": digest},
    }

    smoke.validate_inference_result(result, root=tmp_path)

    wrong_calls = copy.deepcopy(result)
    wrong_calls["actual_transformer_invocations"] = 5
    with pytest.raises(ValueError, match="compute counters"):
        smoke.validate_inference_result(wrong_calls, root=tmp_path)

    wrong_hash = copy.deepcopy(result)
    wrong_hash["output"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="PNG hash"):
        smoke.validate_inference_result(wrong_hash, root=tmp_path)

    reduced_resolution = copy.deepcopy(result)
    reduced_resolution["height"] = 512
    with pytest.raises(ValueError, match="1024x1024"):
        smoke.validate_inference_result(reduced_resolution, root=tmp_path)


def test_bundle_links_reject_cross_run_artifact_splicing() -> None:
    stage1, stage2, manifest1, manifest2, export, snapshot = _bundle_link_fixture()

    smoke._validate_bundle_links(
        stage1=stage1,
        stage2=stage2,
        manifest1=manifest1,
        manifest2=manifest2,
        export_manifest=export,
        automodel_snapshot=snapshot,
    )

    wrong_checkpoint = copy.deepcopy(manifest2)
    wrong_checkpoint["identity"]["model"]["revision"] = "a" * 40
    with pytest.raises(ValueError, match="identities differ"):
        smoke._validate_bundle_links(
            stage1=stage1,
            stage2=stage2,
            manifest1=manifest1,
            manifest2=wrong_checkpoint,
            export_manifest=export,
            automodel_snapshot=snapshot,
        )

    wrong_export_identity = copy.deepcopy(export)
    wrong_export_identity["identity"]["data"] = {"spliced": True}
    with pytest.raises(ValueError, match="export identity"):
        smoke._validate_bundle_links(
            stage1=stage1,
            stage2=stage2,
            manifest1=manifest1,
            manifest2=manifest2,
            export_manifest=wrong_export_identity,
            automodel_snapshot=snapshot,
        )

    wrong_export_checkpoint = copy.deepcopy(export)
    wrong_export_checkpoint["source_checkpoint"]["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="exact step-2"):
        smoke._validate_bundle_links(
            stage1=stage1,
            stage2=stage2,
            manifest1=manifest1,
            manifest2=manifest2,
            export_manifest=wrong_export_checkpoint,
            automodel_snapshot=snapshot,
        )

    wrong_export_source = copy.deepcopy(export)
    wrong_export_source["modelopt_source"]["commit"] = "e" * 40
    with pytest.raises(ValueError, match="training source"):
        smoke._validate_bundle_links(
            stage1=stage1,
            stage2=stage2,
            manifest1=manifest1,
            manifest2=manifest2,
            export_manifest=wrong_export_source,
            automodel_snapshot=snapshot,
        )

    with pytest.raises(ValueError, match="AutoModel snapshot"):
        smoke._validate_bundle_links(
            stage1=stage1,
            stage2=stage2,
            manifest1=manifest1,
            manifest2=manifest2,
            export_manifest=export,
            automodel_snapshot={},
        )

    corrupt_snapshot = copy.deepcopy(snapshot)
    corrupt_snapshot["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tree digest"):
        smoke._validate_bundle_links(
            stage1=stage1,
            stage2=stage2,
            manifest1=manifest1,
            manifest2=manifest2,
            export_manifest=export,
            automodel_snapshot=corrupt_snapshot,
        )

    float_count_snapshot = copy.deepcopy(snapshot)
    float_count_snapshot["package_file_count"] = float(smoke._AUTOMODEL_PACKAGE_FILE_COUNT)
    with pytest.raises(ValueError, match="release identity"):
        smoke._validate_bundle_links(
            stage1=stage1,
            stage2=stage2,
            manifest1=manifest1,
            manifest2=manifest2,
            export_manifest=export,
            automodel_snapshot=float_count_snapshot,
        )


def test_automodel_snapshots_require_identical_bytes_and_no_symlinks(tmp_path: Path) -> None:
    snapshot, automodel = _automodel_snapshot_fixture()
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    payload = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    before.write_text(payload)
    after.write_text(payload)
    assert (
        smoke._load_matching_automodel_snapshots(
            before,
            after,
            expected_automodel=automodel,
        )
        == snapshot
    )

    after.write_text(json.dumps(snapshot, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        smoke._load_matching_automodel_snapshots(
            before,
            after,
            expected_automodel=automodel,
        )

    symlink = tmp_path / "before-symlink.json"
    symlink.symlink_to(before)
    with pytest.raises(ValueError, match="symlink"):
        smoke._load_matching_automodel_snapshots(
            symlink,
            before,
            expected_automodel=automodel,
        )


def test_smoke_artifact_paths_reject_symlinked_stage_inference_and_export(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "artifact.json"
    target.write_text("{}")
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "stage1.json").symlink_to(target)
    inference = run_root / "inference"
    inference.mkdir()
    (inference / "pdd4.json").symlink_to(target)
    (run_root / "export").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        smoke._relative_regular_file(run_root, "stage1.json", name="stage")
    with pytest.raises(ValueError, match="symlink"):
        smoke._relative_regular_file(run_root, "inference/pdd4.json", name="inference")
    with pytest.raises(ValueError, match="symlink"):
        smoke._regular_directory(run_root / "export", name="export")

    target_parent = tmp_path / "target-parent"
    target_parent.mkdir()
    symlinked_parent = tmp_path / "symlinked-parent"
    symlinked_parent.symlink_to(target_parent, target_is_directory=True)
    requested_child = symlinked_parent / "must-not-be-created"
    with pytest.raises(ValueError, match="symlink"):
        smoke._create_run_root(requested_child)
    assert not (target_parent / requested_child.name).exists()
