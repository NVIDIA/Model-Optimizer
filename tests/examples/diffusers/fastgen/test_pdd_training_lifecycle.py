# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic direct-update, committed-cursor, and strict PDD resume evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import shutil
import sys

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))
if str(pathlib.Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

from fastgen_data.replayable_sampler import ReplayableBatchSampler
from pdd_checkpoint import (
    PDDCheckpointManager,
    build_pdd_checkpoint_identity,
    resolve_pdd_training_checkpoint,
)
from pdd_recipe import initialize_pdd_distributed
from pdd_test_utils import SamplerDataset, build_toy_lifecycle, make_batch, ordered_id_sha256
from pdd_training import prepare_qwen_pdd_batch
from verify_readonly_automodel import snapshot_installed_distribution


def _released_sampler(sample_ids: tuple[str, ...]) -> ReplayableBatchSampler:
    sampler_module = pytest.importorskip("nemo_automodel.components.datasets.diffusion.sampler")
    dataset = SamplerDataset(sample_ids)
    sampler = sampler_module.SequentialBucketSampler(
        dataset,
        base_batch_size=1,
        base_resolution=(64, 64),
        drop_last=True,
        shuffle_buckets=True,
        shuffle_within_bucket=True,
        dynamic_batch_size=False,
        seed=31,
        num_replicas=1,
        rank=0,
    )
    return ReplayableBatchSampler(sampler)


def _run_next(lifecycle, sampler):
    sample_ids = sampler.expected_next_sample_ids()
    assert sample_ids
    diagnostics = lifecycle.trainer.train_step(make_batch(sample_ids))
    lifecycle.scheduler.step()
    sampler.commit(sample_ids)
    if sampler.remaining_batches == 0:
        sampler.set_epoch(sampler.epoch + 1)
    return sample_ids, diagnostics


def _checkpointer(lifecycle, checkpoint_dir):
    checkpoint_module = pytest.importorskip("nemo_automodel.components.checkpoint.config")
    config = checkpoint_module.CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(checkpoint_dir),
        model_save_format="torch_save",
        model_repo_id="synthetic-pdd-toy",
        save_consolidated=False,
        is_peft=False,
        model_state_dict_keys=list(lifecycle.student.state_dict()),
    )
    return config.build(dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)


def _identity(lifecycle, scheduler, sample_ids):
    return build_pdd_checkpoint_identity(
        metadata=lifecycle.metadata,
        model_id="synthetic-pdd-toy",
        model_revision=None,
        guidance_scale=None,
        guidance_rescale=1.0,
        guidance_eps=1e-5,
        automodel_snapshot=snapshot_installed_distribution(),
        ordered_train_id_sha256=ordered_id_sha256(sample_ids),
        ordered_heldout_id_sha256="1" * 64,
        dataset_snapshot_sha256="2" * 64,
        local_batch_size=1,
        grad_accumulation_steps=1,
        training_seed=1234,
        validation_seed=2026,
        validation_every_steps=100,
        max_grad_norm=0.5,
        zero_grad_warmup_steps=0,
        activation_checkpointing=False,
        dtype="float32",
        optimizer=lifecycle.optimizer,
        scheduler=scheduler,
    )


def _manager(root, lifecycle, sampler, rng):
    checkpointer = _checkpointer(lifecycle, root)
    manager = PDDCheckpointManager(
        root=root,
        checkpointer=checkpointer,
        model=lifecycle.student,
        optimizer=lifecycle.optimizer,
        scheduler=lifecycle.scheduler,
        trainer=lifecycle.trainer,
        sampler=sampler,
        rng=rng,
        identity=_identity(lifecycle, lifecycle.scheduler, tuple(f"sample-{i}" for i in range(8))),
    )
    return manager, checkpointer


def _optimizer_state_by_name(lifecycle):
    names = {parameter: name for name, parameter in lifecycle.student.named_parameters()}
    return {
        names[parameter]: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
        for parameter, state in lifecycle.optimizer.state.items()
    }


def _refresh_complete_marker(checkpoint: pathlib.Path) -> None:
    manifest_path = checkpoint / "manifest.json"
    marker = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (checkpoint / "COMPLETE").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")


def test_replayable_sampler_commits_consumed_batches_not_prefetch() -> None:
    sample_ids = tuple(f"sample-{index}" for index in range(8))
    sampler = _released_sampler(sample_ids)
    first_expected = sampler.expected_next_sample_ids()
    iterator = iter(sampler)
    next(iterator)
    next(iterator)

    assert sampler.committed_batches == 0
    assert sampler.expected_next_sample_ids() == first_expected
    sampler.commit(first_expected)
    state = sampler.state_dict()

    restored = _released_sampler(sample_ids)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    next_indices = next(iter(restored))
    assert tuple(restored.dataset.metadata[index]["sample_id"] for index in next_indices) == tuple(
        state["next_sample_ids"]
    )
    bad_hash = dict(state, plan_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="plan hash"):
        _released_sampler(sample_ids).load_state_dict(bad_hash)
    bad_ids = dict(state, next_sample_ids=["wrong-id"])
    with pytest.raises(RuntimeError, match="next sample IDs"):
        _released_sampler(sample_ids).load_state_dict(bad_ids)


def test_qwen_batch_preparation_preserves_ids_masks_and_negative_condition() -> None:
    batch = {
        "image_latents": torch.ones(2, 3, 4, 4),
        "text_embeddings": torch.ones(2, 5, 6),
        "text_embeddings_mask": torch.ones(2, 5, dtype=torch.bool),
        "negative_text_embeddings": torch.zeros(5, 6),
        "negative_text_embeddings_mask": torch.ones(5, dtype=torch.bool),
        "metadata": {"sample_ids": ["qwen-a", "qwen-b"]},
    }

    prepared = prepare_qwen_pdd_batch(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        require_negative_condition=True,
    )

    assert prepared.sample_ids == ("qwen-a", "qwen-b")
    assert prepared.valid_mask == (True, True)
    assert prepared.data.shape == (2, 3, 4, 4)
    assert prepared.condition[0].shape == (2, 5, 6)
    assert prepared.negative_condition is not None
    assert prepared.negative_condition[0].shape == (2, 5, 6)

    without_negative = dict(batch)
    without_negative.pop("negative_text_embeddings")
    without_negative.pop("negative_text_embeddings_mask")
    with pytest.raises(ValueError, match="requires negative prompt conditioning"):
        prepare_qwen_pdd_batch(
            without_negative,
            device=torch.device("cpu"),
            dtype=torch.float32,
            require_negative_condition=True,
        )


def test_two_direct_updates_have_finite_gradients_updates_and_targeted_coverage() -> None:
    lifecycle = build_toy_lifecycle(weight_decay=0.02)
    first_batch = make_batch(("a", "b"))
    before = [parameter.detach().clone() for parameter in lifecycle.student.parameters()]
    first = lifecycle.trainer.train_step(
        first_batch,
        noise=torch.tensor([[0.25, -0.5, 1.0], [-0.25, 0.5, -1.0]]),
        n=torch.tensor([0, 1]),
        k=torch.tensor([1, 3]),
    )
    lifecycle.scheduler.step()
    actual_update = math.sqrt(
        sum(
            (parameter.detach() - saved).double().square().sum().item()
            for parameter, saved in zip(lifecycle.student.parameters(), before)
        )
    )
    before_norm = math.sqrt(sum(saved.double().square().sum().item() for saved in before))

    assert math.isfinite(first.loss) and first.loss > 0
    assert math.isfinite(first.grad_norm) and first.grad_norm > 0
    assert first.pdd_projection_update_ratio is not None
    assert first.pdd_projection_update_ratio > 0
    assert math.isfinite(first.student_teacher_velocity_rms_ratio)
    assert first.student_teacher_velocity_rms_ratio >= 0
    assert first.student_adamw_nominal_update_ratio == pytest.approx(
        actual_update / before_norm,
        rel=2e-5,
        abs=1e-8,
    )
    assert all(parameter.grad is None for parameter in lifecycle.teacher.parameters())

    second = lifecycle.trainer.train_step(
        make_batch(("c", "d"), offset=0.25),
        noise=torch.tensor([[0.75, 0.0, -0.5], [-0.75, 0.0, 0.5]]),
        n=torch.tensor([2, 3]),
        k=torch.tensor([2, 3]),
    )
    lifecycle.scheduler.step()
    assert second.completed_step == 2
    lifecycle.trainer.coverage.require_pairs([(0, 1), (1, 3), (2, 2), (3, 3)])


def test_training_hard_aborts_for_teacher_gradient_zero_gradient_and_missing_coverage() -> None:
    teacher_gradient = build_toy_lifecycle()
    teacher_gradient.teacher.scale.grad = torch.ones_like(teacher_gradient.teacher.scale)
    with pytest.raises(RuntimeError, match="teacher received a gradient"):
        teacher_gradient.trainer.train_step(make_batch(("teacher-grad",)))

    zero_gradient = build_toy_lifecycle(zero_student_gradient=True, weight_decay=0.0)
    first = zero_gradient.trainer.train_step(make_batch(("zero-1",)))
    assert first.grad_norm == 0.0
    with pytest.raises(RuntimeError, match="zero for two consecutive"):
        zero_gradient.trainer.train_step(make_batch(("zero-2",)))

    with pytest.raises(RuntimeError, match="did not cover"):
        zero_gradient.trainer.coverage.require_pairs([(3, 3)])

    nonfinite = build_toy_lifecycle()
    bad_batch = make_batch(("nan",))
    bad_batch.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        nonfinite.trainer.train_step(bad_batch)

    nonfinite_gradient = build_toy_lifecycle()
    handle = nonfinite_gradient.projection.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    with pytest.raises(FloatingPointError, match="gradient became non-finite"):
        nonfinite_gradient.trainer.train_step(make_batch(("inf-gradient",)))
    handle.remove()

    invalid_support = build_toy_lifecycle()
    with pytest.raises(RuntimeError, match="k must satisfy"):
        invalid_support.trainer.train_step(
            make_batch(("invalid-support",)),
            n=torch.tensor([2]),
            k=torch.tensor([1]),
        )

    nonfinite_update = build_toy_lifecycle()
    nonfinite_update.optimizer.param_groups[0]["eps"] = float("nan")
    with pytest.raises(FloatingPointError, match="parameter update became non-finite"):
        nonfinite_update.trainer.train_step(
            make_batch(("inf-update",)),
            measure_updates=False,
        )


def test_stock_dcp_resume_recovers_rng_scheduler_cursor_and_next_loss(tmp_path) -> None:
    pytest.importorskip("nemo_automodel")
    rng_module = pytest.importorskip("nemo_automodel.components.training.rng")
    if not torch.distributed.is_initialized():
        initialize_pdd_distributed(backend="gloo", timeout_minutes=1)
    sample_ids = tuple(f"sample-{index}" for index in range(8))

    source = build_toy_lifecycle()
    source_sampler = _released_sampler(sample_ids)
    source_rng = rng_module.StatefulRNG(1234, ranked=True)
    source_manager, source_checkpointer = _manager(
        tmp_path / "checkpoints", source, source_sampler, source_rng
    )
    _run_next(source, source_sampler)
    _run_next(source, source_sampler)
    checkpoint = source_manager.save()
    assert checkpoint.name == "step_00000002"
    assert source_manager.identity["data"]["dataset_snapshot_sha256"] == "2" * 64
    assert source_manager.identity["training"] == {
        "seed": 1234,
        "validation_seed": 2026,
        "validation_every_steps": 100,
        "max_grad_norm": 0.5,
        "zero_grad_warmup_steps": 0,
        "activation_checkpointing": False,
    }
    expected_next_ids, expected_next = _run_next(source, source_sampler)
    expected_model = copy.deepcopy(source.student.state_dict())
    expected_optimizer = _optimizer_state_by_name(source)

    destination = build_toy_lifecycle()
    destination_sampler = _released_sampler(sample_ids)
    destination_rng = rng_module.StatefulRNG(9999, ranked=True)
    destination_manager, destination_checkpointer = _manager(
        tmp_path / "checkpoints",
        destination,
        destination_sampler,
        destination_rng,
    )
    resume = destination_manager.load("LATEST")
    assert resume is not None
    assert resume.completed_steps == 2
    assert resume.sample_slots_consumed == 2
    assert resume.expected_next_sample_ids == expected_next_ids
    resume.verify_first_batch(destination_sampler.expected_next_sample_ids())
    with pytest.raises(RuntimeError, match="first resumed sample IDs"):
        resume.verify_first_batch(("wrong-id",))
    actual_ids, actual_next = _run_next(destination, destination_sampler)

    assert actual_ids == expected_next_ids
    assert actual_next.n == expected_next.n
    assert actual_next.k == expected_next.k
    assert actual_next.loss == expected_next.loss
    assert actual_next.learning_rate == expected_next.learning_rate
    for name, tensor in destination.student.state_dict().items():
        torch.testing.assert_close(tensor, expected_model[name], rtol=0, atol=0)
    actual_optimizer = _optimizer_state_by_name(destination)
    assert actual_optimizer.keys() == expected_optimizer.keys()
    for name in actual_optimizer:
        for key, actual in actual_optimizer[name].items():
            expected = expected_optimizer[name][key]
            if isinstance(actual, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert actual == expected

    resumed_checkpoint = destination_manager.save()
    assert resumed_checkpoint.name == "step_00000003"
    third = build_toy_lifecycle()
    third_sampler = _released_sampler(sample_ids)
    third_rng = rng_module.StatefulRNG(7, ranked=True)
    third_manager, third_checkpointer = _manager(
        tmp_path / "checkpoints", third, third_sampler, third_rng
    )
    second_resume = third_manager.load("LATEST")
    assert second_resume is not None
    assert second_resume.completed_steps == 3
    assert second_resume.expected_next_sample_ids == destination_sampler.expected_next_sample_ids()

    inventory = {path.name.lower() for path in resumed_checkpoint.rglob("*")}
    assert not any(
        token in name
        for name in inventory
        for token in ("fake_score", "discriminator", "ema", "r1", "gan")
    )

    incomplete = tmp_path / "checkpoints" / "step_99999998"
    incomplete.mkdir()
    (tmp_path / "checkpoints" / "LATEST").write_text(incomplete.name + "\n")
    assert third_manager.resolve("LATEST") == resumed_checkpoint.resolve()
    with pytest.raises(RuntimeError, match="incomplete"):
        third_manager.resolve(incomplete.name)

    mismatched = tmp_path / "checkpoints" / "step_99999999"
    shutil.copytree(resumed_checkpoint, mismatched)
    manifest_path = mismatched / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["model"]["id"] = "different-model"
    manifest["completed_steps"] = 99999999
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _refresh_complete_marker(mismatched)
    (tmp_path / "checkpoints" / "LATEST").write_text(mismatched.name + "\n")
    assert third_manager.resolve("LATEST") == resumed_checkpoint.resolve()
    selected, selected_manifest = resolve_pdd_training_checkpoint(
        tmp_path / "checkpoints",
        "LATEST",
        expected_world_size=1,
        expected_identity=third_manager.identity,
    )
    assert selected == resumed_checkpoint.resolve()
    assert selected_manifest["identity"] == third_manager.identity
    with pytest.raises(RuntimeError, match="identity"):
        third_manager.resolve(mismatched.name)

    missing_model = tmp_path / "checkpoints" / "step_99999996"
    shutil.copytree(resumed_checkpoint, missing_model)
    model_payload = next(
        path for path in (missing_model / "model").iterdir() if path.name != ".metadata"
    )
    model_payload.unlink()
    (tmp_path / "checkpoints" / "LATEST").write_text(missing_model.name + "\n")
    assert third_manager.resolve("LATEST") == resumed_checkpoint.resolve()
    with pytest.raises(RuntimeError, match="DCP"):
        third_manager.resolve(missing_model.name)

    corrupt_optimizer = tmp_path / "checkpoints" / "step_99999997"
    shutil.copytree(resumed_checkpoint, corrupt_optimizer)
    optimizer_payload = next(
        path for path in (corrupt_optimizer / "optim").iterdir() if path.name != ".metadata"
    )
    with optimizer_payload.open("ab") as stream:
        stream.write(b"corrupt")
    (tmp_path / "checkpoints" / "LATEST").write_text(corrupt_optimizer.name + "\n")
    assert third_manager.resolve("LATEST") == resumed_checkpoint.resolve()
    with pytest.raises(RuntimeError, match="DCP"):
        third_manager.resolve(corrupt_optimizer.name)

    step_mismatch = tmp_path / "checkpoints" / "step_00000004"
    shutil.copytree(resumed_checkpoint, step_mismatch)
    step_manifest_path = step_mismatch / "manifest.json"
    step_manifest = json.loads(step_manifest_path.read_text())
    step_manifest["completed_steps"] = 4
    step_manifest_path.write_text(json.dumps(step_manifest, indent=2, sort_keys=True) + "\n")
    trainer_state_path = step_mismatch / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text())
    trainer_state["completed_steps"] = 4
    trainer_state_path.write_text(json.dumps(trainer_state, indent=2, sort_keys=True) + "\n")
    _refresh_complete_marker(step_mismatch)
    with pytest.raises(RuntimeError, match="trainer step"):
        third_manager.load(step_mismatch.name)

    lr_mismatch = tmp_path / "checkpoints" / "step_00000005"
    shutil.copytree(resumed_checkpoint, lr_mismatch)
    lr_manifest_path = lr_mismatch / "manifest.json"
    lr_manifest = json.loads(lr_manifest_path.read_text())
    lr_manifest["learning_rates"] = [0.123]
    lr_manifest_path.write_text(json.dumps(lr_manifest, indent=2, sort_keys=True) + "\n")
    lr_trainer_path = lr_mismatch / "trainer_state.json"
    lr_trainer = json.loads(lr_trainer_path.read_text())
    lr_trainer["learning_rates"] = [0.123]
    lr_trainer_path.write_text(json.dumps(lr_trainer, indent=2, sort_keys=True) + "\n")
    _refresh_complete_marker(lr_mismatch)
    with pytest.raises(RuntimeError, match="learning rate"):
        third_manager.load(lr_mismatch.name)

    cursor_mismatch = tmp_path / "checkpoints" / "step_00000006"
    shutil.copytree(resumed_checkpoint, cursor_mismatch)
    sampler_path = cursor_mismatch / "sampler" / "sampler_dp_rank_0.pt"
    sampler_state = torch.load(sampler_path, weights_only=False)
    sampler_state["plan_sha256"] = "0" * 64
    torch.save(sampler_state, sampler_path)
    cursor_manifest_path = cursor_mismatch / "manifest.json"
    cursor_manifest = json.loads(cursor_manifest_path.read_text())
    cursor_manifest["sidecar_sha256"]["sampler/sampler_dp_rank_0.pt"] = hashlib.sha256(
        sampler_path.read_bytes()
    ).hexdigest()
    cursor_manifest_path.write_text(json.dumps(cursor_manifest, indent=2, sort_keys=True) + "\n")
    _refresh_complete_marker(cursor_mismatch)
    with pytest.raises(RuntimeError, match="plan hash"):
        third_manager.load(cursor_mismatch.name)

    source_checkpointer.close()
    destination_checkpointer.close()
    third_checkpointer.close()
