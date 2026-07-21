# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for direct updates, committed cursors, and strict PDD resume."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))
if str(pathlib.Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

from fastgen_data.replayable_sampler import ReplayableBatchSampler
from pdd.checkpoint import (
    PDDCheckpointManager,
    build_pdd_checkpoint_identity,
    resolve_pdd_training_checkpoint,
)
from pdd.recipe import PDDDiffusionRecipe, initialize_pdd_distributed
from pdd.training import prepare_qwen_pdd_batch
from pdd_test_utils import SamplerDataset, build_toy_lifecycle, make_batch, ordered_id_sha256

from modelopt.torch.fastgen.plugins.qwen_image_pdd import QWEN_IMAGE_PDD_EXECUTION


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


def _identity(
    lifecycle,
    scheduler,
    sample_ids,
    *,
    qwen_image_execution=QWEN_IMAGE_PDD_EXECUTION,
):
    return build_pdd_checkpoint_identity(
        qwen_image_execution=qwen_image_execution,
        metadata=lifecycle.metadata,
        model_id="synthetic-pdd-toy",
        model_revision="a" * 40,
        guidance_scale=None,
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


def test_checkpoint_identity_rejects_unbound_qwen_execution() -> None:
    lifecycle = build_toy_lifecycle()
    with pytest.raises(ValueError, match="qwen_image_execution"):
        _identity(
            lifecycle,
            lifecycle.scheduler,
            ("sample-0",),
            qwen_image_execution="canonical_diffusers",
        )


class _StepSchedulerStub:
    def __init__(self, trainer, sampler) -> None:
        self.trainer = trainer
        self.sampler = sampler
        self.loaded_state = None

    def state_dict(self):
        return {"step": self.trainer.completed_steps, "epoch": self.sampler.epoch}

    def load_state_dict(self, state):
        self.loaded_state = dict(state)


class _RecordingCheckpointManager(PDDCheckpointManager):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_calls = []

    def save(self):
        self.save_calls.append(
            {
                "live_step": self.step_scheduler.step,
                "serialized_step": self.step_scheduler.state_dict()["step"],
                "trainer_step": self.trainer.completed_steps,
                "live_epoch": self.step_scheduler.epoch,
                "sampler_epoch": self.sampler.epoch,
            }
        )
        return super().save()


def _manager(root, lifecycle, sampler, rng):
    checkpointer = _checkpointer(lifecycle, root)
    step_scheduler = _StepSchedulerStub(lifecycle.trainer, sampler)
    manager = PDDCheckpointManager(
        root=root,
        checkpointer=checkpointer,
        model=lifecycle.student,
        optimizer=lifecycle.optimizer,
        scheduler=lifecycle.scheduler,
        step_scheduler=step_scheduler,
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


def test_automodel_step_scheduler_serializes_the_completed_yielded_step() -> None:
    scheduler_module = pytest.importorskip("nemo_automodel.components.training.step_scheduler")
    scheduler = scheduler_module.StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=2,
        save_checkpoint_every_epoch=False,
        dataloader=[{"sample": 0}, {"sample": 1}],
        val_every_steps=None,
        start_step=0,
        start_epoch=0,
        num_epochs=1,
        max_steps=2,
    )

    iterator = iter(scheduler)
    assert next(iterator) == [{"sample": 0}]
    assert scheduler.step == 0
    assert scheduler.state_dict() == {"step": 1, "epoch": 0}
    assert next(iterator) == [{"sample": 1}]
    assert scheduler.step == 1
    assert scheduler.is_last_step
    assert scheduler.state_dict() == {"step": 2, "epoch": 0}
    with pytest.raises(StopIteration):
        next(iterator)


def _build_recipe_loop(
    root,
    sample_ids,
    *,
    max_steps,
    num_epochs,
    ckpt_every_steps,
    restore_from=None,
):
    if not dist.is_initialized():
        initialize_pdd_distributed(backend="gloo", timeout_minutes=1)
    scheduler_module = pytest.importorskip("nemo_automodel.components.training.step_scheduler")
    rng_module = pytest.importorskip("nemo_automodel.components.training.rng")

    lifecycle = build_toy_lifecycle()
    sampler = _released_sampler(sample_ids)
    rng = rng_module.StatefulRNG(1234, ranked=True)
    step_scheduler = scheduler_module.StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=ckpt_every_steps,
        save_checkpoint_every_epoch=False,
        dataloader=[None] * len(sampler),
        val_every_steps=None,
        start_step=0,
        start_epoch=0,
        num_epochs=num_epochs,
        max_steps=max_steps,
    )
    checkpointer = _checkpointer(lifecycle, root)
    manager = _RecordingCheckpointManager(
        root=root,
        checkpointer=checkpointer,
        model=lifecycle.student,
        optimizer=lifecycle.optimizer,
        scheduler=lifecycle.scheduler,
        step_scheduler=step_scheduler,
        trainer=lifecycle.trainer,
        sampler=sampler,
        rng=rng,
        identity=_identity(lifecycle, lifecycle.scheduler, sample_ids),
    )
    resume = manager.load(restore_from)
    events = SimpleNamespace(first_ids=[], diagnostics=[], validation_steps=[])

    recipe = object.__new__(PDDDiffusionRecipe)
    recipe.config = SimpleNamespace(
        step_scheduler=SimpleNamespace(local_batch_size=1, log_every=1),
        validation=SimpleNamespace(every_steps=10_000),
        checkpoint=SimpleNamespace(enabled=True),
        device=torch.device("cpu"),
    )
    recipe.training = SimpleNamespace(
        pipeline=lifecycle.pipeline,
        trainer=lifecycle.trainer,
        scheduler=lifecycle.scheduler,
        rng=rng,
    )
    recipe.setup_artifacts = SimpleNamespace(checkpointer=checkpointer)
    recipe.step_scheduler = step_scheduler
    recipe.checkpoint_manager = manager
    recipe.sampler = sampler
    recipe.resume = resume
    recipe.resume_pending = resume is not None
    recipe.rank = 0
    recipe.world_size = 1

    def prepared_batches():
        # Bind this iterator to the current sampler plan. The production loader iterator also
        # exhausts after that plan even though the recipe commits the sampler into its next epoch.
        for _ in range(sampler.remaining_batches):
            expected_ids = sampler.expected_next_sample_ids()
            if recipe.resume_pending:
                assert recipe.resume is not None
                recipe.resume.verify_first_batch(expected_ids)
            events.first_ids.append(expected_ids)
            offset = sum(ord(character) for character in expected_ids[0]) / 10_000
            yield make_batch(expected_ids, offset=offset), expected_ids

    recipe.__dict__["_prepared_training_batches"] = prepared_batches
    recipe.__dict__["_run_validation"] = events.validation_steps.append
    recipe.__dict__["_log_step"] = lambda diagnostics, _data_wait, _step_time: (
        events.diagnostics.append(diagnostics)
    )
    return SimpleNamespace(
        recipe=recipe,
        lifecycle=lifecycle,
        sampler=sampler,
        step_scheduler=step_scheduler,
        manager=manager,
        resume=resume,
        events=events,
    )


def test_recipe_loop_saves_periodic_and_max_step_checkpoints_once(tmp_path) -> None:
    run = _build_recipe_loop(
        tmp_path / "periodic",
        tuple(f"sample-{index}" for index in range(8)),
        max_steps=3,
        num_epochs=4,
        ckpt_every_steps=2,
    )
    run.recipe.run_train_validation_loop()

    assert [call["trainer_step"] for call in run.manager.save_calls] == [2, 3]
    assert [call["live_step"] for call in run.manager.save_calls] == [1, 2]
    assert [call["serialized_step"] for call in run.manager.save_calls] == [2, 3]
    assert sorted(path.name for path in (tmp_path / "periodic").glob("step_*")) == [
        "step_00000002",
        "step_00000003",
    ]
    assert run.events.validation_steps == [3]


def test_recipe_loop_epoch_final_save_normalizes_and_restores_epoch(tmp_path) -> None:
    root = tmp_path / "epoch"
    sample_ids = ("sample-0", "sample-1")
    source = _build_recipe_loop(
        root,
        sample_ids,
        max_steps=100,
        num_epochs=1,
        ckpt_every_steps=100,
    )
    source.recipe.run_train_validation_loop()

    assert len(source.manager.save_calls) == 1
    assert source.manager.save_calls[0] == {
        "live_step": 1,
        "serialized_step": 2,
        "trainer_step": 2,
        "live_epoch": 0,
        "sampler_epoch": 1,
    }
    manifest = json.loads((root / "step_00000002" / "manifest.json").read_text())
    assert manifest["step_scheduler"] == {"step": 2, "epoch": 1}

    resumed = _build_recipe_loop(
        root,
        sample_ids,
        max_steps=3,
        num_epochs=2,
        ckpt_every_steps=100,
        restore_from="step_00000002",
    )
    assert resumed.step_scheduler.step == 2
    assert resumed.step_scheduler.epoch == resumed.sampler.epoch == 1
    assert resumed.resume is not None
    expected_ids = resumed.resume.expected_next_sample_ids
    resumed.recipe.run_train_validation_loop()
    assert resumed.events.first_ids[0] == expected_ids
    assert (root / "step_00000003" / "COMPLETE").is_file()


def test_recipe_loop_resume_matches_uninterrupted_next_update(tmp_path) -> None:
    sample_ids = tuple(f"sample-{index}" for index in range(4))
    control = _build_recipe_loop(
        tmp_path / "control",
        sample_ids,
        max_steps=2,
        num_epochs=2,
        ckpt_every_steps=100,
    )
    control.recipe.run_train_validation_loop()

    staged_root = tmp_path / "staged"
    first = _build_recipe_loop(
        staged_root,
        sample_ids,
        max_steps=1,
        num_epochs=2,
        ckpt_every_steps=100,
    )
    first.recipe.run_train_validation_loop()
    resumed = _build_recipe_loop(
        staged_root,
        sample_ids,
        max_steps=2,
        num_epochs=2,
        ckpt_every_steps=100,
        restore_from="step_00000001",
    )
    assert resumed.resume is not None
    expected_next_ids = resumed.resume.expected_next_sample_ids
    resumed.recipe.run_train_validation_loop()

    assert resumed.events.first_ids == [expected_next_ids]
    assert resumed.events.first_ids[0] == control.events.first_ids[1]
    assert resumed.events.diagnostics == [control.events.diagnostics[1]]
    for name, tensor in resumed.lifecycle.student.state_dict().items():
        torch.testing.assert_close(
            tensor,
            control.lifecycle.student.state_dict()[name],
            rtol=0,
            atol=0,
        )
    actual_optimizer = _optimizer_state_by_name(resumed.lifecycle)
    expected_optimizer = _optimizer_state_by_name(control.lifecycle)
    assert actual_optimizer.keys() == expected_optimizer.keys()
    for name in actual_optimizer:
        for key, actual in actual_optimizer[name].items():
            expected = expected_optimizer[name][key]
            if isinstance(actual, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert actual == expected


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
        expected_latent_channels=3,
        expected_condition_features=6,
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
            expected_latent_channels=3,
            expected_condition_features=6,
        )


def test_qwen_batch_preparation_rejects_every_pre_model_shape_and_dtype_mismatch() -> None:
    base = {
        "image_latents": torch.ones(2, 3, 4, 4),
        "text_embeddings": torch.ones(2, 5, 6),
        "text_embeddings_mask": torch.ones(2, 5, dtype=torch.long),
        "negative_text_embeddings": torch.zeros(2, 5, 6),
        "negative_text_embeddings_mask": torch.ones(2, 5, dtype=torch.bool),
        "metadata": {"sample_ids": ["qwen-a", "qwen-b"]},
    }
    cases = (
        ("image_latents", torch.ones(2, 3, 4, 4, dtype=torch.long), "floating-point dtype"),
        ("image_latents", torch.ones(2, 4, 4, 4), "exactly 3 channels"),
        ("image_latents", torch.ones(2, 3, 3, 4), "positive even spatial dimensions"),
        ("text_embeddings", torch.ones(2, 5, 6, dtype=torch.long), "floating-point dtype"),
        ("text_embeddings", torch.ones(2, 5, 7), "exactly 6 features"),
        ("text_embeddings", torch.ones(2, 5, 6, 1), "must be 2D or 3D"),
        ("text_embeddings_mask", torch.ones(2, 5), "integer or boolean dtype"),
        ("text_embeddings_mask", torch.ones(2, 4, dtype=torch.long), "sequence length"),
        (
            "negative_text_embeddings_mask",
            torch.ones(2, 5),
            "integer or boolean dtype",
        ),
    )

    for field, value, message in cases:
        batch = dict(base)
        batch[field] = value
        with pytest.raises((TypeError, ValueError), match=message):
            prepare_qwen_pdd_batch(
                batch,
                device=torch.device("cpu"),
                dtype=torch.float32,
                require_negative_condition=True,
                expected_latent_channels=3,
                expected_condition_features=6,
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


def test_training_hard_aborts_for_teacher_gradient_zero_gradient_and_missing_coverage(
    monkeypatch,
) -> None:
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

    zero_actual_update = build_toy_lifecycle()
    monkeypatch.setattr(zero_actual_update.trainer, "_projection_update_ratio", lambda before: 0.0)
    with pytest.raises(RuntimeError, match="zero actual projection update"):
        zero_actual_update.trainer.train_step(make_batch(("zero-actual-update",)))

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
    assert source_manager.identity["qwen_image"] == {"execution": "fastgen_mr210"}
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
    moved_model_identity = copy.deepcopy(third_manager.identity)
    moved_model_identity["model"]["revision"] = "b" * 40
    with pytest.raises(RuntimeError, match="identity"):
        resolve_pdd_training_checkpoint(
            tmp_path / "checkpoints",
            resumed_checkpoint.name,
            expected_world_size=1,
            expected_identity=moved_model_identity,
        )
    with pytest.raises(RuntimeError, match="identity"):
        third_manager.resolve(mismatched.name)

    for step, execution in ((99999994, None), (99999995, "canonical_diffusers")):
        incompatible = tmp_path / "checkpoints" / f"step_{step:08d}"
        shutil.copytree(resumed_checkpoint, incompatible)
        incompatible_manifest_path = incompatible / "manifest.json"
        incompatible_manifest = json.loads(incompatible_manifest_path.read_text())
        incompatible_manifest["completed_steps"] = step
        if execution is None:
            incompatible_manifest["identity"].pop("qwen_image")
        else:
            incompatible_manifest["identity"]["qwen_image"] = {"execution": execution}
        incompatible_manifest_path.write_text(
            json.dumps(incompatible_manifest, indent=2, sort_keys=True) + "\n"
        )
        _refresh_complete_marker(incompatible)
        (tmp_path / "checkpoints" / "LATEST").write_text(incompatible.name + "\n")
        assert third_manager.resolve("LATEST") == resumed_checkpoint.resolve()
        with pytest.raises(RuntimeError, match="identity"):
            third_manager.resolve(incompatible.name)

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
    step_manifest["step_scheduler"]["step"] = 4
    step_manifest_path.write_text(json.dumps(step_manifest, indent=2, sort_keys=True) + "\n")
    trainer_state_path = step_mismatch / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text())
    trainer_state["completed_steps"] = 4
    trainer_state["step_scheduler"]["step"] = 4
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


def test_save_chains_validated_parent_without_resolving_latest(tmp_path, monkeypatch) -> None:
    pytest.importorskip("nemo_automodel")
    rng_module = pytest.importorskip("nemo_automodel.components.training.rng")
    if not torch.distributed.is_initialized():
        initialize_pdd_distributed(backend="gloo", timeout_minutes=1)
    sample_ids = tuple(f"sample-{index}" for index in range(8))

    source = build_toy_lifecycle()
    source_sampler = _released_sampler(sample_ids)
    source_manager, source_checkpointer = _manager(
        tmp_path / "checkpoints",
        source,
        source_sampler,
        rng_module.StatefulRNG(1234, ranked=True),
    )
    _run_next(source, source_sampler)
    first = source_manager.save()
    assert json.loads((first / "manifest.json").read_text())["parent_checkpoint"] is None

    resumed = build_toy_lifecycle()
    resumed_sampler = _released_sampler(sample_ids)
    resumed_manager, resumed_checkpointer = _manager(
        tmp_path / "checkpoints",
        resumed,
        resumed_sampler,
        rng_module.StatefulRNG(9999, ranked=True),
    )
    assert resumed_manager.load("LATEST") is not None

    def fail_resolution(restore_from):
        raise AssertionError(f"save re-resolved {restore_from}")

    monkeypatch.setattr(resumed_manager, "_collective_resolve", fail_resolution)
    _run_next(resumed, resumed_sampler)
    second = resumed_manager.save()
    _run_next(resumed, resumed_sampler)
    third = resumed_manager.save()

    assert json.loads((second / "manifest.json").read_text())["parent_checkpoint"] == first.name
    assert json.loads((third / "manifest.json").read_text())["parent_checkpoint"] == second.name
    source_checkpointer.close()
    resumed_checkpointer.close()
