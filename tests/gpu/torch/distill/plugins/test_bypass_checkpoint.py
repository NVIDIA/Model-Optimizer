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

"""Real interrupted/resumed Bypass updates, committed files and finalized reload."""

import copy
import gc
import json
import os
import random
import shutil
import time
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from _test_utils.bypass import assert_exact_state as _equal
from _test_utils.bypass import bypass_problem
from _test_utils.bypass import committed_checkpoint as _committed
from _test_utils.torch.distributed.utils import bounded_collectives
from torch.distributed.device_mesh import init_device_mesh

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill import _bypass_checkpoint as checkpoint_module
from modelopt.torch.distill._bypass_capture import ModuleBypassAdapter
from modelopt.torch.distill._bypass_checkpoint import (
    BypassCheckpoint,
    _batch_hash,
    _model_hash,
    _rng_state,
)
from modelopt.torch.distill._bypass_optimizer import Lion
from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
from modelopt.torch.quantization.utils.native_weight import pin_native_weight


class _Gain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("gain", torch.tensor(1.0), persistent=False)

    def forward(self, value):
        return value * self.gain


def test_checkpoint_refuses_changed_nonpersistent_execution_buffer(tmp_path):
    student, teacher, batches = bypass_problem(activation_factory=_Gain)
    config = mtd.BypassConfig(
        num_steps=1, global_batch_size=2, sequence_length=3, checkpoint_dir=str(tmp_path)
    )
    checkpoint = BypassCheckpoint(student, teacher, config, dataloader=batches, devices=[])
    checkpoint.save(1, fields={}, optimizers={}, data_position={}, rng_state=_rng_state([]))
    assert checkpoint.load(str(tmp_path))[0] == 1
    teacher.activation.gain.fill_(2)
    changed = BypassCheckpoint(student, teacher, config, dataloader=batches, devices=[])
    with pytest.raises(ValueError, match="identity"):
        changed.load(str(tmp_path))


@pytest.mark.parametrize("pointer", ["", ".."])
def test_checkpoint_refuses_reserved_latest_pointer(tmp_path, pointer):
    student, teacher, batches = bypass_problem()
    config = mtd.BypassConfig(
        num_steps=1, global_batch_size=2, sequence_length=3, checkpoint_dir=str(tmp_path)
    )
    checkpoint = BypassCheckpoint(student, teacher, config, dataloader=batches, devices=[])
    (tmp_path / "latest").write_text(pointer)
    with pytest.raises(ValueError, match=r"Invalid Bypass.*pointer"):
        checkpoint.save(1, fields={}, optimizers={}, data_position={}, rng_state=_rng_state([]))
    with pytest.raises(ValueError, match=r"Invalid Bypass.*pointer"):
        checkpoint.load(str(tmp_path))


@pytest.mark.parametrize("failure", ["flag", "filesystem"])
def test_checkpoint_requires_directory_sync_before_training(tmp_path, monkeypatch, failure):
    student, teacher, batches = bypass_problem()
    if failure == "flag":
        monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
    else:
        monkeypatch.setattr(
            checkpoint_module,
            "_sync_directory",
            mock.Mock(side_effect=OSError("unsupported directory sync")),
        )
    with (
        mock.patch.object(
            ModuleBypassAdapter, "teacher_microbatch", side_effect=AssertionError("teacher ran")
        ),
        pytest.raises(ValueError, match="directory sync"),
    ):
        mtd.bypass(
            student,
            teacher,
            batches,
            {
                "num_steps": 1,
                "global_batch_size": 2,
                "sequence_length": 3,
                "checkpoint_dir": str(tmp_path),
            },
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("retry", [False, True])
def test_checkpoint_first_save_syncs_new_ancestor_entries(tmp_path, retry):
    student, teacher, batches = bypass_problem()
    root = tmp_path / "new-parent" / "checkpoints"
    config = mtd.BypassConfig(
        num_steps=1, global_batch_size=2, sequence_length=3, checkpoint_dir=str(root)
    )
    checkpoint = BypassCheckpoint(student, teacher, config, dataloader=batches, devices=[])
    if retry:
        sync_directory = checkpoint_module._sync_directory

        def fail_parent(path):
            if path == tmp_path:
                raise OSError("first ancestor sync failed")
            return sync_directory(path)

        with (
            mock.patch.object(checkpoint_module, "_sync_directory", side_effect=fail_parent),
            pytest.raises(ValueError, match="first ancestor sync failed"),
        ):
            checkpoint.save(1, fields={}, optimizers={}, data_position={}, rng_state=_rng_state([]))
        assert root.is_dir()
    with mock.patch.object(
        checkpoint_module, "_sync_directory", wraps=checkpoint_module._sync_directory
    ) as sync:
        checkpoint.save(1, fields={}, optimizers={}, data_position={}, rng_state=_rng_state([]))
    paths = [call.args[0] for call in sync.call_args_list]
    assert tmp_path in paths and root.parent in paths
    assert checkpoint.load(str(root))[0] == 1


def test_generic_checkpoint_does_not_require_transformers_for_teacher_metadata(tmp_path):
    student, teacher, batches = bypass_problem()
    teacher.config = SimpleNamespace(model_type="custom_generic_teacher")
    config = mtd.BypassConfig(
        num_steps=1, global_batch_size=2, sequence_length=3, checkpoint_dir=str(tmp_path)
    )
    with mock.patch.object(
        checkpoint_module, "version", side_effect=PackageNotFoundError("transformers")
    ):
        checkpoint = BypassCheckpoint(student, teacher, config, dataloader=batches, devices=[])
    assert "transformers" not in checkpoint.identity["runtime"]
    checkpoint.save(1, fields={}, optimizers={}, data_position={}, rng_state=_rng_state([]))
    assert checkpoint.load(str(tmp_path))[0] == 1


def test_checkpoint_refuses_unsafe_initial_loader_cursor(tmp_path):
    class UnsafeCursor(_StatefulCursor):
        def state_dict(self):
            return {"position": self.position, "permutation": np.arange(len(self.batches))}

    student, teacher, batches = bypass_problem()
    with (
        mock.patch.object(
            ModuleBypassAdapter, "teacher_microbatch", side_effect=AssertionError("teacher ran")
        ),
        pytest.raises(ValueError, match=r"[Ww]eights only|[Ww]eights-only"),
    ):
        mtd.bypass(
            student,
            teacher,
            UnsafeCursor(batches),
            {
                "num_steps": 1,
                "global_batch_size": 2,
                "sequence_length": 3,
                "checkpoint_dir": str(tmp_path),
            },
        )
    assert not list(tmp_path.iterdir())


def test_checkpoint_refuses_changed_cuda_rng_scope(tmp_path):
    student, teacher, batches = bypass_problem()
    config = mtd.BypassConfig(
        num_steps=1, global_batch_size=2, sequence_length=3, checkpoint_dir=str(tmp_path)
    )
    # CPU payloads exercise admission before any CUDA restoration or allocation.
    saved = BypassCheckpoint(student, teacher, config, dataloader=batches, devices=[0])
    saved.save(1, fields={}, optimizers={}, data_position={}, rng_state=_rng_state([]))
    same = BypassCheckpoint(student, teacher, config, dataloader=batches, devices=[0])
    assert same.load(str(tmp_path))[0] == 1
    moved = BypassCheckpoint(student, teacher, config, dataloader=batches, devices=[1])
    with pytest.raises(ValueError, match="identity"):
        moved.load(str(tmp_path))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs two visible CUDA devices")
def test_non_distributed_resume_refuses_cuda_device_move(tmp_path):
    initial, teacher, batches = bypass_problem()
    config = {"num_steps": 1, "global_batch_size": 2, "sequence_length": 3}
    mtd.bypass(
        copy.deepcopy(initial).to("cuda:0"),
        copy.deepcopy(teacher).to("cuda:0"),
        [batches[0].to("cuda:0")],
        {**config, "checkpoint_dir": str(tmp_path)},
    )
    moved = copy.deepcopy(initial).to("cuda:1")
    target = copy.deepcopy(teacher).to("cuda:1")
    before = copy.deepcopy(moved.state_dict())
    caller_rng = _rng_state([0, 1])
    with (
        mock.patch("modelopt.torch.distill._bypass_runner.tree_map") as priming_transfer,
        mock.patch("modelopt.torch.distill._bypass_runner._restore_rng_state") as restore_rng,
        pytest.raises(ValueError, match="identity"),
    ):
        mtd.bypass(
            moved,
            target,
            [batches[0].to("cuda:1")],
            {**config, "resume_from": str(tmp_path)},
        )
    priming_transfer.assert_not_called()
    restore_rng.assert_not_called()
    _equal(moved.state_dict(), before)
    _equal(_rng_state([0, 1]), caller_rng)


class _SelfAttention(torch.nn.MultiheadAttention):
    def forward(self, hidden_states):
        return super().forward(hidden_states, hidden_states, hidden_states, need_weights=False)[0]


class _Interrupted(list):
    def __iter__(self):
        for index, batch in enumerate(list.__iter__(self)):
            if index == 2:
                raise OSError("Simulated data-reader interruption")
            random.random()
            np.random.random()
            torch.rand(())
            yield batch


class _RandomCursor(list):
    def __iter__(self):
        for batch in list.__iter__(self):
            random.random()
            np.random.random()
            torch.rand(())
            yield batch


class _WrongNextCursor(_RandomCursor):
    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            yield self[index + 1] if index == 2 else batch


class _StatefulCursor:
    def __init__(
        self, batches, *, interrupted=False, snapshot_failure=False, unsafe=None, native_state=False
    ):
        self.batches, self.position, self.interrupted = batches, 0, interrupted
        self.snapshot_failure, self.unsafe = snapshot_failure, unsafe
        self.native_state = native_state
        self.data_identity = [_batch_hash(batch) for batch in batches]

    def __iter__(self):
        return self

    def __next__(self):
        if self.interrupted and self.position == 2:
            raise OSError("Simulated data-reader interruption")
        if self.position == len(self.batches):
            raise StopIteration
        batch = self.batches[self.position]
        self.position += 1
        if self.unsafe == "priming_batch" and self.position > 2:
            batch = {"hidden_states": batch, "sample_ids": np.arange(batch.shape[0])}
        random.random()
        np.random.random()
        torch.rand(())
        return batch

    def state_dict(self):
        if self.snapshot_failure:
            raise OSError("Simulated data-cursor snapshot failure")
        state = {"position": self.position}
        if self.native_state:
            state["sparse_position"] = torch.tensor([self.position]).to_sparse()
        if self.unsafe == "loader_state" and self.position > 2:
            state["permutation"] = np.arange(len(self.batches))
        return state

    def load_state_dict(self, state):
        if self.native_state:
            self.position = state["sparse_position"].to_dense().item()
            assert self.position == state["position"]
        else:
            self.position = state["position"]


@pytest.mark.parametrize("stateful", [False, True])
def test_bypass_resume_matches_uninterrupted(tmp_path, monkeypatch, caplog, stateful):
    def activation_factory():
        activation = torch.nn.LeakyReLU(0.1) if stateful else torch.nn.SiLU()
        if stateful:
            return torch.nn.Sequential(
                activation, activation, _SelfAttention(256, 2, batch_first=True)
            )
        return torch.nn.Sequential(activation, activation)

    initial, teacher, batches = bypass_problem(activation_factory=activation_factory)

    def fresh_student():
        model = copy.deepcopy(initial)
        native, scale = FP8QTensor.quantize(model.head.weight.detach(), axis=0)
        quantizer = model.head.weight_quantizer
        quantizer.set_from_attribute_config(
            mtq.QuantizerAttributeConfig(num_bits=(4, 3), axis=0, fake_quant=stateful)
        )
        quantizer._set_buffer("_scale", scale)
        if stateful:
            model.head.weight.data.copy_(native.dequantize(scale=scale))
            pin_native_weight(quantizer, native, model.head.weight)
        else:
            mtq.compress(model, config={"compress": {"default": False, "*head*": True}})
            assert isinstance(model.head.weight, QTensorWrapper)
        return model

    def cursor(data, *, interrupted=False):
        if stateful:
            return _StatefulCursor(data, interrupted=interrupted, native_state=True)
        return _Interrupted(data) if interrupted else _RandomCursor(data)

    recipe = {
        "num_steps": 4,
        "global_batch_size": 2,
        "sequence_length": 3,
        "warmup_fraction": 0.25,
        "checkpoint_interval": 2,
    }
    reference = fresh_student()
    mtd.bypass(
        reference,
        copy.deepcopy(teacher),
        cursor(batches),
        {**recipe, "checkpoint_dir": str(tmp_path / "reference")},
    )
    interrupted = fresh_student()
    with pytest.raises(OSError):
        mtd.bypass(
            interrupted,
            copy.deepcopy(teacher),
            cursor(batches, interrupted=True),
            {**recipe, "checkpoint_dir": str(tmp_path / "resumed")},
        )
    generation, saved = _committed(tmp_path / "resumed")
    assert saved["step"] == 2
    incomplete = tmp_path / "resumed" / ".pending-interrupted"
    incomplete.mkdir()
    (incomplete / "rank-0.pt").write_bytes(b"incomplete")
    refusals = [
        (incomplete, cursor(batches), copy.deepcopy(teacher)),
        (generation, cursor([batches[0] + 1, *batches[1:]]), copy.deepcopy(teacher)),
        (generation, cursor(batches), copy.deepcopy(teacher)),
    ]
    if not stateful:
        refusals.append((generation, _WrongNextCursor(batches), copy.deepcopy(teacher)))
    for index, (source, data, target) in enumerate(refusals):
        if index == 2:
            with torch.no_grad():
                target.head.weight.add_(0.001)
        refused = fresh_student()
        before = copy.deepcopy(refused.state_dict())
        with pytest.raises(ValueError):
            mtd.bypass(refused, target, data, {**recipe, "resume_from": str(source)})
        _equal(refused.state_dict(), before)

    for role in ("teacher", "student"):
        changes = (
            ("attention", "duplicate-edge", "execution")
            if stateful
            else ("duplicate-edge", "execution")
        )
        for change in changes:
            refused, target = fresh_student(), copy.deepcopy(teacher)
            changed = target if role == "teacher" else refused
            if change == "attention":
                # MultiheadAttention has empty extra_repr despite execution options.
                changed.activation[2].num_heads = 4
            elif change == "duplicate-edge":
                # Default named_modules() deduplicates the repeated activation.
                # Removing one registered edge still changes actual execution.
                del changed.activation._modules["1"]
            elif stateful:
                changed.activation[0].negative_slope = 0.2
            else:
                activation = torch.nn.ReLU()
                changed.activation = torch.nn.Sequential(activation, activation)
            before = copy.deepcopy(refused.state_dict())
            target_before = copy.deepcopy(target.state_dict())
            caller_rng = _rng_state([])
            with pytest.raises(ValueError):
                mtd.bypass(
                    refused,
                    target,
                    cursor(batches),
                    {**recipe, "resume_from": str(generation)},
                )
            _equal(refused.state_dict(), before)
            _equal(target.state_dict(), target_before)
            _equal(_rng_state([]), caller_rng)

    refused = fresh_student()
    metadata = (
        refused.head.weight_quantizer._native_weight_meta["native_metadata"]
        if stateful
        else refused.head.weight.metadata
    )
    metadata["dtype"] = torch.float64
    before = copy.deepcopy(refused.state_dict())
    with pytest.raises(ValueError):
        mtd.bypass(
            refused,
            copy.deepcopy(teacher),
            cursor(batches),
            {**recipe, "resume_from": str(generation)},
        )
    _equal(refused.state_dict(), before)
    assert metadata["dtype"] == torch.float64

    original_save = torch.save

    def failed_write(state, stream, *args, **kwargs):
        if ".pending-step-000004-" in str(getattr(stream, "name", "")):
            raise OSError("Simulated checkpoint disk failure")
        return original_save(state, stream, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", failed_write)
        with pytest.raises(ValueError, match="Simulated checkpoint disk failure"):
            mtd.bypass(
                fresh_student(),
                copy.deepcopy(teacher),
                cursor(batches),
                {
                    **recipe,
                    "resume_from": str(tmp_path / "resumed"),
                    "checkpoint_dir": str(tmp_path / "resumed"),
                },
            )
    assert _committed(tmp_path / "resumed")[0] == generation
    assert not list((tmp_path / "resumed").glob(".pending-step-*"))
    assert incomplete.exists()  # Never delete a different writer's uncommitted directory.
    if not stateful:
        for phase in ("after_rename", "before_latest", "after_latest", "retirement"):
            root = tmp_path / f"commit-{phase}"
            shutil.copytree(tmp_path / "resumed", root)
            foreign = root / ".latest-other-writer"
            foreign.write_text("untouched")
            original_replace = os.replace
            original_rmtree = shutil.rmtree

            def fail_commit(source, target):
                source = Path(source)
                if phase == "before_latest" and source.name.startswith(".latest-"):
                    raise OSError("Simulated latest pointer write failure")
                result = original_replace(source, target)
                if (phase == "after_rename" and source.name.startswith(".pending-step-")) or (
                    phase == "after_latest" and source.name.startswith(".latest-")
                ):
                    raise OSError("Simulated checkpoint commit failure")
                return result

            def fail_retire(path, *args, **kwargs):
                if Path(path) == root / generation.name:
                    raise OSError("Simulated old-generation retirement failure")
                return original_rmtree(path, *args, **kwargs)

            candidate = fresh_student()
            with monkeypatch.context() as patch:
                patch.setattr(os, "replace", fail_commit)
                if phase == "retirement":
                    patch.setattr(shutil, "rmtree", fail_retire)
                with pytest.raises(ValueError) if phase != "retirement" else nullcontext():
                    mtd.bypass(
                        candidate,
                        copy.deepcopy(teacher),
                        cursor(batches),
                        {**recipe, "resume_from": str(root), "checkpoint_dir": str(root)},
                    )
            current, committed = _committed(root)
            reader = BypassCheckpoint(
                fresh_student(),
                copy.deepcopy(teacher),
                mtd.BypassConfig(**recipe, resume_from=str(root)),
                dataloader=cursor(batches),
                devices=[],
            )
            verified_step, verified = reader.load(str(root))
            assert verified_step == (4 if phase in ("after_latest", "retirement") else 2)
            _equal(verified, committed)
            assert foreign.read_text() == "untouched"
            assert (root / incomplete.name).exists()
            assert not list(root.glob(".pending-step-*"))
            assert set(root.glob(".latest-*")) == {foreign}
            if phase in ("after_latest", "retirement"):
                assert current.name != generation.name and committed["step"] == 4
                assert current.exists()  # Never remove the generation latest publishes.
            else:
                assert current.name == generation.name
                assert list(root.glob("step-*")) == [current]
            if phase == "retirement":
                assert (
                    root / generation.name
                ).is_dir()  # Failed retirement may need manual cleanup.
                _equal(candidate.state_dict(), reference.state_dict())
                _, expected = _committed(tmp_path / "reference")
                for key in ("fields", "optimizers", "rng_state", "data_position"):
                    _equal(committed[key], expected[key])
                assert any(
                    "Simulated old-generation retirement failure" in record.message
                    for record in caplog.records
                )

    resumed = fresh_student()
    random.seed(203)
    np.random.seed(203)
    caller_rng = _rng_state([])
    mtd.bypass(
        resumed,
        copy.deepcopy(teacher),
        cursor(batches),
        {
            **recipe,
            "resume_from": str(tmp_path / "resumed"),
            "checkpoint_dir": str(tmp_path / "resumed"),
        },
    )
    _equal(_rng_state([]), caller_rng)
    final_generation, actual = _committed(tmp_path / "resumed")
    _, expected = _committed(tmp_path / "reference")
    assert actual["step"] == 4
    for key in ("fields", "optimizers", "rng_state", "data_position"):
        _equal(actual[key], expected[key])
    _equal(resumed.state_dict(), reference.state_dict())
    with pytest.raises(ValueError):
        mtd.bypass(
            fresh_student(),
            copy.deepcopy(teacher),
            cursor(batches),
            {**recipe, "checkpoint_dir": str(tmp_path / "resumed")},
        )
    unchanged_generation, unchanged = _committed(tmp_path / "resumed")
    assert unchanged_generation == final_generation
    _equal(unchanged, actual)
    assert actual["data_position"]["consumed_batches"] == 4
    if stateful:
        assert actual["data_position"]["loader_state"]["position"] == 4
        _equal(
            actual["data_position"]["loader_state"]["sparse_position"],
            torch.tensor([4]).to_sparse(),
        )
        assert "prefix_sha256" not in actual["data_position"]
        assert "next_batch_hash" not in actual["data_position"]
    else:
        assert actual["data_position"]["next_batch_hash"] == _batch_hash(batches[4])
    assert all("_local_repair_state" not in name for name in resumed.state_dict())
    relocated = tmp_path / "relocated"
    shutil.copytree(final_generation, relocated)
    restored = fresh_student()
    mtd.bypass(
        restored,
        copy.deepcopy(teacher),
        cursor(batches),
        {**recipe, "resume_from": str(relocated)},
    )
    _equal(restored.state_dict(), reference.state_dict())
    deployed = tmp_path / "model.pt"
    mto.save(resumed, deployed)
    reloaded = mto.restore(copy.deepcopy(teacher), deployed)
    torch.testing.assert_close(reloaded(batches[0]), reference(batches[0]), rtol=0, atol=0)

    if stateful:

        class LargeSequence(_RandomCursor):
            def __len__(self):
                return 1_000_000

            def __getitem__(self, index):
                if index >= 9:
                    raise OSError("Unneeded lazy dataset tail was accessed")
                return super().__getitem__(index)

        data = [*batches, batches[-1] + 0.125]
        bounded = fresh_student()
        mtd.bypass(
            bounded,
            copy.deepcopy(teacher),
            LargeSequence(data),
            {**recipe, "checkpoint_dir": str(tmp_path / "bounded-data")},
        )
        _equal(bounded.state_dict(), reference.state_dict())
        # Future values inside the recipe bound remain part of preflight identity.
        changed = [*data[:8], data[8] + 1]
        refused = fresh_student()
        before = copy.deepcopy(refused.state_dict())
        with pytest.raises(ValueError):
            mtd.bypass(
                refused,
                copy.deepcopy(teacher),
                LargeSequence(changed),
                {**recipe, "resume_from": str(tmp_path / "bounded-data")},
            )
        _equal(refused.state_dict(), before)


def _distributed_resume_worker(rank, size, directory, failure=None):
    threads = torch.get_num_threads()
    try:
        # Per-rank thread pools oversubscribe this small CPU quantization workload.
        torch.set_num_threads(1)
        print(
            f"CPU checkpoint worker rank={rank} threads={torch.get_num_threads()} "
            f"previous_threads={threads} interop_threads={torch.get_num_interop_threads()}",
            flush=True,
        )
        _distributed_resume_case(rank, size, directory, failure)
    finally:
        torch.set_num_threads(threads)


def _distributed_resume_case(rank, size, directory, failure):

    initial, teacher, batches = bypass_problem()
    batches = [sample.unsqueeze(0) for batch in batches[:4] for sample in batch]
    local = batches[rank::size]
    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("dp",))
    recipe = {
        "num_steps": 4,
        "global_batch_size": 2,
        "sequence_length": 3,
        "checkpoint_interval": 2,
        "warmup_fraction": 0.25,
    }
    root = Path(directory)
    if failure in ("shard_cleanup", "commit_cleanup"):
        directory = root / failure
        candidate = copy.deepcopy(initial)
        before = copy.deepcopy(candidate.state_dict())
        caller_rng = _rng_state([])
        original_save, original_replace, original_rmtree = torch.save, os.replace, shutil.rmtree

        def fail_shard(state, stream, *args, **kwargs):
            path = Path(getattr(stream, "name", ""))
            if (
                failure == "shard_cleanup"
                and rank == 1
                and path.parent.name.startswith(".pending-step-000004-")
            ):
                raise OSError("Primary shard failure")
            return original_save(state, stream, *args, **kwargs)

        def fail_commit(source, target):
            result = original_replace(source, target)
            if failure == "commit_cleanup" and Path(source).name.startswith(
                ".pending-step-000004-"
            ):
                raise OSError("Primary commit failure")
            return result

        def fail_cleanup(path, *args, **kwargs):
            path = Path(path)
            if path.parent == directory and path.name.startswith(
                (".pending-step-000004-", "step-000004-")
            ):
                raise OSError("Secondary cleanup failure")
            return original_rmtree(path, *args, **kwargs)

        with bounded_collectives(10):
            if rank == 0:
                (directory / ".pending-other-writer").mkdir(parents=True)
                (directory / ".latest-other-writer").write_text("untouched")
            torch.distributed.barrier()
            with (
                mock.patch.object(torch, "save", fail_shard),
                mock.patch.object(os, "replace", fail_commit),
                mock.patch.object(shutil, "rmtree", fail_cleanup),
            ):
                try:
                    mtd.bypass(
                        candidate,
                        copy.deepcopy(teacher),
                        _RandomCursor(local),
                        {**recipe, "checkpoint_dir": str(directory)},
                        device_mesh=mesh,
                    )
                except ValueError as exc:
                    error = str(exc)
                else:
                    raise AssertionError("Primary checkpoint failure was accepted")
        previous, state = _committed(directory, size=size, rank=rank)
        assert state["step"] == 2
        assert previous.is_dir()
        assert (directory / ".pending-other-writer").is_dir()
        assert (directory / ".latest-other-writer").read_text() == "untouched"
        _equal(candidate.state_dict(), before)
        _equal(_rng_state([]), caller_rng)
        primary = (
            "Primary shard failure" if failure == "shard_cleanup" else "Primary commit failure"
        )
        assert primary in error, error
        return
    if failure in ("loader_state", "priming_batch"):
        candidate = copy.deepcopy(initial)
        before = copy.deepcopy(candidate.state_dict())
        caller_rng = _rng_state([])
        directory = root / failure
        with bounded_collectives(10):
            try:
                mtd.bypass(
                    candidate,
                    copy.deepcopy(teacher),
                    _StatefulCursor(local, unsafe=failure if rank == 1 else None),
                    {**recipe, "checkpoint_dir": str(directory)},
                    device_mesh=mesh,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe stateful checkpoint was accepted")
        previous, state = _committed(directory, size=size, rank=rank)
        assert state["step"] == 2
        assert list(directory.glob("step-*")) == [previous]
        assert not list(directory.glob(".pending-step-*"))
        _equal(candidate.state_dict(), before)
        _equal(_rng_state([]), caller_rng)
        return
    reference = copy.deepcopy(initial)
    mtd.bypass(
        reference,
        copy.deepcopy(teacher),
        _RandomCursor(local),
        {**recipe, "checkpoint_dir": str(root / "reference")},
        device_mesh=mesh,
    )
    with pytest.raises(ValueError):
        mtd.bypass(
            copy.deepcopy(initial),
            copy.deepcopy(teacher),
            _Interrupted(local) if rank == 1 else _RandomCursor(local),
            {**recipe, "checkpoint_dir": str(root / "resumed")},
            device_mesh=mesh,
        )
    if failure is not None:
        candidate = copy.deepcopy(initial)
        before = copy.deepcopy(candidate.state_dict())
        caller_rng = _rng_state([])
        failed = False
        if failure == "priming_transfer":
            target, name = torch.Tensor, "to"
        elif failure == "priming_validation":
            target, name = ModuleBypassAdapter, "validate_batch"
        elif failure == "optimizer_restore":
            target, name = Lion, "load_state_dict"
        else:
            target, name = torch, "set_rng_state"
        original = getattr(target, name)

        def fail_local(*args, **kwargs):
            nonlocal failed
            priming = failure != "priming_transfer" or (
                len(args) == 2 and args[1] == "cpu" and args[0].shape == local[0].shape
            )
            if rank == 1 and priming and not failed:
                failed = True
                raise torch.OutOfMemoryError("Simulated rank-local resume allocation failure")
            return original(*args, **kwargs)

        # Only the unavailable rank-local device failure is injected; resume stays real.
        with (
            bounded_collectives(10),
            mock.patch.object(target, name, fail_local),
            pytest.raises(ValueError),
        ):
            mtd.bypass(
                candidate,
                copy.deepcopy(teacher),
                _RandomCursor(local),
                {**recipe, "resume_from": str(root / "resumed")},
                device_mesh=mesh,
            )
        _equal(candidate.state_dict(), before)
        _equal(_rng_state([]), caller_rng)
        return

    class LateSnapshotFailure(_StatefulCursor):
        def state_dict(self):
            if rank == 1 and self.position >= 4:
                raise OSError("Simulated save-time data-cursor snapshot failure")
            return super().state_dict()

    with (
        bounded_collectives(10),
        pytest.raises(ValueError, match="save-time data-cursor snapshot failure") as snapshot_error,
    ):
        mtd.bypass(
            copy.deepcopy(initial),
            copy.deepcopy(teacher),
            LateSnapshotFailure(local),
            {**recipe, "checkpoint_dir": str(root / "snapshot-failure")},
            device_mesh=mesh,
        )
    errors = [None] * size
    torch.distributed.all_gather_object(errors, str(snapshot_error.value), group=mesh.get_group())
    assert len(set(errors)) == 1
    generation, saved = _committed(root / "snapshot-failure", size=size, rank=rank)
    assert saved["step"] == 2
    assert list((root / "snapshot-failure").glob("step-*")) == [generation]
    assert not list((root / "snapshot-failure").glob(".pending-step-*"))
    resumed = copy.deepcopy(initial)
    mtd.bypass(
        resumed,
        copy.deepcopy(teacher),
        _RandomCursor(local),
        {**recipe, "checkpoint_dir": str(root / "resumed"), "resume_from": str(root / "resumed")},
        device_mesh=mesh,
    )
    _equal(resumed.state_dict(), reference.state_dict())
    _, expected = _committed(root / "reference", size=size, rank=rank)
    _, actual = _committed(root / "resumed", size=size, rank=rank)
    for key in ("fields", "optimizers", "rng_state", "data_position"):
        _equal(actual[key], expected[key])
    single = init_device_mesh("cpu", (size, 1), mesh_dim_names=("owners", "dp"))["dp"]
    with pytest.raises(ValueError):
        mtd.bypass(
            copy.deepcopy(initial),
            copy.deepcopy(teacher),
            _RandomCursor(local),
            {**recipe, "resume_from": str(root / "resumed")},
            device_mesh=single,
        )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "priming_transfer",
        "priming_validation",
        "optimizer_restore",
        "rng_restore",
        "loader_state",
        "priming_batch",
        "shard_cleanup",
        "commit_cleanup",
    ],
)
@pytest.mark.timeout(180)
def test_distributed_resume_commits_every_owner(bypass_cpu_workers, tmp_path, failure):
    timeout, grace = bypass_cpu_workers.job_timeout, bypass_cpu_workers.failure_grace
    try:
        bypass_cpu_workers.job_timeout, bypass_cpu_workers.failure_grace = 60, 2
        bypass_cpu_workers.run(_distributed_resume_worker, str(tmp_path), failure)
    finally:
        bypass_cpu_workers.job_timeout, bypass_cpu_workers.failure_grace = timeout, grace


class _NativeInterrupted(list):
    def __init__(self, batches, after):
        super().__init__(batches)
        self.after = after

    def __iter__(self):
        for index, batch in enumerate(list.__iter__(self)):
            if index == self.after:
                raise OSError("Interrupted native data lane")
            yield batch


def _native_checkpoint_problem(rank, size):
    # Native model builders and device meshes require the optional Transformers integration.
    from _test_utils.transformers_families import build_model, cases
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    torch.cuda.set_device(rank)
    torch.manual_seed(97)
    record = cases(["qwen3_moe"])[0]
    record["minimum_pattern_prefix"] = 2
    record["config"].update(mlp_only_layers=[0], num_experts=4, num_experts_per_tok=1)
    student = build_model(record, torch.device("cuda", rank))
    with torch.no_grad():
        student.model.layers[1].mlp.gate.weight.zero_()
    teacher = copy.deepcopy(student)
    batches = [
        {"input_ids": torch.randint(3, 100, (1, 8), device=torch.device("cuda", rank))}
        for _ in range(16)
    ]
    mtq.quantize(
        student,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*mlp*weight_quantizer*",
                    "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                },
                {
                    "quantizer_name": "*layers.0.mlp*weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
                {"quantizer_name": "*mlp.gate.weight_quantizer*", "enable": False},
                {
                    "quantizer_name": "*experts.down_proj_weight_quantizers*",
                    "cfg": {
                        "num_bits": (2, 1),
                        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    },
                },
            ],
            "algorithm": "max",
        },
        lambda m: m(**batches[0]),
    )
    manager = TransformersDeviceMesh(
        init_device_mesh(
            "cuda", (1, 1, 2, size // 2), mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")
        ),
        init_device_mesh(
            "cuda",
            (1, 1, 2, 1) if size == 2 else (1, 1, 1, 4),
            mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep"),
        ),
    )
    adapter = TransformersBypassAdapter(manager)
    adapter.parallelize(student)
    adapter.parallelize(teacher)
    lane = adapter.context.data_mesh.get_local_rank()
    local = [
        batches[step * 8 + index]
        for step in range(2)
        for index in range(lane, 8, adapter.context.dp_size)
    ]
    return student, teacher, local, manager


def _native_resume_worker(rank, size, directory):
    root = Path(directory)
    torch.cuda.set_device(rank)
    probe = mtq.quantize(
        torch.nn.Sequential(
            torch.nn.Linear(4096, 4096, bias=False, device=f"cuda:{rank}", dtype=torch.bfloat16)
        ),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": 0}},
            ],
            "algorithm": None,
        },
    )
    mtq.compress(probe)
    assert isinstance(probe[0].weight, QTensorWrapper)
    physical_bytes = sum(
        value.numel() * value.element_size() for value in (*probe.parameters(), *probe.buffers())
    )
    torch.cuda.synchronize(rank)
    baseline = torch.cuda.memory_allocated(rank)
    torch.cuda.reset_peak_memory_stats(rank)
    started = time.perf_counter()
    identity = _model_hash(probe)
    torch.cuda.synchronize(rank)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(rank) - baseline
    (root / f"native-identity-{rank}.json").write_text(
        json.dumps(
            {
                "native_checkpoint_identity": {
                    "rank": rank,
                    "shape": [4096, 4096],
                    "physical_bytes": physical_bytes,
                    "incremental_peak_bytes": peak,
                    "seconds": elapsed,
                    "hash_length": len(identity),
                }
            }
        )
    )
    assert peak <= 10 * 1024**2  # Existing 8-MiB hash chunks plus 2 MiB allocator slack.
    del probe
    gc.collect()
    recipe = {
        "num_steps": 2,
        "global_batch_size": 8,
        "sequence_length": 8,
        "warmup_fraction": 0.5,
        "checkpoint_interval": 1,
    }
    reference, teacher, local, mesh = _native_checkpoint_problem(rank, size)
    mtd.bypass(
        reference,
        teacher,
        local,
        {**recipe, "checkpoint_dir": str(root / "reference")},
        device_mesh=mesh,
    )
    student, teacher, local, mesh = _native_checkpoint_problem(rank, size)
    with pytest.raises(ValueError):
        mtd.bypass(
            student,
            teacher,
            _NativeInterrupted(local, 4) if rank == 1 else local,
            {**recipe, "checkpoint_dir": str(root / "resumed")},
            device_mesh=mesh,
        )
    student, teacher, local, mesh = _native_checkpoint_problem(rank, size)

    def frozen_down_projections():
        return {
            name: (value.to_local() if hasattr(value, "to_local") else value).detach().clone()
            for name, value in student.state_dict().items()
            if "experts.down_proj" in name and isinstance(value, torch.Tensor)
        }

    frozen = frozen_down_projections()
    assert frozen and any("quantizer" in name for name in frozen)
    assert any("quantizer" not in name for name in frozen)
    mtd.bypass(
        student,
        teacher,
        local,
        {**recipe, "checkpoint_dir": str(root / "resumed"), "resume_from": str(root / "resumed")},
        device_mesh=mesh,
    )
    _equal(frozen_down_projections(), frozen)
    _, expected = _committed(root / "reference", size=size, rank=rank)
    _, actual = _committed(root / "resumed", size=size, rank=rank)
    assert expected["fields"]
    assert any(value.count_nonzero() for value in expected["fields"].values())
    for key in ("fields", "optimizers", "rng_state", "data_position"):
        _equal(actual[key], expected[key])
    for name, module in student.named_modules():
        if getattr(module, "backend", None) == "ggml":
            reference_module = reference.get_submodule(name)
            _equal(module.state_dict(), reference_module.state_dict())


def _native_pair_identity_worker(rank, size, directory):
    root = Path(directory)
    torch.cuda.set_device(rank)
    # Real 2.147B-parameter frozen pair, assembled one layer at a time so the
    # constructor stages one BF16 matrix rather than an uncompressed student.
    construction_started = time.perf_counter()
    construction_baseline = torch.cuda.memory_allocated(rank)
    torch.cuda.reset_peak_memory_stats(rank)
    pair = {"student": torch.nn.Sequential(), "teacher": torch.nn.Sequential()}
    for _ in range(64):
        layer = mtq.quantize(
            torch.nn.Sequential(
                torch.nn.Linear(4096, 4096, bias=False, device=f"cuda:{rank}", dtype=torch.bfloat16)
            ),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": 0}},
                ],
                "algorithm": None,
            },
        )
        mtq.compress(layer)
        assert isinstance(layer[0].weight, QTensorWrapper)
        pair["student"].append(layer[0])
        pair["teacher"].append(
            torch.nn.Linear(4096, 4096, bias=False, device=f"cuda:{rank}", dtype=torch.bfloat16)
        )
    torch.cuda.synchronize(rank)
    resident = torch.cuda.memory_allocated(rank)
    construction_peak = torch.cuda.max_memory_allocated(rank)
    construction_seconds = time.perf_counter() - construction_started
    model_measurements = {}
    for role, model in pair.items():
        torch.cuda.reset_peak_memory_stats(rank)
        started = time.perf_counter()
        identity = _model_hash(model)
        torch.cuda.synchronize(rank)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(rank) - resident
        model_measurements[role] = {
            "parameters": sum(value.numel() for value in model.parameters()),
            "physical_bytes": sum(
                value.numel() * value.element_size()
                for value in (*model.parameters(), *model.buffers())
            ),
            "seconds": elapsed,
            "incremental_peak_bytes": peak,
            "hash_length": len(identity),
        }
        assert peak <= 10 * 1024**2
    measurement_path = root / f"native-pair-identity-{rank}.json"
    measurements = {}
    measurements["native_checkpoint_pair_identity"] = {
        "rank": rank,
        "layers_per_model": 64,
        "layer_shape": [4096, 4096],
        "models": model_measurements,
        "pair_parameters": sum(value["parameters"] for value in model_measurements.values()),
        "pair_physical_bytes": sum(
            value["physical_bytes"] for value in model_measurements.values()
        ),
        "pair_seconds": sum(value["seconds"] for value in model_measurements.values()),
        "resident_cuda_bytes": resident - construction_baseline,
        "construction_seconds": construction_seconds,
        "construction_peak_bytes": construction_peak - construction_baseline,
        "construction_extra_peak_bytes": construction_peak - resident,
    }
    measurement_path.write_text(json.dumps(measurements))
    del pair, layer, model
    gc.collect()


def _native_transformers_unavailable():
    # Reuse the optional adapter's exact stable-version admission predicate.
    from modelopt.torch.distill.plugins.transformers.bypass import _require_transformers

    try:
        _require_transformers()
    except ImportError:
        return True
    return False


@pytest.mark.skipif(
    _native_transformers_unavailable(), reason="Native Bypass requires stable Transformers 5.23.x"
)
@pytest.mark.manual
@pytest.mark.parametrize("native_workers", [2, 4], indirect=True, ids=["fsdp2", "fsdp2_tp2_ep4"])
@pytest.mark.timeout(300)
def test_large_native_checkpoint_identity_measurement(native_workers, tmp_path, capsys):
    native_workers.run(_native_pair_identity_worker, str(tmp_path))
    with capsys.disabled():
        for rank in range(native_workers.world_size):
            print((tmp_path / f"native-pair-identity-{rank}.json").read_text(), flush=True)


@pytest.mark.skipif(
    _native_transformers_unavailable(), reason="Native Bypass requires stable Transformers 5.23.x"
)
@pytest.mark.parametrize("native_workers", [2, 4], indirect=True, ids=["fsdp2", "fsdp2_tp2_ep4"])
@pytest.mark.timeout(300)
def test_native_resume_with_empty_owners_and_frozen_nvfp4(native_workers, tmp_path, capsys):
    native_workers.run(_native_resume_worker, str(tmp_path))
    with capsys.disabled():
        for rank in range(native_workers.world_size):
            print((tmp_path / f"native-identity-{rank}.json").read_text(), flush=True)
