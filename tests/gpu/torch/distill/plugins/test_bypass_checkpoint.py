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
import random
import shutil
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from _test_utils.bypass import bypass_problem
from _test_utils.torch.distributed.utils import bounded_collectives

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill._bypass_capture import ModuleBypassAdapter
from modelopt.torch.distill._bypass_checkpoint import _batch_hash, _model_hash, _rng_state
from modelopt.torch.distill._bypass_optimizer import Lion
from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
from modelopt.torch.quantization.utils.native_weight import pin_native_weight


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
    def __init__(self, batches, *, interrupted=False, snapshot_failure=False):
        self.batches, self.position, self.interrupted = batches, 0, interrupted
        self.snapshot_failure = snapshot_failure
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
        random.random()
        np.random.random()
        torch.rand(())
        return batch

    def state_dict(self):
        if self.snapshot_failure:
            raise OSError("Simulated data-cursor snapshot failure")
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def _committed(root, *, size=1, rank=0):
    generation = root / (root / "latest").read_text().strip()
    manifest = json.loads((generation / "COMMITTED.json").read_text())
    assert manifest["world_size"] == size
    return generation, torch.load(generation / f"rank-{rank}.pt", weights_only=True)


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("stateful", [False, True])
def test_bypass_resume_matches_uninterrupted(tmp_path, monkeypatch, stateful):
    initial, teacher, batches = bypass_problem()

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
            return _StatefulCursor(data, interrupted=interrupted)
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

    def failed_write(*args, **kwargs):
        raise OSError("Simulated checkpoint disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", failed_write)
        with pytest.raises(ValueError):
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
        assert actual["data_position"]["loader_state"] == {"position": 4}
        assert "prefix_hashes" not in actual["data_position"]
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


def _distributed_resume_worker(rank, size, directory, failure=None):
    from torch.distributed.device_mesh import init_device_mesh

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
    with bounded_collectives(10), pytest.raises(ValueError):
        mtd.bypass(
            copy.deepcopy(initial),
            copy.deepcopy(teacher),
            _StatefulCursor(local, snapshot_failure=rank == 1),
            {**recipe, "checkpoint_dir": str(root / "snapshot-failure")},
            device_mesh=mesh,
        )
    assert not (root / "snapshot-failure" / "latest").exists()
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
    "failure", [None, "priming_transfer", "priming_validation", "optimizer_restore", "rng_restore"]
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
    from _test_utils.transformers_families import build_model, cases
    from torch.distributed.device_mesh import init_device_mesh
    from transformers.distributed.utils import TransformersDeviceMesh

    import modelopt.torch.quantization as mtq
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
    mtd.bypass(
        student,
        teacher,
        local,
        {**recipe, "checkpoint_dir": str(root / "resumed"), "resume_from": str(root / "resumed")},
        device_mesh=mesh,
    )
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


@pytest.mark.parametrize("native_workers", [2, 4], indirect=True, ids=["fsdp2", "fsdp2_tp2_ep4"])
@pytest.mark.timeout(300)
def test_native_resume_with_empty_owners_and_frozen_nvfp4(native_workers, tmp_path, capsys):
    native_workers.run(_native_resume_worker, str(tmp_path))
    with capsys.disabled():
        for rank in range(native_workers.world_size):
            print((tmp_path / f"native-identity-{rank}.json").read_text(), flush=True)
