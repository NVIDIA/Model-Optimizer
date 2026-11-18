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

"""Native packed TP projections and VJPs, mesh adoption and four-GPU placement checks."""

import copy
from collections import OrderedDict
from contextlib import ExitStack
from datetime import timedelta
from functools import cache
from importlib import import_module
from inspect import unwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import (
    DistributedWorkerPool,
    bounded_collectives,
    default_worker_teardown,
    get_live_pool,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.nn import functional as F

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill._bypass_capture import (
    BypassUnit,
    CaptureRecord,
    ModuleBypassAdapter,
    mse_statistics,
)
from modelopt.torch.distill._bypass_optimizer import Lion
from modelopt.torch.distill._bypass_parallel import (
    BypassMeshContext,
    prepare_quantizer_partitions,
    torch_mesh_context,
)
from modelopt.torch.distill._bypass_runner import _BypassRunner
from modelopt.torch.distill._bypass_runtime import DistributedBypassRuntime
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor, QTensorWrapper
from modelopt.torch.quantization.repair import IQRepairPolicy, plan_iq_repair
from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp
from modelopt.torch.quantization.utils.native_weight import get_pinned_native_weight


@cache
def _runtime_mesh(axis):
    # Reuse bounded groups until persistent-worker teardown, including native tensor collectives.
    group = dist.new_group(timeout=timedelta(seconds=10))
    return DeviceMesh.from_group(group, "cpu", mesh_dim_names=(axis,))


@cache
def _singleton_mesh(rank):
    group = dist.new_group([rank], timeout=timedelta(seconds=10), use_local_synchronization=True)
    return DeviceMesh.from_group(group, "cpu", mesh_dim_names=("dp",))


@cache
def _data_tensor_mesh():
    groups = [
        dist.new_group(ranks, timeout=timedelta(seconds=10))
        for ranks in ([0, 2], [1, 3], [0, 1], [2, 3])
    ]
    data_rank, tensor_rank = divmod(dist.get_rank(), 2)
    return DeviceMesh.from_group(
        [groups[tensor_rank], groups[2 + data_rank]],
        "cpu",
        mesh=torch.arange(4).reshape(2, 2),
        mesh_dim_names=("dp", "tp"),
    )


def _measurement_metadata_worker(rank, size, path):
    """Check batched ownership agreement and object-free healthy metadata on real Gloo."""
    mesh = _runtime_mesh("tp")
    config = mtd.BypassConfig(global_batch_size=2, micro_batch_size=1, sequence_length=4)
    runtime = DistributedBypassRuntime(torch_mesh_context(mesh), config)
    model = torch.nn.Identity()
    runner = _BypassRunner(model, model, [], None, config, ModuleBypassAdapter(), runtime)
    batch = torch.ones(2, 4, 8)
    reductions, objects, transfers = [], [], []
    reduce, gather, tolist = dist.all_reduce, dist.all_gather_object, torch.Tensor.tolist

    def observed_reduce(value, *args, **kwargs):
        if not value.is_floating_point():
            reductions.append(value.numel())
        return reduce(value, *args, **kwargs)

    def observed_gather(*args, **kwargs):
        objects.append(True)
        return gather(*args, **kwargs)

    def observed_tolist(value, *args, **kwargs):
        if value.dtype == torch.int32:
            transfers.append(value.numel())
        return tolist(value, *args, **kwargs)

    with (
        bounded_collectives(10),
        patch.object(dist, "all_reduce", observed_reduce),
        patch.object(dist, "all_gather_object", observed_gather),
        patch.object(torch.Tensor, "tolist", observed_tolist),
    ):
        if path == "boundaries":
            units = [BypassUnit(name, "", "") for name in ("one", "two", "three")]
            for unit in units:
                runtime.record(unit, CaptureRecord((), {}, batch))
            for _ in range(2):
                statistics = {unit.name: [torch.tensor(12.0), torch.tensor(6)] for unit in units}
                actual = runtime.reduce_statistics(statistics, list(statistics))
                assert all(error == 12 and count == 6 for error, count in actual.values())
            assert reductions.count(6) == 2, reductions
            assert transfers == [6, 6]
            assert not objects
            # Placement changes must still refuse after successful measurements, including
            # peer disagreement and a locally mixed full/sharded record history.
            runtime.boundary_modes["one"] = 2 if rank else 1
            with pytest.raises(ValueError, match="TP boundary loss ownership"):
                runtime.reduce_statistics(statistics, list(statistics))
            runtime.boundary_modes["one"] = 3
            with pytest.raises(ValueError, match="TP boundary loss ownership"):
                runtime.reduce_statistics(statistics, list(statistics))
        elif path == "window":
            assert runner._window(iter([batch]))[0] is batch
            assert not objects
            # Equal sample totals with different microbatch schedules still refuse.
            batches = [batch] if rank == 0 else [batch[:1], batch[1:]]
            config.micro_batch_size = 2
            with pytest.raises(ValueError, match="unequal microbatch schedules"):
                runner._window(iter(batches))
            assert not objects
        else:
            validated = list(runner._validation_batches([batch]))
            assert len(validated) == 1 and validated[0] is batch
            assert not objects
            with pytest.raises(ValueError, match="unequal validation schedules"):
                list(runner._validation_batches([batch] if rank == 0 else []))
            assert not objects

            def broken_reader():
                if rank:
                    raise OSError("held-out read failed")
                yield batch

            with pytest.raises(ValueError, match="held-out read failed"):
                list(runner._validation_batches(broken_reader()))
            assert objects  # Message exchange is reserved for actual reader failures.


@pytest.mark.parametrize("path", ["boundaries", "window", "validation"])
@pytest.mark.timeout(180)
def test_measurement_metadata_uses_batched_integer_collectives(bypass_cpu_workers, path):
    bypass_cpu_workers.run(_measurement_metadata_worker, path)


def _replicated_field_worker(rank, size, expert, full_boundary=False):
    torch.manual_seed(701)
    inputs = [torch.randn(1, 3, 256) * scale for scale in (0.5, 2.0)]
    if full_boundary:
        inputs = [inputs[0]] * size  # Full-loss TP boundaries consume identical samples.
    teacher = torch.nn.Linear(256, 16, bias=False)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda model: model(inputs[0]),
    )
    reference = copy.deepcopy(student)
    if expert:
        student._is_expert_parallel = True
    mesh = _runtime_mesh("tp")
    context = BypassMeshContext(
        device_mesh=mesh, tensor_mesh=mesh, expert_replica_mesh=mesh if expert else None
    )
    runtime = DistributedBypassRuntime(context, mtd.BypassConfig(global_batch_size=1))
    runtime.student = student
    policy, expected_policy = IQRepairPolicy(), IQRepairPolicy()
    try:
        fields, expected_fields = policy.prepare(student), expected_policy.prepare(reference)
        if full_boundary:
            # Explicit False declares native partial VJPs, not unknown full-loss replicas.
            runtime.layouts = {
                id(binding.quantizer): dict.fromkeys(dict(state.named_parameters()), False)
                for binding, state, _, _ in policy.adapters
            }
        runtime.register(student, policy)
        runtime.validate_gradient_schedules({"mlp": fields})
        # Native partial VJPs must sum even when the boundary target is a full replica.
        target = teacher(inputs[rank]).detach()
        if not full_boundary:
            target = DTensor.from_local(target, mesh, [Shard(-1)])
        runtime.record(BypassUnit("mlp", "", ""), CaptureRecord((), {}, target))

        def contribution(model, x, owner):
            prediction, target = model(x), teacher(x).detach()
            if full_boundary:
                prediction = prediction.chunk(size, dim=-1)[owner]
                target = target.chunk(size, dim=-1)[owner]
            return F.mse_loss(prediction, target, reduction="sum")

        sum(contribution(reference, x, owner) for owner, x in enumerate(inputs)).backward()
        contribution(student, inputs[rank], rank).backward()
        # Each rank contributes a different real native-field VJP.
        assert any(not torch.allclose(p.grad, q.grad) for p, q in zip(fields, expected_fields))
        runtime.reduce_gradients(fields)
        for actual, expected in zip(fields, expected_fields, strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=1e-8)
        expected_norm = torch.nn.utils.clip_grad_norm_(expected_fields, 1e-4)
        actual_norm = runtime.clip_gradients(fields, 1e-4)
        torch.testing.assert_close(actual_norm, expected_norm)
        for actual, expected in zip(fields, expected_fields, strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=1e-10)
        for masters, active in ((fields, policy), (expected_fields, expected_policy)):
            Lion(masters, lr=1e-3).step()
            active.project_()
            active.finalize_()
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(student.state_dict()[name], expected, rtol=0, atol=0)
    finally:
        policy.close()
        expected_policy.close()


@pytest.mark.parametrize(
    ("expert", "full_boundary"),
    [(False, False), (True, False), (False, True)],
    ids=["dense-partial", "expert-partial", "explicit-false-full-boundary"],
)
@pytest.mark.timeout(180)
def test_replicated_native_fields_reduce_once_and_match_reference(
    bypass_cpu_workers, expert, full_boundary
):
    bypass_cpu_workers.run(_replicated_field_worker, expert, full_boundary)


class _TPBoundary(torch.nn.Module):
    def __init__(self, mixed):
        super().__init__()
        self.dense = torch.nn.Linear(256, 16, bias=False)
        self.parallel = torch.nn.Linear(256, 16, bias=False) if mixed else None
        self.mesh, self.layout = None, "full"

    def forward(self, inputs):
        output = self.dense(inputs)
        if self.parallel is not None:
            partial = self.parallel(inputs)
            output = output + (partial.full_tensor() if isinstance(partial, DTensor) else partial)
        if self.mesh is not None and self.layout in (
            "replicate",
            "shard",
            "sequence_shard",
            "partial",
        ):
            placement = Replicate()
            if self.layout in ("shard", "sequence_shard"):
                dimension = -1 if self.layout == "shard" else 1
                output = output.chunk(self.mesh.size(), dim=dimension)[self.mesh.get_local_rank()]
                placement = Shard(dimension)
            elif self.layout == "partial":
                output = output / self.mesh.size()
                placement = Partial()
            output = DTensor.from_local(output, self.mesh, [placement])
        return output


class _TPModel(torch.nn.Module):
    def __init__(self, mixed):
        super().__init__()
        self.mlp = _TPBoundary(mixed)

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


def _public_tp_loss_worker(rank, size, layout, max_norm):
    torch.manual_seed(704)
    cross_data = layout == "dp_tp"
    unsupported = layout in (
        "sequence_shard",
        "partial",
        "asymmetric_sequence_shard",
        "asymmetric_partial",
    )
    teacher = _TPModel(layout == "mixed" or unsupported)
    batch = torch.randn(2 if cross_data else 1, 4, 256)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda model: model(batch),
    )
    reference = copy.deepcopy(student)
    config = {
        "num_steps": 1,
        "global_batch_size": batch.shape[0],
        "micro_batch_size": 1,
        "sequence_length": 4,
        "max_grad_norm": max_norm,
        "warmup_fraction": 0,
        "scheduler": "constant",
    }
    mesh = _data_tensor_mesh() if cross_data else _runtime_mesh("tp")
    tensor_mesh = mesh["tp"] if cross_data else mesh
    data = (
        {"hidden_states": batch, "valid_mask": torch.tensor([[True] * 4, [False] * 4])}
        if cross_data
        else batch
    )
    observations = []
    original = DistributedBypassRuntime.clip_gradients

    def observe(runtime, fields, bound):
        before = [p.grad.detach().clone() for p in fields]
        norm = original(runtime, fields, bound)
        observations.append(
            (
                before,
                norm.detach().clone(),
                [p.grad.detach().clone() for p in fields],
                [id(p) in runtime.tp_parameters for p in fields],
            )
        )
        return norm

    DistributedBypassRuntime.clip_gradients = observe
    try:
        mtd.bypass(
            reference, copy.deepcopy(teacher), [data], config, device_mesh=_singleton_mesh(rank)
        )
        for model in (student, teacher):
            model.mlp.mesh, model.mlp.layout = tensor_mesh, layout
            if cross_data:
                model.mlp.layout = "shard" if mesh["dp"].get_local_rank() else "full"
            if layout == "disagree":
                model.mlp.layout = "shard" if rank else "full"
            if unsupported:
                model.mlp.layout = layout.removeprefix("asymmetric_")
                if layout.startswith("asymmetric_") and rank == 0:
                    model.mlp.layout = "full"
            if layout == "mixed" or unsupported:
                parallelize_module(
                    model.mlp.parallel, mesh, ColwiseParallel(use_local_output=False)
                )
        if unsupported:
            before = {
                name: (value.to_local() if isinstance(value, DTensor) else value).clone()
                for name, value in student.state_dict().items()
            }
            with (
                bounded_collectives(10),
                pytest.raises(ValueError, match="Unsupported TP boundary target placement"),
            ):
                mtd.bypass(student, teacher, [data], config, device_mesh=mesh)
            for name, value in student.state_dict().items():
                local = value.to_local() if isinstance(value, DTensor) else value
                assert local.dtype == before[name].dtype
                torch.testing.assert_close(local, before[name], rtol=0, atol=0)
            assert not any("_local_repair_state" in name for name in student.state_dict())
            return
        if layout == "disagree":
            before = copy.deepcopy(student.state_dict())
            with pytest.raises(ValueError, match="TP boundary loss ownership"):
                mtd.bypass(student, teacher, [data], config, device_mesh=mesh)
            for name, value in student.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            return
        if cross_data:
            lane = mesh["dp"].get_local_rank()
            data = {name: value[lane : lane + 1] for name, value in data.items()}
        mtd.bypass(student, teacher, [data], config, device_mesh=mesh)
    finally:
        DistributedBypassRuntime.clip_gradients = original
    expected, actual = observations
    assert expected[1] > 0
    for index in (0, 2):
        for value, target, sharded in zip(actual[index], expected[index], actual[3], strict=True):
            if sharded:
                target = target.chunk(size, dim=0)[rank]
            # Mixed FP32 matmul/VJP ordering uses a field-scale cancellation budget.
            # Recompute after clipping; final native fields still require exact equality.
            atol = max(1e-10, 2e-5 * target.abs().amax().item()) if layout == "mixed" else 1e-10
            torch.testing.assert_close(value, target, rtol=2e-5, atol=atol)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=1e-10)
    if max_norm < expected[1]:
        assert any(not torch.equal(a, b) for a, b in zip(actual[0], actual[2], strict=True))
    for name, target in reference.state_dict().items():
        value = student.state_dict()[name]
        value = value.full_tensor() if isinstance(value, DTensor) else value
        if value.shape != target.shape:
            target = target.chunk(size, dim=0)[rank]
        assert value.dtype == target.dtype, name
        torch.testing.assert_close(value.to(device=target.device), target, rtol=0, atol=0)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("layout", ["full", "replicate", "shard", "mixed"])
@pytest.mark.parametrize("max_norm", [1e8, 1e-4])
def test_public_tp_bypass_counts_each_boundary_loss_once(bypass_cpu_workers, layout, max_norm):
    bypass_cpu_workers.run(_public_tp_loss_worker, layout, max_norm)


@pytest.mark.timeout(180)
def test_public_tp_boundary_ownership_disagreement_refuses_without_update(bypass_cpu_workers):
    bypass_cpu_workers.run(_public_tp_loss_worker, "disagree", 1e8)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "layout", ["sequence_shard", "partial", "asymmetric_sequence_shard", "asymmetric_partial"]
)
def test_public_tp_unsupported_target_placement_refuses_without_update(bypass_cpu_workers, layout):
    timeout = bypass_cpu_workers.job_timeout
    try:
        bypass_cpu_workers.job_timeout = 60
        bypass_cpu_workers.run(_public_tp_loss_worker, layout, 1e8)
    finally:
        bypass_cpu_workers.job_timeout = timeout


def _plain_tp_owner_worker(rank, size, layout):
    torch.manual_seed(718)
    teacher = torch.nn.Sequential(OrderedDict([("mlp", torch.nn.Linear(256, 16, bias=False))]))
    if layout == "equal_slice":
        with torch.no_grad():
            teacher.mlp.weight[8:].copy_(teacher.mlp.weight[:8])
    batch = torch.randn(1, 4, 256)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda model: model(batch),
    )
    mesh = _runtime_mesh("tp")
    for model in (student, teacher):
        parallelize_module(
            model.mlp,
            mesh,
            ColwiseParallel(
                output_layouts=Replicate() if layout == "local_full" else Shard(-1),
                use_local_output=not (layout == "asymmetric" and rank == 0),
            ),
        )
    if layout == "equal_slice":
        actual = teacher(batch)
        peers = [torch.empty_like(actual) for _ in range(size)]
        dist.all_gather(peers, actual, group=mesh.get_group())
        assert all(torch.equal(actual, peer) for peer in peers)
    before = {
        name: (value.to_local() if isinstance(value, DTensor) else value).clone()
        for name, value in student.state_dict().items()
    }
    replays = []
    handle = student.mlp.register_forward_pre_hook(lambda *args: replays.append(True))
    try:
        with bounded_collectives(10), pytest.raises(ValueError):
            mtd.bypass(
                student,
                teacher,
                [batch],
                {"num_steps": 1, "global_batch_size": 1, "sequence_length": 4},
                device_mesh=mesh,
            )
    finally:
        handle.remove()
    assert not replays, "Ambiguous plain output must refuse before native peer replay"
    for name, value in student.state_dict().items():
        local = value.to_local() if isinstance(value, DTensor) else value
        assert local.dtype == before[name].dtype
        torch.testing.assert_close(local, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["local_slice", "equal_slice", "local_full", "asymmetric"])
@pytest.mark.timeout(180)
def test_plain_tp_native_owner_refuses_erased_output(bypass_cpu_workers, layout):
    timeout = bypass_cpu_workers.job_timeout
    try:
        bypass_cpu_workers.job_timeout = 60
        bypass_cpu_workers.run(_plain_tp_owner_worker, layout)
    finally:
        bypass_cpu_workers.job_timeout = timeout


def _hf_plain_projection_worker(rank, size, style):
    # The native HF runtime is optional; this worker executes its real resolved plan.
    from transformers import LlamaConfig, LlamaForCausalLM

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    torch.manual_seed(719)
    teacher = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=False,
        )
    ).eval()
    teacher.config._attn_implementation = "eager"
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    path = "model.layers.0.mlp.gate_proj"
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": path + ".weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda model: model(**batch),
    )
    reference = copy.deepcopy(student)
    config = {
        "num_steps": 1,
        "global_batch_size": 1,
        "sequence_length": 4,
        "modules": [path],
        "max_grad_norm": 1e-5,
        "scheduler": "constant",
        "warmup_fraction": 0,
    }
    observations = []
    original_clip = DistributedBypassRuntime.clip_gradients

    def observe(runtime, fields, bound):
        before = [p.grad.detach().clone() for p in fields]
        norm = original_clip(runtime, fields, bound)
        observations.append(
            (before, norm.detach().clone(), [p.grad.detach().clone() for p in fields])
        )
        return norm

    counts = []
    original_statistics = DistributedBypassRuntime.reduce_statistics

    def count_statistics(runtime, statistics, names):
        result = original_statistics(runtime, statistics, names)
        if runtime.context.tensor_mesh is not None and runtime.context.tensor_mesh.size() > 1:
            counts.append(result[path][1].item())
        return result

    mesh = _runtime_mesh("tp")
    adapter = TransformersBypassAdapter(mesh)
    with (
        patch.object(DistributedBypassRuntime, "clip_gradients", observe),
        patch.object(DistributedBypassRuntime, "reduce_statistics", count_statistics),
    ):
        mtd.bypass(
            reference, copy.deepcopy(teacher), [batch], config, device_mesh=_singleton_mesh(rank)
        )
        for model in (student, teacher):
            plan = dict(model.tp_plan)
            native_style = "colwise" if style == "conflict" else style
            if style == "colwise_rep":
                plan.update(
                    {
                        "model.layers.*.mlp.up_proj": "colwise_rep",
                        "model.layers.*.mlp.down_proj": "rowwise_split_input",
                    }
                )
            if style == "local_params":
                # Keep a valid full MLP while the exact selected output style is unknown.
                plan.update(
                    {
                        "model.layers.*.mlp." + name: "replicate"
                        for name in ("gate_proj", "up_proj", "down_proj")
                    }
                )
            plan[path.replace(".0.", ".*.")] = native_style
            model._tp_plan = plan
            expanded = model._expand_parallel_plan(plan)
            assert expanded[path] == expanded[path + ".weight"] == native_style
            print("HF exact plan", rank, style, path, expanded[path], flush=True)
            adapter.parallelize(model)
            # Fresh public adapters must consume applied facts, not these edited rules.
            model._tp_plan = {"model.layers.*.mlp.gate_proj": "local_params"}
        if style == "conflict":
            teacher.get_submodule(path).register_forward_hook(
                lambda module, args, output: DTensor.from_local(output, mesh, [Replicate()])
            )
        if style in ("local_params", "conflict"):
            before = {
                name: (value.to_local() if isinstance(value, DTensor) else value).clone()
                for name, value in student.state_dict().items()
            }
            with bounded_collectives(10), pytest.raises(ValueError):
                mtd.bypass(student, teacher, [batch], config, device_mesh=mesh)
            for name, value in student.state_dict().items():
                local = value.to_local() if isinstance(value, DTensor) else value
                torch.testing.assert_close(local, before[name], rtol=0, atol=0)
            return
        boundary = teacher.get_submodule(path)
        outputs = []
        handle = boundary.register_forward_hook(
            lambda module, args, output: outputs.append(tuple(output.shape))
        )
        try:
            mtd.bypass(student, teacher, [batch], config, device_mesh=mesh)
        finally:
            handle.remove()
        width = 512 // size if style == "colwise" else 512
        assert outputs and all(shape == (1, 4, width) for shape in outputs), outputs
        assert counts == [4 * 512], counts
        print("HF boundary output/count", rank, style, outputs, counts, flush=True)
    expected, actual = observations
    for index in (0, 2):
        for value, target in zip(actual[index], expected[index], strict=True):
            target = target.chunk(size, dim=0)[rank] if target.shape != value.shape else target
            torch.testing.assert_close(value, target, rtol=2e-4, atol=1e-9)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-4, atol=1e-9)
    for name, target in reference.state_dict().items():
        value = student.state_dict()[name]
        value = value.full_tensor() if isinstance(value, DTensor) else value
        if value.shape != target.shape:
            target = target.chunk(size, dim=0)[rank]
        assert value.dtype == target.dtype, name
        torch.testing.assert_close(value, target, rtol=0, atol=0)


@pytest.mark.parametrize("style", ["colwise", "colwise_rep", "local_params", "conflict"])
@pytest.mark.timeout(180)
def test_plain_tp_native_hf_projection_matches_reference(bypass_cpu_workers, style):
    pytest.importorskip("transformers", minversion="5.23")
    timeout = bypass_cpu_workers.job_timeout
    try:
        bypass_cpu_workers.job_timeout = 90
        bypass_cpu_workers.run(_hf_plain_projection_worker, style)
    finally:
        bypass_cpu_workers.job_timeout = timeout


@pytest.fixture
def data_tensor_cpu_workers(_dist_pools):
    pool = get_live_pool(
        _dist_pools,
        ("bypass_cpu", 4),
        lambda: DistributedWorkerPool(
            world_size=4, backend="gloo", teardown_fn=default_worker_teardown, ready_timeout=180
        ),
    )
    pool.job_timeout = 120
    return pool


@pytest.mark.timeout(180)
def test_public_dp_tp_bypass_with_empty_partitioned_lane(data_tensor_cpu_workers):
    data_tensor_cpu_workers.run(_public_tp_loss_worker, "dp_tp", 1e-4)


def _functional_child_worker(rank, size, route):
    native = route != "generic"
    # Optional Transformers integration must not gate the generic Torch workers.
    from modelopt.torch.distill.plugins.transformers.functional_fsdp import (
        install_functional_fsdp_calls,
    )

    class FunctionalParent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Conv1d(4, 8, 3, bias=False)

        def forward(self, inputs):
            return F.conv1d(inputs, self.conv1d.weight)

    torch.manual_seed(702)
    mesh = _runtime_mesh("fsdp")
    if native:
        from _test_utils.transformers_families import build_model, cases
        from transformers.distributed.configuration_utils import DistributedConfig
        from transformers.distributed.utils import TransformersDeviceMesh

        from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

        local = _singleton_mesh(rank).get_group()
        groups = [local, mesh.get_group(), local]
        ranks = torch.arange(size).reshape(1, size, 1)
        manager = TransformersDeviceMesh(
            DeviceMesh.from_group(groups, "cpu", ranks, mesh_dim_names=("pp", "fsdp", "tp")),
            DeviceMesh.from_group(groups, "cpu", ranks, mesh_dim_names=("pp", "efsdp", "ep")),
        )
        model = build_model(cases(["qwen3_next"])[0], "cpu").eval()
        decoder = next(block for block in model.model.layers if hasattr(block, "linear_attn"))
        parent = decoder.linear_attn
        path = next(name for name, module in model.named_modules() if module is parent.conv1d)
        config = DistributedConfig(
            fsdp_size=size,
            **(
                {"fsdp_plan": {"model.layers.*.linear_attn.conv1d": "free_full_weight"}}
                if route == "native-adopted"
                else {}
            ),
        )
        model.config.distributed_config = config
        plan = model._resolve_parallel_plans(config)[2]
        reference = copy.deepcopy(parent)
        original_forward = parent.__dict__.get("forward")
        inputs = torch.randn(1, 4, 256, dtype=next(reference.parameters()).dtype)
        adapter = TransformersBypassAdapter(manager)
        if route == "native-adopted":
            assert plan[path] == "free_full_weight"
            # HF places the model; no ModelOpt saved placement exists before fresh adoption.
            model.maybe_distribute_model(model, config, manager)
            assert not hasattr(model, "_modelopt_bypass_placement")
            assert parent.__dict__.get("forward") is original_forward
            adapter.adopt(model)
        else:
            assert path not in plan
            adapter.parallelize(model)
        retained = adapter.context.parallel_metadata[id(model)]["plans"]["fsdp"]
        assert tuple(retained.items()) == tuple(plan.items())
        outer_group = fully_shard.state(decoder)._fsdp_param_group
        if route == "native-default":
            assert not isinstance(parent.conv1d, torch.distributed.fsdp.FSDPModule)
            assert parent.__dict__.get("forward") is original_forward
            assert any(p._module_info.module is parent.conv1d for p in outer_group.fsdp_params)
        else:
            child_group = fully_shard.state(parent.conv1d)._fsdp_param_group
            assert child_group is not outer_group
            assert {id(p.sharded_param) for p in child_group.fsdp_params}.isdisjoint(
                id(p.sharded_param) for p in outer_group.fsdp_params
            )
    else:
        model = torch.nn.Module()
        model.config = SimpleNamespace(model_type="qwen3_next")
        model.model = torch.nn.Module()
        block = torch.nn.Module()
        block.linear_attn = FunctionalParent()
        model.model.layers = torch.nn.ModuleList([block])
        parent = block.linear_attn
        reference = copy.deepcopy(parent)
        inputs = torch.randn(2, 4, 6)
        fully_shard(parent.conv1d, mesh=mesh, reshard_after_forward=True)
    expected_shape = tuple(reference.conv1d.weight.shape)
    install_functional_fsdp_calls(model)
    forward = parent.forward
    install_functional_fsdp_calls(model)
    if route == "native-default":
        assert parent.__dict__.get("forward") is original_forward
    else:
        assert parent.forward is forward
    reads = []
    if native:
        native_module = import_module(type(reference).__module__)
        conv = unwrap(native_module.causal_conv1d_fn)

        def checked_conv(inputs, weight, *args, **kwargs):
            assert tuple(weight.shape) == (expected_shape[0], expected_shape[2])
            torch.testing.assert_close(weight, reference.conv1d.weight.squeeze(1), rtol=0, atol=0)
            reads.append(tuple(weight.shape))
            return conv(inputs, weight, *args, **kwargs)

    with ExitStack() as stack, bounded_collectives(10), torch.no_grad():
        if native:
            # CPU ownership/lifecycle needs native CPU math, independent of installed CUDA kernels.
            for name in (
                "causal_conv1d_fn",
                "causal_conv1d_update",
                "torch_chunk_gated_delta_rule",
                "torch_recurrent_gated_delta_rule",
            ):
                stack.enter_context(
                    patch.object(native_module, name, unwrap(getattr(native_module, name)))
                )
        for _ in range(2):
            expected = reference(inputs)
            if native:
                # Native parent weights belong to the decoder; enter its actual FSDP forward.
                with (
                    patch.object(decoder, "forward", lambda value: parent(value)),
                    patch.object(native_module, "causal_conv1d_fn", checked_conv),
                ):
                    actual = decoder(inputs)
            else:
                actual = parent(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            weight = parent.conv1d.weight
            assert isinstance(weight, DTensor) and weight.placements == (Shard(0),)
            assert tuple(weight.to_local().shape) == (
                expected_shape[0] // size,
                *expected_shape[1:],
            )
    if native:
        assert len(reads) == 2
    if route != "native-default":
        parent.conv1d = fully_shard(copy.deepcopy(reference.conv1d), mesh=mesh)
        with pytest.raises(ValueError):
            install_functional_fsdp_calls(model)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("route", ["generic", "native-default", "native-adopted"])
def test_functional_fsdp_child_gathers_weight_and_reshards(bypass_cpu_workers, route):
    pytest.importorskip("transformers", minversion="5.23")
    timeout = bypass_cpu_workers.job_timeout
    try:
        bypass_cpu_workers.job_timeout = 120
        bypass_cpu_workers.run(_functional_child_worker, route)
    finally:
        bypass_cpu_workers.job_timeout = timeout


class _DocumentTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="bypass_document_test", _attn_implementation="sdpa"
        )
        self.mlp = torch.nn.Linear(256, 16, bias=False)
        self.group = None
        self.seen = []

    def forward(self, input_ids, position_ids=None, attention_mask=None, **kwargs):
        if self.group is not None:
            dist.all_reduce(torch.zeros(1), group=self.group)
        self.seen.append((input_ids.shape[-1], attention_mask is None))
        hidden = F.one_hot(input_ids, 256).float().cumsum(1)
        return self.mlp(hidden)


def _unequal_documents_worker(rank, size):
    # Optional Transformers integration must not gate the generic Torch workers.
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    torch.manual_seed(703)
    teacher = _DocumentTeacher()
    tokens = torch.tensor([[1, 3, 5, 7], [9, 11, 13, 15]])
    docs = torch.tensor([[1, 1, 2, 2], [1, 1, 1, 1]])
    batch = {
        "input_ids": tokens,
        "_packed_seq_ids": docs,
        "attention_mask": torch.ones(2, 1, 4, 4, dtype=torch.bool).tril_(),
    }
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda model: model(tokens),
    )
    reference = copy.deepcopy(student)
    units = [BypassUnit("mlp", "mlp", "mlp")]
    config = mtd.BypassConfig(
        num_steps=1,
        global_batch_size=2,
        sequence_length=4,
        max_grad_norm=1e-5,
        warmup_fraction=0,
        scheduler="constant",
    )
    expected_adapter = TransformersBypassAdapter()
    expected_adapter.student = reference
    records = []
    expected_adapter.teacher_microbatch(
        teacher, units, batch, lambda unit, record: records.append(record)
    )
    expected_statistics = [torch.zeros(()), torch.zeros((), dtype=torch.long)]
    for record in records:
        error, count = mse_statistics(
            expected_adapter.replay(reference.mlp, record), record.target, record.valid_mask
        )
        expected_statistics[0] += error.detach()
        expected_statistics[1] += count
    assert expected_statistics[1] == 8 * 16
    expected_gradients = []
    original_step = Lion.step

    def observed_step(optimizer, closure=None):
        expected_gradients.append(
            [p.grad.clone() for group in optimizer.param_groups for p in group["params"]]
        )
        return original_step(optimizer, closure)

    def run(model, target, adapter, data, runtime=None):
        plan = plan_iq_repair([model.mlp], root_model=model)
        runner = _BypassRunner(model, target, units, plan, config, adapter, runtime)
        # Observe the actual training statistics without substituting runtime math.
        original = runner._update

        def update(statistics, step):
            for actual, expected in zip(statistics["mlp"], expected_statistics, strict=True):
                torch.testing.assert_close(actual, expected)
            return original(statistics, step)

        runner._update = update
        runner.run([data])

    Lion.step = observed_step
    try:
        run(reference, copy.deepcopy(teacher), expected_adapter, batch)
        mesh = _runtime_mesh("dp")
        adapter = TransformersBypassAdapter()
        adapter.student = student
        adapter.context = torch_mesh_context(mesh)
        teacher.group = mesh.get_group()
        teacher.seen.clear()
        runtime = DistributedBypassRuntime(adapter.context, config)
        local = {key: value[rank : rank + 1] for key, value in batch.items()}
        objects = []
        original_gather = dist.all_gather_object

        def observed_gather(*args, **kwargs):
            objects.append(args[1])
            return original_gather(*args, **kwargs)

        dist.all_gather_object = observed_gather
        try:
            with bounded_collectives(10):
                adapter.teacher_microbatch(teacher, units, local, lambda *_: None)
                assert not objects
                invalid = {**local, "_packed_seq_ids": -local["_packed_seq_ids"]} if rank else local
                with pytest.raises(ValueError) as refusal:
                    adapter.teacher_microbatch(teacher, units, invalid, lambda *_: None)
                assert "nonnegative" in str(refusal.value)
                assert len(objects) == 1
        finally:
            dist.all_gather_object = original_gather
        teacher.seen.clear()
        with bounded_collectives(10):
            run(student, teacher, adapter, local, runtime)
    finally:
        Lion.step = original_step
    for actual, expected in zip(expected_gradients[1], expected_gradients[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-10)
    # Rank 1's second forward is one safe all-invalid token with no caller mask.
    assert teacher.seen == ([(2, False), (2, False)] if rank == 0 else [(4, False), (1, False)])
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(student.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.timeout(180)
def test_unequal_document_slots_have_zero_contribution(bypass_cpu_workers):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_unequal_documents_worker)


def _asymmetric_gradient_schedule_worker(rank, size, mismatch):
    from torch.distributed.device_mesh import init_device_mesh

    from modelopt.torch.distill._bypass_parallel import BypassMeshContext
    from modelopt.torch.distill._bypass_runtime import DistributedBypassRuntime

    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("dp",))
    runtime = DistributedBypassRuntime(
        BypassMeshContext(device_mesh=mesh, data_mesh=mesh, data_context_mesh=mesh),
        mtd.BypassConfig(global_batch_size=size),
    )
    parameter = torch.nn.Parameter(torch.ones(2 if rank == 0 or mismatch != "shape" else 3))
    runtime.parameter_names = {id(parameter): "mlp.weight_quantizer.global_delta"}
    runtime.parameter_meshes[id(parameter)] = [mesh]
    fields = [] if mismatch == "missing-group" and rank else [parameter]
    with pytest.raises(ValueError, match=r"gradient.*schedule"):
        runtime.validate_gradient_schedules({"mlp": fields})


@pytest.mark.parametrize("mismatch", ["missing-group", "shape"])
def test_gradient_schedule_asymmetry_is_refused_before_reductions(bypass_cpu_workers, mismatch):
    bypass_cpu_workers.run(_asymmetric_gradient_schedule_worker, mismatch)


def _unique_batch_worker(rank, size, axis, unequal):
    from torch.distributed.device_mesh import init_device_mesh

    from modelopt.torch.distill._bypass_optimizer import Lion

    torch.manual_seed(67)
    teacher = torch.nn.Sequential(
        OrderedDict(
            [
                ("mlp", torch.nn.Linear(256, 256, bias=False)),
                ("activation", torch.nn.SiLU()),
                ("moe", torch.nn.Linear(256, 256, bias=False)),
            ]
        )
    )
    batch_size = 2 if axis == "fsdp" else 1
    batches = [torch.randn(batch_size, 4, 256) for _ in range(4)]
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda m: m(batches[0]),
    )
    config = {
        "num_steps": 2,
        "global_batch_size": 2 * batch_size,
        "micro_batch_size": 1,
        "sequence_length": 4,
        "warmup_fraction": 0,
        "scheduler": "constant",
        "max_grad_norm": 1e-8,
    }
    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=(axis,))
    single = init_device_mesh("cpu", (size, 1), mesh_dim_names=("owners", "dp"))["dp"]
    reference = copy.deepcopy(student)
    gradients, norms = [], []
    original_step = Lion.step

    def observed_step(optimizer, closure=None):
        values = [
            p.grad.detach().clone() for group in optimizer.param_groups for p in group["params"]
        ]
        gradients.append(values)
        norms.append(sum(g.square().sum() for g in values).sqrt())
        return original_step(optimizer, closure)

    Lion.step = observed_step
    try:
        mtd.bypass(reference, copy.deepcopy(teacher), batches, config, device_mesh=single)
    finally:
        Lion.step = original_step
    expected_gradients = gradients[:]
    gradients.clear()
    if axis == "fsdp":
        from torch.distributed.fsdp import fully_shard

        for model in (student, teacher):
            fully_shard(model.mlp, mesh=mesh)
            fully_shard(model.moe, mesh=mesh)
            fully_shard(model, mesh=mesh)
    local = batches[rank::size] if axis in ("dp", "fsdp") else batches
    if unequal and rank == 1:
        local = local[:1]
    if unequal:
        before = copy.deepcopy(student.state_dict())
        with pytest.raises(ValueError):
            mtd.bypass(student, teacher, local, config, device_mesh=mesh)
        # No partially assembled global update may advance native fields.
        if axis == "dp":
            assert not any("_local_repair_state" in key for key in student.state_dict())
        assert all(torch.equal(value, before[name]) for name, value in student.state_dict().items())
        return
    forward_sizes, handles = {}, []
    if axis == "fsdp":
        for name, module in (
            ("student", student),
            ("teacher", teacher),
            ("mlp", student.mlp),
            ("moe", student.moe),
        ):
            sizes = forward_sizes[name] = []
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, args, sizes=sizes: sizes.append(args[0].shape[0])
                )
            )
    Lion.step = observed_step
    try:
        mtd.bypass(student, teacher, local, config, device_mesh=mesh)
    finally:
        Lion.step = original_step
        for handle in handles:
            handle.remove()
    if axis == "fsdp":
        # One bounded warmup precedes four training microbatches on each data lane.
        assert forward_sizes == {
            "student": [1],
            "teacher": [1] * 5,
            "mlp": [1] * 5,
            "moe": [1] * 5,
        }, forward_sizes
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-14)
    assert all(norm <= config["max_grad_norm"] for norm in norms)
    for name, value in reference.state_dict().items():
        actual = student.state_dict()[name]
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
        assert actual.dtype == value.dtype, name
        actual = actual.to(device=value.device)
        if value.is_floating_point():
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        else:
            assert torch.equal(actual, value), name


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("axis", "unequal"), [("dp", False), ("cp", False), ("fsdp", False), ("dp", True)]
)
def test_distributed_bypass_matches_unique_batch(bypass_cpu_workers, axis, unequal):
    bypass_cpu_workers.run(_unique_batch_worker, axis, unequal)


def _validation_refusal_worker(rank, size, problem):
    from _test_utils.torch.distributed.utils import bounded_collectives
    from torch.distributed.device_mesh import init_device_mesh

    group = dist.new_group(timeout=timedelta(seconds=10))

    class CollectiveTeacher(torch.nn.Sequential):
        def forward(self, inputs):
            inputs = inputs.clone()
            dist.all_reduce(inputs, group=group)
            return super().forward(inputs / size)

    class ResolveStudent(torch.nn.Sequential):
        def get_submodule(self, target):
            if getattr(self, "fail_lookup", False):
                raise OSError()
            return super().get_submodule(target)

    torch.manual_seed(97)
    teacher = CollectiveTeacher(OrderedDict([("mlp", torch.nn.Linear(256, 256, bias=False))]))
    batch = torch.randn(2, 4, 256)
    student = mtq.quantize(
        ResolveStudent(OrderedDict([("mlp", copy.deepcopy(teacher.mlp))])),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda model: model(batch),
    )
    before = copy.deepcopy(student.state_dict())
    axis = "dp" if problem == "global_batch" else "cp"
    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=(axis,))
    recipe = {"num_steps": 1, "global_batch_size": 2, "sequence_length": 4}
    if problem == "global_batch" and rank == 0:
        recipe["global_batch_size"] = 1
    if problem in ("iterator", "length", "advance", "microbatches"):
        reference = copy.deepcopy(student)
        mtd.bypass(reference, copy.deepcopy(teacher), [batch], recipe, device_mesh=mesh)
        before = copy.deepcopy(reference.state_dict())
    student.fail_lookup = problem == "resolve" and rank == 1

    class Validation:
        def __iter__(self):
            if problem == "iterator" and rank == 1:
                raise OSError("validation iterator failure")
            return self.read()

        def read(self):
            yield batch[:1] if problem == "microbatches" and rank == 1 else batch
            if problem == "advance" and rank == 1:
                raise OSError("validation reader failure")
            if problem != "length" or rank == 0:
                yield batch

    class Training:
        def __iter__(self):
            if problem == "train_advance" and rank == 1:
                raise RuntimeError()
            yield batch

    validation = None if problem == "presence" and rank == 1 else Validation()
    try:
        with bounded_collectives(seconds=10), pytest.raises(ValueError) as refusal:
            mtd.bypass(
                student,
                teacher,
                Training(),
                recipe,
                validation_dataloader=validation,
                device_mesh=mesh,
            )
        if problem == "global_batch":
            errors = [None] * size
            dist.all_gather_object(errors, str(refusal.value), group=group)
            assert len(set(errors)) == 1
        assert all(torch.equal(value, before[name]) for name, value in student.state_dict().items())
        assert not any("_local_repair_state" in name for name in student.state_dict())
    finally:
        dist.destroy_process_group(group)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "problem",
    [
        "presence",
        "iterator",
        "length",
        "advance",
        "microbatches",
        "train_advance",
        "resolve",
        "global_batch",
    ],
)
def test_distributed_bypass_refuses_unequal_validation(bypass_cpu_workers, problem):
    timeout = bypass_cpu_workers.job_timeout
    try:
        bypass_cpu_workers.job_timeout = 60
        bypass_cpu_workers.run(_validation_refusal_worker, problem)
    finally:
        bypass_cpu_workers.job_timeout = timeout


def _native_geometry_preflight_worker(rank, size):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.placement_types import _StridedShard
    from transformers import Qwen3MoeConfig
    from transformers.distributed.expert_parallel import MoEParamShard
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    from modelopt.torch.distill._bypass_parallel import (
        _tp_quantizer_layouts,
        apply_tp_quantizer_fields,
    )
    from modelopt.torch.distill.plugins.transformers.parallel import (
        _validate_parameter_geometry,
        apply_hf_tensor_parallelism,
        validate_hf_tensor_parallelism,
    )
    from modelopt.torch.quantization.plugins.huggingface import register_fused_experts_on_the_fly

    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("tp",))
    expert_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("ep",))
    torch.manual_seed(54)
    config = Qwen3MoeConfig(
        hidden_size=8, moe_intermediate_size=8, num_experts=4, num_experts_per_tok=1
    )
    config._experts_implementation = "eager"
    original = torch.nn.Module()
    original.experts = Qwen3MoeExperts(config)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.normal_(std=0.3)
    register_fused_experts_on_the_fly(original)
    mtq.quantize(
        original,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*gate_up_proj_weight_quantizers*",
                    "cfg": {"num_bits": 8, "axis": 0},
                },
            ],
            "algorithm": "max",
        },
        lambda model: model.experts(
            torch.randn(4, 8), torch.arange(4).reshape(4, 1), torch.ones(4, 1)
        ),
    )
    before = copy.deepcopy(original.state_dict())
    parameter = original.experts.gate_up_proj
    quantizers = original.experts.gate_up_proj_weight_quantizers
    with pytest.raises(ValueError, match="physical matrix or fused expert views"):
        validate_hf_tensor_parallelism(original, mesh, {"experts.gate_up_proj": "grouped_gemm"})
    # The same whole-expert placement is legal on EP and must remain metadata-only here.
    _validate_parameter_geometry(
        original.experts,
        "gate_up_proj",
        ALL_PARALLEL_STYLES["grouped_gemm"],
        expert_mesh,
        "experts.gate_up_proj",
        allow_expert_axis=True,
    )
    with pytest.raises(ValueError, match="physical matrix or fused expert views"):
        _validate_parameter_geometry(
            original.experts,
            "gate_up_proj",
            MoEParamShard(_StridedShard(0, split_factor=2), shards_expert_dim=True),
            expert_mesh,
            "experts.gate_up_proj",
            allow_expert_axis=True,
        )
    assert original.experts.gate_up_proj is parameter
    assert original.experts.num_experts == 4 and len(quantizers) == 4
    assert all(torch.equal(value, before[name]) for name, value in original.state_dict().items())
    for style in ("packed_colwise", "rowwise"):
        model = copy.deepcopy(original)
        plan = {"experts.gate_up_proj": style}
        validate_hf_tensor_parallelism(model, mesh, plan)
        apply_hf_tensor_parallelism(model, mesh, plan)
        for quantizer, fields, _, _ in _tp_quantizer_layouts(model):
            apply_tp_quantizer_fields(quantizer, fields)
        local = model.experts.gate_up_proj.to_local()
        for index, quantizer in enumerate(model.experts.gate_up_proj_weight_quantizers):
            reference_weight = quantizers[index](parameter[index])
            amax = quantizers[index].amax
            if style == "packed_colwise":
                rows = torch.cat((torch.arange(4) + rank * 4, torch.arange(4) + 8 + rank * 4))
                reference_weight, amax = reference_weight[rows], amax[rows]
            else:
                reference_weight = reference_weight[:, rank * 4 : (rank + 1) * 4]
            torch.testing.assert_close(quantizer.amax, amax, rtol=0, atol=0)
            inputs = torch.randn(3, local.shape[-1], requires_grad=True)
            expected_inputs = inputs.detach().clone().requires_grad_()
            actual = F.linear(inputs, quantizer(local[index]))
            expected = F.linear(expected_inputs, reference_weight)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=0, atol=0)


@pytest.mark.timeout(180)
def test_native_geometry_preflight_matches_physical_field_ownership(bypass_cpu_workers):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_native_geometry_preflight_worker)


def _native_packed_tp_worker(rank, size, format_name, style):
    # The native HF TP interface is optional and requires Transformers 5.23.
    from modelopt.torch.distill.plugins.transformers.parallel import (
        apply_hf_tensor_parallelism,
        torch_mesh_context,
        validate_hf_tensor_parallelism,
    )

    assert size == 2
    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("tp",))
    context = torch_mesh_context(mesh)
    torch.manual_seed(73)
    model = mtq.quantize(
        torch.nn.Sequential(torch.nn.Linear(256, 16, bias=False)),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    layer = model[0]
    quantizer = layer.weight_quantizer
    if format_name == "fp8":
        native, scale = FP8QTensor.quantize(layer.weight.detach(), axis=0)
        double_scale = None
        attributes = {"num_bits": (4, 3), "axis": 0}
    else:
        native, scale, double_scale = NVFP4QTensor.quantize(
            layer.weight.detach(), 16, try_tensorrt=False
        )
        attributes = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        }
    quantizer.set_from_attribute_config(
        QuantizerAttributeConfig(**attributes, fake_quant=False, enable=True)
    )
    quantizer._set_buffer("_scale", scale)
    if double_scale is not None:
        quantizer._set_buffer("_double_scale", double_scale)
    layer.weight = QTensorWrapper(native)
    original_codes = native._quantized_data.contiguous().view(torch.uint8).clone()
    decoded = native.dequantize(
        scale=scale, double_scale=double_scale, block_sizes=quantizer.block_sizes
    )
    plan = {"0.weight": style, "0": style}
    validate_hf_tensor_parallelism(model, mesh, plan)
    prepare_packed_weights_for_tp(model)
    apply_hf_tensor_parallelism(model, mesh, plan)
    prepare_quantizer_partitions(model, context)

    dim = 0 if style == "colwise" else 1
    width = decoded.shape[dim] // size
    expected_weight = decoded.narrow(dim, rank * width, width)
    assert isinstance(layer.weight, DTensor)
    assert layer.weight.device_mesh == mesh
    assert layer.weight.placements == (Shard(dim),)
    assert tuple(layer.weight.shape) == (16, 256)
    assert not layer.weight.requires_grad
    local = layer.weight.to_local()
    assert local.shape == expected_weight.shape
    torch.testing.assert_close(local, expected_weight, rtol=0, atol=0)
    assert quantizer.is_enabled and quantizer.fake_quant
    pinned = get_pinned_native_weight(quantizer, local)
    assert type(pinned) is type(native)
    assert pinned.metadata["shape"] == tuple(expected_weight.shape)
    columns_per_byte = 2 if format_name == "nvfp4" else 1
    physical_codes = original_codes.reshape(16, 256 // columns_per_byte)
    code_width = width // columns_per_byte if dim == 1 else width
    expected_codes = physical_codes.narrow(dim, rank * code_width, code_width)
    assert torch.equal(quantizer._native_weight_codes.flatten(), expected_codes.flatten())
    torch.testing.assert_close(quantizer(local), expected_weight, rtol=0, atol=0)

    inputs = torch.randn(3, expected_weight.shape[1], requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    projection = F.linear(inputs, quantizer(local))
    reference_projection = F.linear(reference_inputs, expected_weight)
    torch.testing.assert_close(projection, reference_projection, rtol=0, atol=0)
    cotangent = torch.randn_like(projection)
    projection.backward(cotangent)
    reference_projection.backward(cotangent)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)

    dense_inputs = torch.randn(3, 256, requires_grad=True)
    dense_output = F.linear(dense_inputs, decoded)
    dense_cotangent = torch.randn_like(dense_output)
    dense_output.backward(dense_cotangent)
    # HF colwise accepts replicated inputs; rowwise accepts the local input-column shard.
    model_inputs = (
        (
            dense_inputs.detach()
            if dim == 0
            else dense_inputs.detach().narrow(1, rank * width, width)
        )
        .clone()
        .requires_grad_()
    )
    actual = model(model_inputs)
    assert not isinstance(actual, DTensor)
    expected = (
        dense_output.detach().narrow(1, rank * width, width) if dim == 0 else dense_output.detach()
    )
    output_cotangent = (
        dense_cotangent.narrow(1, rank * width, width) if dim == 0 else dense_cotangent
    )
    # Only rowwise forward and colwise input backward reorder sums across ranks.
    tolerance = {"rtol": 1e-5, "atol": 1e-6}
    exact = {"rtol": 0, "atol": 0}
    torch.testing.assert_close(actual, expected, **(exact if dim == 0 else tolerance))
    actual.backward(output_cotangent)
    expected_grad = (
        dense_inputs.grad if dim == 0 else dense_inputs.grad.narrow(1, rank * width, width)
    )
    torch.testing.assert_close(
        model_inputs.grad, expected_grad, **(tolerance if dim == 0 else exact)
    )
    assert layer.weight.grad is None
    assert get_pinned_native_weight(quantizer, layer.weight.to_local()) is not None
    assert torch.equal(quantizer._native_weight_codes.flatten(), expected_codes.flatten())


@pytest.mark.parametrize("format_name", ["fp8", "nvfp4"])
@pytest.mark.parametrize("style", ["colwise", "rowwise"])
@pytest.mark.timeout(180)
def test_native_packed_tp_preserves_codes_projection_and_input_vjp(
    bypass_cpu_workers, format_name, style
):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_native_packed_tp_worker, format_name, style)


@pytest.mark.parametrize("format_name", ["iq", "fp8", "nvfp4"])
@pytest.mark.timeout(60)
def test_native_mesh_adoption_preserves_forward_and_hooks(tmp_path, format_name):
    pytest.importorskip("transformers", minversion="5.23")
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    if dist.is_initialized():
        pytest.skip("This single-process adoption case owns its isolated Gloo group")
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        manager = TransformersDeviceMesh(
            init_device_mesh(
                "cpu", (1, 1, 1, 1), mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")
            ),
            init_device_mesh(
                "cpu", (1, 1, 1, 1), mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep")
            ),
        )
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=256,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        teacher = LlamaForCausalLM(config).eval()
        batch = {"input_ids": torch.randint(3, 32, (1, 4))}
        adapter = TransformersBypassAdapter(manager)
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    *[
                        {"quantizer_name": pattern, "cfg": {"num_bits": "iq2_s", "backend": "ggml"}}
                        for pattern in adapter.weight_patterns(teacher)
                    ],
                ],
                "algorithm": "max",
            },
            lambda m: m(**batch),
        )
        if format_name != "iq":
            layer = student.model.layers[0].self_attn.q_proj
            attributes = (
                {"num_bits": (4, 3), "axis": 0}
                if format_name == "fp8"
                else {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                }
            )
            quantizer = layer.weight_quantizer = TensorQuantizer(
                QuantizerAttributeConfig(**attributes)
            )
            quantizer.pre_quant_scale = torch.linspace(0.75, 1.25, 256)
            mtq.compress(
                student,
                config={"compress": {"default": False, "*self_attn.q_proj*": True}},
            )
            layer._use_real_quant_gemm = True
            packed = layer.weight
            assert isinstance(packed, QTensorWrapper)
            native = packed.get_qtensor()
            assert type(native) is (FP8QTensor if format_name == "fp8" else NVFP4QTensor)
            native_metadata = copy.deepcopy(native.metadata)
            codes = native._quantized_data.contiguous().view(torch.uint8).clone()
            buffers = {name: value.clone() for name, value in quantizer.named_buffers()}
            assert not quantizer.fake_quant
            assert quantizer._enable_pre_quant_scale and not quantizer.rotate_is_enabled
        with torch.no_grad():
            expected = student(**batch).logits
        hooks = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in student.modules()]
        assert (
            mtd.bypass(student, teacher, [batch], {"num_steps": 0}, device_mesh=manager) is student
        )
        if format_name != "iq":
            assert layer.weight is packed
            assert isinstance(layer.weight, QTensorWrapper)
            restored = layer.weight.get_qtensor()
            assert type(restored) is type(native)
            assert restored.metadata == native_metadata
            assert torch.equal(restored._quantized_data.contiguous().view(torch.uint8), codes)
            assert dict(quantizer.named_buffers()).keys() == buffers.keys()
            for name, value in quantizer.named_buffers():
                assert torch.equal(
                    value.contiguous().reshape(-1).view(torch.uint8),
                    buffers[name].contiguous().reshape(-1).view(torch.uint8),
                )
            assert not quantizer.fake_quant
            assert quantizer._enable_pre_quant_scale and not quantizer.rotate_is_enabled
            assert layer._use_real_quant_gemm
        assert student._mesh_manager is manager
        assert adapter.parallelize(student) is student
        assert adapter.parallelize(student) is student
        with torch.no_grad():
            torch.testing.assert_close(student(**batch).logits, expected, rtol=0, atol=0)
        assert all(
            len(m._forward_hooks) == after and len(m._forward_pre_hooks) == before
            for m, after, before in hooks
        )
    finally:
        dist.destroy_process_group()


def _native_projection_worker(rank, size, frozen_format, layout=None):
    from _test_utils.transformers_families import build_model, cases
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill._bypass_runtime import DistributedBypassRuntime
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    assert size == 4
    torch.cuda.set_device(rank)
    torch.manual_seed(61)
    record = cases(["qwen3_moe"])[0]
    record["minimum_pattern_prefix"] = 2
    record["config"]["mlp_only_layers"] = [0]
    record["config"]["num_experts"] = 4
    record["config"]["num_experts_per_tok"] = 1
    model = build_model(record, torch.device("cuda", rank))
    with torch.no_grad():
        model.model.layers[1].mlp.gate.weight.zero_()
    teacher = copy.deepcopy(model) if layout is not None else None
    batch = {"input_ids": torch.randint(3, 100, (2, 8), device=torch.device("cuda", rank))}
    frozen_cfg = (
        {"num_bits": (4, 3), "axis": None}
        if frozen_format == "fp8"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*mlp*weight_quantizer*",
                    "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                },
                {"quantizer_name": "*mlp.gate.weight_quantizer*", "enable": False},
                {"quantizer_name": "*mlp.experts.down_proj_weight_quantizers*", "cfg": frozen_cfg},
            ],
            "algorithm": "max",
        },
        lambda m: m(**batch),
    )
    with torch.no_grad():
        expected = model(**batch).logits.detach().cpu()
    batches = [
        {"input_ids": torch.randint(3, 100, (1, 8), device=torch.device("cuda", rank))}
        for _ in range(16)
    ]
    recipe = {
        "num_steps": 2,
        "global_batch_size": 8,
        "sequence_length": 8,
        "warmup_fraction": 0,
        "scheduler": "constant",
        "max_grad_norm": 1e-8,
    }
    traces, phase, trace_index = [], "reference", 0
    original_clip = DistributedBypassRuntime.clip_gradients

    def observed_clip(runtime, fields, max_norm):
        nonlocal trace_index
        selected = {id(p) for p in fields}
        values = {}
        for quantizer in runtime.student.modules():
            state = getattr(quantizer, "_local_repair_state", None)
            if state is None:
                continue
            for field, parameter in state.named_parameters():
                if id(parameter) in selected:
                    values[state.field_prefix + "." + field] = (
                        None if parameter.grad is None else parameter.grad.detach().float().cpu(),
                        state.weight_shape,
                        state.format.block_size,
                    )
        norm = original_clip(runtime, fields, max_norm)
        if phase == "reference":
            traces.append((values, norm.detach().cpu()))
        else:
            expected_values, expected_norm = traces[trace_index]
            trace_index += 1
            torch.testing.assert_close(norm.detach().cpu(), expected_norm, rtol=0.03, atol=0)
            for key, (actual, shape, block) in values.items():
                expected, full_shape, _ = expected_values[key]
                if expected is None:
                    assert actual is None or not bool(actual.count_nonzero())
                    continue
                assert actual is not None
                if shape != full_shape:
                    # This fixture has ordinary gate/up/down matrices: align
                    # native TP rows or complete K blocks with the full oracle.
                    view = expected.reshape(
                        full_shape[0], full_shape[1] // block, *expected.shape[1:]
                    )
                    dimension = 0 if shape[0] != full_shape[0] else 1
                    expected = view.chunk(runtime.context.tensor_mesh.size(), dim=dimension)[
                        runtime.context.tensor_mesh.get_local_rank()
                    ].reshape_as(actual)
                error = (actual - expected).norm() / expected.norm().clamp_min(1e-20)
                assert error < 0.03, (key, error.item())
        return norm

    if layout is not None:
        single = init_device_mesh("cuda", (4, 1), mesh_dim_names=("owners", "dp"))["dp"]
        reference = copy.deepcopy(model)
        DistributedBypassRuntime.clip_gradients = observed_clip
        try:
            mtd.bypass(reference, copy.deepcopy(teacher), batches, recipe, device_mesh=single)
        finally:
            DistributedBypassRuntime.clip_gradients = original_clip
        with torch.no_grad():
            expected = reference(**batch).logits.detach().cpu()
    fsdp, tp, efsdp, ep = (4, 1, 2, 2) if layout == "ep2-replica2" else (2, 2, 1, 4)
    manager = TransformersDeviceMesh(
        init_device_mesh(
            "cuda", (1, 1, fsdp, tp), mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")
        ),
        init_device_mesh(
            "cuda", (1, 1, efsdp, ep), mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep")
        ),
    )
    adapter = TransformersBypassAdapter(manager)
    adapter.parallelize(model)
    metadata = adapter.context.parallel_metadata[id(model)]
    assert all(len(ids) == 4 // ep for ids in metadata["expert_global_ids"].values())
    if layout is not None:
        adapter.parallelize(teacher)
        frozen = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if "experts.down_proj" in name or ".mlp.gate." in name
        }
        lane = adapter.context.data_mesh.get_local_rank()
        local = [
            batches[step * 8 + index]
            for step in range(2)
            for index in range(lane, 8, adapter.context.dp_size)
        ]
        phase = "native"
        DistributedBypassRuntime.clip_gradients = observed_clip
        try:
            mtd.bypass(model, teacher, local, recipe, device_mesh=manager)
        finally:
            DistributedBypassRuntime.clip_gradients = original_clip
        assert trace_index == len(traces)
        for name, value in frozen.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    with torch.no_grad():
        actual = model(**batch).logits
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.02)


@pytest.mark.parametrize("frozen_format", ["fp8", "nvfp4"])
@pytest.mark.timeout(300)
def test_native_iq_placement_preserves_projection(native_workers, frozen_format):
    native_workers.run(_native_projection_worker, frozen_format)


@pytest.mark.parametrize("layout", ["tp2-fsdp2-ep4", "ep2-replica2"])
@pytest.mark.parametrize("frozen_format", ["fp8", "nvfp4"])
@pytest.mark.timeout(300)
def test_native_owner_updates_match_reference(native_workers, layout, frozen_format):
    native_workers.run(_native_projection_worker, frozen_format, layout)


def _native_ep_transport_policy_worker(rank, size):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from transformers.distributed.configuration_utils import DistributedConfig
    from transformers.distributed.expert_parallel import AllReduceEP, AllToAllEP
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    manager = TransformersDeviceMesh(
        init_device_mesh(
            "cpu", (1, 1, 1, size), mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")
        ),
        init_device_mesh(
            "cpu", (1, 1, 1, size), mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep")
        ),
    )
    torch.manual_seed(109)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        use_cache=False,
    )
    config._attn_implementation = config._experts_implementation = "eager"
    reference = Qwen3MoeForCausalLM(config).eval()
    with torch.no_grad():
        reference.model.layers[0].mlp.gate.weight.zero_()
    tokens = torch.tensor([[1, 3, 5, 7]])
    with torch.no_grad():
        expected = reference(tokens).logits

    calls, routed = {}, set()
    original_all_to_all, original_all_reduce = AllToAllEP.__call__, AllReduceEP.__call__

    def observe(implementation, name, inputs, indices, weights):
        calls[name] = calls.get(name, 0) + 1
        local = indices.to_local() if isinstance(indices, DTensor) else indices
        routed.update(local.reshape(-1).tolist())
        original = original_all_to_all if name == "all_to_all" else original_all_reduce
        return original(implementation, inputs, indices, weights)

    def traced_all_to_all(implementation, inputs, indices, weights):
        return observe(implementation, "all_to_all", inputs, indices, weights)

    def traced_all_reduce(implementation, inputs, indices, weights):
        return observe(implementation, "all_reduce", inputs, indices, weights)

    AllToAllEP.__call__, AllReduceEP.__call__ = traced_all_to_all, traced_all_reduce
    try:
        for policy in ("auto", "all_to_all", "all_reduce"):
            calls.clear()
            routed.clear()
            model = copy.deepcopy(reference)
            model.config.distributed_config = DistributedConfig(
                tp_size=size, fsdp_size=1, ep_size=size, ep_implementation=policy
            )
            adapter = TransformersBypassAdapter(manager)
            if policy == "all_reduce":
                before = copy.deepcopy(model.state_dict())
                parameters = dict(model.named_parameters())
                plans = {
                    name: (name in model.__dict__, copy.deepcopy(getattr(model, name, None)))
                    for name in ("_tp_plan", "_ep_plan", "_fsdp_plan")
                }
                hooks = [
                    (module, tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                    for module in model.modules()
                ]
                with pytest.raises(ValueError, match="ep_implementation"):
                    adapter.parallelize(model)
                assert all(model.get_parameter(name) is value for name, value in parameters.items())
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                for name, (owned, plan) in plans.items():
                    assert (name in model.__dict__) == owned
                    assert getattr(model, name, None) == plan
                for module, forward, pre in hooks:
                    assert tuple(module._forward_hooks) == forward
                    assert tuple(module._forward_pre_hooks) == pre
                assert getattr(model, "_mesh_manager", None) is None
                assert "_modelopt_bypass_placement" not in model.__dict__
                assert id(model) not in adapter.context.parallel_metadata
                assert not calls
                with torch.no_grad():
                    torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)
                continue
            adapter.parallelize(model)
            metadata = adapter.context.parallel_metadata[id(model)]
            assert metadata["ep_implementation"] == policy
            with torch.no_grad():
                actual = model(tokens).logits
                actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            transport = "all_reduce" if policy == "auto" else "all_to_all"
            assert calls == {transport: 1}, (policy, calls)
            assert len(routed) == 1
            local_ids = metadata["expert_global_ids"]["model.layers.0.mlp.experts"]
            empty_owner = torch.tensor(int(not routed.intersection(local_ids)))
            dist.all_reduce(empty_owner)
            assert empty_owner.item() == size - 1
    finally:
        AllToAllEP.__call__, AllReduceEP.__call__ = original_all_to_all, original_all_reduce


@pytest.mark.timeout(240)
def test_native_ep_transport_policy_reaches_hf_dispatcher(bypass_cpu_workers):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_native_ep_transport_policy_worker)


def _context_worker(rank, size, fallback):
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import LlamaConfig, LlamaForCausalLM

    from modelopt.torch.distill._bypass_optimizer import Lion
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
    from modelopt.torch.quantization.ggml.scale_fields import (
        decode_scale_fields,
        replace_scale_fields,
    )

    torch.cuda.set_device(rank)
    torch.manual_seed(71)
    device = torch.device("cuda", rank)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    teacher = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
    batches = [{"input_ids": torch.randint(3, 32, (1, 8), device=device)} for _ in range(8)]
    if fallback:
        for batch in batches:
            batch["attention_mask"] = torch.ones(1, 1, 8, 8, dtype=torch.bool, device=device).tril()
            batch["valid_mask"] = torch.ones(1, 8, dtype=torch.bool, device=device)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*mlp*weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda m: m(**{k: v for k, v in batches[0].items() if k != "valid_mask"}),
    )
    recipe = {
        "num_steps": 2,
        "global_batch_size": 4,
        "sequence_length": 8,
        "scheduler": "constant",
        "warmup_fraction": 0,
    }
    reference = copy.deepcopy(student)
    single = init_device_mesh("cuda", (size, 1), mesh_dim_names=("owners", "dp"))["dp"]
    gradients = []
    original_step = Lion.step

    def observed_step(optimizer, closure=None):
        gradients.append(
            [
                p.grad.detach().float().cpu()
                for group in optimizer.param_groups
                for p in group["params"]
            ]
        )
        return original_step(optimizer, closure)

    Lion.step = observed_step
    try:
        mtd.bypass(reference, copy.deepcopy(teacher), batches, recipe, device_mesh=single)
    finally:
        Lion.step = original_step
    expected_gradients = gradients[:]
    gradients.clear()
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "cp"))
    adapter = TransformersBypassAdapter(mesh)
    adapter.parallelize(student)
    adapter.parallelize(teacher)
    local = [
        batches[step * 4 + index]
        for step in range(2)
        for index in range(mesh["dp"].get_local_rank(), 4, 2)
    ]
    Lion.step = observed_step
    try:
        mtd.bypass(student, teacher, local, recipe, device_mesh=mesh)
    finally:
        Lion.step = original_step
    # BF16 dW rounds before each CP owner's sum; native attention also changes
    # its reduction tree. Validate the actual normalized VJP, rather than
    # demanding bitwise scale updates after Lion amplifies a near-zero sign.
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        for a, b in zip(actual, expected, strict=True):
            relative_error = (a - b).norm() / b.norm().clamp_min(1e-20)
            assert relative_error < 0.03, relative_error.item()
    assert student._modelopt_bypass_execution["context_parallel_modes"] == [
        "replicated_full_context" if fallback else "native_sdpa"
    ]
    for name, module in student.named_modules():
        if isinstance(module, TensorQuantizer) and module.backend == "ggml":
            fmt = IQ_FORMAT_REGISTRY["iq2_s"]
            actual = module._ggml_pinned_iq2_s
            expected = reference.get_submodule(name)._ggml_pinned_iq2_s
            global_scale, local_scale, _ = decode_scale_fields(expected, fmt)
            assert torch.equal(
                replace_scale_fields(actual, fmt, global_scale, local_scale), expected
            )
            shape = torch.tensor((actual.shape[0], actual.shape[1] * fmt.block_size))
            torch.testing.assert_close(
                fmt.dequantize(actual, shape, dtype=torch.float32),
                fmt.dequantize(expected, shape, dtype=torch.float32),
                rtol=0.006,
                atol=1e-6,
            )


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.timeout(300)
def test_native_context_and_replicated_fallback_match_reference(native_workers, fallback):
    import importlib

    if not fallback:
        try:
            native = importlib.import_module(
                "torch.distributed.tensor.experimental._context_parallel._attention"
            )
        except ImportError:
            pytest.skip("Native CP needs Torch's scoped module-wrapper attention API")
        if not hasattr(native, "_ContextParallel"):
            pytest.skip("Native CP needs Torch's scoped module-wrapper attention API")
    native_workers.run(_context_worker, fallback)
