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

"""Native IQ payload placement, mesh adoption and real four-GPU projection checks."""

import copy
from collections import OrderedDict
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight
from modelopt.torch.quantization.nn import TensorQuantizer


def test_hybrid_fsdp_plan_wraps_only_present_functional_children():
    pytest.importorskip("transformers", minversion="5.23")
    from _test_utils.transformers_families import build_model, cases
    from transformers.distributed.configuration_utils import DistributedConfig

    from modelopt.torch.distill.plugins.transformers.functional_fsdp import extend_family_fsdp_plan

    record = cases(["qwen3_next"])[0]
    model = build_model(record, "meta")
    plan = model._resolve_parallel_plans(DistributedConfig(fsdp_size=2))[2]
    original = dict(plan)
    extended = extend_family_fsdp_plan(model, plan)
    modules = dict(model.named_modules())
    children = {name for name in modules if name.endswith("linear_attn.conv1d")}
    assert children and any(hasattr(block, "self_attn") for block in model.model.layers)
    assert children <= extended.keys() <= modules.keys()
    assert all(extended[name] == "free_full_weight" for name in children)
    assert plan == original


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
    batches = [torch.randn(1, 4, 256) for _ in range(4)]
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
        "global_batch_size": 2,
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
    local = batches[rank::size] if axis == "dp" else batches
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
    Lion.step = observed_step
    try:
        mtd.bypass(student, teacher, local, config, device_mesh=mesh)
    finally:
        Lion.step = original_step
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-14)
    assert all(norm <= config["max_grad_norm"] for norm in norms)
    for name, value in reference.state_dict().items():
        if value.is_floating_point():
            torch.testing.assert_close(student.state_dict()[name], value, rtol=0, atol=0)
        else:
            assert torch.equal(student.state_dict()[name], value), name


@pytest.mark.timeout(180)
@pytest.mark.parametrize(("axis", "unequal"), [("dp", False), ("cp", False), ("dp", True)])
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
    student.fail_lookup = problem == "resolve" and rank == 1
    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("cp",))

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
        with bounded_collectives(seconds=10), pytest.raises(ValueError):
            mtd.bypass(
                student,
                teacher,
                Training(),
                {"num_steps": 1, "global_batch_size": 2, "sequence_length": 4},
                validation_dataloader=validation,
                device_mesh=mesh,
            )
        assert all(torch.equal(value, before[name]) for name, value in student.state_dict().items())
        assert not any("_local_repair_state" in name for name in student.state_dict())
    finally:
        dist.destroy_process_group(group)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "problem",
    ["presence", "iterator", "length", "advance", "microbatches", "train_advance", "resolve"],
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


@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
@pytest.mark.parametrize("dim", [0, 1])
def test_native_iq_partition_preserves_pinned_projection(name, dim):
    from modelopt.torch.quantization.repair_partition import (
        NativeWeightPartition,
        native_weight_partition,
    )

    torch.manual_seed(59)
    fmt = IQ_FORMAT_REGISTRY[name]
    weight = torch.randn(4, 512)
    importance = 1 + torch.rand(512) if fmt.weighted else None
    packed, shape = fmt.quantize(
        weight, **({"importance": importance} if importance is not None else {})
    )
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=name, backend="ggml", block_sizes={-1: fmt.block_size})
    )
    search = fmt.payload_scales(packed.reshape(-1, fmt.block_bytes)) if fmt.payload_scales else None
    pin_packed_weight(quantizer, name, packed, importance, source=weight, search_scales=search)
    expected = fmt.dequantize(packed, shape, dtype=torch.float32)
    view = NativeWeightPartition((4, 512), dim, 1, 2, split_factor=2 if dim == 0 else 1)
    local = view.select(weight)
    fields, _, _ = native_weight_partition(quantizer, view, source=local)
    for key, value in fields.items():
        quantizer._set_buffer(key, value) if isinstance(value, torch.Tensor) else setattr(
            quantizer, key, value
        )
    x = torch.randn(3, local.shape[-1], requires_grad=True)
    actual = F.linear(x, quantizer(local))
    reference = F.linear(x, view.select(expected))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.square().sum().backward()
    actual_grad = x.grad.clone()
    x.grad = None
    reference.square().sum().backward()
    torch.testing.assert_close(actual_grad, x.grad, rtol=0, atol=0)
    before = {n: v.clone() for n, v in quantizer.state_dict().items()}
    invalid = NativeWeightPartition((4, 512), 1, 0, 2 * (512 // fmt.block_size))
    with pytest.raises(ValueError):
        native_weight_partition(quantizer, invalid)
    assert all(torch.equal(v, before[n]) for n, v in quantizer.state_dict().items())


def test_packed_frozen_projection_staging_and_restore(tmp_path):
    import modelopt.torch.opt as mto
    from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
    from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp

    torch.manual_seed(60)
    model = mtq.quantize(
        torch.nn.Sequential(torch.nn.Linear(256, 16, bias=False)),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    layer = model[0]
    native, scale = FP8QTensor.quantize(layer.weight.detach(), axis=0)
    layer.weight_quantizer.set_from_attribute_config(
        QuantizerAttributeConfig(num_bits=(4, 3), axis=0, fake_quant=False)
    )
    layer.weight_quantizer._set_buffer("_scale", scale)
    layer.weight = QTensorWrapper(native)
    decoded = native.dequantize(scale=scale)
    prepare_packed_weights_for_tp(model)
    x = torch.randn(3, 256, requires_grad=True)
    expected = F.linear(x, decoded)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    grad = x.grad.clone()
    x.grad = None
    expected.square().sum().backward()
    torch.testing.assert_close(grad, x.grad, rtol=0, atol=0)
    assert not layer.weight.requires_grad
    path = tmp_path / "staged.pt"
    mto.save(model, path)
    restored = mto.restore(torch.nn.Sequential(torch.nn.Linear(256, 16, bias=False)), path)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    assert torch.equal(
        restored[0].weight_quantizer._native_weight_codes, native._quantized_data.view(torch.uint8)
    )


@pytest.mark.timeout(60)
def test_native_mesh_adoption_preserves_forward_and_hooks(tmp_path):
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
        with torch.no_grad():
            expected = student(**batch).logits
        assert adapter.parallelize(student) is student
        hooks = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in student.modules()]
        assert adapter.parallelize(student) is student
        with torch.no_grad():
            torch.testing.assert_close(student(**batch).logits, expected, rtol=0, atol=0)
        assert (
            mtd.bypass(student, teacher, [batch], {"num_steps": 0}, device_mesh=manager) is student
        )
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
