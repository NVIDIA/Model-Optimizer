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

"""Native overlapping HF tensor/expert meshes against a unique-data reference."""

import copy
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from _test_utils.transformers_families import hf_torch_kernels
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter
from modelopt.torch.distill.plugins.transformers.functional_fsdp import (
    extend_family_fsdp_plan,
    install_functional_fsdp_calls,
)

try:
    from transformers import BailingHybridConfig, BailingHybridForCausalLM
except ImportError:
    BailingHybridConfig = BailingHybridForCausalLM = None


def _repair_models(dtype=torch.float32, native_format="int4"):
    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        dtype=str(dtype).removeprefix("torch."),
        attn_implementation="sdpa",
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._experts_implementation = "eager"
    teacher = (
        Qwen3MoeForCausalLM(config).to(device="cuda", dtype=dtype).eval().requires_grad_(False)
    )
    # Deliberately leave entire expert owners unused; their native collectives must still finish.
    teacher.model.embed_tokens.weight[:, 0].fill_(1)
    teacher.model.layers[0].mlp.gate.weight.zero_()
    teacher.model.layers[0].mlp.gate.weight[:, 0].copy_(
        torch.tensor([0.01, 0.02, -0.01, -0.02], device="cuda")
    )
    student = copy.deepcopy(teacher)
    weight_cfg = (
        {"num_bits": 4, "axis": 0}
        if native_format == "int4"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    mtq.quantize(
        student,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*self_attn*weight_quantizer",
                    "cfg": weight_cfg,
                },
                {"quantizer_name": "*experts*weight_quantizer*", "cfg": weight_cfg},
            ],
            "algorithm": "max",
        },
    )
    return teacher, student


def _repair_case(rank, batch_size, dtype=torch.float32, native_format="int4"):
    teacher, student = _repair_models(dtype, native_format)
    reference, reference_teacher = copy.deepcopy(student), copy.deepcopy(teacher)
    generator = torch.Generator(device="cuda").manual_seed(43)
    tokens = torch.randint(3, 120, (batch_size, 16), device="cuda", generator=generator)
    global_batch = {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
        "valid_mask": torch.ones_like(tokens, dtype=torch.bool),
    }
    repair = {
        "units": ["model.layers.0"],
        "updates": 2,
        "global_batch_size": batch_size,
        "micro_batch_size": 1,
        "warmup_fraction": 0,
        "scheduler": "constant",
        "gradient_clip_norm": 1e-6,
    }
    singleton = DeviceMesh("cuda", [rank], mesh_dim_names=("dp",), _init_backend=False)
    expected = mtd.local_distill(
        reference,
        reference_teacher,
        [global_batch, global_batch],
        repair,
        adapter=TransformersLocalDistillationAdapter(singleton),
    )
    return teacher, student, reference, global_batch, repair, expected


def _assert_matches_reference(student, reference, observed, expected, *, cp=False):
    for actual_row, reference_row in zip(
        observed.metrics["history"], expected.metrics["history"], strict=True
    ):
        for field in ("mse", "gradient_norm"):
            torch.testing.assert_close(
                actual_row["units"]["model.layers.0"][field],
                reference_row["units"]["model.layers.0"][field],
                rtol=0.03 if cp else 0.003,
                atol=1e-7 if cp else 1e-8,
            )
    reference_parameters = dict(reference.model.layers[0].named_parameters())
    for name, parameter in student.model.layers[0].named_parameters():
        value = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
        other = reference_parameters[name]
        torch.testing.assert_close(
            value, other, rtol=0.025 if cp else 0.003, atol=0.0002 if cp else 1e-6, msg=name
        )
        if cp:
            relative_error = (
                value.float() - other.float()
            ).norm() / other.float().norm().clamp_min(1e-12)
            if relative_error >= 0.003:
                # Ring and serial BF16 attention can put a learned scale on opposite
                # sides of a rounding boundary. Keep the aggregate bound unless every
                # element differs by at most one adjacent representable BF16 value.
                assert value.dtype == other.dtype == torch.bfloat16, name
                lower = torch.nextafter(other, torch.full_like(other, -torch.inf))
                upper = torch.nextafter(other, torch.full_like(other, torch.inf))
                assert bool(((value >= lower) & (value <= upper)).all()), name


@pytest.mark.skipif(
    "ep_size" not in DistributedConfig.__dataclass_fields__,
    reason="Requires native HF separate TP/EP plans",
)
@pytest.mark.parametrize(
    ("route", "native_format"),
    [
        ("masked", "int4"),
        ("dispatch", "int4"),
        ("dispatch_cp", "int4"),
        ("masked", "nvfp4"),
        ("dispatch_cp", "nvfp4"),
    ],
)
@pytest.mark.timeout(180)
def test_native_transformers_tp_ep_fsdp_matches_unique_data_reference(route, native_format):
    if int(os.environ.get("WORLD_SIZE", 1)) != 4:
        pytest.skip("Launch on four ranks")
    # Optional API introduced after the frozen HF 5.18 regression environment.
    from transformers.distributed.utils import TransformersDeviceMesh

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    cp = route == "dispatch_cp"
    tp_size, ep_size = (1, 2) if cp else (2, 2 if route == "masked" else 4)
    if cp:
        dense = init_device_mesh("cuda", (1, 2, 2, 1), mesh_dim_names=("pp", "cp", "fsdp", "tp"))
        expert = init_device_mesh("cuda", (1, 2, 1, 2), mesh_dim_names=("pp", "cp", "efsdp", "ep"))
    else:
        dense = init_device_mesh("cuda", (1, 2, 2), mesh_dim_names=("pp", "fsdp", "tp"))
        expert = init_device_mesh(
            "cuda", (1, 4 // ep_size, ep_size), mesh_dim_names=("pp", "efsdp", "ep")
        )
    manager = TransformersDeviceMesh(dense, expert)
    teacher, student, reference, global_batch, repair, expected = _repair_case(
        rank, 2, dtype=torch.bfloat16 if cp else torch.float32, native_format=native_format
    )
    ep_plan = None
    if route == "masked":
        ep_plan = {
            "model.layers.*.mlp.gate": "ep_router",
            "model.layers.*.mlp.experts": "moe_tp_experts",
        }
    distributed = DistributedConfig(tp_size=tp_size, fsdp_size=2, ep_size=ep_size, ep_plan=ep_plan)
    teacher.config.distributed_config = copy.deepcopy(distributed)
    student.config.distributed_config = copy.deepcopy(distributed)
    adapter = TransformersLocalDistillationAdapter(manager)
    if cp:
        assert adapter.context.context_mesh is not None
        assert adapter.context.context_mesh.size() == 2
        assert adapter.context.device_mesh.size() == 4
        assert adapter.context.data_context_mesh.size() == 4
        assert adapter.context.expert_replica_mesh.size() == 2
        # CP peers must retain the same expert owner; swapping one replica's owners breaks that contract.
        mismatched = expert.mesh.clone()
        mismatched[:, 1] = mismatched[:, 1].flip(-1)
        incompatible = DeviceMesh("cuda", mismatched, mesh_dim_names=expert.mesh_dim_names)
        with pytest.raises(ValueError, match="context parallel rank groups"):
            TransformersLocalDistillationAdapter(TransformersDeviceMesh(dense, incompatible))
    adapter.parallelize(teacher)
    adapter.parallelize(student)
    if route == "masked":
        forwards = tuple(
            (name, id(module.forward))
            for name, module in student.named_modules()
            if "forward" in module.__dict__
        )
        parameters = tuple((name, id(value)) for name, value in student.named_parameters())
        adopted = TransformersLocalDistillationAdapter(manager)
        adopted.parallelize(student)
        assert (
            tuple(
                (name, id(module.forward))
                for name, module in student.named_modules()
                if "forward" in module.__dict__
            )
            == forwards
        )
        assert tuple((name, id(value)) for name, value in student.named_parameters()) == parameters
    lane = dense["fsdp"].get_local_rank()
    batch = {key: value[lane : lane + 1] for key, value in global_batch.items()}
    observed = mtd.local_distill(student, teacher, [batch, batch], repair, adapter=adapter)
    _assert_matches_reference(student, reference, observed, expected, cp=cp)
    if native_format == "nvfp4":
        from examples.local_quantization_distillation.evaluate import (
            diagnostic_statistics,
            evaluate_pair,
        )

        diagnostic = dict(batch)
        diagnostic["labels"] = diagnostic["input_ids"].roll(-1, -1)
        diagnostic["labels"][:, -1] = -100
        with torch.no_grad():
            target = teacher(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            prediction = student(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            if isinstance(target, DTensor):
                target = target.full_tensor()
            if isinstance(prediction, DTensor):
                prediction = prediction.full_tensor()
            baseline = diagnostic_statistics(
                target, prediction, diagnostic["labels"], batch["valid_mask"]
            )
        dist.all_reduce(baseline, group=dense["fsdp"].get_group())
        malformed = copy.deepcopy(diagnostic)
        if dist.get_rank() == 0:
            malformed["labels"][0, 0] = 128
        with pytest.raises(ValueError, match="outside the output vocabulary"):
            evaluate_pair(
                teacher, student, [malformed], adapter=adapter, token_chunk=5, vocabulary_chunk=17
            )
        chunked = evaluate_pair(
            teacher, student, [diagnostic], adapter=adapter, token_chunk=5, vocabulary_chunk=17
        )
        torch.testing.assert_close(
            chunked["kld_teacher_candidate"],
            (baseline[2] / baseline[3]).item(),
            rtol=0.01,
            atol=2e-6,
        )
        torch.testing.assert_close(
            chunked["mean_cross_entropy"], (baseline[0] / baseline[1]).item(), rtol=1e-5, atol=2e-6
        )
        assert chunked["kl_tokens"] == 32
        assert "full_logits" not in chunked["projection_modes"]
    if cp:
        assert observed.metrics["execution"]["context_parallel_modes"] == ["native_sdpa"]
    actual_experts, reference_experts = (
        student.model.layers[0].mlp.experts,
        reference.model.layers[0].mlp.experts,
    )
    width = 4 // ep_size
    start = expert["ep"].get_local_rank() * width
    actual_counts = observed.metrics["units"]["model.layers.0"]["expert_coverage"]["mlp.experts"][
        "routed_assignments"
    ]
    reference_counts = expected.metrics["units"]["model.layers.0"]["expert_coverage"][
        "mlp.experts"
    ]["routed_assignments"]
    if route.startswith("dispatch"):
        assert actual_counts == reference_counts
        owner_ids = adapter._native_parallel_metadata[id(student)]["expert_global_ids"][
            "model.layers.0.mlp.experts"
        ]
        assert list(owner_ids) == list(range(start, start + width))
        assert [actual_counts[index] for index in owner_ids] == reference_counts[
            start : start + width
        ]
    else:
        assert actual_counts == reference_counts[start : start + width]
    for projection in ("gate_up_proj", "down_proj"):
        quantizers = getattr(actual_experts, projection + "_weight_quantizers")
        assert len(quantizers) == width
        reference_quantizers = getattr(reference_experts, projection + "_weight_quantizers")
        for index, quantizer in enumerate(quantizers):
            torch.testing.assert_close(
                quantizer._amax,
                reference_quantizers[start + index]._amax,
                rtol=0.025 if cp else 0.003,
                atol=0.0002 if cp else 1e-7,
            )
    dist.barrier()


@pytest.mark.timeout(180)
def test_transformers_hsdp_matches_four_unique_data_lanes():
    if int(os.environ.get("WORLD_SIZE", 1)) != 4:
        pytest.skip("Launch on four ranks")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard"))
    teacher, student, reference, global_batch, repair, expected = _repair_case(rank, 4)
    for model in (teacher, student):
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
    adapter = TransformersLocalDistillationAdapter(mesh)
    lane = adapter.context.data_mesh.get_local_rank()
    batch = {key: value[lane : lane + 1] for key, value in global_batch.items()}
    observed = mtd.local_distill(student, teacher, [batch, batch], repair, adapter=adapter)
    _assert_matches_reference(student, reference, observed, expected)
    actual_counts = observed.metrics["units"]["model.layers.0"]["expert_coverage"]["mlp.experts"][
        "routed_assignments"
    ]
    reference_counts = expected.metrics["units"]["model.layers.0"]["expert_coverage"][
        "mlp.experts"
    ]["routed_assignments"]
    assert actual_counts == reference_counts
    dist.barrier()


@pytest.mark.skipif(BailingHybridConfig is None, reason="Requires native Bailing hybrid modeling")
@pytest.mark.timeout(180)
def test_native_functional_conv_fsdp_preserves_attention_replay_backward():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size not in (4, 8):
        pytest.skip("Launch on four or eight ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    torch.manual_seed(91)
    config = BailingHybridConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        num_mtp_layers=0,
        pad_token_id=0,
        layer_types=["linear_attention"],
        use_cache=False,
    )
    config.linear_num_heads = 2
    config.linear_head_dim = 64
    model = BailingHybridForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    attention = model.model.layers[0].self_attn
    assert (attention.num_heads, attention.head_dim) == (2, 64)
    attention.conv1d.float()
    attention.forget_gate.A_log = torch.nn.Parameter(attention.forget_gate.A_log.float())
    attention.forget_gate.dt_bias = torch.nn.Parameter(attention.forget_gate.dt_bias.float())
    reference = copy.deepcopy(attention)
    _, _, plan = model._resolve_parallel_plans(DistributedConfig(tp_size=1, fsdp_size=world_size))
    plan = extend_family_fsdp_plan(model, plan)
    prefix = "model.layers.0.self_attn"
    assert prefix + ".forget_gate.f_proj" in plan
    for name in plan:
        if name.startswith(prefix + "."):
            fully_shard(model.get_submodule(name), mesh=mesh, reshard_after_forward=True)
    fully_shard(attention, mesh=mesh, reshard_after_forward=True)
    with pytest.raises(RuntimeError, match=r"mixed.*Tensor.*DTensor"):
        torch.nn.functional.conv1d(
            torch.zeros(2, attention.conv_dim, 8, device="cuda"),
            attention.conv1d.weight,
            groups=attention.conv_dim,
        )
    install_functional_fsdp_calls(model)
    forward = attention.forward
    install_functional_fsdp_calls(model)
    assert attention.forward is forward
    adapter = TransformersLocalDistillationAdapter()
    adapter.student = model
    inputs = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    with hf_torch_kernels(model):
        actual = adapter.replay_unit(attention, (inputs,), {})
        expected = reference(other)
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.0002)
        assert isinstance(attention.conv1d.weight, DTensor)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
    torch.testing.assert_close(inputs.grad, other.grad, rtol=0.025, atol=0.0002)
    relative_input_error = (
        inputs.grad.float() - other.grad.float()
    ).norm() / other.grad.float().norm()
    assert torch.isfinite(relative_input_error) and relative_input_error < 0.025
    actual_parameters = dict(attention.named_parameters())
    for name, parameter in reference.named_parameters():
        if parameter.grad is None:
            continue
        gradient = actual_parameters[name].grad
        assert gradient is not None, name
        if isinstance(gradient, DTensor):
            gradient = gradient.full_tensor()
        torch.testing.assert_close(gradient, parameter.grad, rtol=0.025, atol=0.0002, msg=name)
    assert isinstance(attention.conv1d.weight, DTensor)
    dist.barrier()


def _preparation_identity(model):
    """Capture observable ownership before a refused preparation attempt."""
    return (
        tuple((name, id(value)) for name, value in model.named_parameters()),
        tuple(
            (
                name,
                id(module.__dict__.get("forward")),
                tuple((key, id(value)) for key, value in module._forward_pre_hooks.items()),
                tuple((key, id(value)) for key, value in module._forward_hooks.items()),
            )
            for name, module in model.named_modules()
        ),
        id(model.config.distributed_config),
        tuple(
            (
                name,
                name in model.__dict__,
                id(model.__dict__.get(name)),
                repr(getattr(model, name, None)),
            )
            for name in ("_tp_plan", "_ep_plan", "_fsdp_plan")
        ),
        tuple(
            (name, id(module))
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.ModuleList)
        ),
    )


@pytest.mark.skipif(
    "ep_size" not in DistributedConfig.__dataclass_fields__,
    reason="Requires native HF separate TP/EP plans",
)
@pytest.mark.parametrize(
    "refusal", ["block_geometry", "unsafe_adopted_masked", "empty_resolved_ep_plan"]
)
@pytest.mark.timeout(180)
def test_native_transformers_refusal_preserves_model(refusal):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size not in (4, 8):
        pytest.skip("Launch on four or eight ranks")
    from transformers.distributed.utils import TransformersDeviceMesh

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    fsdp_size = world_size // 2
    dense = init_device_mesh("cuda", (1, fsdp_size, 2), mesh_dim_names=("pp", "fsdp", "tp"))
    expert = init_device_mesh("cuda", (1, fsdp_size, 2), mesh_dim_names=("pp", "efsdp", "ep"))
    manager = TransformersDeviceMesh(dense, expert)
    _, student = _repair_models()
    distributed = DistributedConfig(
        tp_size=2,
        fsdp_size=fsdp_size,
        ep_size=2,
        ep_plan={
            "model.layers.*.mlp.gate": "ep_router",
            "model.layers.*.mlp.experts": "moe_tp_experts",
        },
    )
    student.config.distributed_config = distributed
    if refusal == "block_geometry":
        quantizer = student.model.layers[0].self_attn.o_proj.weight_quantizer
        quantizer.set_from_attribute_config(
            {"num_bits": (2, 1), "block_sizes": {-1: 128, "type": "dynamic", "scale_bits": (4, 3)}}
        )
        message = "complete native quantization blocks"
    elif refusal == "empty_resolved_ep_plan":
        distributed.ep_plan = None
        student.ep_plan = {
            "model.layers.0.mlp.experts.gate_up_proj": "grouped_gemm",
            "model.layers.0.mlp.experts.down_proj": "grouped_gemm",
            "model.layers.0.mlp.experts": "ep_dispatch_experts",
        }
        message = "resolved EP plan is empty"
    else:
        student.maybe_distribute_model(student, distributed, manager)
        message = "safe masked"
    before = _preparation_identity(student)
    with pytest.raises(ValueError, match=message):
        TransformersLocalDistillationAdapter(manager).parallelize(student)
    assert _preparation_identity(student) == before
    dist.barrier()
