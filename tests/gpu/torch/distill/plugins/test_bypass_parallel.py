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
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig




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


@pytest.fixture
def native_workers(_dist_pools):
    if torch.cuda.device_count() < 4:
        pytest.skip("Native overlapping TP/FSDP/EP projection checks require four GPUs")
    from _test_utils.torch.distributed.utils import (
        DistributedWorkerPool,
        default_worker_teardown,
        get_live_pool,
    )

    return get_live_pool(
        _dist_pools,
        4,
        lambda: DistributedWorkerPool(
            world_size=4, backend="nccl", teardown_fn=default_worker_teardown
        ),
    )


def _native_projection_worker(rank, size, frozen_format):
    from _test_utils.transformers_families import build_model, cases
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    assert size == 4
    torch.cuda.set_device(rank)
    torch.manual_seed(61)
    record = cases(["qwen3_moe"])[0]
    record["minimum_pattern_prefix"] = 2
    record["config"]["mlp_only_layers"] = [0]
    model = build_model(record, torch.device("cuda", rank))
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
    manager = TransformersDeviceMesh(
        init_device_mesh("cuda", (1, 1, 2, 2), mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")),
        init_device_mesh(
            "cuda", (1, 1, 1, 4), mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep")
        ),
    )
    adapter = TransformersBypassAdapter(manager)
    adapter.parallelize(model)
    with torch.no_grad():
        actual = model(**batch).logits
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.02)


@pytest.mark.parametrize("frozen_format", ["fp8", "nvfp4"])
@pytest.mark.timeout(300)
def test_native_iq_placement_preserves_projection(native_workers, frozen_format):
    native_workers.run(_native_projection_worker, frozen_format)


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
