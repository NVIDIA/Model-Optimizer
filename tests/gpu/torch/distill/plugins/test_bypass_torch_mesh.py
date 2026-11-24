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

"""Raw Torch mesh provenance and collective preflight on two CPU/Gloo ranks."""

import copy
from datetime import timedelta
from functools import cache

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import bounded_collectives
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import FP8QTensor
from modelopt.torch.quantization.utils.native_weight import pin_native_weight


def _tiny_model(num_hidden_layers=1, *, experts=False):
    # Transformers is optional, and the workers import this module before the test's skip check.
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

    config_type, model_type = (
        (Qwen3MoeConfig, Qwen3MoeForCausalLM) if experts else (LlamaConfig, LlamaForCausalLM)
    )
    config = config_type(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
        **(
            {"num_experts": 4, "num_experts_per_tok": 1, "moe_intermediate_size": 16}
            if experts
            else {}
        ),
    )
    config._attn_implementation = config._experts_implementation = "eager"
    return model_type(config).eval()


def _snapshot(model):
    tensors = {
        **dict(model.named_parameters(remove_duplicate=False)),
        **dict(model.named_buffers(remove_duplicate=False)),
    }
    values = {
        name: (value.to_local() if isinstance(value, DTensor) else value).detach().clone()
        for name, value in tensors.items()
    }
    hooks = [
        (
            module,
            {
                name: dict(value)
                for name, value in vars(module).items()
                if "_hook" in name and isinstance(value, dict)
            },
        )
        for module in model.modules()
    ]
    forwards = [(module, module.forward) for module in model.modules()]
    markers = {
        name: (name in model.__dict__, getattr(model, name, None))
        for name in (
            "_modelopt_bypass_torch_mesh",
            "_modelopt_bypass_placement",
            "_mesh_manager",
            "_device_mesh",
            "_tp_size",
            "_fsdp_size",
            "_is_distributed_loading_by_transformers",
            "_tp_plan",
            "_ep_plan",
            "_fsdp_plan",
        )
    }
    config = getattr(model.config, "distributed_config", None)
    config_state = (
        "distributed_config" in vars(model.config),
        config,
        copy.deepcopy(vars(config)) if config is not None else None,
    )
    containers = [
        (module, name, getattr(module, name), tuple(getattr(module, name)))
        for module in model.modules()
        if hasattr(module, "_first_proj_weight_quantizers_attr")
        for name in (module._first_proj_weight_quantizers_attr, "down_proj_weight_quantizers")
    ]
    native_meta = [
        (module, copy.deepcopy(module._native_weight_meta))
        for module in model.modules()
        if getattr(module, "_native_weight_meta", None) is not None
    ]
    activation_state = [
        (
            module,
            {
                name: (hasattr(module, name), getattr(module, name, None))
                for name in (
                    "_native_tp_activation_layout",
                    "_native_tp_activation_fields",
                    "_quantizer_cache",
                    "_block_reshape_size",
                    "_original_shape",
                    "_padding",
                    "_slices",
                )
            },
        )
        for module in model.modules()
        if isinstance(module, TensorQuantizer)
    ]
    return (
        tensors,
        values,
        hooks,
        forwards,
        markers,
        config_state,
        containers,
        native_meta,
        activation_state,
    )


def _assert_unchanged(model, snapshot):
    (
        tensors,
        values,
        hooks,
        forwards,
        markers,
        config_state,
        containers,
        native_meta,
        activation_state,
    ) = snapshot
    for module, state in activation_state:
        for name, (present, value) in state.items():
            assert hasattr(module, name) == present
            assert getattr(module, name, None) is value
    for module, name, container, members in containers:
        assert getattr(module, name) is container
        assert tuple(container) == members
    current = {
        **dict(model.named_parameters(remove_duplicate=False)),
        **dict(model.named_buffers(remove_duplicate=False)),
    }
    assert current.keys() == tensors.keys()
    replaced = [name for name, tensor in current.items() if tensor is not tensors[name]]
    assert not replaced, f"Parameters/buffers replaced before refusal: {replaced}"
    for name, tensor in current.items():
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        torch.testing.assert_close(local, values[name], rtol=0, atol=0)
    for module, original in hooks:
        current_hooks = {
            name: dict(value)
            for name, value in vars(module).items()
            if "_hook" in name and isinstance(value, dict)
        }
        assert current_hooks == original
    for module, forward in forwards:
        assert module.forward == forward
    for name, (present, marker) in markers.items():
        assert (name in model.__dict__) == present
        assert getattr(model, name, None) is marker

    present, config, config_values = config_state
    assert ("distributed_config" in vars(model.config)) == present
    assert getattr(model.config, "distributed_config", None) is config
    if config is not None:
        assert vars(config) == config_values
    for module, metadata in native_meta:
        assert module._native_weight_meta == metadata


@cache
def _mesh_group():
    # Mesh caches retain group names; keep the bounded group until worker teardown.
    return dist.new_group(timeout=timedelta(seconds=10))


@cache
def _native_mesh(expert_parallel=False):
    # Keep native TP on the same bounded group for the persistent workers' lifetime.
    from transformers.distributed.utils import TransformersDeviceMesh

    group = _mesh_group()
    singletons = [dist.new_group([rank], timeout=timedelta(seconds=10)) for rank in range(2)]
    local = singletons[dist.get_rank()]
    ranks = torch.arange(2)
    return TransformersDeviceMesh(
        DeviceMesh.from_group(
            [local, local, local, group],
            "cpu",
            ranks.reshape(1, 1, 1, 2),
            mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp"),
        ),
        DeviceMesh.from_group(
            [local, local, local, group] if expert_parallel else [local, local, group, local],
            "cpu",
            ranks.reshape(1, 1, 1, 2) if expert_parallel else ranks.reshape(1, 1, 2, 1),
            mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep"),
        ),
    )


def _torch_mesh_state_worker(rank, size, route, state):
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    assert size == 2
    fields = state.startswith("adopt_fields")
    experts = state == "adopt_fields_experts"
    mesh = (
        DeviceMesh.from_group(_mesh_group(), "cpu", mesh_dim_names=("tp",))
        if route == "raw"
        else _native_mesh(expert_parallel=experts)
    )
    model = _tiny_model(experts=experts)
    original_containers = []
    if fields:
        model = mtq.quantize(
            model,
            {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
        )
        original_containers = [
            (module, name, getattr(module, name))
            for module in model.modules()
            if hasattr(module, "_first_proj_weight_quantizers_attr")
            for name in (module._first_proj_weight_quantizers_attr, "down_proj_weight_quantizers")
        ]
    adapter = TransformersBypassAdapter(mesh)
    with bounded_collectives(10):
        assert adapter.parallelize(model) is model
    assert isinstance(model.model.layers[0].self_attn.q_proj.weight, DTensor)
    if fields:
        from transformers.distributed.configuration_utils import DistributedConfig

        del model._modelopt_bypass_placement
        for module, name, container in original_containers:
            assert isinstance(module.gate_up_proj, DTensor)
            assert module.gate_up_proj.to_local().shape[0] == 2
            setattr(module, name, container)
        layer = model.model.layers[0].self_attn.q_proj
        quantizer = layer.weight_quantizer
        quantizer.set_from_attribute_config(
            QuantizerAttributeConfig(num_bits=(4, 3), axis=0, enable=True)
        )
        local = layer.weight.to_local()
        native, scale = FP8QTensor.quantize(local.detach(), axis=0)
        quantizer._set_buffer("_scale", scale)
        pin_native_weight(quantizer, native, local)
        valid_metadata = copy.deepcopy(quantizer._native_weight_meta)
        if state == "adopt_fields" or rank == 1:
            quantizer._native_weight_meta["source_shape"] = (30, 32)
        if state == "adopt_fields":
            del model.config.distributed_config
        else:
            model.config.distributed_config = DistributedConfig(
                tp_size=2,
                fsdp_size=1,
                ep_size=2 if experts else 1,
                tp_plan={"model.layers.*.self_attn.q_proj": "colwise"},
            )
    if state == "adopt_missing" and rank == 1:
        delattr(model, "_mesh_manager" if route == "native" else "_modelopt_bypass_torch_mesh")
    elif state == "adopt_wrong_mesh":
        other = DeviceMesh.from_group(_mesh_group(), "cpu", mesh_dim_names=("tp",))
        if rank == 1:
            model._modelopt_bypass_torch_mesh = other
    elif state == "adopt_fresh" or (state == "prepared_fresh" and rank == 1):
        model = _tiny_model()
    elif state == "marker_loss" and rank == 1:
        del model._modelopt_bypass_placement
    elif state == "adopted_fresh":
        del model._modelopt_bypass_placement
        if rank == 1:
            model = _tiny_model()
    elif state == "mesh_policy":
        if route == "raw":
            other = DeviceMesh.from_group(_mesh_group(), "cpu", mesh_dim_names=("tp",))
            adapter = TransformersBypassAdapter(other if rank == 0 else mesh)
        elif rank == 0:
            # A saved ordering can be stale on just one rank.
            model._modelopt_bypass_placement["dense_ranks"] = []
    # A new adapter exposes any context writes made before collective refusal.
    if route == "native":
        adapter = TransformersBypassAdapter(mesh)
    context_state = None
    if fields:
        adapter.context.parallel_metadata[id(model)] = {"previous": object()}
        adapter.context.tp_quantizer_layouts[id(model)] = {"previous": object()}
        context_state = [
            (mapping, dict(mapping))
            for mapping in (adapter.context.parallel_metadata, adapter.context.tp_quantizer_layouts)
        ]
    snapshot = _snapshot(model)
    error = None
    with bounded_collectives(10):
        try:
            if state.startswith("adopt_"):
                assert adapter.adopt(model) is None
                if state == "adopt_reentry":
                    assert adapter.adopt(model) is None
            else:
                assert adapter.parallelize(model) is model
        except Exception as exc:
            error = (type(exc).__name__, str(exc))
    # Keep result collection independent of a failed placement/validation subgroup.
    errors = [None] * size
    dist.all_gather_object(errors, error)
    if fields:
        assert all(item is not None and item[0] == "ValueError" for item in errors), errors
        assert errors[0] == errors[1], errors
    _assert_unchanged(model, snapshot)
    if state in ("reentry", "adopt_reentry"):
        assert errors == [None, None], errors
        if route == "native":
            assert id(model) in adapter.context.parallel_metadata
    else:
        assert all(item is not None and item[0] == "ValueError" for item in errors), errors
        assert errors[0] == errors[1], errors
        if route == "native":
            if fields:
                for mapping, previous in context_state:
                    assert mapping.keys() == previous.keys()
                    assert all(mapping[key] is value for key, value in previous.items())
                assert adapter.context.parallel_metadata is context_state[0][0]
                assert adapter.context.tp_quantizer_layouts is context_state[1][0]
            else:
                assert not adapter.context.parallel_metadata
                assert not adapter.context.tp_quantizer_layouts
    if fields:
        # A refused adoption is retryable before and after correcting the peer's fields.
        with bounded_collectives(10), pytest.raises(ValueError) as retry:
            adapter.adopt(model)
        assert str(retry.value) == errors[rank][1]
        _assert_unchanged(model, snapshot)
        quantizer._native_weight_meta = valid_metadata
        with bounded_collectives(10):
            adapter.adopt(model)
        for module, name, container in original_containers:
            assert tuple(getattr(module, name)) == tuple(container[rank * 2 : (rank + 1) * 2])
        placed = _snapshot(model)
        with bounded_collectives(10):
            adapter.adopt(model)
        _assert_unchanged(model, placed)


@pytest.mark.parametrize(
    ("route", "state"),
    [
        (route, state)
        for route in ("raw", "native")
        for state in ("prepared_fresh", "reentry", "mesh_policy")
    ]
    + [("native", "marker_loss"), ("native", "adopted_fresh")]
    + [
        (route, state)
        for route in ("raw", "native")
        for state in ("adopt_missing", "adopt_reentry")
    ]
    + [("raw", "adopt_wrong_mesh"), ("native", "adopt_fresh")]
    + [
        ("native", state) for state in ("adopt_fields", "adopt_fields_peer", "adopt_fields_experts")
    ],
)
@pytest.mark.timeout(180)
def test_torch_mesh_placement_state_is_collective(bypass_cpu_workers, route, state):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_torch_mesh_state_worker, route, state)


def _torch_mesh_validation_worker(rank, size, failure):
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    assert size == 2
    group = _mesh_group()
    mesh = DeviceMesh.from_group(group, "cpu", mesh_dim_names=("tp",))
    model = mtq.quantize(
        _tiny_model(),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    layer = model.model.layers[0].self_attn.q_proj
    quantizer = layer.weight_quantizer
    quantizer.set_from_attribute_config(
        QuantizerAttributeConfig(num_bits=(4, 3), axis=0, enable=True)
    )
    native, scale = FP8QTensor.quantize(layer.weight.detach(), axis=0)
    quantizer._set_buffer("_scale", scale)
    with torch.no_grad():
        layer.weight.copy_(native.dequantize(scale=scale))
    pin_native_weight(quantizer, native, layer.weight)
    if rank == 0:
        if failure == "wrong_shape":
            quantizer._native_weight_meta["source_shape"] = (30, 32)
        else:
            del quantizer._native_weight_codes
    snapshot = _snapshot(model)
    metadata = copy.deepcopy(quantizer._native_weight_meta)
    error = None
    with bounded_collectives(10):
        try:
            TransformersBypassAdapter(mesh).parallelize(model)
        except Exception as exc:
            error = (type(exc).__name__, str(exc))
    # Use the healthy default group after a possible bounded placement failure.
    errors = [None] * size
    dist.all_gather_object(errors, error)
    assert all(item is not None and item[0] == "ValueError" for item in errors), errors
    assert errors[0] == errors[1], errors
    assert not hasattr(model, "_modelopt_bypass_torch_mesh")
    assert quantizer._native_weight_meta == metadata
    _assert_unchanged(model, snapshot)


@pytest.mark.parametrize("failure", ["wrong_shape", "missing_codes"])
@pytest.mark.timeout(180)
def test_raw_torch_mesh_validation_is_collective(bypass_cpu_workers, failure):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_torch_mesh_validation_worker, failure)


def _input_alias_case(rank, size, route, sharing, mesh):
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    sequential = "sequential" in sharing
    colwise = sharing in ("input_sequential_colwise", "input_sequential_colwise_shared")
    geometry = sharing == "input_geometry"
    legacy = sharing == "input_legacy_reentry"
    prepared = sharing == "input_prepared_unplanned" or legacy
    retained = sharing in ("input_disabled_amax", "input_disabled_amax_single")
    unplanned = sharing in ("input_unplanned", "input_unplanned_compatible")
    compatible = sharing in ("input_compatible", "input_field_compatible", "input_geometry")
    disabled = sharing in (
        "input_sequential_disabled",
        "input_sequential_disabled_scale",
        "input_disabled_scale",
        "input_disabled_amax",
        "input_disabled_amax_single",
    )
    accept = (
        compatible
        or (colwise and sharing != "input_sequential_colwise_shared")
        or sharing
        in (
            "input_sequential_disabled",
            "input_unplanned_compatible",
            "input_disabled_amax_single",
            "input_legacy_reentry",
        )
    )
    tokens = torch.tensor([[1, 3, 5, 7]])
    first_name = "model.layers.0.self_attn.q_proj"
    second_name = (
        "model.layers.0.self_attn.k_proj"
        if geometry
        else "model.layers.1.self_attn.o_proj"
        if compatible
        else "model.layers.0.self_attn.o_proj"
    )
    if (compatible and not geometry) or prepared:
        first_name = "model.layers.0.self_attn.o_proj"
    if prepared:
        second_name = "model.layers.1.self_attn.o_proj"
    if unplanned:
        first_name = "lm_head"
        if sharing == "input_unplanned_compatible":
            second_name = "model.layers.0.self_attn.q_proj"
    target = first_name if colwise else second_name
    model = mtq.quantize(
        _tiny_model(num_hidden_layers=2),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                *(
                    [
                        {
                            "quantizer_name": f"{target}.input_quantizer",
                            "cfg": [{"num_bits": 8, "axis": -1}, {"num_bits": 8, "axis": -1}],
                        }
                    ]
                    if sequential
                    else [
                        {
                            "quantizer_name": f"{name}.input_quantizer",
                            "cfg": {
                                "num_bits": 8,
                                "axis": -1 if retained else None,
                                "enable": True,
                            },
                        }
                        for name in (first_name, second_name)
                    ]
                ),
            ],
            "algorithm": "max",
        },
        lambda m: m(tokens),
    )
    first, second = model.get_submodule(first_name), model.get_submodule(second_name)
    if geometry:
        assert first.weight.shape == (32, 32) and second.weight.shape == (16, 32)
        model._tp_plan = dict(model.tp_plan)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            model._tp_plan[f"model.layers.*.self_attn.{name}"] = "rowwise_rep"
    if unplanned or sharing == "input_prepared_unplanned":
        model._tp_plan = {
            name: style for name, style in model.tp_plan.items() if not name.startswith("lm_head")
        }
        assert "lm_head.weight" not in model._expand_parallel_plan(model.tp_plan)
    if retained:
        assert first.input_quantizer._amax.shape == (1, 1, 32)
    if not sequential and not retained:
        assert first.input_quantizer._amax.numel() == 1
        first.input_quantizer.pre_quant_scale = torch.linspace(0.75, 1.25, 32)
        second.input_quantizer.pre_quant_scale = torch.linspace(0.75, 1.25, 32)
    if sequential:
        chain = model.get_submodule(target).input_quantizer
        assert isinstance(chain, SequentialQuantizer)
        assert all(member._amax.shape == (1, 1, 32) for member in chain)
        if sharing == "input_sequential_disabled_scale":
            chain[0].pre_quant_scale = torch.linspace(0.75, 1.25, 32)
    if disabled:
        for module in (first, second):
            module.input_quantizer.disable()
    with torch.no_grad():
        assert model(tokens).logits.shape == (1, 4, 32)
    if route == "native_adopt" and not prepared:
        # Simulate external native placement retaining full calibrated input fields.
        original_inputs = [(module, module.input_quantizer) for module in (first, second)]
        for module, _ in original_inputs:
            module.input_quantizer = copy.deepcopy(module.weight_quantizer)
        with bounded_collectives(10):
            TransformersBypassAdapter(mesh).parallelize(model)
        del model._modelopt_bypass_placement
        for module, quantizer in original_inputs:
            module.input_quantizer = quantizer
    if (
        not sequential
        and not legacy
        and sharing != "input_disabled_amax_single"
        and (sharing != "input_peer" or rank == 1)
    ):
        if sharing.startswith("input_field"):
            second.input_quantizer._set_buffer(
                "_pre_quant_scale", first.input_quantizer.pre_quant_scale
            )
            assert first.input_quantizer is not second.input_quantizer
        else:
            second.input_quantizer = first.input_quantizer
    if sharing == "input_sequential_colwise_shared":
        second.input_quantizer = first.input_quantizer[0]
    if sequential and sharing == "input_sequential_peer" and rank == 0:
        second.input_quantizer.disable()
    if route != "native_adopt":
        with torch.no_grad():
            assert model(tokens).logits.shape == (1, 4, 32)
    if prepared:
        with bounded_collectives(10):
            TransformersBypassAdapter(mesh).parallelize(model)
        with torch.no_grad():
            assert model(tokens).logits.shape == (1, 4, 32)
        assert first.input_quantizer.pre_quant_scale.shape == (16,)
        if legacy:
            for module in (first, second):
                del module.input_quantizer._native_tp_activation_fields
        else:
            assert not isinstance(model.lm_head.weight, DTensor)
            model.lm_head.input_quantizer = first.input_quantizer
        del model._modelopt_bypass_placement
    original_scale = getattr(first.input_quantizer, "pre_quant_scale", None)
    snapshot = _snapshot(model)
    inputs = [(module, module.input_quantizer) for module in (first, second)]
    adapter = TransformersBypassAdapter(mesh)
    adapter.context.parallel_metadata[id(model)] = {"previous": object()}
    adapter.context.tp_quantizer_layouts[id(model)] = {"previous": object()}
    context_state = [
        (name, getattr(adapter.context, name), dict(getattr(adapter.context, name)))
        for name in ("parallel_metadata", "tp_quantizer_layouts")
    ]
    error = None
    with bounded_collectives(10):
        try:
            if route == "native_adopt":
                adapter.adopt(model)
            else:
                adapter.parallelize(model)
        except Exception as exc:
            error = (type(exc).__name__, str(exc))
    errors = [None] * size
    dist.all_gather_object(errors, error)
    if accept:
        assert errors == [None, None], errors
        if unplanned:
            assert not isinstance(model.lm_head.weight, DTensor)
        if legacy:
            current = dict(model.named_parameters(remove_duplicate=False))
            current.update(model.named_buffers(remove_duplicate=False))
            assert all(current[name] is tensor for name, tensor in snapshot[0].items())
            assert all(module.forward == forward for module, forward in snapshot[3])
            assert not hasattr(first.input_quantizer, "_native_tp_activation_fields")
        if compatible:
            assert first.input_quantizer.pre_quant_scale is second.input_quantizer.pre_quant_scale
            torch.testing.assert_close(
                first.input_quantizer.pre_quant_scale,
                original_scale.narrow(0, rank * 16, 16),
                rtol=0,
                atol=0,
            )
        with torch.no_grad():
            assert model(tokens).logits.shape == (1, 4, 32)
    else:
        assert all(item is not None and item[0] == "ValueError" for item in errors), errors
        assert errors[0] == errors[1], errors
        _assert_unchanged(model, snapshot)
        for module, quantizer in inputs:
            assert module.input_quantizer is quantizer
        for name, mapping, previous in context_state:
            assert getattr(adapter.context, name) is mapping
            assert mapping.keys() == previous.keys()
            assert all(mapping[key] is value for key, value in previous.items())


def _torch_mesh_alias_worker(rank, size, route, sharing):
    # Native HF is optional; exercise its actual mesh manager and placement implementation.
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
    from modelopt.torch.distill.plugins.transformers.parallel import validate_hf_tensor_parallelism

    assert size == 2
    planned_module = sharing in ("shared_module", "shared_module_peer", "shared_module_override")
    group = _mesh_group()
    mesh = DeviceMesh.from_group(group, "cpu", mesh_dim_names=("tp",))
    if route in ("native", "native_adopt"):
        mesh = TransformersDeviceMesh(
            DeviceMesh(
                "cpu",
                torch.arange(size).reshape(1, 1, 1, size),
                mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp"),
            ),
            DeviceMesh(
                "cpu",
                torch.arange(size).reshape(1, 1, size, 1),
                mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep"),
            ),
        )
    if sharing.startswith("input_"):
        return _input_alias_case(rank, size, route, sharing, mesh)
    model = mtq.quantize(
        _tiny_model(num_hidden_layers=2),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    first = model.model.layers[0].self_attn.q_proj
    second = (
        model.model.layers[1].self_attn.q_proj
        if sharing in ("compatible", "tied_compatible")
        else model.model.layers[0].self_attn.o_proj
    )
    assert first.weight.shape == second.weight.shape == (32, 32)
    if sharing in ("tied_parameter", "tied_compatible"):
        second.weight = first.weight
        assert first is not second
        assert second.weight is first.weight
    elif sharing == "module_alias" or (
        planned_module and (sharing != "shared_module_peer" or rank == 1)
    ):
        model.model.layers[0].self_attn.o_proj = first
        second = model.model.layers[0].self_attn.o_proj
        assert second is first
        assert second.weight is first.weight
    if sharing in ("tied_parameter", "tied_compatible", "module_alias") or (
        planned_module and (sharing != "shared_module_peer" or rank == 1)
    ):
        second_name = (
            "model.layers.1.self_attn.q_proj.weight"
            if sharing == "tied_compatible"
            else "model.layers.0.self_attn.o_proj.weight"
        )
        assert second_name not in dict(model.named_parameters())
        assert dict(model.named_parameters(remove_duplicate=False))[second_name] is first.weight
        if sharing in ("tied_parameter", "tied_compatible"):
            assert second_name in model._expand_parallel_plan(model.tp_plan)
        else:
            assert "model.layers.0.self_attn.o_proj" not in dict(model.named_modules())
    if sharing == "compatible":
        model.unplanned_alias = first
        assert model.unplanned_alias is first
        assert not any(
            name.startswith("unplanned_alias")
            for name in model._expand_parallel_plan(model.tp_plan)
        )
    for layer in (first, second):
        layer.weight_quantizer.set_from_attribute_config(
            QuantizerAttributeConfig(num_bits=8, axis=0, enable=True)
        )
        # Real calibrated per-output geometry; values are distinct across output rows.
        layer.weight_quantizer._set_buffer(
            "_amax", layer.weight.detach().abs().amax(dim=1, keepdim=True)
        )
    shared = first.weight_quantizer._amax
    assert shared.shape == (32, 1)
    if sharing != "shared_module_peer" and (sharing != "rank_local" or rank == 0):
        second.weight_quantizer._set_buffer("_amax", shared)
    if sharing == "shared_module_override":
        from transformers.distributed.configuration_utils import DistributedConfig

        model.config.distributed_config = DistributedConfig(
            tp_size=2,
            tp_plan={"model.layers.*.self_attn.o_proj": "colwise"},
        )
    if planned_module:
        # The ordinary unplaced graph works; refusal must precede native parameter replacement.
        with torch.no_grad():
            assert model(torch.tensor([[1, 3, 5, 7]])).logits.shape == (1, 4, 32)
    snapshot = _snapshot(model)
    error = None
    adapter = TransformersBypassAdapter(mesh)
    context_state = [
        (name, getattr(adapter.context, name), dict(getattr(adapter.context, name)))
        for name in ("parallel_metadata", "tp_quantizer_layouts")
    ]
    with bounded_collectives(10):
        try:
            if route == "explicit_plan":
                # HF expansion drops shared modules; an explicit plan must still resolve every owner.
                plan = {"model.layers.0.self_attn.q_proj.weight": "colwise"}
                if sharing in ("canonical_module", "unknown_key"):
                    key = (
                        "model.layers.0.self_attn.q_proj"
                        if sharing == "canonical_module"
                        else "missing.weight"
                    )
                    plan = {key: "colwise"}
                elif sharing == "expanded_parameter":
                    pass
                elif sharing == "module_alias":
                    plan["model.layers.0.self_attn.o_proj"] = "colwise"
                else:
                    plan["model.layers.0.self_attn.o_proj.weight"] = "rowwise"
                validate_hf_tensor_parallelism(model, mesh, plan)
            else:
                adapter.parallelize(model)
        except Exception as exc:
            error = (type(exc).__name__, str(exc))
    errors = [None] * size
    dist.all_gather_object(errors, error)
    if sharing == "expanded_parameter":
        assert errors == [None, None], errors
        _assert_unchanged(model, snapshot)
    elif sharing == "compatible":
        assert errors == [None, None], errors
        assert first.weight_quantizer._amax is second.weight_quantizer._amax
        torch.testing.assert_close(
            first.weight_quantizer._amax, shared.narrow(0, rank * 16, 16), rtol=0, atol=0
        )
        placed = _snapshot(model)
        with bounded_collectives(10):
            adapter.parallelize(model)
        _assert_unchanged(model, placed)
    else:
        assert all(item is not None and item[0] == "ValueError" for item in errors), errors
        assert errors[0] == errors[1], errors
        _assert_unchanged(model, snapshot)
        for name, mapping, previous in context_state:
            assert getattr(adapter.context, name) is mapping
            assert mapping.keys() == previous.keys()
            assert all(mapping[key] is value for key, value in previous.items())
        if sharing in ("tied_compatible", "module_alias") or (
            planned_module and route != "explicit_plan"
        ):
            assert "alias" in errors[0][1], errors
            assert (
                "model.layers.1.self_attn.q_proj.weight"
                if sharing == "tied_compatible"
                else "model.layers.0.self_attn.o_proj"
            ) in errors[0][1], errors
        elif sharing in ("canonical_module", "unknown_key"):
            assert key in errors[0][1], errors
        else:
            assert "incompatible TP layouts" in errors[0][1], errors


@pytest.mark.parametrize(
    ("route", "sharing"),
    [
        (route, sharing)
        for route in ("raw", "native")
        for sharing in (
            "incompatible",
            "rank_local",
            "compatible",
            "tied_parameter",
            "tied_compatible",
            "shared_module",
            "shared_module_peer",
        )
    ]
    + [
        (route, sharing)
        for route in ("raw", "native")
        for sharing in (
            "input_incompatible",
            "input_peer",
            "input_field",
            "input_compatible",
            "input_field_compatible",
            "input_geometry",
            "input_sequential",
            "input_sequential_peer",
            "input_sequential_colwise",
            "input_sequential_colwise_shared",
            "input_sequential_disabled",
            "input_sequential_disabled_scale",
            "input_disabled_scale",
            "input_unplanned",
            "input_unplanned_compatible",
        )
    ]
    + [
        ("native_adopt", sharing)
        for sharing in (
            "input_incompatible",
            "input_peer",
            "input_field",
            "input_sequential",
            "input_unplanned",
            "input_prepared_unplanned",
            "input_legacy_reentry",
        )
    ]
    + [("native", "input_disabled_amax"), ("native", "input_disabled_amax_single")]
    + [("native", "shared_module_override")]
    + [
        ("explicit_plan", sharing)
        for sharing in (
            "shared_module",
            "module_alias",
            "canonical_module",
            "unknown_key",
            "expanded_parameter",
        )
    ],
)
@pytest.mark.timeout(180)
def test_shared_quantizer_fields_preflight(bypass_cpu_workers, route, sharing):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_torch_mesh_alias_worker, route, sharing)
