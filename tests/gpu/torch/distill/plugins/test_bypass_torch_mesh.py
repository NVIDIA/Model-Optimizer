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
from modelopt.torch.quantization.qtensor import FP8QTensor
from modelopt.torch.quantization.utils.native_weight import pin_native_weight


def _tiny_model(num_hidden_layers=1):
    # Transformers is optional, and the workers import this module before the test's skip check.
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


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
        for name in ("_modelopt_bypass_torch_mesh", "_modelopt_bypass_placement", "_mesh_manager")
    }
    return tensors, values, hooks, forwards, markers


def _assert_unchanged(model, snapshot):
    tensors, values, hooks, forwards, markers = snapshot
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


@cache
def _mesh_group():
    # Mesh caches retain group names; keep the bounded group until worker teardown.
    return dist.new_group(timeout=timedelta(seconds=10))


def _torch_mesh_identity_worker(rank, size):
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    assert size == 2
    group = _mesh_group()
    mesh_a = DeviceMesh.from_group(group, "cpu", mesh_dim_names=("tp",))
    mesh_b = DeviceMesh.from_group(group, "cpu", mesh_dim_names=("tp",))
    assert mesh_a is not mesh_b
    model = _tiny_model()
    adapter = TransformersBypassAdapter(mesh_a)
    with bounded_collectives(10):
        assert adapter.parallelize(model) is model
        assert isinstance(model.model.layers[0].self_attn.q_proj.weight, DTensor)
        snapshot = _snapshot(model)
        assert adapter.parallelize(model) is model
        _assert_unchanged(model, snapshot)
        with pytest.raises(ValueError):
            TransformersBypassAdapter(mesh_b).parallelize(model)
        assert model._modelopt_bypass_torch_mesh is mesh_a
        _assert_unchanged(model, snapshot)


@pytest.mark.timeout(180)
def test_raw_torch_mesh_reentry_preserves_placement(bypass_cpu_workers):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_torch_mesh_identity_worker)


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


def _torch_mesh_alias_worker(rank, size, route, sharing):
    # Native HF is optional; exercise its actual mesh manager and placement implementation.
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
    from modelopt.torch.distill.plugins.transformers.parallel import validate_hf_tensor_parallelism

    assert size == 2
    group = _mesh_group()
    mesh = DeviceMesh.from_group(group, "cpu", mesh_dim_names=("tp",))
    if route == "native":
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
    elif sharing in ("shared_module", "module_alias"):
        model.model.layers[0].self_attn.o_proj = first
        second = model.model.layers[0].self_attn.o_proj
        assert second is first
        assert second.weight is first.weight
    if sharing in ("tied_parameter", "tied_compatible", "shared_module", "module_alias"):
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
    if sharing != "rank_local" or rank == 0:
        second.weight_quantizer._set_buffer("_amax", shared)
    snapshot = _snapshot(model)
    error = None
    adapter = TransformersBypassAdapter(mesh)
    with bounded_collectives(10):
        try:
            if route == "explicit_plan":
                # HF expansion drops shared modules; an explicit plan must still resolve every owner.
                plan = {"model.layers.0.self_attn.q_proj.weight": "colwise"}
                if sharing == "module_alias":
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
    if sharing == "compatible":
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
        _assert_unchanged(model, snapshot)
        assert all(item is not None and item[0] == "ValueError" for item in errors), errors
        assert errors[0] == errors[1], errors
        if sharing in ("tied_compatible", "module_alias"):
            assert "alias" in errors[0][1], errors
            assert (
                "model.layers.1.self_attn.q_proj.weight"
                if sharing == "tied_compatible"
                else "model.layers.0.self_attn.o_proj"
            ) in errors[0][1], errors
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
        )
    ]
    + [("explicit_plan", "shared_module"), ("explicit_plan", "module_alias")],
)
@pytest.mark.timeout(180)
def test_shared_quantizer_fields_preflight(bypass_cpu_workers, route, sharing):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_torch_mesh_alias_worker, route, sharing)
