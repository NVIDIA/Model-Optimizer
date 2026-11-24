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


def _tiny_model():
    # Transformers is optional, and the workers import this module before the test's skip check.
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def _snapshot(model):
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    values = {
        name: (value.to_local() if isinstance(value, DTensor) else value).detach().clone()
        for name, value in tensors.items()
    }
    hooks = [
        (module, dict(module._forward_hooks), dict(module._forward_pre_hooks))
        for module in model.modules()
    ]
    return tensors, values, hooks


def _assert_unchanged(model, snapshot):
    tensors, values, hooks = snapshot
    current = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    assert current.keys() == tensors.keys()
    for name, tensor in current.items():
        assert tensor is tensors[name]
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        torch.testing.assert_close(local, values[name], rtol=0, atol=0)
    for module, after, before in hooks:
        assert dict(module._forward_hooks) == after
        assert dict(module._forward_pre_hooks) == before


def _torch_mesh_identity_worker(rank, size):
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    assert size == 2
    group = dist.new_group(timeout=timedelta(seconds=10))
    try:
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
    finally:
        dist.destroy_process_group(group)


@pytest.mark.timeout(180)
def test_raw_torch_mesh_reentry_preserves_placement(bypass_cpu_workers):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_torch_mesh_identity_worker)


def _torch_mesh_validation_worker(rank, size, failure):
    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    assert size == 2
    group = dist.new_group(timeout=timedelta(seconds=10))
    try:
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
    finally:
        dist.destroy_process_group(group)


@pytest.mark.parametrize("failure", ["wrong_shape", "missing_codes"])
@pytest.mark.timeout(180)
def test_raw_torch_mesh_validation_is_collective(bypass_cpu_workers, failure):
    pytest.importorskip("transformers", minversion="5.23")
    bypass_cpu_workers.run(_torch_mesh_validation_worker, failure)
