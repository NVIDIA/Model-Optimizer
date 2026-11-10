# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import weakref
from unittest import mock

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils import (
    convert_quantization_axis_to_reduce_axis,
    core_utils,
    reduce_amax,
    reduce_block_amax,
)
from modelopt.torch.quantization.utils.core_utils import (
    _disable_fsdp_unshard_reshard,
    _fsdp2_owned_params,
    _fsdp2_owns,
    _fsdp2_param_groups,
    enable_fake_quant,
    fsdp2_weight_access_and_writeback_context,
    persistent_materialization,
)
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector


@pytest.mark.parametrize(
    ("block_sizes", "test_input", "expected_scales"),
    [
        (
            {-1: 2, -2: 2},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([5.0, 7.0], dtype=torch.bfloat16),
        ),
        (
            {-1: 4, -2: 2},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([7.0, 15.0], dtype=torch.bfloat16),
        ),
        (
            {-2: 4},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([[12.0, 13.0, 14.0, 15.0]], dtype=torch.bfloat16),
        ),
    ],
)
def test_reduce_block_amax(block_sizes, test_input, expected_scales):
    scales = reduce_block_amax(test_input, block_sizes)

    torch.allclose(scales, expected_scales)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
def test_reduce_amax_fp8(fp8_dtype, axis):
    """FP8 tensors have no reduction/abs kernels; reduce_amax must upcast them.

    Regression test for ``NotImplementedError: "max_all_cuda" not implemented for
    'Float8_e4m3fn'`` when calibrating models with natively FP8 weights (e.g. DeepSeek-V3).
    """
    # Values chosen to be exactly representable in both FP8 formats so the upcast is lossless.
    ref = torch.tensor([[1.0, -3.0, 2.0], [0.5, -0.25, 4.0]])
    x_fp8 = ref.to(fp8_dtype)

    out = reduce_amax(x_fp8, axis=axis)
    expected = reduce_amax(ref, axis=axis)

    assert out.dtype == torch.get_default_dtype()
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    ("shape", "quant_axis", "expected_reduce_axis"),
    [
        ((2, 3, 4), None, None),  # Per-tensor quantization (None axis)
        ((2, 3, 4), 0, [1, 2]),  # Single axis cases
        ((2, 3, 4), -2, [0, 2]),  # Negative indices
        ((2, 3, 4, 5), [0, -1], [1, 2]),  # Multiple axes
        ((2, 3, 4), (0, 2), [1]),  # Tuple instead of list
    ],
)
def test_convert_quantization_axis_to_reduce_axis(shape, quant_axis, expected_reduce_axis):
    """Test converting quantization axes to reduction axes."""
    # Create a tensor with the specified shape
    input_tensor = torch.randn(shape)

    # Convert quantization axis to reduce axis
    result = convert_quantization_axis_to_reduce_axis(input_tensor, quant_axis)

    # Check if the result matches expected
    assert result == expected_reduce_axis, (
        f"For shape {shape} and quant_axis {quant_axis}, expected {expected_reduce_axis} but got {result}"
    )

    # Additional sanity check: if we sum-reduce along the returned axes,
    # we should get a tensor with the same shape as if we kept the quantization axes
    if result is not None and len(result) > 0:
        # Create a ones tensor for easy shape verification
        ones = torch.ones(shape)

        # Reduce sum across the returned axes
        reduced = ones
        for axis in sorted(
            result, reverse=True
        ):  # Reduce from highest dim to avoid changing indices
            reduced = reduced.sum(dim=axis, keepdim=True)

        # Build expected shape after reduction
        expected_shape = list(shape)
        for axis in sorted(result, reverse=True):
            expected_shape[axis] = 1

        assert reduced.shape == tuple(expected_shape), (
            f"Reduction result shape {reduced.shape} doesn't match expected {tuple(expected_shape)}"
        )


def test_layer_activation_collector_support_api(monkeypatch):
    class _SupportedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    class _UnsupportedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

    supported = _SupportedModel()
    unsupported = _UnsupportedModel()

    def _supports_layers(model):
        return hasattr(model, "layers")

    def _discover_layers(model):
        return model.layers

    monkeypatch.setattr(LayerActivationCollector, "_decoder_layer_support", [])
    LayerActivationCollector.register_decoder_layer_support(_supports_layers, _discover_layers)

    assert LayerActivationCollector.is_supported(supported)
    assert LayerActivationCollector.get_decoder_layers(supported) is supported.layers
    assert not LayerActivationCollector.is_supported(unsupported)
    assert LayerActivationCollector.get_decoder_layers(unsupported) is None


def test_layer_activation_collector_decoder_discoverer_resolution_order(monkeypatch):
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    calls = {"first": 0, "second": 0}

    def _supported(_model):
        return True

    def _first_discoverer(_model):
        calls["first"] += 1

    def _second_discoverer(model):
        calls["second"] += 1
        return model.layers

    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(_supported, _first_discoverer), (_supported, _second_discoverer)],
    )

    model = _Model()
    resolved = LayerActivationCollector.get_decoder_layers(model)
    assert resolved is model.layers
    assert calls["first"] == 1
    assert calls["second"] == 1


def test_layer_activation_collector_decoder_discoverer_no_match(monkeypatch):
    class _Model(torch.nn.Module):
        pass

    def _unsupported(_model):
        return False

    def _discoverer(_model):
        return torch.nn.ModuleList([torch.nn.Identity()])

    monkeypatch.setattr(
        LayerActivationCollector, "_decoder_layer_support", [(_unsupported, _discoverer)]
    )

    model = _Model()
    assert LayerActivationCollector.get_decoder_layers(model) is None
    assert not LayerActivationCollector.is_supported(model)


def test_layer_activation_collector_decoder_discoverer_dedup(monkeypatch):
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def _supported(model):
        return hasattr(model, "layers")

    def _discoverer(model):
        return model.layers

    monkeypatch.setattr(LayerActivationCollector, "_decoder_layer_support", [])
    LayerActivationCollector.register_decoder_layer_support(_supported, _discoverer)
    LayerActivationCollector.register_decoder_layer_support(_supported, _discoverer)

    assert len(LayerActivationCollector._decoder_layer_support) == 1


def test_layer_activation_collector_skip_forward_captures_correct_inputs(monkeypatch):
    """The skip/run/capture strategy produces the same inputs as a plain forward."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.scale = scale

        def forward(self, hidden_states):
            return hidden_states * self.scale

    class _ToyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_ToyLayer(2.0), _ToyLayer(0.5), _ToyLayer(3.0)])

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )

    model = _ToyDecoder()
    batches = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]

    def _forward_loop(m):
        for b in batches:
            m(b)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        inp0 = collector.get_input_activations(model.layers[0], _forward_loop)
        inp1 = collector.get_input_activations(model.layers[1], _forward_loop)
        inp2 = collector.get_input_activations(model.layers[2], _forward_loop)
    finally:
        collector._unpatch_all_layers()

    expected_0 = batches
    expected_1 = [model.layers[0](b) for b in batches]
    expected_2 = [model.layers[1](b) for b in expected_1]

    for (args, _kw), exp in zip(inp0, expected_0):
        assert torch.allclose(args[0], exp)
    for (args, _kw), exp in zip(inp1, expected_1):
        assert torch.allclose(args[0], exp)
    for (args, _kw), exp in zip(inp2, expected_2):
        assert torch.allclose(args[0], exp)


def test_layer_activation_collector_run_uses_cached_inputs_not_parent(monkeypatch):
    """Verify that the 'run' layer uses cached inputs, not garbage from skip layers."""

    call_log = []

    class _ToyLayer(torch.nn.Module):
        def __init__(self, name, bias):
            super().__init__()
            self.layer_name = name
            self.bias = bias

        def forward(self, hidden_states):
            call_log.append(self.layer_name)
            return hidden_states + self.bias

    class _ToyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [_ToyLayer("L0", 1.0), _ToyLayer("L1", 2.0), _ToyLayer("L2", 3.0)]
            )

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )

    model = _ToyDecoder()
    batches = [torch.tensor([[10.0]])]

    def _forward_loop(m):
        for b in batches:
            m(b)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        collector.get_input_activations(model.layers[0], _forward_loop)
        call_log.clear()
        collector.get_input_activations(model.layers[1], _forward_loop)
        call_log_for_layer1 = list(call_log)
        call_log.clear()
        inp2 = collector.get_input_activations(model.layers[2], _forward_loop)
    finally:
        collector._unpatch_all_layers()

    assert "L0" in call_log_for_layer1
    assert "L1" not in call_log_for_layer1

    assert torch.allclose(inp2[0][0][0], torch.tensor([[10.0 + 1.0 + 2.0]]))


def test_enable_fake_quant_restores_each_stage_when_the_window_raises():
    """A W4A8 weight quantizer holds the flag on each of its stages. A window that raises, such as
    an FSDP2 unshard running out of memory, must give every stage its own flag back."""
    module = torch.nn.Module()
    module.weight_quantizer = SequentialQuantizer(
        TensorQuantizer(QuantizerAttributeConfig(fake_quant=False)), TensorQuantizer()
    )

    with pytest.raises(RuntimeError), enable_fake_quant(module):
        assert [q._fake_quant for q in module.weight_quantizer] == [True, True]
        raise RuntimeError

    assert [q._fake_quant for q in module.weight_quantizer] == [False, True]


def _forward_through_other_layers_while_one_is_materialized(rank, size, writeback):
    mesh = init_device_mesh("cpu", (size,))
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    for layer in model:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    x = torch.randn(2, 8)
    expected = model(x)
    # Layerwise calibration runs the whole model to collect the next layer's inputs while one layer
    # is materialized; the other FSDP modules must still unshard.
    with persistent_materialization(model[0], writeback=writeback):
        torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(model(x), expected)  # FSDP unshard/reshard restored on exit


# writeback=False nests the unshard/reshard override (via _fsdp2_unshard_context).
@pytest.mark.parametrize("writeback", [True, False])
def test_persistent_materialization_only_skips_unshard_for_its_layer(cpu_dist_workers, writeback):
    cpu_dist_workers.run(_forward_through_other_layers_while_one_is_materialized, writeback)


def _shared_group_override_restores_prior_attributes(rank, size):
    mesh = init_device_mesh("cpu", (size,))
    block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    fully_shard(list(block), mesh=mesh)  # both linears share one FSDPParamGroup
    fully_shard(block, mesh=mesh)
    group = fully_shard.state(block[0])._fsdp_param_group
    prior_reshard = group.reshard  # pre-existing instance override must survive
    group.reshard = prior_reshard
    with _disable_fsdp_unshard_reshard(block):
        with _disable_fsdp_unshard_reshard(block):
            pass
        assert group.unshard() is None  # outer no-op survives the inner exit
        assert "unshard" in group.__dict__
    assert "unshard" not in group.__dict__
    assert group.__dict__["reshard"] is prior_reshard
    # A layer that is not itself FSDP-wrapped was not gathered, so its FSDP submodules must unshard.
    with _disable_fsdp_unshard_reshard(torch.nn.Sequential(*block)):
        assert "unshard" not in group.__dict__


def test_disable_fsdp_unshard_reshard_shared_group_and_nesting(cpu_dist_workers):
    cpu_dist_workers.run(_shared_group_override_restores_prior_attributes)


def _fsdp2_owns_its_params_in_every_state_job(rank, size):
    mesh = init_device_mesh("cpu", (size,))
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(8, 8)
    model.layer = torch.nn.Linear(8, 8)
    fully_shard(model.layer, mesh=mesh)  # a unit of its own, inside the model's
    fully_shard(model, mesh=mesh)
    layer = model.layer
    sharded = layer.weight
    owned = _fsdp2_owned_params(model)
    assert _fsdp2_owns(owned, model.dense.weight) and _fsdp2_owns(owned, sharded)
    # An id kept for a parameter does not make another object at that address FSDP2's.
    assert not _fsdp2_owns({id(sharded): weakref.ref(model.dense.weight)}, sharded)
    layer.unshard()
    unsharded = layer.weight
    try:
        assert unsharded is not sharded, "FSDP2 registered the same parameter when unsharded"
        assert _fsdp2_owns(_fsdp2_owned_params(model, refresh=True), unsharded)
    finally:
        layer.reshard()
    # The set is kept for the unit: opening the window again costs the module's parameters only.
    with mock.patch.object(core_utils, "_fsdp2_param_groups", wraps=_fsdp2_param_groups) as scans:
        for _ in range(3):
            with fsdp2_weight_access_and_writeback_context(model.dense, model):
                pass
    assert scans.call_count == 0


def test_fsdp2_owns_its_parameters_in_every_shard_state(cpu_dist_workers):
    """FSDP2 registers another parameter object once a unit is unsharded (or resharded after the
    forward, under HSDP), and the weight-access window gathers by ownership: each counts as FSDP2's,
    or a weight it does own would be left sharded in the window. So do the parameters of a unit
    nested in it. The window keeps the set per unit rather than scanning it on every entry."""
    cpu_dist_workers.run(_fsdp2_owns_its_params_in_every_state_job)


def _window_leaves_what_fsdp2_does_not_own_job(rank, size):
    ep_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("ep",))
    dp_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("dp",))
    layer = torch.nn.Module()
    layer.dense = torch.nn.Linear(8, 8)
    layer.experts = torch.nn.Module()
    for name in ("gate_up_proj", "down_proj"):
        weight = distribute_tensor(torch.randn(2 * size, 8), ep_mesh, [Shard(0)])
        layer.experts.register_parameter(name, torch.nn.Parameter(weight))
    fully_shard(layer, mesh=dp_mesh, ignored_params=set(layer.experts.parameters()))
    with fsdp2_weight_access_and_writeback_context(layer, layer):
        assert not isinstance(layer.dense.weight, DTensor)  # FSDP2's own: gathered
        for weight in layer.experts.parameters():
            assert isinstance(weight, DTensor)
            assert weight.to_local().shape[0] == 2
    # Every expert weight misses the kept set, and one scan of the unit answers them all.
    with (
        mock.patch.object(core_utils, "_fsdp2_param_groups", wraps=_fsdp2_param_groups) as scans,
        fsdp2_weight_access_and_writeback_context(layer, layer),
    ):
        pass
    assert scans.call_count == 1
    assert all(weight.to_local().shape[0] == 2 for weight in layer.experts.parameters())


def test_fsdp2_weight_access_leaves_parameters_fsdp2_does_not_own(cpu_dist_workers):
    """An expert-parallel weight is a DTensor sharded over the experts that fully_shard was told to
    ignore. The weight-access window gathers FSDP2's parameters only, so it stays sharded: gathering
    it would hand every rank every expert. Each such weight misses the window's kept set, which it
    scans again once, not once per weight."""
    cpu_dist_workers.run(_window_leaves_what_fsdp2_does_not_own_job)
