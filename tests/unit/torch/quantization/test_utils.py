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
from contextlib import ExitStack, nullcontext
from unittest import mock

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

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
    enable_weight_access_and_writeback,
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


def _fsdp2_owns_its_params_in_every_state_job(rank, size, asymmetric, receipt_stage="dispatcher"):
    native_tp = receipt_stage == "native_tp"
    world_mesh = (
        init_device_mesh("cpu", (size, 1), mesh_dim_names=("dp", "tp")) if native_tp else None
    )
    mesh = world_mesh["dp"] if native_tp else init_device_mesh("cpu", (size,))
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(8, 8)
    model.layer = torch.nn.Linear(8, 8)
    model.shared = torch.nn.Module()
    model.shared.weight = model.dense.weight
    model.ignored = torch.nn.Linear(8, 8)
    if native_tp:
        for name, parameter in list(model.layer.named_parameters()):
            model.layer.register_parameter(
                name,
                torch.nn.Parameter(
                    distribute_tensor(parameter.detach(), world_mesh["tp"], [Shard(0)])
                ),
            )
    fully_shard(model.layer, mesh=mesh)  # a unit of its own, inside the model's
    fully_shard(model, mesh=mesh, ignored_params=set(model.ignored.parameters()))
    layer = model.layer
    sharded = layer.weight
    owned = _fsdp2_owned_params(model)
    dense_weight = model.dense.weight
    assert model.shared.weight is dense_weight
    assert _fsdp2_owns(owned, model.dense.weight) and _fsdp2_owns(owned, sharded)
    # An id kept for a parameter does not make another object at that address FSDP2's.
    assert not _fsdp2_owns({id(sharded): weakref.ref(model.dense.weight)}, sharded)
    shard_before = sharded.to_local().detach().clone()
    _fsdp2_owned_params(layer)
    layer.unshard()
    unsharded = layer.weight
    full_before = unsharded.detach().clone()
    bias_before = layer.bias.detach().clone()
    if native_tp:
        try:
            assert isinstance(unsharded, DTensor)
            assert torch.distributed.get_process_group_ranks(unsharded.device_mesh.get_group()) != (
                torch.distributed.get_process_group_ranks(mesh.get_group())
            )
            with ExitStack() as window:
                with pytest.warns(
                    RuntimeWarning, match="native-full.*weight.*caller.*publish.*reshard"
                ) as notices:
                    window.enter_context(enable_weight_access_and_writeback(layer, model))
                assert len(notices) == 1
                with torch.no_grad():
                    assert layer.weight is unsharded
                    layer.weight.add_(1)
            assert layer.weight is unsharded
            torch.testing.assert_close(layer.weight, full_before + 1, rtol=0, atol=0)
            with (
                pytest.raises(ValueError, match="reshard"),
                enable_weight_access_and_writeback(layer, model, rollback=True),
            ):
                pytest.fail("Rollback exposed a native-full TP registration")
        finally:
            # Native full views belong to their outer caller; reshard does not publish edits.
            with torch.no_grad():
                unsharded.copy_(full_before)
            layer.reshard()
        assert layer.weight is sharded
        torch.testing.assert_close(sharded.to_local(), shard_before, rtol=0, atol=0)
        inputs = torch.ones(2, 8)
        expected = torch.nn.functional.linear(
            inputs, full_before.to_local(), bias_before.to_local()
        )
        actual = layer(distribute_tensor(inputs, world_mesh["tp"], [Replicate()]))
        torch.testing.assert_close(actual.to_local(), expected)
        return
    try:
        assert unsharded is not sharded, "FSDP2 registered the same parameter when unsharded"
        if not asymmetric:
            # Discover the native registration once, then keep the warm ownership cache.
            with (
                mock.patch.object(
                    core_utils, "_fsdp2_param_groups", wraps=_fsdp2_param_groups
                ) as scans,
                mock.patch.object(
                    torch.distributed, "all_reduce", wraps=torch.distributed.all_reduce
                ) as collectives,
            ):
                for _ in range(2):
                    with enable_weight_access_and_writeback(layer, model):
                        assert layer.weight is unsharded
            assert scans.call_count == 1
            assert collectives.call_count == 6
        assert _fsdp2_owns(_fsdp2_owned_params(model, refresh=True), unsharded)
        if asymmetric and rank != 0:
            layer.reshard()
        registered = dict(layer.named_parameters())
        reached = False
        with (
            pytest.raises((ValueError, RuntimeError)),
            enable_weight_access_and_writeback(layer, model, rollback=True),
        ):
            reached = True
        assert not reached
        assert all(layer.get_parameter(name) is parameter for name, parameter in registered.items())
        torch.testing.assert_close(sharded.to_local(), shard_before, rtol=0, atol=0)
        if not asymmetric:
            with enable_weight_access_and_writeback(layer, model, writeback=False):
                assert layer.weight is unsharded
    finally:
        layer.reshard()
    assert layer.weight is sharded
    torch.testing.assert_close(sharded.to_local(), shard_before, rtol=0, atol=0)
    with enable_weight_access_and_writeback(layer, model, rollback=True), torch.no_grad():
        layer.weight.add_(1)
    layer.unshard()
    try:
        torch.testing.assert_close(layer.weight, full_before + 1, rtol=0, atol=0)
        inputs = torch.ones(2, 8)
        expected = torch.nn.functional.linear(inputs, full_before + 1, bias_before)
        torch.testing.assert_close(layer(inputs), expected)
    finally:
        layer.reshard()
    torch.testing.assert_close(sharded.to_local(), shard_before + 1, rtol=0, atol=0)
    with fsdp2_weight_access_and_writeback_context(model.dense, model):
        assert not isinstance(model.dense.weight, DTensor)
    # The set is kept for the unit: opening the window again costs the module's parameters only.
    with mock.patch.object(core_utils, "_fsdp2_param_groups", wraps=_fsdp2_param_groups) as scans:
        for _ in range(3):
            with fsdp2_weight_access_and_writeback_context(model.dense, model):
                pass
    assert scans.call_count == 0
    model.unshard()
    try:
        ignored = model.ignored.weight
        with enable_weight_access_and_writeback(model.ignored, model, rollback=True):
            assert model.ignored.weight is ignored
    finally:
        model.reshard()

    if asymmetric:
        return
    try:
        from accelerate import cpu_offload

        from modelopt.torch.quantization.plugins import accelerate as offload_plugin
        from modelopt.torch.quantization.plugins.accelerate import (
            weight_access_and_writeback_context as offload_weight_access,
        )
    except ImportError:
        return

    # Direct and dispatched windows expire overlapping receipts while preserving disjoint undo.
    alias = torch.nn.Module()
    alias.weight = torch.nn.Parameter(sharded.to_local().detach())
    weights = {"weight": alias.weight.detach()}
    cpu_offload(alias, execution_device="cpu", state_dict=weights)
    independent = torch.nn.Linear(4, 4, bias=False)
    independent_weights = {"weight": independent.weight.detach()}
    independent_before = independent.weight.detach().clone()
    cpu_offload(independent, execution_device="cpu", state_dict=independent_weights)
    with offload_weight_access(independent, rollback=True) as independent_undo, torch.no_grad():
        independent.weight.add_(1)
    if receipt_stage == "direct_entry":
        with offload_weight_access(alias, rollback=True) as undo:
            pass
    access = (
        enable_weight_access_and_writeback
        if receipt_stage == "dispatcher"
        else fsdp2_weight_access_and_writeback_context
    )
    before_publication = sharded.to_local().detach().clone()
    scope_failure = receipt_stage == "direct_scope_failure"
    scope_ready = False
    inspect_scope = offload_plugin._rollback_scope

    def post_publication_scope(owner, materialized):
        if scope_ready and owner is layer and owner.weight is sharded:
            raise RuntimeError("post-publication scope inspection failed")
        return inspect_scope(owner, materialized)

    with (
        mock.patch.object(offload_plugin, "_rollback_scope", post_publication_scope)
        if scope_failure
        else nullcontext(),
        access(layer, model),
        torch.no_grad(),
    ):
        if receipt_stage == "direct_entry":
            with pytest.raises(RuntimeError):
                undo()
        else:
            # The materialized view hides the persistent shard that this receipt can mutate.
            with offload_weight_access(alias, rollback=True) as undo:
                pass
        layer.weight.add_(1)
        scope_ready = scope_failure
    published = sharded.to_local().detach().clone()
    torch.testing.assert_close(published, before_publication + 1, rtol=0, atol=0)
    try:
        with pytest.raises(RuntimeError):
            undo()
    finally:
        assert layer.weight is sharded
        torch.testing.assert_close(sharded.to_local(), published, rtol=0, atol=0)
    torch.testing.assert_close(
        independent_weights["weight"], independent_before + 1, rtol=0, atol=0
    )
    if scope_failure:
        with pytest.raises(RuntimeError):
            independent_undo()
        torch.testing.assert_close(
            independent_weights["weight"], independent_before + 1, rtol=0, atol=0
        )
    else:
        independent_undo()
        torch.testing.assert_close(
            independent_weights["weight"], independent_before, rtol=0, atol=0
        )


@pytest.mark.parametrize(
    ("asymmetric", "receipt_stage"),
    [
        (False, "dispatcher"),
        (True, "dispatcher"),
        (False, "direct_entry"),
        (False, "direct_publication"),
        (False, "native_tp"),
        (False, "direct_scope_failure"),
    ],
)
def test_fsdp2_owns_its_parameters_in_every_shard_state(
    cpu_dist_workers, asymmetric, receipt_stage
):
    """FSDP2 registers another parameter object once a unit is unsharded (or resharded after the
    forward, under HSDP), and the weight-access window gathers by ownership: each counts as FSDP2's,
    or a weight it does own would be left sharded in the window. So do the parameters of a unit
    nested in it. The window keeps the set per unit rather than scanning it on every entry."""
    if asymmetric:
        # Import and initialize on the workers before applying the collective-hang bound.
        cpu_dist_workers.run(_fsdp2_owns_its_params_in_every_state_job, False)
    timeout, grace = cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace
    try:
        if asymmetric:
            cpu_dist_workers.job_timeout = 15
            cpu_dist_workers.failure_grace = 2
        cpu_dist_workers.run(_fsdp2_owns_its_params_in_every_state_job, asymmetric, receipt_stage)
    finally:
        cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace = timeout, grace


def _fsdp2_receipt_inspection_failure_job(rank, size, direct, inject, stage):
    # Accelerate is optional; the parent integration test skips when it is unavailable.
    from accelerate import cpu_offload

    from modelopt.torch.quantization.plugins import accelerate as offload_plugin

    mesh = init_device_mesh("cpu", (size,))
    layer = torch.nn.Linear(8, 8)
    root = layer
    if stage in ("ignored_entry", "ignored_exit"):
        root = torch.nn.Sequential(layer, torch.nn.Linear(8, 8))
        fully_shard(root, mesh=mesh, ignored_params=set(layer.parameters()))
    else:
        fully_shard(root, mesh=mesh)

    def local(value):
        return value.to_local() if isinstance(value, DTensor) else value

    parameters = dict(layer.named_parameters())
    before = {name: local(value).detach().clone() for name, value in parameters.items()}
    alias = torch.nn.Linear(8, 4, bias=False)
    alias.weight = torch.nn.Parameter(local(layer.weight).detach())
    cpu_offload(alias, execution_device="cpu")
    # A resident TP/FSDP shard may coexist with the independent offloaded owner.
    mixed = torch.nn.ModuleDict({"offloaded": alias, "resident": torch.nn.Module()})
    mixed["resident"].register_parameter("weight", layer.weight)
    with offload_plugin.weight_access_and_writeback_context(mixed, rollback=True) as undo:
        pass
    failure = RuntimeError("rank-local receipt inspection failure")
    inspect = offload_plugin._rollback_scope
    redistribute = DTensor.redistribute
    gathers = 0
    body_done = False

    def inspect_scope(owner, materialized):
        at_stage = (
            stage in ("preflight", "read_entry", "ignored_entry")
            or (stage in ("read_exit", "ignored_exit") and body_done)
            or (stage == "publication" and body_done and owner.weight is parameters["weight"])
        )
        if inject and rank == 0 and owner is layer and at_stage:
            raise failure
        return inspect(owner, materialized)

    def count_gathers(tensor, *args, **kwargs):
        nonlocal gathers
        gathers += 1
        return redistribute(tensor, *args, **kwargs)

    access = (
        fsdp2_weight_access_and_writeback_context if direct else enable_weight_access_and_writeback
    )
    error = None
    entered = False
    try:
        with (
            mock.patch.object(offload_plugin, "_rollback_scope", inspect_scope),
            mock.patch.object(DTensor, "redistribute", count_gathers),
            access(layer, root, writeback=not stage.startswith("read_")),
        ):
            entered = True
            if stage in ("read_exit", "publication", "ignored_exit"):
                with offload_plugin.weight_access_and_writeback_context(
                    alias, rollback=True
                ) as undo:
                    pass
            if stage == "publication":
                with torch.no_grad():
                    layer.weight.add_(1)
            body_done = True
    except RuntimeError as exc:
        error = exc
    if inject and stage == "preflight":
        assert error is not None and not entered
        assert gathers == 0, "Receipt inspection must fail every rank before gathering"
        if rank == 0:
            assert error is failure
    else:
        assert error is None and entered, repr(error)
    with pytest.raises(RuntimeError):
        undo()
    for name, parameter in parameters.items():
        assert layer.get_parameter(name) is parameter
        expected = before[name] + (stage == "publication" and name == "weight")
        torch.testing.assert_close(local(parameter), expected, rtol=0, atol=0)
    torch.distributed.barrier()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "stage",
    ["preflight", "read_entry", "read_exit", "publication", "ignored_entry", "ignored_exit"],
)
def test_fsdp2_receipt_inspection_failure_is_coordinated(cpu_dist_workers, direct, stage):
    pytest.importorskip("accelerate")
    cpu_dist_workers.run(_fsdp2_receipt_inspection_failure_job, direct, False, stage)
    timeout, grace = cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace
    try:
        cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace = 15, 2
        cpu_dist_workers.run(_fsdp2_receipt_inspection_failure_job, direct, True, stage)
    finally:
        cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace = timeout, grace


@mock.patch.object(torch.cuda, "is_available", return_value=False)
def _window_leaves_what_fsdp2_does_not_own_job(rank, size, failure_stage, _cuda_available):
    ep_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("ep",))
    dp_mesh = (
        init_device_mesh("cpu", (size, 1), mesh_dim_names=("replicate", "dp"))["dp"]
        if failure_stage
        in ("singleton_body", "singleton_default", "nested_same", "nested_disjoint")
        else init_device_mesh("cpu", (size,), mesh_dim_names=("dp",))
    )
    layer = torch.nn.Module()
    layer.dense = torch.nn.Linear(8, 8)
    if failure_stage == "plain_ignored":
        layer.dense.adapter = torch.nn.Parameter(torch.randn(8))
    if failure_stage.startswith("nested_"):
        layer.other = torch.nn.Linear(8, 8)
    layer.experts = torch.nn.Module()
    for name in ("gate_up_proj", "down_proj"):
        weight = distribute_tensor(torch.randn(2 * size, 8), ep_mesh, [Shard(0)])
        layer.experts.register_parameter(name, torch.nn.Parameter(weight))
    layer.requires_grad_(False)
    ignored = set(layer.experts.parameters())
    if failure_stage == "plain_ignored":
        ignored.add(layer.dense.adapter)
    fully_shard(layer, mesh=dp_mesh, ignored_params=ignored)
    if failure_stage == "plain_ignored":
        _fsdp2_owned_params(layer)
        adapter = layer.dense.adapter
        before = adapter.detach().clone()
        with mock.patch.object(
            core_utils, "_fsdp2_param_groups", wraps=_fsdp2_param_groups
        ) as scans:
            for _ in range(3):
                with fsdp2_weight_access_and_writeback_context(layer.dense, layer):
                    assert layer.dense.adapter is adapter
                    assert not isinstance(layer.dense.weight, DTensor)
        assert scans.call_count == 0
        torch.testing.assert_close(adapter, before, rtol=0, atol=0)
        return
    if failure_stage.startswith("nested_"):
        parameters = dict(layer.named_parameters())
        before = {name: value.to_local().detach().clone() for name, value in parameters.items()}
        same = failure_stage == "nested_same"
        try:
            with fsdp2_weight_access_and_writeback_context(layer.dense, layer), torch.no_grad():
                outer_weight = layer.dense.weight
                if same:
                    entered = False
                    with (
                        pytest.raises(ValueError, match="overlap"),
                        fsdp2_weight_access_and_writeback_context(
                            layer.dense, layer, rollback=True
                        ),
                    ):
                        entered = True
                        layer.dense.weight.add_(1)
                        raise RuntimeError("nested body modified the outer alias")
                    assert not entered
                else:
                    with fsdp2_weight_access_and_writeback_context(
                        layer.other, layer, rollback=True
                    ):
                        layer.other.weight.add_(2)
                assert layer.dense.weight is outer_weight
        finally:
            for name, parameter in parameters.items():
                assert layer.get_parameter(name) is parameter
                delta = 2 if not same and name == "other.weight" else 0
                torch.testing.assert_close(
                    parameter.to_local(), before[name] + delta, rtol=0, atol=0
                )
        with fsdp2_weight_access_and_writeback_context(layer.dense, layer, rollback=True):
            pass
        return
    if failure_stage == "ignored_only":
        originals = dict(layer.experts.named_parameters())
        before = {name: parameter.to_local().clone() for name, parameter in originals.items()}
        with (
            mock.patch.object(
                torch.distributed, "all_reduce", wraps=torch.distributed.all_reduce
            ) as collectives,
            enable_weight_access_and_writeback(layer.experts, layer),
        ):
            assert all(
                layer.experts.get_parameter(name) is value for name, value in originals.items()
            )
        assert collectives.call_count == 0
        for name, parameter in originals.items():
            assert layer.experts.get_parameter(name) is parameter
            torch.testing.assert_close(parameter.to_local(), before[name], rtol=0, atol=0)
        return
    dense_storage = layer.dense.weight.to_local().untyped_storage().data_ptr()
    group_type = type(fully_shard.state(layer)._fsdp_param_group)
    with (
        mock.patch.object(
            group_type,
            "is_unsharded",
            new_callable=mock.PropertyMock,
            side_effect=AttributeError("Opt-in native state is unavailable"),
        )
        if failure_stage == "singleton_default"
        else nullcontext(),
        fsdp2_weight_access_and_writeback_context(layer, layer),
    ):
        if failure_stage == "singleton_default":
            assert layer.dense.weight.untyped_storage().data_ptr() == dense_storage
        assert not layer.dense.weight.requires_grad
        assert not layer.dense.bias.requires_grad
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

    if failure_stage in ("none", "singleton_default"):
        return
    originals = dict(layer.named_parameters())
    before = {name: parameter.to_local().detach().clone() for name, parameter in originals.items()}
    if failure_stage in (
        "rank0_discovery",
        "rank0_first_copy_default",
        "rank0_final_copy_default",
        "rank0_final_copy",
        "rank0_restore",
        "rank0_last_prepare_default",
    ):
        failure = RuntimeError("rank-local FSDP discovery/publication failure")
        discover = core_utils._fsdp2_owned_params
        set_parameter = core_utils._set_parameter
        copy_tensor = torch.Tensor.copy_
        target_name = "dense.weight" if "first_copy" in failure_stage else "dense.bias"
        target = originals[target_name].to_local()
        target_pointer = target.untyped_storage().data_ptr()
        armed = True

        def discover_once(*args, **kwargs):
            nonlocal armed
            if failure_stage == "rank0_discovery" and rank == 0 and armed:
                armed = False
                raise failure
            return discover(*args, **kwargs)

        def copy_once(tensor, source, *args, **kwargs):
            nonlocal armed
            if (
                failure_stage
                not in ("rank0_discovery", "rank0_restore", "rank0_last_prepare_default")
                and rank == 0
                and armed
                and tensor.untyped_storage().data_ptr() == target_pointer
            ):
                armed = False
                raise failure
            return copy_tensor(tensor, source, *args, **kwargs)

        def restore_once(owner, name, value):
            nonlocal armed
            if (
                failure_stage in ("rank0_restore", "rank0_last_prepare_default")
                and rank == 0
                and armed
                and name == target_name
                and (value is originals[name]) == (failure_stage == "rank0_restore")
            ):
                armed = False
                raise failure
            return set_parameter(owner, name, value)

        error = None
        entered = False
        try:
            with (
                mock.patch.object(core_utils, "_fsdp2_owned_params", discover_once),
                mock.patch.object(torch.Tensor, "copy_", copy_once),
                mock.patch.object(core_utils, "_set_parameter", restore_once),
                enable_weight_access_and_writeback(
                    layer, layer, rollback=failure_stage == "rank0_final_copy"
                ),
                torch.no_grad(),
            ):
                entered = True
                if failure_stage == "rank0_last_prepare_default":
                    torch.distributed.barrier()
                layer.dense.weight.add_(1)
                layer.dense.bias.add_(2)
        except RuntimeError as exc:
            error = exc
        assert error is not None, "A rank-local failure must fail every participating rank"
        if failure_stage == "rank0_last_prepare_default":
            assert not entered, "Final preparation must fail every rank before the body"
        if rank == 0:
            assert error is failure
        if failure_stage == "rank0_restore" and rank == 0:
            assert layer.get_parameter(target_name) is not originals[target_name]
            # The caller repairs a reported incomplete restoration before opening another window.
            set_parameter(layer, target_name, originals[target_name])
        assert all(layer.get_parameter(name) is parameter for name, parameter in originals.items())
        for name, parameter in originals.items():
            delta = 0
            if failure_stage not in ("rank0_discovery", "rank0_last_prepare_default"):
                if name == "dense.weight":
                    delta = 0 if "first_copy" in failure_stage and rank == 0 else 1
                elif name == "dense.bias" and (
                    ("final_copy" in failure_stage and rank != 0)
                    or failure_stage == "rank0_restore"
                ):
                    delta = 2
            torch.testing.assert_close(parameter.to_local(), before[name] + delta, rtol=0, atol=0)
        # Do not let a retry accidentally match the peer's stranded entry collective.
        if failure_stage == "rank0_last_prepare_default":
            torch.distributed.barrier()
        if failure_stage != "rank0_discovery":
            with enable_weight_access_and_writeback(layer, layer):
                assert not isinstance(layer.dense.weight, DTensor)
        return
    failure = RuntimeError("FSDP weight access " + failure_stage)
    body_failure = RuntimeError("original context body failure")
    redistribute = DTensor.redistribute
    armed = True

    def fail_once(tensor, *args, **kwargs):
        nonlocal armed
        if failure_stage == "entry" and tensor is originals["dense.bias"] and armed:
            armed = False
            with torch.no_grad():
                layer.dense.weight.add_(7)
            raise failure
        result = redistribute(tensor, *args, **kwargs)
        if (
            failure_stage == "rank0_entry"
            and rank == 0
            and tensor is originals["dense.weight"]
            and armed
        ):
            armed = False
            raise failure
        if (
            failure_stage
            not in ("entry", "rank0_body", "rank0_body_default", "singleton_body", "rank0_entry")
            and tensor.placements[0].is_replicate()
            and armed
        ):
            # Both ranks finish this collective before the injected failure.
            armed = False
            raise failure
        return result

    error = None
    try:
        with (
            mock.patch.object(DTensor, "redistribute", fail_once),
            enable_weight_access_and_writeback(
                layer, layer, rollback=failure_stage != "rank0_body_default"
            ),
            torch.no_grad(),
        ):
            layer.dense.weight.add_(1)
            layer.dense.bias.add_(2)
            if failure_stage in ("body_writeback", "singleton_body") or (
                failure_stage in ("rank0_body", "rank0_body_default") and rank == 0
            ):
                raise body_failure
    except RuntimeError as exc:
        error = exc
    assert all(layer.get_parameter(name) is parameter for name, parameter in originals.items())
    if failure_stage in (
        "entry",
        "body_writeback",
        "singleton_body",
        "rank0_entry",
        "rank0_body",
        "rank0_body_default",
    ):
        for name, parameter in originals.items():
            torch.testing.assert_close(parameter.to_local(), before[name], rtol=0, atol=0)
    assert error is not None, "A peer's body failure must fail the context on every rank"
    if not (failure_stage in ("rank0_entry", "rank0_body", "rank0_body_default") and rank != 0):
        assert error is (
            body_failure
            if failure_stage
            in ("body_writeback", "rank0_body", "rank0_body_default", "singleton_body")
            else failure
        )
    if failure_stage == "writeback":
        # Transactional staging coordination stops later owners from publishing after failure.
        torch.testing.assert_close(
            originals["dense.weight"].to_local(), before["dense.weight"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            originals["dense.bias"].to_local(), before["dense.bias"], rtol=0, atol=0
        )
    with fsdp2_weight_access_and_writeback_context(layer, layer):
        assert not isinstance(layer.dense.weight, DTensor)


@pytest.mark.parametrize(
    "failure_stage",
    [
        "none",
        "ignored_only",
        "plain_ignored",
        "entry",
        "writeback",
        "body_writeback",
        "rank0_body",
        "rank0_body_default",
        "singleton_body",
        "singleton_default",
        "nested_same",
        "nested_disjoint",
        "rank0_entry",
        "rank0_discovery",
        "rank0_first_copy_default",
        "rank0_final_copy_default",
        "rank0_final_copy",
        "rank0_restore",
        "rank0_last_prepare_default",
    ],
)
def test_fsdp2_weight_access_leaves_parameters_fsdp2_does_not_own(cpu_dist_workers, failure_stage):
    """An expert-parallel weight is a DTensor sharded over the experts that fully_shard was told to
    ignore. The weight-access window gathers FSDP2's parameters only, so it stays sharded: gathering
    it would hand every rank every expert. Each such weight misses the window's kept set, which it
    scans again once, not once per weight."""
    bounded = failure_stage in (
        "rank0_entry",
        "rank0_body_default",
        "ignored_only",
        "rank0_discovery",
        "rank0_first_copy_default",
        "rank0_final_copy_default",
        "rank0_final_copy",
        "rank0_restore",
        "rank0_last_prepare_default",
    )
    if bounded:
        # Warm imports before bounding the injected collective-order failure.
        cpu_dist_workers.run(_window_leaves_what_fsdp2_does_not_own_job, "none")
    timeout, grace = cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace
    try:
        if bounded:
            cpu_dist_workers.job_timeout = 15
            cpu_dist_workers.failure_grace = 2
        cpu_dist_workers.run(_window_leaves_what_fsdp2_does_not_own_job, failure_stage)
    finally:
        cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace = timeout, grace


def _nested_fsdp_mesh_mismatch_job(rank, size, rollback, hsdp_prefix):
    if hsdp_prefix:
        child_mesh = init_device_mesh("cpu", (1, size), mesh_dim_names=("replicate", "dp"))
        parent_mesh = child_mesh["replicate"]
    else:
        child_mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("dp",))
        parent_mesh = init_device_mesh("cpu", (size, 1), mesh_dim_names=("replicate", "dp"))["dp"]
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    fully_shard(model[0], mesh=child_mesh)
    fully_shard(model, mesh=parent_mesh)
    # Model a cache built before FSDP introduced the child's current parameter objects.
    cached = _fsdp2_owned_params(model)
    for parameter in model[0].parameters():
        cached.pop(id(parameter), None)
    originals = dict(model.named_parameters())
    before = {name: value.to_local().clone() for name, value in originals.items()}
    with (
        mock.patch.object(
            DTensor, "redistribute", autospec=True, side_effect=DTensor.redistribute
        ) as redistributes,
        pytest.raises(ValueError, match="same sharding mesh"),
        enable_weight_access_and_writeback(model, model, rollback=rollback),
    ):
        pytest.fail(f"A writable window exposed nested weight shape {tuple(model[0].weight.shape)}")
    assert redistributes.call_count == 0
    for name, value in originals.items():
        assert model.get_parameter(name) is value
        torch.testing.assert_close(value.to_local(), before[name], rtol=0, atol=0)
    # Accessing each unit on its actual mesh remains supported.
    for owner in (model[0], model[1]):
        with enable_weight_access_and_writeback(owner, model, rollback=rollback):
            assert not isinstance(owner.weight, DTensor)


@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("hsdp_prefix", [False, True])
def test_writable_nested_fsdp_requires_same_sharding_mesh(cpu_dist_workers, rollback, hsdp_prefix):
    if hsdp_prefix:
        # Warm worker imports before bounding any mismatched-collective regression.
        cpu_dist_workers.run(_nested_fsdp_mesh_mismatch_job, rollback, False)
    timeout, grace = cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace
    try:
        if hsdp_prefix:
            cpu_dist_workers.job_timeout = 15
            cpu_dist_workers.failure_grace = 2
        cpu_dist_workers.run(_nested_fsdp_mesh_mismatch_job, rollback, hsdp_prefix)
    finally:
        cpu_dist_workers.job_timeout, cpu_dist_workers.failure_grace = timeout, grace
