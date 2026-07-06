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

"""Tests for the QuantModule / QuantInputBase / QuantLinearConvBase class machinery.

This file targets ``modelopt/torch/quantization/nn/modules/quant_module.py`` directly:
conversion plumbing, quantizer attachment in ``_setup``, forward wrapping, the
``quantize_weight`` context, ``fold_weight``, ``modelopt_post_restore``, parallel state
handling and the class-level ``QuantModuleRegistry`` interplay.

Intentionally NOT covered here (already exercised elsewhere):
- ``_forward_pre_dm`` recursion guard for MRO-bound forwards -> test_forward_patching.py
- ``mtq.register``/``replace_quant_module`` model-level flows -> test_module_registry.py,
  test_quantize_replace.py
- QuantLinear (legacy mixin) numerics against tensor_quant -> test_quant_linear.py
"""

import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.nn.modules.quant_module import (
    QuantInputBase,
    QuantLinearConvBase,
    QuantModule,
)
from modelopt.torch.quantization.utils import export_torch_mode
from modelopt.torch.utils.distributed import ParallelState

IN_FEATURES = 8
OUT_FEATURES = 4


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(1234)


class ToyLinear(nn.Module):
    """A bare linear module (not an nn.Linear subclass) to exercise base-class conversion."""

    def __init__(self, in_features=IN_FEATURES, out_features=OUT_FEATURES):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class StatelessAdd(nn.Module):
    """A module without any parameters or buffers."""

    def forward(self, x):
        return x + 1.0


class TupleOut(nn.Module):
    """A module whose forward returns a tuple."""

    def forward(self, x):
        return x + 1.0, "aux"


@pytest.fixture
def toy_linear_registered():
    QuantModuleRegistry.register({ToyLinear: "test.ToyLinear"})(QuantLinearConvBase)
    try:
        yield
    finally:
        QuantModuleRegistry.unregister(ToyLinear)


@pytest.fixture
def stateless_add_registered():
    QuantModuleRegistry.register({StatelessAdd: "test.StatelessAdd"})(QuantInputBase)
    try:
        yield
    finally:
        QuantModuleRegistry.unregister(StatelessAdd)


@pytest.fixture
def tuple_out_registered():
    QuantModuleRegistry.register({TupleOut: "test.TupleOut"})(QuantInputBase)
    try:
        yield
    finally:
        QuantModuleRegistry.unregister(TupleOut)


def _convert_toy_linear() -> QuantLinearConvBase:
    return QuantModuleRegistry.convert(ToyLinear())


class TestRegistryClassConversion:
    """QuantModuleRegistry interplay at the class level."""

    def test_convert_linear_returns_same_object_with_quant_class(self):
        lin = nn.Linear(IN_FEATURES, OUT_FEATURES)
        q = QuantModuleRegistry.convert(lin)

        assert q is lin  # in-place conversion
        assert isinstance(q, nn.Linear)
        assert isinstance(q, QuantLinearConvBase)
        assert isinstance(q, QuantModule)
        assert isinstance(q, DynamicModule)
        assert type(q).__name__ == "QuantLinear"

    def test_registry_lookup_is_cached_and_string_keyed(self):
        dm_cls = QuantModuleRegistry.get(nn.Linear)
        assert dm_cls is not None
        assert QuantModuleRegistry.get(nn.Linear) is dm_cls  # cached identity
        assert QuantModuleRegistry[nn.Linear] is dm_cls
        assert QuantModuleRegistry.get("nn.Linear") is dm_cls
        assert "nn.Linear" in QuantModuleRegistry
        assert nn.Linear in QuantModuleRegistry

    def test_subclass_with_shared_forward_is_convertible(self):
        class PlainLinearSub(nn.Linear):
            pass

        # A subclass sharing nn.Linear.forward is registered through the nn.Linear entry
        # (this caches a dynamic class for PlainLinearSub inside the registry; there is no
        # public API to drop a single generated class without unregistering nn.Linear).
        assert PlainLinearSub in QuantModuleRegistry
        dm_cls = QuantModuleRegistry.get(PlainLinearSub)
        assert dm_cls is not QuantModuleRegistry.get(nn.Linear)
        assert dm_cls.__name__ == "QuantPlainLinearSub"

        q = QuantModuleRegistry.convert(PlainLinearSub(IN_FEATURES, OUT_FEATURES))
        assert isinstance(q, PlainLinearSub)
        assert isinstance(q.input_quantizer, TensorQuantizer)

    def test_subclass_with_custom_forward_is_not_registered(self):
        class CustomForwardLinear(nn.Linear):
            def forward(self, x):
                return super().forward(x) * 2

        assert CustomForwardLinear not in QuantModuleRegistry
        assert QuantModuleRegistry.get(CustomForwardLinear) is None
        with pytest.raises(KeyError):
            QuantModuleRegistry.convert(CustomForwardLinear(IN_FEATURES, OUT_FEATURES))

    def test_convert_bare_registered_class(self, stateless_add_registered):
        assert StatelessAdd in QuantModuleRegistry
        assert "test.StatelessAdd" in QuantModuleRegistry

        q = QuantModuleRegistry.convert(StatelessAdd())
        assert type(q).__name__ == "QuantStatelessAdd"
        assert isinstance(q, StatelessAdd)
        assert isinstance(q, QuantInputBase)
        assert isinstance(q.input_quantizer, TensorQuantizer)
        assert isinstance(q.output_quantizer, TensorQuantizer)

    def test_unregister_removes_class(self):
        QuantModuleRegistry.register({StatelessAdd: "test.StatelessAdd"})(QuantInputBase)
        assert StatelessAdd in QuantModuleRegistry
        QuantModuleRegistry.unregister(StatelessAdd)
        assert StatelessAdd not in QuantModuleRegistry
        assert QuantModuleRegistry.get(StatelessAdd) is None


class TestQuantizerDefaultWiring:
    """The _setup contracts: default quantizer attachment on input/weight/output."""

    def test_quantizers_are_registered_submodules(self, toy_linear_registered):
        q = _convert_toy_linear()
        children = dict(q.named_children())
        assert set(children) == {"input_quantizer", "output_quantizer", "weight_quantizer"}
        assert all(isinstance(child, TensorQuantizer) for child in children.values())

    def test_default_descriptor_values(self, toy_linear_registered):
        q = _convert_toy_linear()
        # Defaults come from QUANT_DESC_8BIT_PER_TENSOR on the base classes.
        assert q.input_quantizer.num_bits == 8
        assert q.input_quantizer.axis is None
        assert q.input_quantizer.is_enabled
        assert q.input_quantizer.fake_quant

        assert q.weight_quantizer.num_bits == 8
        assert q.weight_quantizer.axis is None
        assert q.weight_quantizer.is_enabled

        # output quantizer is attached but disabled by default
        assert not q.output_quantizer.is_enabled
        assert q.output_quantizer.num_bits == 8

        # weight quantization is off outside the quantize_weight context
        assert q._enable_weight_quantization is False

    def test_subclass_default_quant_desc_is_honored(self):
        class FourBitToyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(OUT_FEATURES, IN_FEATURES))

            def forward(self, x):
                return F.linear(x, self.weight)

        class FourBitQuantLinear(QuantLinearConvBase):
            default_quant_desc_input = QuantizerAttributeConfig(num_bits=4)
            default_quant_desc_weight = QuantizerAttributeConfig(num_bits=4, axis=0)

        QuantModuleRegistry.register({FourBitToyLinear: "test.FourBitToyLinear"})(
            FourBitQuantLinear
        )
        try:
            q = QuantModuleRegistry.convert(FourBitToyLinear())
            assert q.input_quantizer.num_bits == 4
            assert q.weight_quantizer.num_bits == 4
            assert q.weight_quantizer.axis == 0
        finally:
            QuantModuleRegistry.unregister(FourBitToyLinear)

    def test_set_quantizer_attributes_partial_propagates(self, toy_linear_registered):
        q = _convert_toy_linear()
        mtq.set_quantizer_attributes_partial(q, "*input_quantizer", {"num_bits": 4})
        assert q.input_quantizer.num_bits == 4
        assert q.input_quantizer.is_enabled  # merge semantics: enable untouched
        assert q.weight_quantizer.num_bits == 8  # non-matching quantizers untouched
        assert q.output_quantizer.num_bits == 8


class TestForwardWrapping:
    """QuantInputBase/QuantLinearConvBase forward behavior."""

    def test_forward_matches_manual_quantizer_reference(self, toy_linear_registered):
        q = _convert_toy_linear()
        x = torch.randn(5, IN_FEATURES)

        out = q(x)
        expected = F.linear(q.input_quantizer(x), q.weight_quantizer(q.weight), q.bias)
        assert torch.equal(out, expected)
        # quantization is actually lossy on random data
        assert not torch.equal(out, F.linear(x, q.weight, q.bias))

    def test_disabled_quantizers_are_exact_passthrough(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw_w = q.weight.detach().clone()
        raw_b = q.bias.detach().clone()
        q.input_quantizer.disable()
        q.weight_quantizer.disable()

        x = torch.randn(5, IN_FEATURES)
        assert torch.equal(q(x), F.linear(x, raw_w, raw_b))

    def test_output_quantizer_disabled_by_default(self, stateless_add_registered):
        q = QuantModuleRegistry.convert(StatelessAdd())
        q.input_quantizer.disable()
        x = torch.randn(3, 7)
        assert torch.equal(q(x), x + 1.0)

    def test_output_quantizer_applied_when_enabled(self, stateless_add_registered):
        q = QuantModuleRegistry.convert(StatelessAdd())
        q.input_quantizer.disable()
        q.output_quantizer.enable()
        x = torch.randn(3, 7)

        out = q(x)
        expected = q.output_quantizer(x + 1.0)
        assert torch.equal(out, expected)
        assert not torch.equal(out, x + 1.0)

    def test_tuple_output_quantizes_only_first_element(self, tuple_out_registered):
        q = QuantModuleRegistry.convert(TupleOut())
        q.input_quantizer.disable()
        q.output_quantizer.enable()
        x = torch.randn(2, 6)

        out = q(x)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert torch.equal(out[0], q.output_quantizer(x + 1.0))
        assert out[1] == "aux"  # non-tensor tail is forwarded untouched

    def test_tuple_output_passthrough_when_output_quantizer_disabled(self, tuple_out_registered):
        q = QuantModuleRegistry.convert(TupleOut())
        q.input_quantizer.disable()
        x = torch.randn(2, 6)
        out = q(x)
        assert torch.equal(out[0], x + 1.0)
        assert out[1] == "aux"

    def test_monkey_patched_forward_is_kept_and_receives_quantized_input(self):
        lin = nn.Linear(IN_FEATURES, OUT_FEATURES)
        seen = {}
        orig_forward = lin.forward

        def patched_forward(x):
            seen["input"] = x
            return orig_forward(x)

        lin.forward = patched_forward
        with pytest.warns(UserWarning, match="monkey patched forward"):
            QuantModuleRegistry.convert(lin)

        # the pre-conversion forward is stashed for the wrapped forward to dispatch to
        assert lin._forward_pre_dm is patched_forward

        x = torch.randn(3, IN_FEATURES)
        out = lin(x)

        # the patch runs *inside* the quant wrapper: it sees the quantized input ...
        expected_in = lin.input_quantizer(x)
        assert torch.equal(seen["input"], expected_in)
        assert not torch.equal(seen["input"], x)

        # ... and its weight accesses resolve to the quantized weight
        with lin.quantize_weight():
            expected_w = lin.weight.detach().clone()
        assert torch.equal(out, F.linear(expected_in, expected_w, lin.bias))

        # final export restores the original monkey-patched forward on the instance
        exported = lin.export()
        assert type(exported) is nn.Linear
        assert exported.forward is patched_forward
        assert torch.equal(exported(x), F.linear(x, exported.weight, exported.bias))


class TestQuantizeWeightMachinery:
    """quantize_weight context, _get_quantized_weight and the dynamic weight attribute."""

    def test_weight_is_raw_outside_context(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw = q.weight
        assert isinstance(raw, nn.Parameter)
        # dynamic attribute resolves to the pytorch-managed parameter storage
        assert raw.data_ptr() == q.get_parameter("weight").data_ptr()

    def test_quantize_weight_context_toggles_flag_and_value(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw = q.weight.detach().clone()

        assert q._enable_weight_quantization is False
        with q.quantize_weight():
            assert q._enable_weight_quantization is True
            inside = q.weight
            assert torch.equal(inside, q.weight_quantizer(raw))
            assert not torch.equal(inside, raw)
        assert q._enable_weight_quantization is False
        assert torch.equal(q.weight, raw)

    def test_quantize_weight_flag_reset_on_exception(self, toy_linear_registered):
        q = _convert_toy_linear()
        with pytest.raises(RuntimeError, match="boom"), q.quantize_weight():
            assert q._enable_weight_quantization is True
            raise RuntimeError("boom")
        assert q._enable_weight_quantization is False

    def test_get_quantized_weight_identity_when_disabled(self, toy_linear_registered):
        q = _convert_toy_linear()
        w = torch.randn(OUT_FEATURES, IN_FEATURES)
        # outside the context (and outside export mode) the weight is returned as-is
        assert QuantLinearConvBase._get_quantized_weight(q, w) is w

    def test_torch_export_mode_quantizes_weight_without_context(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw = q.weight.detach().clone()
        q.weight_quantizer.amax = raw.abs().amax()
        q.input_quantizer.amax = torch.tensor(2.0)
        x = torch.randn(4, IN_FEATURES)

        with export_torch_mode():
            # plain attribute access resolves to the quantized weight in export mode
            expected_w = q.weight_quantizer(raw)
            assert torch.equal(q.weight, expected_w)
            # the flag is never toggled: export path bypasses the context manager
            out = q(x)
            assert q._enable_weight_quantization is False
            assert torch.equal(out, F.linear(q.input_quantizer(x), expected_w, q.bias))

        # back to raw once the export context exits
        assert torch.equal(q.weight, raw)


class TestFoldWeight:
    """fold_weight contracts (CPU-testable fake-quant paths).

    NOTE: fold_weight derives the weight attribute name as ``name[:-10]`` which strips the
    ``_quantizer`` suffix, not ``_weight_quantizer`` as the inline comment claims. For custom
    per-weight quantizers such as ``gate_up_proj_weight_quantizer`` (weight attr
    ``gate_up_proj``, cf. quantizer_attr_names) this derives ``gate_up_proj_weight`` and
    would trip the hasattr assertion. Only the standard ``weight``/``weight_quantizer``
    layout is exercised here.
    """

    def test_fold_weight_replaces_weight_and_strips_state(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw = q.weight.detach().clone()
        q.weight_quantizer.amax = raw.abs().amax()
        q.weight_quantizer.pre_quant_scale = torch.ones(IN_FEATURES)
        expected = q.weight_quantizer(raw.float()).to(raw.dtype)

        q.fold_weight()

        assert torch.equal(q.weight.detach(), expected)
        assert not q.weight_quantizer.is_enabled
        assert not hasattr(q.weight_quantizer, "_amax")
        assert not hasattr(q.weight_quantizer, "_pre_quant_scale")
        assert q.weight_quantizer.amax is None

    def test_forward_after_fold_matches_prefold_quantized_weight(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw = q.weight.detach().clone()
        q.weight_quantizer.amax = raw.abs().amax()
        expected_w = q.weight_quantizer(raw.float()).to(raw.dtype)

        q.fold_weight()
        x = torch.randn(5, IN_FEATURES)

        # weight quantizer is now a disabled no-op: forward uses the folded weight directly
        with q.quantize_weight():
            assert torch.equal(q.weight, expected_w)
        assert torch.equal(q(x), F.linear(q.input_quantizer(x), expected_w, q.bias))

    def test_fold_weight_keep_attrs_preserves_state(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw = q.weight.detach().clone()
        amax = raw.abs().amax()
        q.weight_quantizer.amax = amax
        q.weight_quantizer.pre_quant_scale = torch.ones(IN_FEATURES)
        expected = q.weight_quantizer(raw.float()).to(raw.dtype)

        q.fold_weight(keep_attrs=True)

        assert torch.equal(q.weight.detach(), expected)
        assert not q.weight_quantizer.is_enabled  # still disabled to avoid double quantization
        assert hasattr(q.weight_quantizer, "_amax")
        assert torch.equal(q.weight_quantizer._amax, amax)
        assert hasattr(q.weight_quantizer, "_pre_quant_scale")

    def test_fold_weight_skips_non_fake_quant(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw = q.weight.detach().clone()
        q.weight_quantizer.amax = raw.abs().amax()
        q.weight_quantizer.set_from_attribute_config({"fake_quant": False})
        assert not q.weight_quantizer.fake_quant

        q.fold_weight()

        assert torch.equal(q.weight.detach(), raw)  # untouched
        assert q.weight_quantizer.is_enabled  # not disabled either
        assert hasattr(q.weight_quantizer, "_amax")


class TestIterWeightsForCalibration:
    def test_yields_weight_and_quantizer_pair(self, toy_linear_registered):
        q = _convert_toy_linear()
        pairs = list(q.iter_weights_for_calibration())
        assert len(pairs) == 1
        weight, quantizer = pairs[0]
        assert weight.data_ptr() == q.get_parameter("weight").data_ptr()
        assert quantizer is q.weight_quantizer

    def test_yields_nothing_without_weights(self, stateless_add_registered):
        q = QuantModuleRegistry.convert(StatelessAdd())
        assert list(q.iter_weights_for_calibration()) == []


class TestModeloptPostRestore:
    def test_moves_quantizer_states_to_non_quantizer_device(
        self, toy_linear_registered, monkeypatch
    ):
        q = _convert_toy_linear()
        amax = torch.tensor(0.5)
        q.input_quantizer.amax = amax

        recorded = []
        original_to = TensorQuantizer.to

        def recording_to(self, *args, **kwargs):
            recorded.append(args)
            return original_to(self, *args, **kwargs)

        monkeypatch.setattr(TensorQuantizer, "to", recording_to)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # must not warn: weight anchors device
            q.modelopt_post_restore()

        # all three attached quantizers moved to the weight's device
        assert len(recorded) == 3
        assert all(args == (torch.device("cpu"),) for args in recorded)
        # quantizer state survives the post-restore intact
        assert torch.equal(q.input_quantizer.amax, amax)

    def test_warns_without_non_quantizer_state(self, stateless_add_registered):
        q = QuantModuleRegistry.convert(StatelessAdd())
        # quantizer-owned state must NOT be used as a device anchor
        q.input_quantizer.amax = torch.tensor(1.0)
        assert len(q.state_dict()) == 1  # only input_quantizer._amax
        with pytest.warns(UserWarning, match="Could not identify the device"):
            q.modelopt_post_restore()


class TestParallelState:
    def test_convert_initializes_default_parallel_state(self):
        with warnings.catch_warnings():
            # no "distributed initialized but no parallel_state" warning on single process
            warnings.simplefilter("error", UserWarning)
            q = QuantModuleRegistry.convert(nn.Linear(IN_FEATURES, OUT_FEATURES))
        assert isinstance(q.parallel_state, ParallelState)
        assert not q.parallel_state.data_parallel_group.is_initialized()
        assert not q.parallel_state.tensor_parallel_group.is_initialized()

    def test_parallel_state_setter_roundtrip(self):
        q = QuantModuleRegistry.convert(nn.Linear(IN_FEATURES, OUT_FEATURES))
        new_state = ParallelState(data_parallel_group=-1)
        q.parallel_state = new_state
        assert q.parallel_state is new_state

    def test_parallel_state_setter_rejects_other_types(self):
        q = QuantModuleRegistry.convert(nn.Linear(IN_FEATURES, OUT_FEATURES))
        with pytest.raises(AssertionError, match="must be a ParallelState"):
            q.parallel_state = "not a parallel state"


class TestExport:
    """Export must undo everything _setup wired up."""

    def test_export_restores_original_class_and_weight(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw_w = q.weight.detach().clone()
        raw_b = q.bias.detach().clone()

        exported = q.export()

        assert exported is q  # in-place
        assert type(exported) is ToyLinear
        assert not isinstance(exported, DynamicModule)
        # temporary attributes (quantizers, flag) are gone
        assert not hasattr(exported, "input_quantizer")
        assert not hasattr(exported, "output_quantizer")
        assert not hasattr(exported, "weight_quantizer")
        assert not hasattr(exported, "_enable_weight_quantization")
        assert len(list(exported.children())) == 0
        # dynamic weight dissolved back into a plain parameter with raw values
        assert isinstance(exported.weight, nn.Parameter)
        assert torch.equal(exported.weight.detach(), raw_w)

        x = torch.randn(5, IN_FEATURES)
        assert torch.equal(exported(x), F.linear(x, raw_w, raw_b))

    def test_export_after_fold_keeps_folded_weight(self, toy_linear_registered):
        q = _convert_toy_linear()
        raw = q.weight.detach().clone()
        q.weight_quantizer.amax = raw.abs().amax()
        folded = q.weight_quantizer(raw.float()).to(raw.dtype)

        q.fold_weight()
        exported = q.export()

        assert type(exported) is ToyLinear
        assert torch.equal(exported.weight.detach(), folded)
        x = torch.randn(5, IN_FEATURES)
        assert torch.equal(exported(x), F.linear(x, folded, exported.bias))
