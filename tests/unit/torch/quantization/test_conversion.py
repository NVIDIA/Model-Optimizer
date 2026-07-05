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

"""Unit tests for modelopt.torch.quantization.conversion.

Covers the mode conversion/restore entrypoints, module replacement bookkeeping,
quantizer-config application (full/partial/by-cfg), quantizer state extraction and
restore, the temporary-config context manager, and the register/unregister API.
"""

import contextlib
import copy

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.models import SimpleConvLinear, SimpleLinear

from modelopt.torch.opt.conversion import ApplyModeError, ModeloptStateManager
from modelopt.torch.quantization.config import QuantizeConfig, QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import (
    _normalize_fused_experts_quantizer_name,
    convert_to_quantized_model,
    maybe_promote_nvfp4_static_quantizer,
    quantizer_state,
    register,
    replace_quant_module,
    restore_quantized_model,
    restore_quantizer_state,
    set_quantizer_attribute,
    set_quantizer_attributes_full,
    set_quantizer_attributes_partial,
    set_quantizer_by_cfg,
    set_quantizer_by_cfg_context,
    unregister,
    update_quantize_metadata,
)
from modelopt.torch.quantization.nn import (
    NVFP4StaticQuantizer,
    QuantModuleRegistry,
    SequentialQuantizer,
    TensorQuantizer,
)
from modelopt.torch.quantization.utils import is_quantized

# Names of the TensorQuantizer modules inserted into SimpleLinear by replace_quant_module.
SIMPLE_LINEAR_QUANTIZER_NAMES = {
    f"net.{i}.{q}"
    for i in (0, 2, 4)
    for q in ("input_quantizer", "weight_quantizer", "output_quantizer")
}


def _replaced(model_cls=SimpleLinear):
    """Return a fresh model with quantized modules inserted (no calibration)."""
    model = model_cls()
    replace_quant_module(model)
    return model


def _quantizers(model, suffix=""):
    """Return {name: module} for all TensorQuantizers whose name ends with ``suffix``.

    CAUTION: SequentialQuantizer subclasses nn.Sequential, NOT TensorQuantizer, and its
    children are named ``<suffix>.0/.1`` — so this helper finds neither the container nor
    its sub-quantizers. Use :func:`_seq_quantizers` for SequentialQuantizer containers,
    and always assert the returned dict is non-empty before looping over it.
    """
    return {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, TensorQuantizer) and n.endswith(suffix)
    }


def _seq_quantizers(model, suffix="weight_quantizer"):
    """Return {name: module} for all SequentialQuantizer containers ending with ``suffix``."""
    return {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, SequentialQuantizer) and n.endswith(suffix)
    }


# SimpleLinear has three nn.Linear layers -> three weight quantizers after replacement.
SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS = 3


def _weight_quantizers(model):
    """TensorQuantizer weight quantizers of a SimpleLinear-based model, guaranteed non-empty.

    Guards against vacuously-passing loops: if the weight quantizers were (unexpectedly)
    upgraded to SequentialQuantizer containers, this fails instead of returning {}.
    """
    found = _quantizers(model, "weight_quantizer")
    assert len(found) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
    return found


class _RootQuantHolder(nn.Module):
    """Module holding a quantizer directly at the root (name has no dots)."""

    def __init__(self):
        super().__init__()
        self.q = TensorQuantizer()


class _FusedExperts(nn.Module):
    """Mimics fused-experts modules that keep per-expert quantizers in a ModuleList."""

    def __init__(self, num_experts=2):
        super().__init__()
        self.weight_quantizers = nn.ModuleList(TensorQuantizer() for _ in range(num_experts))
        self.input_quantizers = nn.ModuleList([TensorQuantizer()])
        self.output_quantizer = TensorQuantizer()


class _PlainAct(nn.SiLU):
    """Activation subclass so register/unregister tests don't touch global nn classes."""


class _QuantPlainAct(_PlainAct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        self.input_quantizer = TensorQuantizer(QuantizerAttributeConfig())

    def forward(self, x):
        return super().forward(self.input_quantizer(x))


class TestReplaceQuantModule:
    # NOTE: quantizer insertion on registered Linear/Conv children is already covered by
    # test_quantize_replace.py::test_quantize_replace; not duplicated here.

    def test_unregistered_modules_left_untouched(self):
        model = _replaced(SimpleConvLinear)
        # ReLU and Flatten have no registered quantized counterpart: their exact
        # (non-dynamic) classes must survive the replacement pass.
        assert any(type(m) is nn.ReLU for m in model.modules())
        assert any(type(m) is nn.Flatten for m in model.modules())

    def test_parameters_preserved_bitwise(self):
        model = SimpleConvLinear()
        params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        replace_quant_module(model)
        params_after = dict(model.named_parameters())
        assert params_before.keys() == params_after.keys()
        for name, before in params_before.items():
            assert torch.equal(before, params_after[name]), name

    def test_default_quantizer_states(self):
        """Input/weight quantizers start enabled; output quantizers start disabled."""
        model = _replaced()
        for name, module in _quantizers(model).items():
            if name.endswith("output_quantizer"):
                assert not module.is_enabled, name
            else:
                assert module.is_enabled, name

    def test_raises_on_already_quantized_model(self):
        model = _replaced()
        with pytest.raises(AssertionError, match="Model must not be quantized!"):
            replace_quant_module(model)

    def test_root_module_is_converted_in_place(self):
        """A registered root module (not just children) is class-patched in place."""
        model = nn.Linear(8, 8)
        replace_quant_module(model)
        assert isinstance(model.weight_quantizer, TensorQuantizer)
        assert isinstance(model.input_quantizer, TensorQuantizer)


class TestRegisterUnregister:
    def test_register_replace_unregister_cycle(self):
        # Cross-reference: test_module_registry.py::test_register_custom_module covers the
        # same API but registers/unregisters the global nn.ReLU class; this variant uses a
        # private activation subclass so the global registry is never left patched.
        model = nn.Sequential(nn.Linear(4, 4), _PlainAct())
        try:
            register(original_cls=_PlainAct, quantized_cls=_QuantPlainAct)
            replace_quant_module(model)
            assert isinstance(model[1], _QuantPlainAct)
            assert isinstance(model[1].input_quantizer, TensorQuantizer)
            # Pin the routing: with a static amax of 1.0 the activation input is
            # fake-quant clamped to [-1, 1]; silu is monotonic increasing, so a
            # bypassed quantizer would produce silu(100) ~= 100 >> silu(1).
            model[1].input_quantizer.amax = torch.tensor(1.0)
            out = model[1](torch.full((2, 4), 100.0))
            assert torch.all(out <= torch.nn.functional.silu(torch.tensor(1.0)) + 1e-6)
        finally:
            unregister(_PlainAct)

        assert QuantModuleRegistry.get(_PlainAct) is None
        # After unregistering, a fresh model is no longer converted.
        model2 = nn.Sequential(nn.Linear(4, 4), _PlainAct())
        replace_quant_module(model2)
        assert type(model2[1]) is _PlainAct
        assert isinstance(model2[0].weight_quantizer, TensorQuantizer)  # Linear still is

    def test_register_requires_setup_method(self):
        class _NoSetup(nn.Module):
            pass

        with pytest.raises(AssertionError, match="_setup"):
            register(original_cls=_PlainAct, quantized_cls=_NoSetup)
        # The failed registration must not leave a partial entry behind.
        assert QuantModuleRegistry.get(_PlainAct) is None

    def test_unregister_unknown_class_raises(self):
        class _NeverRegistered(nn.Module):
            pass

        with pytest.raises(KeyError, match="not registered"):
            unregister(_NeverRegistered)


class TestSetQuantizerByCfg:
    # The same logical config expressed in the three accepted formats.
    _DENY_ALL_THEN_WEIGHT4 = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
    ]
    _LEGACY_LIST = [
        {"*": {"enable": False}},
        {"*weight_quantizer": {"num_bits": 4, "axis": 0}},
    ]
    _LEGACY_FLAT_DICT = {
        "*": {"enable": False},
        "*weight_quantizer": {"num_bits": 4, "axis": 0},
    }

    @pytest.mark.parametrize(
        ("quant_cfg", "is_legacy"),
        [
            (_DENY_ALL_THEN_WEIGHT4, False),
            (_LEGACY_LIST, True),
            (_LEGACY_FLAT_DICT, True),
        ],
        ids=["new_format", "legacy_list", "legacy_flat_dict"],
    )
    def test_deny_all_then_enable_pattern_all_formats(self, quant_cfg, is_legacy):
        """The canonical deny-all + selective-enable pattern works in every cfg format.

        Legacy formats still apply correctly but are pinned to emit a DeprecationWarning.
        """
        model = _replaced()
        warn_ctx = pytest.warns(DeprecationWarning) if is_legacy else contextlib.nullcontext()
        with warn_ctx:
            set_quantizer_by_cfg(model, quant_cfg)
        for name, module in _quantizers(model).items():
            if name.endswith("weight_quantizer"):
                assert module.is_enabled, name  # cfg without enable implies enable=True
                assert module.num_bits == 4
                assert module.axis == 0
            else:
                assert not module.is_enabled, name

    def test_enable_only_entries_preserve_attributes(self):
        """enable-only entries toggle state without resetting configured attributes."""
        model = _replaced()
        set_quantizer_by_cfg(
            model, [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}}]
        )
        set_quantizer_by_cfg(model, [{"quantizer_name": "*weight_quantizer", "enable": False}])
        for module in _weight_quantizers(model).values():
            assert not module.is_enabled
            assert module.num_bits == 4
            assert module.axis == 0
        set_quantizer_by_cfg(model, [{"quantizer_name": "*weight_quantizer", "enable": True}])
        for module in _weight_quantizers(model).values():
            assert module.is_enabled
            assert module.num_bits == 4
            assert module.axis == 0

    # NOTE: full-replacement atomicity and last-entry-wins ordering are already covered by
    # test_quantize_cpu.py::test_atomicity_unset_fields_revert_to_defaults and
    # test_quantize_cpu.py::test_ordering_later_entry_overrides_earlier; not duplicated here.

    def test_parent_class_scopes_matching(self):
        """A parent_class entry only touches quantizers whose parent is that class."""
        model = _replaced(SimpleConvLinear)
        set_quantizer_by_cfg(
            model,
            [
                {
                    "parent_class": "nn.Linear",
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": 4},
                }
            ],
        )
        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert module.weight_quantizer.num_bits == 4
            elif isinstance(module, nn.Conv2d):
                assert module.weight_quantizer.num_bits == 8  # untouched default

    def test_unknown_parent_class_raises(self):
        model = _replaced()
        with pytest.raises(ValueError, match="not found in QuantModuleRegistry"):
            set_quantizer_by_cfg(
                model,
                [{"parent_class": "nn.NoSuchModule", "quantizer_name": "*", "enable": False}],
            )

    def test_list_cfg_creates_sequential_quantizer_in_order(self):
        model = _replaced()
        set_quantizer_by_cfg(
            model,
            [
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": [{"num_bits": 4}, {"num_bits": 8, "axis": 0}],
                }
            ],
        )
        # The containers are invisible to _quantizers() (SequentialQuantizer subclasses
        # nn.Sequential, not TensorQuantizer), so collect them explicitly and prove the
        # upgrade actually happened before asserting on the contents.
        containers = _seq_quantizers(model)
        assert len(containers) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
        for module in containers.values():
            assert len(module) == 2
            assert module[0].num_bits == 4
            assert module[1].num_bits == 8
            assert module[1].axis == 0

    def test_single_cfg_downgrades_sequential_quantizer(self):
        model = _replaced()
        set_quantizer_by_cfg(
            model,
            [
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": [{"num_bits": 4}, {"num_bits": 8}],
                }
            ],
        )
        assert len(_seq_quantizers(model)) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS  # upgraded
        set_quantizer_by_cfg(
            model, [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 6}}]
        )
        # Every container must have been downgraded back to a plain TensorQuantizer.
        assert not any(isinstance(m, SequentialQuantizer) for m in model.modules())
        weight_quantizers = _quantizers(model, "weight_quantizer")
        assert len(weight_quantizers) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
        for module in weight_quantizers.values():
            assert module.num_bits == 6

    def test_enable_only_broadcasts_to_sequential_sub_quantizers(self):
        model = _replaced()
        set_quantizer_by_cfg(
            model,
            [
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": [{"num_bits": 4}, {"num_bits": 8}],
                }
            ],
        )
        set_quantizer_by_cfg(model, [{"quantizer_name": "*weight_quantizer", "enable": False}])
        containers = _seq_quantizers(model)
        assert len(containers) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS  # not downgraded
        for module in containers.values():
            assert all(not q.is_enabled for q in module)
            assert [q.num_bits for q in module] == [4, 8]  # attributes preserved


class TestSetQuantizerAttributesFull:
    def test_callable_filter_selects_quantizers(self):
        model = _replaced()
        set_quantizer_attributes_full(
            model,
            lambda name: name.endswith("input_quantizer"),
            QuantizerAttributeConfig(num_bits=4),
        )
        for name, module in _quantizers(model).items():
            expected = 4 if name.endswith("input_quantizer") else 8
            assert module.num_bits == expected, name

    def test_filter_function_sees_only_quantizer_names(self):
        """The filter is invoked with quantizer module names, never with other modules."""
        model = _replaced()
        seen = []

        def _filter(name):
            seen.append(name)
            return False

        set_quantizer_attributes_full(model, _filter, QuantizerAttributeConfig(num_bits=4))
        assert set(seen) == SIMPLE_LINEAR_QUANTIZER_NAMES

    # NOTE: rejection of a plain dict for `attributes` is already covered by
    # test_quantize_cpu.py::TestSetQuantizerAttributesFull::test_invalid_attributes_type_raises.

    def test_list_with_non_config_elements_rejected(self):
        model = _replaced()
        with pytest.raises(ValueError, match="must be of type QuantizerAttributeConfig"):
            set_quantizer_attributes_full(
                model, "*weight_quantizer", [QuantizerAttributeConfig(num_bits=4), {"num_bits": 8}]
            )

    def test_length_mismatch_with_existing_sequential_warns(self):
        model = _replaced()
        set_quantizer_attributes_full(
            model,
            "*weight_quantizer",
            [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)],
        )
        with pytest.warns(UserWarning, match="does not match the number"):
            set_quantizer_attributes_full(
                model, "*weight_quantizer", [QuantizerAttributeConfig(num_bits=2)]
            )

    def test_double_list_application_nests_sequential_quantizers(self):
        # NOTE: documents current behavior; arguably a bug because applying the same
        # list-valued attributes twice should be idempotent. On the second pass the
        # existing SequentialQuantizer's children ("...weight_quantizer.0") are
        # re-matched by "*weight_quantizer" through the fused-experts name
        # normalization and each TensorQuantizer sub-slot is replaced by a *nested*
        # SequentialQuantizer.
        model = _replaced()
        attrs = [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)]
        set_quantizer_attributes_full(model, "*weight_quantizer", attrs)
        wq = model.net[0].weight_quantizer
        assert isinstance(wq, SequentialQuantizer)
        assert all(type(q) is TensorQuantizer for q in wq)

        set_quantizer_attributes_full(model, "*weight_quantizer", attrs)
        wq = model.net[0].weight_quantizer
        assert isinstance(wq, SequentialQuantizer)
        assert all(isinstance(q, SequentialQuantizer) for q in wq)  # nested (buggy)

    def test_parent_class_with_root_level_quantizer(self):
        """A dot-free quantizer name resolves its parent to the model itself."""
        model = _RootQuantHolder()
        set_quantizer_attributes_full(
            model, "*", QuantizerAttributeConfig(num_bits=4), parent_class=_RootQuantHolder
        )
        assert model.q.num_bits == 4

        model2 = _RootQuantHolder()
        set_quantizer_attributes_full(
            model2, "*", QuantizerAttributeConfig(num_bits=4), parent_class=nn.Linear
        )
        assert model2.q.num_bits == 8  # parent class did not match -> untouched


class TestSetQuantizerAttributesPartial:
    def test_merge_preserves_unspecified_attributes(self):
        model = _replaced()
        set_quantizer_attributes_full(
            model, "*weight_quantizer", QuantizerAttributeConfig(num_bits=8, axis=0)
        )
        set_quantizer_attributes_partial(model, "*weight_quantizer", {"num_bits": 4})
        for module in _weight_quantizers(model).values():
            assert module.num_bits == 4
            assert module.axis == 0  # preserved, unlike the full-replacement API

    def test_dict_broadcasts_to_sequential_sub_quantizers(self):
        model = _replaced()
        set_quantizer_attributes_full(
            model,
            "*weight_quantizer",
            [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)],
        )
        # Use a callable filter (matches only the container, never its ".0/.1" children)
        # so this test pins the container dict-broadcast branch specifically. A wildcard
        # would also reach the children directly via fused-experts name normalization,
        # masking a broken broadcast branch.
        set_quantizer_attributes_partial(
            model, lambda name: name.endswith("weight_quantizer"), {"enable": False}
        )
        containers = _seq_quantizers(model)
        assert len(containers) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
        for module in containers.values():
            assert all(not q.is_enabled for q in module)
            assert [q.num_bits for q in module] == [4, 8]

    def test_list_on_plain_tensor_quantizer_raises(self):
        model = _replaced()
        with pytest.raises(ValueError, match="not a SequentialQuantizer"):
            set_quantizer_attributes_partial(
                model, "*input_quantizer", [{"enable": False}, {"enable": True}]
            )

    def test_list_on_sequential_via_wildcard_raises_through_sub_quantizers(self):
        # NOTE: documents current behavior; arguably a bug because the docstring
        # promises list attributes work when "the matched module [is] already a
        # SequentialQuantizer". With a wildcard like "*weight_quantizer" the
        # SequentialQuantizer's own children ("...weight_quantizer.0") also match via
        # the fused-experts name normalization, and since they are TensorQuantizers
        # the call raises after having already configured the parent container.
        model = _replaced()
        set_quantizer_attributes_full(
            model,
            "*weight_quantizer",
            [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)],
        )
        with pytest.raises(ValueError, match="not a SequentialQuantizer"):
            set_quantizer_attributes_partial(
                model, "*weight_quantizer", [{"enable": False}, {"enable": True}]
            )

    def test_list_on_sequential_via_callable_filter_applies_per_position(self):
        """A callable filter (no name normalization) makes per-position lists work."""
        model = _replaced()
        set_quantizer_attributes_full(
            model,
            "*weight_quantizer",
            [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)],
        )
        set_quantizer_attributes_partial(
            model,
            lambda name: name.endswith("weight_quantizer"),
            [{"enable": False}, {"enable": True}],
        )
        containers = _seq_quantizers(model)
        assert len(containers) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
        for module in containers.values():
            assert not module[0].is_enabled
            assert module[1].is_enabled

    @pytest.mark.parametrize("bad_attributes", [42, "enable", [{"enable": False}, 42]])
    def test_invalid_attribute_containers_rejected(self, bad_attributes):
        model = _replaced()
        with pytest.raises(ValueError, match=r"must be of type dict|expected dictionary"):
            set_quantizer_attributes_partial(model, "*", bad_attributes)

    def test_unsupported_wildcard_type_raises(self):
        model = _replaced()
        with pytest.raises(NotImplementedError, match="Unsupported type"):
            set_quantizer_attributes_partial(model, 42, {"enable": False})

    def test_non_matching_wildcard_is_noop(self):
        model = _replaced()
        state_before = quantizer_state(model)
        set_quantizer_attributes_partial(model, "*no_such_quantizer", {"enable": False})
        assert quantizer_state(model) == state_before


class TestFusedExpertsNameNormalization:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # plural ModuleList entries collapse to the singular wildcard-friendly form
            ("model.gate_up_proj_weight_quantizers.0", "model.gate_up_proj_weight_quantizer"),
            ("experts.input_quantizers.12", "experts.input_quantizer"),
            # singular naming variant with an index is also collapsed
            ("layers.0.weight_quantizer.3", "layers.0.weight_quantizer"),
            # index in the middle of the name is stripped too
            ("weight_quantizers.0.sub", "weight_quantizer.sub"),
            # no index -> unchanged
            ("layers.0.weight_quantizers", "layers.0.weight_quantizers"),
            # non weight/input quantizer lists are not normalized
            ("layers.0.output_quantizer.0", "layers.0.output_quantizer.0"),
            # non-numeric suffix -> unchanged
            ("layers.0.weight_quantizers.x", "layers.0.weight_quantizers.x"),
        ],
    )
    def test_normalization(self, name, expected):
        assert _normalize_fused_experts_quantizer_name(name) == expected

    def test_singular_wildcard_matches_per_expert_module_list(self):
        """Stock '*weight_quantizer' patterns reach per-expert ModuleList quantizers."""
        model = _FusedExperts(num_experts=2)
        set_quantizer_attributes_partial(model, "*weight_quantizer", {"enable": False})
        assert all(not q.is_enabled for q in model.weight_quantizers)
        assert model.input_quantizers[0].is_enabled  # different suffix -> untouched
        assert model.output_quantizer.is_enabled

    def test_raw_module_list_names_still_match(self):
        """Patterns written against the raw dotted names keep working."""
        model = _FusedExperts(num_experts=2)
        set_quantizer_attributes_partial(model, "weight_quantizers.1", {"enable": False})
        assert model.weight_quantizers[0].is_enabled
        assert not model.weight_quantizers[1].is_enabled


class TestQuantizerStateAndMetadata:
    def test_state_covers_every_quantizer_module(self):
        model = _replaced()
        assert quantizer_state(model).keys() == SIMPLE_LINEAR_QUANTIZER_NAMES

    def test_identically_configured_models_have_equal_state(self):
        model_a = _replaced()
        model_b = _replaced()
        assert quantizer_state(model_a) == quantizer_state(model_b)

        set_quantizer_attributes_partial(model_b, "*weight_quantizer", {"num_bits": 4})
        assert quantizer_state(model_a) != quantizer_state(model_b)

    def test_sequential_quantizer_state_entries(self):
        """A SequentialQuantizer contributes its own entry plus one per sub-quantizer."""
        model = _replaced()
        set_quantizer_by_cfg(
            model,
            [
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": [{"num_bits": 4}, {"num_bits": 8}],
                }
            ],
        )
        state = quantizer_state(model)
        assert state["net.0.weight_quantizer"] == {
            "num_quantizers": 2,
            "is_sequential_quantizer": True,
        }
        assert "net.0.weight_quantizer.0" in state
        assert "net.0.weight_quantizer.1" in state

    def test_update_quantize_metadata_records_state(self):
        model = _replaced()
        config = QuantizeConfig()
        metadata = {}
        update_quantize_metadata(model, config, metadata)
        assert metadata["quantizer_state"] == quantizer_state(model)

    def test_update_quantize_metadata_pops_stale_shared_state(self):
        """Stale shared_quant_states entries are removed when the model has none."""
        model = _replaced()
        metadata = {"shared_quant_states": {"stale": True}}
        update_quantize_metadata(model, QuantizeConfig(), metadata)
        assert "shared_quant_states" not in metadata


class TestRestoreQuantizerState:
    _CFG = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
    ]

    def _make_saved_metadata(self):
        model = _replaced()
        set_quantizer_by_cfg(model, copy.deepcopy(self._CFG))
        metadata = {}
        update_quantize_metadata(model, QuantizeConfig(), metadata)
        return model, metadata

    def test_roundtrip_restores_attributes(self):
        model_src, metadata = self._make_saved_metadata()
        model_dst = _replaced()  # default attributes, different from model_src
        restored = restore_quantizer_state(model_dst, QuantizeConfig(), metadata)
        assert restored is model_dst
        assert quantizer_state(model_dst) == quantizer_state(model_src)
        for name, module in _quantizers(model_dst).items():
            if name.endswith("weight_quantizer"):
                assert module.num_bits == 4
                assert module.axis == 0
                assert module.is_enabled  # cfg entry after deny-all implies enable=True
            else:
                assert not module.is_enabled  # still off from the deny-all entry

    def test_restore_recreates_buffer_placeholders(self):
        """Buffers recorded in the state (e.g. amax) are re-created with correct shape."""
        model_src = _replaced()
        model_src.net[0].weight_quantizer.amax = torch.tensor(1.5)
        metadata = {}
        update_quantize_metadata(model_src, QuantizeConfig(), metadata)

        model_dst = _replaced()
        assert not hasattr(model_dst.net[0].weight_quantizer, "_amax")
        restore_quantizer_state(model_dst, QuantizeConfig(), metadata)
        wq = model_dst.net[0].weight_quantizer
        assert hasattr(wq, "_amax")
        assert wq._amax.shape == model_src.net[0].weight_quantizer._amax.shape

    def test_metadata_without_quantizer_state_is_noop(self):
        """MCore sharded checkpoints carry no quantizer_state; restore must pass through."""
        model = _replaced()
        state_before = quantizer_state(model)
        restored = restore_quantizer_state(model, QuantizeConfig(), {})
        assert restored is model
        assert quantizer_state(model) == state_before

    def test_unmatched_checkpoint_keys_raise(self):
        """Keys present in the checkpoint but missing from the model are 'unmatched'."""
        _, metadata = self._make_saved_metadata()
        metadata["quantizer_state"]["bogus.weight_quantizer"] = {}
        model_dst = _replaced()
        with pytest.raises(ApplyModeError, match=r"Unmatched keys.*bogus\.weight_quantizer"):
            restore_quantizer_state(model_dst, QuantizeConfig(), metadata)

    def test_extra_model_keys_raise(self):
        """Quantizers in the model that have no checkpoint entry are 'extra'."""
        _, metadata = self._make_saved_metadata()
        del metadata["quantizer_state"]["net.0.weight_quantizer"]
        model_dst = _replaced()
        with pytest.raises(ApplyModeError, match=r"Extra keys.*net\.0\.weight_quantizer"):
            restore_quantizer_state(model_dst, QuantizeConfig(), metadata)

    @pytest.mark.parametrize(
        ("state", "expect_promoted"),
        [
            ({"_is_nvfp4_static_quantizer": True}, True),
            ({"_is_nvfp4_static_quantizer": False}, False),
            ({}, False),
        ],
    )
    def test_nvfp4_static_promotion(self, state, expect_promoted):
        """Quantizers flagged in the saved state are promoted to NVFP4StaticQuantizer."""
        quantizer = TensorQuantizer()
        maybe_promote_nvfp4_static_quantizer(quantizer, state)
        assert isinstance(quantizer, NVFP4StaticQuantizer) is expect_promoted


class TestConvertRestoreEntrypoints:
    def _config(self):
        return QuantizeConfig(
            quant_cfg=[
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
                {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8}},
            ],
            algorithm="max",
        )

    def _model_with_state(self):
        model = SimpleLinear()
        ModeloptStateManager(model, init_state=True)
        return model

    def test_convert_returns_model_and_metadata(self):
        model = self._model_with_state()
        config = self._config()
        converted, metadata = convert_to_quantized_model(model, config)
        assert converted is model  # conversion is in-place
        assert is_quantized(converted)
        assert metadata["quantizer_state"] == quantizer_state(converted)
        for module in _weight_quantizers(converted).values():
            assert module.num_bits == 4
            assert module.axis == 0

    def test_restore_quantized_model_roundtrip(self):
        config = self._config()
        model_src = self._model_with_state()
        _, metadata = convert_to_quantized_model(model_src, config)
        # Simulate a calibrated checkpoint: amax exists on one quantizer.
        model_src.net[0].weight_quantizer.amax = torch.tensor(2.0)
        update_quantize_metadata(model_src, config, metadata)

        model_dst = self._model_with_state()
        restored = restore_quantized_model(model_dst, config, metadata)
        assert restored is model_dst
        assert quantizer_state(restored) == quantizer_state(model_src)
        assert hasattr(restored.net[0].weight_quantizer, "_amax")


class TestSetQuantizerByCfgContext:
    def test_temporary_override_and_restore(self):
        model = _replaced()
        with set_quantizer_by_cfg_context(
            model, [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": None}}]
        ):
            for module in _weight_quantizers(model).values():
                assert module.num_bits == 4
                assert module.axis is None
        for module in _weight_quantizers(model).values():
            assert module.num_bits == 8  # default restored
            assert module.axis == 0  # default weight axis restored

    def test_enable_states_restored(self):
        model = _replaced()
        with set_quantizer_by_cfg_context(model, [{"quantizer_name": "*", "enable": False}]):
            assert all(not m.is_enabled for m in _quantizers(model).values())
        for name, module in _quantizers(model).items():
            expected = not name.endswith("output_quantizer")
            assert module.is_enabled is expected, name

    # NOTE: restore-on-exception is already covered (more thoroughly) by
    # test_tensor_quant_cpu.py::test_set_quantizer_cxt_restores_on_exception.

    def test_rejects_sequential_cfg_lists(self):
        model = _replaced()
        with (
            pytest.raises(ValueError, match="Sequential cfg lists are not allowed"),
            set_quantizer_by_cfg_context(
                model,
                [{"quantizer_name": "*weight_quantizer", "cfg": [{"num_bits": 4}]}],
            ),
        ):
            pass

    def test_sequential_downgrade_is_reverted_on_exit(self):
        """A single-cfg entry downgrades a SequentialQuantizer; exit re-creates it."""
        model = _replaced()
        set_quantizer_by_cfg(
            model,
            [
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": [{"num_bits": 4}, {"num_bits": 8, "axis": 0}],
                }
            ],
        )
        with set_quantizer_by_cfg_context(
            model, [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        ):
            wq = model.net[0].weight_quantizer
            assert isinstance(wq, TensorQuantizer)
            assert not isinstance(wq, SequentialQuantizer)
            assert wq.num_bits == 8
        wq = model.net[0].weight_quantizer
        assert isinstance(wq, SequentialQuantizer)
        assert len(wq) == 2
        assert [q.num_bits for q in wq] == [4, 8]
        assert wq[1].axis == 0

    def test_manual_upgrade_inside_context_is_reverted(self):
        """Even a manual TensorQuantizer->SequentialQuantizer swap in the body is undone."""
        model = _replaced()
        with set_quantizer_by_cfg_context(model, []):
            set_quantizer_attributes_full(
                model,
                "*weight_quantizer",
                [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)],
            )
            assert isinstance(model.net[0].weight_quantizer, SequentialQuantizer)
        wq = model.net[0].weight_quantizer
        assert isinstance(wq, TensorQuantizer)
        assert not isinstance(wq, SequentialQuantizer)
        assert wq.num_bits == 8
        assert wq.axis == 0  # original default weight attributes restored


class TestDeprecatedSetQuantizerAttribute:
    def test_warns_and_merges_like_partial(self):
        model = _replaced()
        set_quantizer_attributes_full(
            model, "*weight_quantizer", QuantizerAttributeConfig(num_bits=8, axis=0)
        )
        with pytest.warns(DeprecationWarning, match="set_quantizer_attribute is deprecated"):
            set_quantizer_attribute(model, "*weight_quantizer", {"num_bits": 4})
        for module in _weight_quantizers(model).values():
            assert module.num_bits == 4
            assert module.axis == 0  # merged, not replaced
