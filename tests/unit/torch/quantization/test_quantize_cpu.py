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

"""High-level tests for quantization."""

import copy

import pytest
import torch
from _test_utils.torch.quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear
from _test_utils.torch.quantization.quantize_common import (
    INT4_AWQ_CLIP_CFG,
    INT4_AWQ_FULL_CFG,
    INT4_SVDQUANT_CFG,
    quantize_model_and_forward,
    save_restore_test,
)
from pydantic import ValidationError

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.calib import MaxCalibrator
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import set_quantizer_attributes_full
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    SequentialQuantizer,
    TensorQuantizer,
)

# A test config with double-quant (using `SequentialQuantizers`)
WINT4INT8_CFG = {
    "quant_cfg": [
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": [
                {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                {"num_bits": 8, "axis": 0},
            ],
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": 8, "axis": None},
            "enable": True,
        },
    ],
    "algorithm": "awq_lite",
}

# Test configs for per channel MSE calibration
INT8_MSE_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
    ],
    "algorithm": "mse",
}

STATIC_WEIGHT_DYNAMIC_ACTIVATION_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": 8, "axis": 0},
        },  # Per-channel quantization
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": 8, "axis": (0, 1), "type": "dynamic"},
        },  # Dynamic per-token quantization
    ],
    "algorithm": "max",
}


class NewMaxCalibrator(MaxCalibrator):
    def compute_amax(self):
        return 2 * self._calib_amax


quant_cfg_custom_calib = {
    "quant_cfg": [
        {
            "quantizer_name": "*",
            "cfg": {
                "num_bits": 4,
                "axis": None,
                "calibrator": (NewMaxCalibrator, (4, None, False)),
            },
            "enable": True,
        }
    ],
    "algorithm": "max",
}


@pytest.mark.parametrize("model_cls", [SimpleLinear, SimpleConv, SimpleConvLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        INT4_SVDQUANT_CFG,
        INT4_AWQ_CLIP_CFG,
        INT4_AWQ_FULL_CFG,
        WINT4INT8_CFG,
        INT8_MSE_CFG,
    ],
)
def test_quantize(model_cls, config):
    """Test quantize function can run without problems."""
    model = model_cls()
    calib_data = [model.get_input() for _ in range(2)]
    quantize_model_and_forward(model, config, calib_data)

    # For fast testing, lets just test one config
    if config == mtq.INT8_DEFAULT_CFG:
        mtq.print_quant_summary(model)


@pytest.mark.parametrize(
    ("model_cls", "quant_config"),
    [
        (SimpleLinear, mtq.INT8_SMOOTHQUANT_CFG),
        (SimpleConvLinear, quant_cfg_custom_calib),
        (SimpleConvLinear, mtq.INT8_DEFAULT_CFG),
        (SimpleLinear, INT4_SVDQUANT_CFG),
    ],
)
def test_save_restore(model_cls, quant_config):
    save_restore_test(model_cls, "cpu", quant_config)


def test_quantize_invalid_cfg():
    model = SimpleLinear()
    config_invalid = {
        "quant_cfg": [
            {"quantizer_name": "*", "cfg": {"num_bits": 4, "axis": 0, "block_sizes": {-1: 128}}}
        ],
        "algorithm": "max",
    }
    with pytest.raises(ValidationError, match="axis must be None when block_sizes is not None."):
        model = mtq.quantize(model, config_invalid)


def test_inplace_backward_compatibility():
    model = SimpleLinear()
    calib_data = [model.get_input() for _ in range(2)]

    def forward_loop():
        for batch in calib_data:
            model(batch)

    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=forward_loop)


def test_custom_calib_config():
    model_ref = SimpleLinear()
    model_ref = mtq.quantize(
        model_ref, quant_cfg_custom_calib, lambda model: model(model.get_input())
    )

    model_quant = SimpleLinear()
    model_quant = mto.restore_from_modelopt_state(model_quant, mto.modelopt_state(model_ref))
    model_quant.load_state_dict(model_ref.state_dict())

    inputs = model_ref.get_input()
    assert torch.allclose(model_ref(inputs), model_quant(inputs))

    for name, module in model_quant.named_modules():
        if name.endswith("quantizer"):
            assert module._calibrator.__class__ == NewMaxCalibrator


def test_class_wise_config():
    model = SimpleConvLinear()
    config = {
        "quant_cfg": [
            {
                "parent_class": "nn.Linear",
                "quantizer_name": "*",
                "cfg": {"num_bits": 4, "axis": -1},
                "enable": True,
            },
            {
                "parent_class": "nn.Conv2d",
                "quantizer_name": "*",
                "cfg": {"num_bits": 8},
                "enable": True,
            },
            {"parent_class": "nn.BatchNorm2d", "quantizer_name": "*", "enable": False},
            {"quantizer_name": "*output_quantizer", "cfg": {"num_bits": 8}, "enable": True},
        ],
        "algorithm": "max",
    }

    model = mtq.quantize(model, config, lambda model: model(model.get_input()))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for sub_quantizer in (module.weight_quantizer, module.input_quantizer):
                assert sub_quantizer.num_bits == 4
                assert sub_quantizer.axis == -1
                assert sub_quantizer.is_enabled
        elif isinstance(module, torch.nn.Conv2d):
            for sub_quantizer in (module.weight_quantizer, module.input_quantizer):
                assert sub_quantizer.num_bits == 8
                assert sub_quantizer.is_enabled
        elif isinstance(module, torch.nn.BatchNorm2d):
            assert module.input_quantizer.is_enabled is False

        if name.endswith("output_quantizer"):
            assert module.is_enabled
            assert module.num_bits == 8


def test_static_weight_dynamic_activations():
    model = SimpleLinear()
    inputs = model.get_input()

    model = mtq.quantize(
        model, STATIC_WEIGHT_DYNAMIC_ACTIVATION_CFG, lambda model: model(model.get_input())
    )
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer"):
            assert module.amax is not None
    # Test that model forward works
    model(inputs)

    # Lets test mtq.quantize without forward_loop
    model = SimpleLinear()
    model = mtq.quantize(model, STATIC_WEIGHT_DYNAMIC_ACTIVATION_CFG)
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer"):
            assert module.amax is not None


def test_block_sizes_axis_model():
    REF_QUANT_CFG = {  # noqa: N806
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {
                "quantizer_name": "*input_quantizer",
                "cfg": {"num_bits": 8, "axis": None, "type": "dynamic"},
            },
        ],
        "algorithm": "max",
    }
    QUANT_CFG = {  # noqa: N806
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 8, "block_sizes": {1: None}},
            },
            {
                "quantizer_name": "*input_quantizer",
                "cfg": {"num_bits": 8, "block_sizes": {0: None, 1: None}, "type": "dynamic"},
            },
        ],
        "algorithm": "max",
    }
    model_ref = SimpleLinear()
    model = copy.deepcopy(model_ref)
    inputs = model_ref.get_input()

    mtq.quantize(model_ref, REF_QUANT_CFG, lambda model: model(inputs))
    mtq.quantize(model, QUANT_CFG, lambda model: model(inputs))

    assert torch.allclose(model_ref(inputs), model(inputs))

    # compare the calibrated amax of all quantizers
    for (name_ref, module_ref), (name, module) in zip(
        model_ref.named_modules(), model.named_modules()
    ):
        if hasattr(module, "weight_quantizer"):
            assert name_ref == name
            assert torch.allclose(module_ref.weight_quantizer.amax, module.weight_quantizer.amax)


def test_quantize_twice():
    """Test that calling mtq.quantize twice on the same model works."""
    model = SimpleLinear()
    inputs = model.get_input()

    def forward_loop(model):
        return model(inputs)

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=forward_loop)
    out1 = model(inputs)
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=forward_loop)
    out2 = model(inputs)

    assert torch.allclose(out1, out2), "Re-quantization with same config should be idempotent"


class TestSetQuantizerAttributesFull:
    """Tests for set_quantizer_attributes_full and its atomicity semantics."""

    def _quantize(self, model):
        return mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(m.get_input()))

    def test_basic_full_replacement(self):
        """set_quantizer_attributes_full replaces all attributes on matched quantizers."""
        model = self._quantize(SimpleLinear())
        attrs = QuantizerAttributeConfig(num_bits=4, axis=0)
        set_quantizer_attributes_full(model, "*weight_quantizer", attrs)
        for name, module in model.named_modules():
            if name.endswith("weight_quantizer"):
                assert isinstance(module, TensorQuantizer)
                assert module.num_bits == 4
                assert module.axis == 0

    def test_atomicity_unset_fields_revert_to_defaults(self):
        """A full replacement reverts unspecified fields to QuantizerAttributeConfig defaults."""
        model = self._quantize(SimpleLinear())
        # First configure with axis=0 (non-default)
        set_quantizer_attributes_full(
            model, "*weight_quantizer", QuantizerAttributeConfig(num_bits=8, axis=0)
        )
        for name, module in model.named_modules():
            if name.endswith("weight_quantizer"):
                assert module.axis == 0

        # Now replace with only num_bits=4; axis should revert to default (None)
        set_quantizer_attributes_full(
            model, "*weight_quantizer", QuantizerAttributeConfig(num_bits=4)
        )
        default_axis = QuantizerAttributeConfig().axis
        for name, module in model.named_modules():
            if name.endswith("weight_quantizer"):
                assert module.num_bits == 4
                assert module.axis == default_axis

    def test_parent_class_filter(self):
        """parent_class restricts which quantizers are affected."""
        model = self._quantize(SimpleConvLinear())
        # Only set num_bits=4 for quantizers inside nn.Linear modules
        set_quantizer_attributes_full(
            model,
            "*weight_quantizer",
            QuantizerAttributeConfig(num_bits=4),
            parent_class=torch.nn.Linear,
        )
        for name, module in model.named_modules():
            if not name.endswith("weight_quantizer"):
                continue
            parent_name = name.rpartition(".")[0]
            parent = model.get_submodule(parent_name)
            if isinstance(parent, torch.nn.Linear):
                assert module.num_bits == 4
            else:
                # Conv2d weight_quantizers should be unchanged (still 8-bit from INT8_DEFAULT_CFG)
                assert module.num_bits == 8

    def test_wildcard_no_match_is_noop(self):
        """A wildcard that matches nothing silently does nothing."""
        model = self._quantize(SimpleLinear())
        # Record state before
        bits_before = {
            n: m.num_bits for n, m in model.named_modules() if isinstance(m, TensorQuantizer)
        }
        set_quantizer_attributes_full(
            model, "*nonexistent_quantizer*", QuantizerAttributeConfig(num_bits=4)
        )
        bits_after = {
            n: m.num_bits for n, m in model.named_modules() if isinstance(m, TensorQuantizer)
        }
        assert bits_before == bits_after

    def test_invalid_attributes_type_raises(self):
        """Passing a plain dict instead of QuantizerAttributeConfig raises ValueError."""
        model = self._quantize(SimpleLinear())
        with pytest.raises((ValueError, AttributeError)):
            set_quantizer_attributes_full(model, "*weight_quantizer", {"num_bits": 4})  # type: ignore[arg-type]

    def test_list_attributes_creates_sequential_quantizer(self):
        """A list of QuantizerAttributeConfig replaces TensorQuantizer with SequentialQuantizer."""
        model = self._quantize(SimpleLinear())
        attrs = [
            QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 128}),
            QuantizerAttributeConfig(num_bits=8, axis=0),
        ]
        set_quantizer_attributes_full(model, "*weight_quantizer", attrs)
        for name, module in model.named_modules():
            if name.endswith("weight_quantizer"):
                assert isinstance(module, SequentialQuantizer)
                assert len(module) == 2


def test_ordering_later_entry_overrides_earlier():
    """Later entries in quant_cfg override earlier ones for the same quantizer."""
    model = SimpleLinear()
    config = {
        "quant_cfg": [
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        ],
        "algorithm": "max",
    }
    model = mtq.quantize(model, config, lambda m: m(m.get_input()))
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer"):
            assert module.num_bits == 4, "Later entry (num_bits=4) should override earlier (8)"
        if name.endswith("input_quantizer"):
            assert module.num_bits == 8


def test_enable_only_entry_preserves_attributes():
    """An enable-only entry toggles the quantizer without resetting its attributes."""
    model = SimpleLinear()
    config = {
        "quant_cfg": [
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            # This enable-only entry should disable without resetting num_bits/axis
            {"quantizer_name": "*weight_quantizer", "enable": False},
        ],
        "algorithm": "max",
    }
    model = mtq.quantize(model, config, lambda m: m(m.get_input()))
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer"):
            assert not module.is_enabled, "weight_quantizer should be disabled"
            assert module.num_bits == 4, "num_bits should be preserved by enable-only entry"
            assert module.axis == 0, "axis should be preserved by enable-only entry"


def test_atomicity_later_cfg_entry_does_not_inherit_earlier():
    """When two cfg-bearing entries match the same quantizer, the second fully replaces the first."""
    model = SimpleLinear()
    config = {
        "quant_cfg": [
            # Entry 1: set axis=0
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            # Entry 2: only set num_bits=4, no axis — axis should revert to default (None), not 0
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        ],
        "algorithm": "max",
    }
    model = mtq.quantize(model, config, lambda m: m(m.get_input()))
    default_axis = QuantizerAttributeConfig().axis
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer"):
            assert module.num_bits == 4
            assert module.axis == default_axis, (
                f"axis should revert to default ({default_axis}), not inherit 0 from earlier entry"
            )


def test_legacy_dict_format_end_to_end():
    """Old dict-format quant_cfg works end-to-end through mtq.quantize via normalization."""
    model = SimpleLinear()
    # Old-style dict config with "default" key and wildcard keys
    old_config = {
        "quant_cfg": {
            "default": {"enable": False},
            "*weight_quantizer": {"num_bits": 8, "axis": 0},
            "*input_quantizer": {"num_bits": 8, "axis": None},
        },
        "algorithm": "max",
    }
    model = mtq.quantize(model, old_config, lambda m: m(m.get_input()))
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if name.endswith(("weight_quantizer", "input_quantizer")):
                assert module.is_enabled
                assert module.num_bits == 8
            elif name.endswith("output_quantizer"):
                # "default" key → quantizer_name="*" with enable=False disables everything,
                # but weight/input quantizers are re-enabled by subsequent entries.
                # output_quantizer is NOT re-enabled so it stays disabled.
                assert not module.is_enabled
