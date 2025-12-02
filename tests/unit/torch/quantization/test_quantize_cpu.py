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

# A test config with double-quant (using `SequentialQuantizers`)
WINT4INT8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": 8, "axis": 0, "enable": True},
        ],
        "*input_quantizer": {"num_bits": 8, "axis": None, "enable": True},
    },
    "algorithm": "awq_lite",
}

INT8_MSE_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
    },
    "algorithm": "mse",
}

STATIC_WEIGHT_DYNAMIC_ACTIVATION_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 8,
            "axis": 0,
        },  # Per-channel quantization
        "*input_quantizer": {
            "num_bits": 8,
            "axis": (0, 1),
            "type": "dynamic",
        },  # Dynamic per-token quantization
        "default": {"enable": False},
    },
    "algorithm": "max",
}


class NewMaxCalibrator(MaxCalibrator):
    def compute_amax(self):
        return 2 * self._calib_amax


quant_cfg_custom_calib = {
    "quant_cfg": {
        "*": {
            "num_bits": 4,
            "axis": None,
            "enable": True,
            "calibrator": (NewMaxCalibrator, (4, None, False)),
        }
    },
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
        "quant_cfg": {"*": {"num_bits": 4, "axis": 0, "block_sizes": {-1: 128}}},
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
        "quant_cfg": {
            "nn.Linear": {"*": {"num_bits": 4, "axis": -1, "enable": True}},
            "nn.Conv2d": {"*": {"num_bits": 8, "enable": True}},
            "nn.BatchNorm2d": {"*": {"enable": False}},
            "*output_quantizer": {"num_bits": 8, "enable": True},
        },
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
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": 8,
                "axis": 0,
            },
            "*input_quantizer": {
                "num_bits": 8,
                "axis": None,
                "type": "dynamic",
            },
            "default": {"enable": False},
        },
        "algorithm": "max",
    }
    QUANT_CFG = {  # noqa: N806
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": 8,
                "block_sizes": {1: None},
            },
            "*input_quantizer": {
                "num_bits": 8,
                "block_sizes": {0: None, 1: None},
                "type": "dynamic",
            },
            "default": {"enable": False},
        },
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
