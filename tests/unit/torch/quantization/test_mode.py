# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.config import ModeloptField
from modelopt.torch.opt.mode import _ModeRegistryCls
from modelopt.torch.quantization.config import MaxCalibConfig, QuantizeAlgorithmConfig
from modelopt.torch.quantization.mode import (
    BaseCalibrateModeDescriptor,
    CalibrateModeRegistry,
    QuantizeModeRegistry,
    get_modelike_from_algo_cfg,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def test_modes():
    for mode in ["max", "smoothquant", "awq_full", "awq_lite", "svdquant", None]:
        mode_name = BaseCalibrateModeDescriptor._get_mode_name(mode)
        assert mode_name in CalibrateModeRegistry

    for mode in ["quantize", "auto_quantize"]:
        assert mode in QuantizeModeRegistry


def test_calibrate_mode_registry_with_custom_mode():
    class TestConfig(QuantizeAlgorithmConfig):
        method: Literal["test"] = ModeloptField("test")

    @CalibrateModeRegistry.register_mode
    class TestCalibrateModeDescriptor(BaseCalibrateModeDescriptor):
        @property
        def config_class(self) -> QuantizeAlgorithmConfig:
            return TestConfig

        _calib_func = None

    assert BaseCalibrateModeDescriptor._get_mode_name("test") in CalibrateModeRegistry
    assert isinstance(_ModeRegistryCls.get_from_any("test_calibrate"), TestCalibrateModeDescriptor)

    # This should result in an error
    with pytest.raises(
        AssertionError,
        match="Mode descriptor for `_CalibrateModeRegistryCls` must be a subclass of `BaseCalibrateModeDescriptor`!",
    ):

        @CalibrateModeRegistry.register_mode
        class TestIncorrectCalibrateModeDescriptor:
            pass


def test_get_modelike_from_algo_cfg_accepts_quantize_algorithm_config():
    """Regression test for #201: ``QuantizeAlgorithmConfig`` instances must be accepted."""
    cfg_obj = MaxCalibConfig(method="max", distributed_sync=False)
    cfg_dict = cfg_obj.model_dump()

    from_obj = get_modelike_from_algo_cfg(cfg_obj)
    from_dict = get_modelike_from_algo_cfg(cfg_dict)

    assert from_obj == from_dict
    assert from_obj[0][1] == cfg_dict

    # List handling should also accept ``QuantizeAlgorithmConfig`` instances.
    from_list = get_modelike_from_algo_cfg([cfg_obj])
    assert from_list == from_obj


@pytest.mark.parametrize("mode_name", sorted(CalibrateModeRegistry._name2descriptor))
def test_scopable_matches_the_calibration_function_signature(mode_name):
    """``scopable`` is hand-written; this keeps it from drifting from the code it describes."""
    import inspect

    descriptor = type(CalibrateModeRegistry._name2descriptor[mode_name])
    func = descriptor._calib_func
    if func is None:  # the no-op mode writes nothing, so any scope is vacuously honoured
        return
    takes_mask = "should_process" in inspect.signature(func).parameters
    assert descriptor.scopable == takes_mask, (
        f"{mode_name}: scopable={descriptor.scopable} but "
        f"{func.__name__}() {'takes' if takes_mask else 'does not take'} should_process"
    )


@pytest.mark.parametrize(
    ("algo", "cfg", "expected"),
    [
        # local_hessian defaults fp8_scale_sweep=True, so a *sparse* cfg -- which is all a plan
        # entry carries -- must still resolve to the static grid that sweep searches. Reading
        # the raw dict instead of the config class disagrees here, in the unsafe direction.
        ("local_hessian", {}, "static"),
        ("local_hessian", {"fp8_scale_sweep": False}, None),
        ("mse", {}, None),
        ("mse", {"fp8_scale_sweep": True}, "static"),
        ("awq_lite", {}, "dynamic"),
        ("gptq", {}, None),
    ],
)
def test_weight_grid_resolves_through_the_config_class(algo, cfg, expected):
    name = BaseCalibrateModeDescriptor._get_mode_name(algo)
    descriptor = type(CalibrateModeRegistry._name2descriptor[name])
    assert descriptor.grid_for_cfg({"method": algo, **cfg}) == expected


# --- prepare(): the weight-grid upgrade ---

NVFP4_DYNAMIC = [
    {"quantizer_name": "*", "enable": False},
    {
        "quantizer_name": "*weight_quantizer",
        "cfg": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        },
    },
]


class _Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(32, 32, bias=False)
        self.b = nn.Linear(32, 32, bias=False)

    def forward(self, x):
        return self.b(self.a(x))


def _nvfp4_pair():
    torch.manual_seed(0)
    model = mtq.quantize(_Pair().eval(), {"quant_cfg": NVFP4_DYNAMIC, "algorithm": None}, None)
    return model, {n: q for n, q in model.named_modules() if isinstance(q, TensorQuantizer)}


def _descriptor(algo):
    return type(CalibrateModeRegistry[BaseCalibrateModeDescriptor._get_mode_name(algo)])


def test_prepare_upgrades_only_the_quantizers_in_its_scope():
    """One ``QuantizerAttributeConfig`` serves every quantizer it matched, and the setter stores
    its ``block_sizes`` uncopied -- so an in-place upgrade would flip quantizers outside the
    scope, which never reach the ``reset_amax`` and would hold a dynamic grid's amax under a
    static one.
    """
    model, quantizers = _nvfp4_pair()
    assert (
        quantizers["a.weight_quantizer"].block_sizes is quantizers["b.weight_quantizer"].block_sizes
    )

    changed = _descriptor("local_hessian").prepare(model, ["a.weight_quantizer"], {})

    assert changed
    assert quantizers["a.weight_quantizer"].block_sizes["type"] == "static"
    assert quantizers["b.weight_quantizer"].block_sizes["type"] == "dynamic"


def test_prepare_refuses_to_downgrade_a_static_grid():
    model, quantizers = _nvfp4_pair()
    _descriptor("local_hessian").prepare(model, ["a.weight_quantizer"], {})

    with pytest.raises(ValueError, match="dynamic NVFP4 weight block scales"):
        _descriptor("awq_lite").prepare(model, ["a.weight_quantizer"], {})


def test_prepare_is_a_no_op_for_an_algorithm_with_no_grid_requirement():
    model, quantizers = _nvfp4_pair()
    assert not _descriptor("gptq").prepare(model, ["a.weight_quantizer"], {})
    assert quantizers["a.weight_quantizer"].block_sizes["type"] == "dynamic"
