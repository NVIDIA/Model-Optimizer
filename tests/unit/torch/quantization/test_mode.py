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

from modelopt.torch.opt.config import ModeloptField
from modelopt.torch.opt.mode import _ModeRegistryCls
from modelopt.torch.quantization.config import (
    MaxCalibConfig,
    QuantizeAlgorithmConfig,
    SmoothQuantCalibConfig,
)
from modelopt.torch.quantization.mode import (
    BaseCalibrateModeDescriptor,
    CalibrateModeRegistry,
    QuantizeModeRegistry,
    get_modelike_from_algo_cfg,
)


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


def test_get_modelike_from_algo_cfg_accepts_config_instance():
    """Regression test for GitHub issue #201.

    A ``QuantizeAlgorithmConfig`` instance passed as the ``algorithm`` config is
    normalized to a dict and accepted, instead of raising ``ValueError``.
    """
    mode_name, algo_cfg = get_modelike_from_algo_cfg(MaxCalibConfig(distributed_sync=False))[0]
    assert mode_name == "max_calibrate"
    assert isinstance(algo_cfg, dict)
    assert algo_cfg["distributed_sync"] is False
    MaxCalibConfig(**algo_cfg)

    modes = get_modelike_from_algo_cfg([MaxCalibConfig(), SmoothQuantCalibConfig()])
    assert [name for name, _ in modes] == ["max_calibrate", "smoothquant_calibrate"]
    assert all(isinstance(cfg, dict) for _, cfg in modes)
