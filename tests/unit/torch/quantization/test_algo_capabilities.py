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

"""Tests for the per-algorithm capability declarations."""

from typing import Literal

import pytest
from pydantic import ValidationError

from modelopt.torch.quantization.algo_cfg import ACTS, WEIGHT, WRITABLE_TOKENS, capabilities_for
from modelopt.torch.quantization.config import QuantizeAlgorithmConfig
from modelopt.torch.quantization.mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry


def _known_algorithms():
    names = getattr(CalibrateModeRegistry, "_name2descriptor", {})
    return sorted(
        n.removesuffix("_calibrate")
        for n in names
        if n.endswith("_calibrate") and not n.startswith("_")
    )


def test_every_registered_algorithm_declares_capabilities():
    for algo in _known_algorithms():
        assert capabilities_for(algo) is not None, algo


def test_a_custom_algorithm_inherits_conservative_capabilities():
    class _CustomConfig(QuantizeAlgorithmConfig):
        method: Literal["my_custom_algo"] = "my_custom_algo"

    @CalibrateModeRegistry.register_mode
    class _CustomDescriptor(BaseCalibrateModeDescriptor):
        _calib_func = None

        @property
        def config_class(self):
            return _CustomConfig

    try:
        caps = capabilities_for("my_custom_algo")
        assert caps is not None, "a registered algorithm must have capabilities"
        assert caps.may_write == WRITABLE_TOKENS, "assume it writes everything"
        assert not caps.scopable, "assume it cannot be scoped"
    finally:
        CalibrateModeRegistry.remove_mode("my_custom_algo_calibrate")


def test_calib_mutates_weights_false_is_rejected_for_every_weight_writing_algorithm():
    checked = []
    for algo in _known_algorithms():
        if WEIGHT not in capabilities_for(algo).may_write:
            continue
        config_class = CalibrateModeRegistry[
            BaseCalibrateModeDescriptor._get_mode_name(algo)
        ].config_class
        with pytest.raises(ValidationError, match="mutates layer weights in-place"):
            config_class(layerwise={"enable": True, "calib_mutates_weights": False})
        checked.append(algo)

    assert checked, "no weight-writing algorithm found -- the check would pass vacuously"


@pytest.mark.parametrize(
    ("algo", "key"),
    [("lsq", "scale_algorithm"), ("nvfp4_act_headroom", "weight_scale_algorithm")],
)
def test_a_delegating_algorithm_inherits_its_sub_algorithms_capabilities(algo, key):
    # `local_hessian` reads activations; the delegating algorithm must say so on its behalf.
    # (`nvfp4_act_headroom` needs activations for its own work, so it declares ACTS either way.)
    assert ACTS in capabilities_for(algo, {key: {"method": "local_hessian"}}).requires


def test_lsq_only_reads_activations_when_its_sub_algorithm_does():
    assert ACTS not in capabilities_for("lsq", {"scale_algorithm": {"method": "max"}}).requires
    assert (
        ACTS in capabilities_for("lsq", {"scale_algorithm": {"method": "local_hessian"}}).requires
    )


def test_a_delegating_algorithm_falls_back_to_the_conservative_upper_bound():
    caps = capabilities_for("lsq", {"scale_algorithm": {"method": "not_an_algorithm"}})
    assert caps.may_write >= WRITABLE_TOKENS
