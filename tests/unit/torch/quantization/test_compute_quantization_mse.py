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

"""Unit tests for mtq.compute_quantization_mse()."""

import pytest
import torch
from _test_utils.torch.quantization.models import SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

INT8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
    },
    "algorithm": "max",
}


def _make_quantized_model():
    model = SimpleLinear()
    calib_data = [model.get_input() for _ in range(4)]

    def forward_loop(m):
        for batch in calib_data:
            m(batch)

    mtq.quantize(model, INT8_CFG, forward_loop)
    return model, forward_loop


class TestComputeQuantizationMse:
    def test_returns_nonnegative_values(self):
        """MSE values must be >= 0 for all quantizers."""
        model, forward_loop = _make_quantized_model()
        mse = mtq.compute_quantization_mse(model, forward_loop)
        assert len(mse) > 0
        assert all(v >= 0.0 for v in mse.values())

    def test_wildcard_star_covers_all_enabled_fake_quant(self):
        """Default wildcard '*' should return an entry for every enabled fake-quant quantizer."""
        model, forward_loop = _make_quantized_model()
        mse = mtq.compute_quantization_mse(model, forward_loop, wildcards="*")

        expected_names = {
            name
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer) and module._if_quant and module._fake_quant
        }
        assert set(mse.keys()) == expected_names

    def test_wildcard_filters_by_suffix(self):
        """A suffix pattern should restrict results to matching quantizer names."""
        model, forward_loop = _make_quantized_model()
        mse = mtq.compute_quantization_mse(model, forward_loop, wildcards="*weight_quantizer")
        assert len(mse) > 0
        assert all("weight_quantizer" in k for k in mse.keys())
        # No input quantizers should appear
        assert not any("input_quantizer" in k for k in mse.keys())

    def test_list_of_wildcards(self):
        """A list of patterns should return the union of matched quantizers."""
        model, forward_loop = _make_quantized_model()
        mse_weight = mtq.compute_quantization_mse(model, forward_loop, wildcards="*weight_quantizer")
        mse_input = mtq.compute_quantization_mse(model, forward_loop, wildcards="*input_quantizer")
        mse_both = mtq.compute_quantization_mse(
            model, forward_loop, wildcards=["*weight_quantizer", "*input_quantizer"]
        )
        assert set(mse_both.keys()) == set(mse_weight.keys()) | set(mse_input.keys())

    def test_callable_filter(self):
        """A callable wildcard should select quantizers by arbitrary predicate."""
        model, forward_loop = _make_quantized_model()
        # Pick only quantizers belonging to the first linear layer (net.0)
        mse = mtq.compute_quantization_mse(
            model, forward_loop, wildcards=lambda n: "net.0" in n
        )
        assert len(mse) > 0
        assert all("net.0" in k for k in mse.keys())

    def test_disabled_quantizer_absent_from_result(self):
        """A quantizer disabled after calibration must not appear in the output."""
        model, forward_loop = _make_quantized_model()

        # Disable one quantizer and record its name
        disabled_name = None
        for name, module in model.named_modules():
            if isinstance(module, TensorQuantizer) and module._if_quant and module._fake_quant:
                module.disable()
                disabled_name = name
                break

        assert disabled_name is not None, "No enabled quantizer found to disable"

        mse = mtq.compute_quantization_mse(model, forward_loop)
        assert disabled_name not in mse

    def test_no_matching_wildcard_returns_empty_dict(self):
        """A pattern that matches nothing should return an empty dict."""
        model, forward_loop = _make_quantized_model()
        mse = mtq.compute_quantization_mse(
            model, forward_loop, wildcards="*nonexistent_quantizer_xyz*"
        )
        assert mse == {}

    def test_does_not_modify_model_parameters(self):
        """Running MSE measurement must leave model weights unchanged."""
        model, forward_loop = _make_quantized_model()
        params_before = {k: v.clone() for k, v in model.named_parameters()}
        mtq.compute_quantization_mse(model, forward_loop)
        for k, v in model.named_parameters():
            assert torch.equal(v, params_before[k]), f"Parameter {k} was modified"

    def test_hooks_removed_after_call(self):
        """All forward hooks registered during the call must be cleaned up."""
        model, forward_loop = _make_quantized_model()

        hooks_before = sum(
            len(m._forward_hooks)
            for m in model.modules()
            if isinstance(m, TensorQuantizer)
        )
        mtq.compute_quantization_mse(model, forward_loop)
        hooks_after = sum(
            len(m._forward_hooks)
            for m in model.modules()
            if isinstance(m, TensorQuantizer)
        )
        assert hooks_after == hooks_before
