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

"""Focused CPU coverage for registry-based NVFP4 Conv3d HF export."""

import copy
import json

import pytest
import torch
import torch.nn as nn

pytest.importorskip("diffusers")

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.hf_export_handlers import (
    _export_quant_conv3d,
    _export_quantized_conv3d_weight,
)
from modelopt.torch.export.registry import ExportModuleRegistry
from modelopt.torch.export.unified_export_hf import _process_quantized_modules, export_hf_checkpoint
from modelopt.torch.quantization.qtensor import NVFP4QTensor

BLOCK_SIZE = 16


class WanCausalConv3d(nn.Conv3d):
    """Tiny stand-in that pins base-class/MRO dispatch for the Wan Conv3d type."""


class TinyWanVaeComponent(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 7,
        kernel_size: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = WanCausalConv3d(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            bias=True,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.conv(sample)


def _quantized_component(
    *,
    in_channels: int = 5,
    out_channels: int = 7,
    kernel_size: int = 1,
    groups: int = 1,
    quant_config: dict | None = None,
) -> TinyWanVaeComponent:
    model = TinyWanVaeComponent(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
    ).eval()

    def forward_loop(module):
        module(torch.randn(1, in_channels, 3, 4, 4))

    mtq.quantize(
        model,
        copy.deepcopy(quant_config or mtq.NVFP4_DEFAULT_CFG),
        forward_loop=forward_loop,
    )
    model.eval()
    return model


def _flatten_and_pad(weight: torch.Tensor) -> torch.Tensor:
    flattened = weight.reshape(weight.shape[0], -1)
    padded_k = ((flattened.shape[-1] + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    return torch.nn.functional.pad(flattened, (0, padded_k - flattened.shape[-1]))


def test_registry_matches_wan_conv3d_mro_only():
    model = _quantized_component()

    assert ExportModuleRegistry.match(model.conv) is _export_quant_conv3d
    plain_conv = nn.Conv3d(2, 2, 1)
    plain_conv.weight_quantizer = nn.Identity()
    assert ExportModuleRegistry.match(plain_conv) is None

    conv_transpose = nn.ConvTranspose3d(2, 2, 1)
    conv_transpose.weight_quantizer = nn.Identity()
    assert ExportModuleRegistry.match(conv_transpose) is None


def test_conv3d_export_preserves_calibrated_global_scale_and_packing():
    model = _quantized_component()
    original_weight = model.conv.weight.detach().clone()

    # Pin a calibrated global amax that is intentionally different from the
    # tensor absmax. Export must match the live TensorQuantizer convention.
    model.conv.weight_quantizer._amax = torch.tensor(0.5)
    expected_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(
        model.conv.weight_quantizer
    )
    flattened = _flatten_and_pad(original_weight)
    expected_packed, expected_scale, _ = NVFP4QTensor.quantize(
        flattened,
        BLOCK_SIZE,
        weights_scaling_factor_2=expected_scale_2,
    )

    _export_quantized_conv3d_weight(model.conv, torch.float32)

    assert model.conv.weight.dtype == torch.uint8
    assert model.conv.weight.shape == (7, BLOCK_SIZE // 2)
    assert torch.equal(model.conv.weight, expected_packed._quantized_data)
    assert torch.equal(model.conv.weight_scale, expected_scale)
    assert torch.equal(model.conv.weight_scale_2, expected_scale_2)
    assert model.conv.input_scale.dtype == torch.float32
    assert model.conv.input_scale.numel() == 1


@pytest.mark.parametrize(
    ("model_factory", "error"),
    [
        (
            lambda: _quantized_component(
                in_channels=4,
                out_channels=4,
                groups=2,
            ),
            "Grouped Conv3d",
        ),
        (
            lambda: _quantized_component(quant_config=mtq.W4A16_NVFP4_CFG),
            "input quantizer must be enabled",
        ),
    ],
)
def test_unsupported_conv3d_configs_fail_without_mutating_weight(model_factory, error):
    model = model_factory()
    original_weight = model.conv.weight.detach().clone()

    with pytest.raises(NotImplementedError, match=error):
        _process_quantized_modules(model, torch.float32)

    assert model.conv.weight.dtype.is_floating_point
    assert torch.equal(model.conv.weight, original_weight)
    assert not hasattr(model.conv, "weight_scale")
    assert not hasattr(model.conv, "weight_scale_2")


def test_static_block_conv3d_fails_without_mutating_weight():
    model = _quantized_component()
    original_weight = model.conv.weight.detach().clone()
    model.conv.weight_quantizer.block_sizes = {
        **model.conv.weight_quantizer.block_sizes,
        "type": "static",
    }

    with pytest.raises(NotImplementedError, match="dynamic block-16 NVFP4"):
        _process_quantized_modules(model, torch.float32)

    assert torch.equal(model.conv.weight, original_weight)
    assert not hasattr(model.conv, "weight_scale")


def test_partial_conv3d_quantization_fails_instead_of_silently_skipping():
    model = _quantized_component()
    original_weight = model.conv.weight.detach().clone()
    model.conv.weight_quantizer.disable()

    with pytest.raises(NotImplementedError, match="weight quantizer must be enabled"):
        _process_quantized_modules(model, torch.float32)

    assert torch.equal(model.conv.weight, original_weight)


def test_uncalibrated_conv3d_input_fails_without_mutating_weight():
    model = _quantized_component()
    original_weight = model.conv.weight.detach().clone()
    model.conv.input_quantizer._amax = None

    with pytest.raises(ValueError, match="input quantizer must be calibrated"):
        _process_quantized_modules(model, torch.float32)

    assert torch.equal(model.conv.weight, original_weight)
    assert not hasattr(model.conv, "weight_scale")


def test_fully_disabled_conv3d_remains_unpacked():
    model = _quantized_component()
    original_weight = model.conv.weight.detach().clone()
    for quantizer_name in ("weight_quantizer", "input_quantizer", "output_quantizer"):
        getattr(model.conv, quantizer_name).disable()

    _process_quantized_modules(model, torch.float32)

    assert torch.equal(model.conv.weight, original_weight)
    assert not hasattr(model.conv, "weight_scale")


def test_nvfp4_config_targets_are_opt_in_and_stably_sorted():
    quant_config = {"quantization": {"quant_algo": "NVFP4", "group_size": 16}}

    legacy = convert_hf_quant_config_format(quant_config)
    typed = convert_hf_quant_config_format(
        quant_config,
        target_types=["Linear", "Conv3d", "Linear"],
    )

    assert legacy["config_groups"]["group_0"]["targets"] == ["Linear"]
    assert typed["config_groups"]["group_0"]["targets"] == ["Conv3d", "Linear"]


def test_public_diffusers_export_writes_conv3d_target_and_schema(tmp_path):
    model = _quantized_component()
    export_hf_checkpoint(model, dtype=torch.float32, export_dir=tmp_path)

    with open(tmp_path / "config.json") as file:
        config = json.load(file)
    quant_config = config["quantization_config"]
    assert quant_config["quant_algo"] == "NVFP4"
    assert quant_config["config_groups"]["group_0"]["targets"] == ["Conv3d"]

    state_dict = {}
    for path in tmp_path.glob("*.safetensors"):
        state_dict.update(load_file(str(path)))

    assert not any("_quantizer" in key for key in state_dict)
    assert state_dict["conv.weight"].dtype == torch.uint8
    assert state_dict["conv.weight_scale"].dtype == torch.float8_e4m3fn
    assert state_dict["conv.weight_scale_2"].dtype == torch.float32
    assert state_dict["conv.input_scale"].dtype == torch.float32
    assert state_dict["conv.weight"].shape[1] == state_dict["conv.weight_scale"].shape[1] * 8


@pytest.mark.parametrize(
    "postprocess_kwargs", [{"padding_strategy": "row"}, {"enable_swizzle_layout": True}]
)
def test_public_diffusers_export_rejects_gemm_layout_options_for_conv3d(
    tmp_path,
    postprocess_kwargs,
):
    model = _quantized_component()
    original_weight = model.conv.weight.detach().clone()

    with pytest.raises(NotImplementedError, match="padding/swizzling"):
        export_hf_checkpoint(
            model,
            dtype=torch.float32,
            export_dir=tmp_path,
            **postprocess_kwargs,
        )

    assert torch.equal(model.conv.weight, original_weight)
