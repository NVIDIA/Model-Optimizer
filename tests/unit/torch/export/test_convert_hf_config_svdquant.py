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

"""Unit tests for the NVFP4_SVD (SVDQuant) HF quantization-config conversion."""

from modelopt.torch.export.convert_hf_config import (
    _quant_algo_to_group_config,
    convert_hf_quant_config_format,
)


def test_nvfp4_svd_group_config_mirrors_awq_with_pre_quant_scale():
    """The NVFP4_SVD config group is NVFP4 weights/activations + a pre_quant_scale flag."""
    group = _quant_algo_to_group_config("NVFP4_SVD", group_size=16)
    assert group["pre_quant_scale"] is True
    assert group["weights"] == {
        "dynamic": False,
        "num_bits": 4,
        "type": "float",
        "group_size": 16,
    }
    assert group["input_activations"]["num_bits"] == 4
    assert group["input_activations"]["type"] == "float"
    assert group["input_activations"]["group_size"] == 16


def test_convert_hf_quant_config_format_nvfp4_svd():
    """A full NVFP4_SVD quantization dict converts to a complete compressed-tensors config."""
    input_config = {
        "producer": {"name": "modelopt", "version": "0.0.0"},
        "quantization": {
            "quant_algo": "NVFP4_SVD",
            "group_size": 16,
            "has_zero_point": False,
            "pre_quant_scale": True,
            "lora_rank": 32,
            "exclude_modules": ["transformer_blocks.0.*", "proj_out"],
            "kv_cache_quant_algo": None,
        },
    }

    out = convert_hf_quant_config_format(input_config)

    # A real config group is emitted (not a bare {"quant_algo": ...} fallback).
    assert "config_groups" in out
    group = out["config_groups"]["group_0"]
    assert group["pre_quant_scale"] is True
    assert group["lora_rank"] == 32
    assert group["weights"]["num_bits"] == 4
    assert group["weights"]["type"] == "float"
    assert group["weights"]["group_size"] == 16
    assert group["input_activations"]["num_bits"] == 4
    assert group["targets"] == ["Linear"]

    # Top-level metadata is preserved.
    assert out["quant_algo"] == "NVFP4_SVD"
    assert out["ignore"] == ["transformer_blocks.0.*", "proj_out"]
    assert out["quant_method"] == "modelopt"


def test_convert_hf_quant_config_format_nvfp4_svd_without_rank():
    """lora_rank is optional; omitting it must not break the conversion."""
    input_config = {
        "quantization": {
            "quant_algo": "NVFP4_SVD",
            "group_size": 16,
            "pre_quant_scale": True,
        },
    }
    out = convert_hf_quant_config_format(input_config)
    group = out["config_groups"]["group_0"]
    assert "lora_rank" not in group
    assert group["pre_quant_scale"] is True
