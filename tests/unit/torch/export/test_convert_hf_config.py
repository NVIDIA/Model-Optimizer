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

import pytest

from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _revert_hf_quant_config_names


def test_convert_mixed_kv_cache_config_preserves_layer_map():
    layer_map = {
        "model.layers.0.self_attn": {"quant_algo": "FP8"},
        "model.layers.1.self_attn": {"quant_algo": "NVFP4"},
    }
    converted = convert_hf_quant_config_format(
        {
            "producer": {"name": "modelopt", "version": "test"},
            "quantization": {
                "kv_cache_quant_algo": "MIXED_PRECISION",
                "kv_cache_quantized_layers": layer_map,
                "kv_cache_schema_version": 1,
            },
        }
    )

    assert converted["quant_method"] == "modelopt"
    assert "quant_algo" not in converted
    assert "config_groups" not in converted
    assert converted["kv_cache_quant_algo"] == "MIXED_PRECISION"
    assert converted["kv_cache_quantized_layers"] == layer_map
    assert converted["kv_cache_schema_version"] == 1


def test_convert_uniform_kv_cache_config_preserves_layer_map():
    layer_map = {"model.layers.0.self_attn": {"quant_algo": "FP8"}}
    converted = convert_hf_quant_config_format(
        {
            "producer": {"name": "modelopt", "version": "test"},
            "quantization": {
                "kv_cache_quant_algo": "FP8",
                "kv_cache_quantized_layers": layer_map,
                "kv_cache_schema_version": 1,
            },
        }
    )

    assert "quant_algo" not in converted
    assert "config_groups" not in converted
    assert converted["kv_cache_scheme"] == {
        "dynamic": False,
        "num_bits": 8,
        "type": "float",
    }
    assert converted["kv_cache_quantized_layers"] == layer_map
    assert converted["kv_cache_schema_version"] == 1


def test_convert_uniform_asymmetric_kv_cache_preserves_semantic_algo():
    layer_map = {"model.layers.0.self_attn": {"quant_algo": "FP8_K_NVFP4_V"}}
    converted = convert_hf_quant_config_format(
        {
            "producer": {"name": "modelopt", "version": "test"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {},
                "kv_cache_quant_algo": "FP8_K_NVFP4_V",
                "kv_cache_quantized_layers": layer_map,
                "kv_cache_schema_version": 1,
            },
        }
    )

    assert converted["quant_algo"] == "MIXED_PRECISION"
    assert converted["kv_cache_quant_algo"] == "FP8_K_NVFP4_V"
    assert "kv_cache_scheme" not in converted
    assert converted["kv_cache_quantized_layers"] == layer_map


def test_reverse_quant_config_name_mapping_is_atomic():
    quant_config = {
        "quantization": {
            "exclude_modules": ["model.good"],
            "kv_cache_quantized_layers": {"model.bad": {"quant_algo": "FP8"}},
        }
    }

    def failing_mapper(name):
        if name == "model.bad":
            raise RuntimeError("unsupported mapping")
        return f"hub.{name}"

    with pytest.raises(RuntimeError, match="unsupported mapping"):
        _revert_hf_quant_config_names(quant_config, failing_mapper)

    assert quant_config == {
        "quantization": {
            "exclude_modules": ["model.good"],
            "kv_cache_quantized_layers": {"model.bad": {"quant_algo": "FP8"}},
        }
    }
