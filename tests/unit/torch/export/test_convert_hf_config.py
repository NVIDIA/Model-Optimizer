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

import json

import torch

from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _write_hf_export_config


def test_convert_mixed_kv_cache_config_preserves_layer_map():
    layer_map = {
        "model.layers.0.self_attn": {"quant_algo": "FP8"},
        "model.layers.1.self_attn": {"quant_algo": "NVFP4"},
    }
    converted = convert_hf_quant_config_format(
        {
            "producer": {"name": "modelopt", "version": "test"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {},
                "kv_cache_quant_algo": "MIXED_PRECISION",
                "kv_cache_quantized_layers": layer_map,
                "kv_cache_schema_version": 1,
            },
        }
    )

    assert converted["quant_method"] == "modelopt"
    assert converted["quant_algo"] == "MIXED_PRECISION"
    assert converted["config_groups"] == {}
    assert converted["kv_cache_quant_algo"] == "MIXED_PRECISION"
    assert converted["kv_cache_quantized_layers"] == layer_map
    assert converted["kv_cache_schema_version"] == 1


def test_write_hf_export_config_writes_mapped_kv_autoquant_report(tmp_path):
    layer_name = "model.layers.0.self_attn"
    model = torch.nn.Module()
    model._modelopt_kv_cache_auto_quantize_state = {
        "best": {"recipe": {layer_name: "fp8"}},
        "layers": {layer_name: {"selected": "fp8"}},
        "search_signature": {"layers": [{"name": layer_name}]},
    }
    quant_config = {
        "producer": {"name": "modelopt", "version": "test"},
        "quantization": {
            "quant_algo": None,
            "kv_cache_quant_algo": "MIXED_PRECISION",
            "kv_cache_quantized_layers": {layer_name: {"quant_algo": "FP8"}},
            "kv_cache_schema_version": 1,
        },
    }
    (tmp_path / "config.json").write_text("{}")

    _write_hf_export_config(
        model,
        quant_config,
        tmp_path,
        name_mapper=lambda name: f"hub.{name}",
    )

    report = json.loads((tmp_path / "kv_cache_auto_quantize_report.json").read_text())
    assert report["best"]["recipe"] == {f"hub.{layer_name}": "fp8"}
    assert report["layers"] == {f"hub.{layer_name}": {"selected": "fp8"}}
    assert report["search_signature"]["layers"] == [{"name": f"hub.{layer_name}"}]
    assert model._modelopt_kv_cache_auto_quantize_state["layers"] == {
        layer_name: {"selected": "fp8"}
    }
    assert model._modelopt_kv_cache_auto_quantize_state["best"]["recipe"] == {layer_name: "fp8"}
    assert (tmp_path / "hf_quant_config.json").is_file()
    exported_config = json.loads((tmp_path / "config.json").read_text())
    assert exported_config["quantization_config"]["kv_cache_quant_algo"] == "MIXED_PRECISION"
