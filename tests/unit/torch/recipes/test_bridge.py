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

"""Tests for recipe bridge (bridge.py)."""

from modelopt.torch.recipes.bridge import recipe_to_hf_ptq_args, summarize_recipe


def test_quantize_config_mapping():
    resolved = {
        "quantize_config": {
            "quant_cfg": {"*weight_quantizer": {"num_bits": (4, 3)}},
            "algorithm": "max",
        }
    }
    result = recipe_to_hf_ptq_args(resolved)
    assert "_resolved_quantize_config" in result


def test_auto_quantize_bridge():
    """Auto-quantize path in recipe_to_hf_ptq_args."""
    resolved = {"auto_quantize_kwargs": {"quantization_formats": [], "method": "gradient"}}
    result = recipe_to_hf_ptq_args(resolved)
    assert "_resolved_auto_quantize_kwargs" in result


def test_calibration_mapping():
    resolved = {
        "calibration": {
            "dataset": "cnn_dailymail",
            "num_samples": 512,
            "max_sequence_length": 2048,
            "batch_size": 4,
        }
    }
    result = recipe_to_hf_ptq_args(resolved)
    assert result["dataset"] == "cnn_dailymail"
    assert result["calib_size"] == 512
    assert result["seq_len"] == 2048
    assert result["batch_size"] == 4


def test_export_mapping():
    resolved = {
        "export": {
            "output_dir": "./my-output",
            "tensor_parallel": 4,
        }
    }
    result = recipe_to_hf_ptq_args(resolved)
    assert result["export_path"] == "./my-output"
    assert result["tp_size"] == 4


def test_summarize_quantize_recipe():
    raw_yaml = {
        "quantization": {
            "preset": "fp8",
            "algorithm": {"method": "awq_lite"},
            "kv_cache": {"format": "fp8"},
            "overrides": [{"pattern": "*layers.0*", "enable": False}],
        }
    }
    summary = summarize_recipe("test.yaml", {}, raw_yaml)
    assert summary["type"] == "quantize"
    assert summary["preset"] == "fp8"
    assert summary["algorithm"] == "awq_lite"
    assert summary["kv_cache"] == "fp8"
    assert summary["overrides_count"] == 1


def test_summarize_auto_quantize_recipe():
    raw_yaml = {
        "auto_quantize": {
            "effective_bits": 4.5,
            "method": "gradient",
            "formats": [{"preset": "fp8"}, {"preset": "nvfp4"}],
        }
    }
    summary = summarize_recipe("test.yaml", {}, raw_yaml)
    assert summary["type"] == "auto_quantize"
    assert summary["algorithm"] == "gradient"
    assert summary["formats_count"] == 2


def test_summarize_empty_recipe():
    """Summarize with no quantization or auto_quantize."""
    summary = summarize_recipe("test.yaml", {}, {})
    assert summary["type"] == "unknown"
