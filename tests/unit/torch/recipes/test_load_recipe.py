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

"""Tests for end-to-end recipe loading (__init__.py)."""

import pytest

from modelopt.torch.recipes import load_recipe


def test_load_fp8(examples_dir):
    result = load_recipe(examples_dir / "ptq" / "ptq_fp8.yaml")
    assert "quantize_config" in result
    qcfg = result["quantize_config"]
    assert "quant_cfg" in qcfg
    assert "*weight_quantizer" in qcfg["quant_cfg"]


def test_load_nvfp4_awq(examples_dir):
    result = load_recipe(examples_dir / "ptq" / "ptq_nvfp4_awq.yaml")
    assert "quantize_config" in result
    assert "export" in result


def test_load_auto_quantize(examples_dir):
    result = load_recipe(examples_dir / "auto" / "auto_quantize.yaml")
    assert "auto_quantize_kwargs" in result
    kwargs = result["auto_quantize_kwargs"]
    assert "quantization_formats" in kwargs
    assert len(kwargs["quantization_formats"]) > 0


def test_load_all_examples(examples_dir):
    for yaml_file in sorted(examples_dir.rglob("*.yaml")):
        if "experiments" in yaml_file.parts:
            continue  # skip sweep/experiment configs
        result = load_recipe(yaml_file)
        assert isinstance(result, dict), f"Failed: {yaml_file.name}"
        assert len(result) > 0, f"Empty result: {yaml_file.name}"


def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_recipe("/nonexistent/path.yaml")
