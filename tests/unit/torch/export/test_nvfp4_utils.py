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

"""Tests for NVFP4 utility functions (pad, swizzle, metadata) and _postprocess_safetensors."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from modelopt.torch.export.diffusers_utils import (
    build_layerwise_quant_metadata,
    pad_nvfp4_weights,
    swizzle_nvfp4_scales,
)
from modelopt.torch.export.unified_export_hf import _postprocess_safetensors


def _make_nvfp4_state_dict(rows=32, cols=64):
    """Create a minimal NVFP4 state dict with one quantized layer."""
    return {
        "layer0.weight": torch.randint(0, 255, (rows, cols), dtype=torch.uint8),
        "layer0.weight_scale": torch.randn(rows, cols // 16).to(torch.float8_e4m3fn),
        "layer0.weight_scale_2": torch.randn(rows, 1),
        "layer0.bias": torch.randn(rows),
    }


# ---------------------------------------------------------------------------
# _find_nvfp4_layers (tested implicitly via pad / swizzle that rely on it)
# ---------------------------------------------------------------------------


class TestBuildLayerwiseQuantMetadata:
    def test_basic(self):
        sd = _make_nvfp4_state_dict()
        cfg = {"quant_algo": "NVFP4"}
        result = json.loads(build_layerwise_quant_metadata(sd, cfg))

        assert result["format_version"] == "1.0"
        assert "layer0" in result["layers"]
        assert result["layers"]["layer0"]["format"] == "nvfp4"

    def test_no_quantized_layers(self):
        sd = {"linear.weight": torch.randn(4, 4), "linear.bias": torch.randn(4)}
        result = json.loads(build_layerwise_quant_metadata(sd, {"quant_algo": "FP8"}))
        assert result["layers"] == {}

    def test_multiple_layers(self):
        sd = {**_make_nvfp4_state_dict()}
        sd["layer1.weight"] = torch.randint(0, 255, (16, 32), dtype=torch.uint8)
        sd["layer1.weight_scale"] = torch.randn(16, 2).to(torch.float8_e4m3fn)
        sd["layer1.weight_scale_2"] = torch.randn(16, 1)

        result = json.loads(build_layerwise_quant_metadata(sd, {"quant_algo": "NVFP4"}))
        assert "layer0" in result["layers"]
        assert "layer1" in result["layers"]


class TestPadNvfp4Weights:
    def test_row_padding(self):
        sd = _make_nvfp4_state_dict(rows=20, cols=64)
        result = pad_nvfp4_weights(sd, "row")

        assert result["layer0.weight"].shape[0] % 16 == 0
        assert result["layer0.weight_scale"].shape[0] % 16 == 0
        assert result["layer0.weight"].shape[0] == 32

    def test_row_col_padding(self):
        sd = _make_nvfp4_state_dict(rows=20, cols=48)
        result = pad_nvfp4_weights(sd, "row_col")

        w = result["layer0.weight"]
        s = result["layer0.weight_scale"]
        assert w.shape[0] % 16 == 0
        assert w.shape[1] % 16 == 0
        assert s.shape[0] % 16 == 0
        assert s.shape[1] % 16 == 0

    def test_row_col_padding_scale_only(self):
        sd = _make_nvfp4_state_dict(rows=32, cols=48)
        result = pad_nvfp4_weights(sd, "row_col")

        assert result["layer0.weight"].shape == (32, 48)
        assert result["layer0.weight_scale"].shape == (32, 16)

    def test_already_aligned(self):
        sd = _make_nvfp4_state_dict(rows=32, cols=64)
        orig_w_shape = sd["layer0.weight"].shape
        result = pad_nvfp4_weights(sd, "row")

        assert result["layer0.weight"].shape == orig_w_shape

    def test_invalid_strategy(self):
        sd = _make_nvfp4_state_dict()
        with pytest.raises(ValueError, match="padding_strategy"):
            pad_nvfp4_weights(sd, "invalid")

    def test_non_nvfp4_tensors_untouched(self):
        sd = _make_nvfp4_state_dict(rows=20, cols=64)
        bias_before = sd["layer0.bias"].clone()
        pad_nvfp4_weights(sd, "row")
        assert torch.equal(sd["layer0.bias"], bias_before)


class TestSwizzleNvfp4Scales:
    def test_shape_preserved(self):
        sd = _make_nvfp4_state_dict(rows=128, cols=64)
        orig_shape = sd["layer0.weight_scale"].shape
        result = swizzle_nvfp4_scales(sd)

        assert result["layer0.weight_scale"].shape == orig_shape

    def test_dtype_is_fp8(self):
        sd = _make_nvfp4_state_dict(rows=128, cols=64)
        result = swizzle_nvfp4_scales(sd)

        assert result["layer0.weight_scale"].dtype == torch.float8_e4m3fn

    def test_non_nvfp4_tensors_untouched(self):
        sd = _make_nvfp4_state_dict(rows=128, cols=64)
        bias_before = sd["layer0.bias"].clone()
        swizzle_nvfp4_scales(sd)
        assert torch.equal(sd["layer0.bias"], bias_before)

    def test_small_scale_needs_internal_padding(self):
        """Scales with rows < 128 trigger internal padding in _to_blocked."""
        sd = _make_nvfp4_state_dict(rows=16, cols=64)
        result = swizzle_nvfp4_scales(sd)
        # _to_blocked pads rows up to the next multiple of 128
        assert result["layer0.weight_scale"].shape == (128, 64 // 16)


class TestPostprocessSafetensors:
    def test_metadata_injection(self, tmp_path):
        sd = {"weight": torch.randn(4, 4)}
        save_file(sd, str(tmp_path / "model.safetensors"))

        hf_quant_config = {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}
        _postprocess_safetensors(
            tmp_path,
            hf_quant_config=hf_quant_config,
            enable_layerwise_quant_metadata=True,
        )

        reloaded = load_file(str(tmp_path / "model.safetensors"))
        assert torch.allclose(reloaded["weight"], sd["weight"])
        with safe_open(str(tmp_path / "model.safetensors"), framework="pt", device="cpu") as f:
            metadata = f.metadata()
        assert json.loads(metadata["quantization_config"]) == hf_quant_config
        assert json.loads(metadata["_quantization_metadata"]) == {
            "format_version": "1.0",
            "layers": {},
        }

    def test_padding_and_swizzle(self, tmp_path):
        sd = _make_nvfp4_state_dict(rows=20, cols=64)
        save_file(sd, str(tmp_path / "model.safetensors"))

        _postprocess_safetensors(
            tmp_path,
            padding_strategy="row",
            enable_swizzle_layout=True,
            enable_layerwise_quant_metadata=False,
        )

        reloaded = load_file(str(tmp_path / "model.safetensors"))
        assert reloaded["layer0.weight"].shape[0] == 32
        assert reloaded["layer0.weight_scale"].dtype == torch.float8_e4m3fn
        assert reloaded["layer0.weight_scale"].shape == (128, 64 // 16)

    def test_sharded_guard(self, tmp_path):
        save_file({"w": torch.randn(2, 2)}, str(tmp_path / "model.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text("{}")

        with pytest.raises(NotImplementedError, match="sharded"):
            _postprocess_safetensors(
                tmp_path,
                merged_base_safetensor_path="/fake/path.safetensors",
                enable_layerwise_quant_metadata=True,
            )

    def test_preserves_existing_metadata(self, tmp_path):
        """Simulate save_pretrained output: safetensors with pre-existing metadata."""
        sd = _make_nvfp4_state_dict(rows=20, cols=64)
        preexisting_metadata = {"format": "pt", "_class_name": "MyModel"}
        save_file(sd, str(tmp_path / "model.safetensors"), metadata=preexisting_metadata)

        hf_quant_config = {"quant_algo": "NVFP4"}
        _postprocess_safetensors(
            tmp_path,
            hf_quant_config=hf_quant_config,
            padding_strategy="row",
            enable_swizzle_layout=True,
            enable_layerwise_quant_metadata=True,
        )

        reloaded = load_file(str(tmp_path / "model.safetensors"))
        assert reloaded["layer0.weight"].shape[0] == 32
        assert reloaded["layer0.weight_scale"].shape == (128, 64 // 16)

        with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as f:
            metadata = f.metadata()
        assert metadata["format"] == "pt"
        assert metadata["_class_name"] == "MyModel"
        assert json.loads(metadata["quantization_config"]) == hf_quant_config
        layer_meta = json.loads(metadata["_quantization_metadata"])
        assert "layer0" in layer_meta["layers"]

    def test_no_safetensor_files(self, tmp_path):
        _postprocess_safetensors(tmp_path)

    def test_unknown_kwargs_silently_ignored(self, tmp_path):
        sd = {"weight": torch.randn(4, 4)}
        save_file(sd, str(tmp_path / "model.safetensors"))

        _postprocess_safetensors(tmp_path, bad_option=True)

        reloaded = load_file(str(tmp_path / "model.safetensors"))
        assert torch.allclose(reloaded["weight"], sd["weight"])

    def test_merge_requires_pipe(self, tmp_path):
        save_file({"w": torch.randn(2, 2)}, str(tmp_path / "model.safetensors"))

        with pytest.raises(ValueError, match="`pipe` must be provided"):
            _postprocess_safetensors(
                tmp_path,
                merged_base_safetensor_path="/fake/path.safetensors",
            )
