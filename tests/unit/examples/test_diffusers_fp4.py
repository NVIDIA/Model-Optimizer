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

import importlib.util
import logging
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn

pytest.importorskip("onnx")
pytest.importorskip("onnx_graphsurgeon")
pytest.importorskip("diffusers")

import modelopt.torch.quantization as mtq
from examples.diffusers.quantization.onnx_utils import export as diffusion_export
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins.diffusion import diffusers as diffusers_plugin

_QUANTIZATION_EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples" / "diffusers" / "quantization"
)
_LOCAL_IMPORT_NAMES = (
    "calib.plugin_calib",
    "calib",
    "calibration",
    "config",
    "models_utils",
    "pipeline_manager",
    "quantize_config",
    "utils",
)


def _load_quantize_example():
    spec = importlib.util.spec_from_file_location(
        "diffusers_quantize_example", _QUANTIZATION_EXAMPLE / "quantize.py"
    )
    assert spec is not None and spec.loader is not None

    original_modules = {
        name: sys.modules.pop(name) for name in _LOCAL_IMPORT_NAMES if name in sys.modules
    }
    sys.path.insert(0, str(_QUANTIZATION_EXAMPLE))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        for name in _LOCAL_IMPORT_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)
    return module


_quantize = _load_quantize_example()
ModelType = _quantize.ModelType
ModelConfig = _quantize.ModelConfig
QuantFormat = _quantize.QuantFormat
QuantizationConfig = _quantize.QuantizationConfig
Quantizer = _quantize.Quantizer
_infer_restored_quantization_format = _quantize._infer_restored_quantization_format


class _RecipeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=False)
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(16, 16, bias=False)
        self.attn.to_k = nn.Linear(16, 16, bias=False)
        self.attn.to_v = nn.Linear(16, 16, bias=False)
        self.conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)


def _quantizer(*, num_bits, enabled=True, block_sizes=None):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=num_bits, axis=None, block_sizes=block_sizes)
    )
    quantizer.amax = torch.tensor(448.0)
    if not enabled:
        quantizer.disable()
    return quantizer


_FP8_QUANTIZER_CONFIG = {"num_bits": (4, 3)}
_NVFP4_QUANTIZER_CONFIG = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
}


@pytest.mark.parametrize("model_type", [ModelType.SDXL_BASE, ModelType.SDXL_TURBO])
def test_sdxl_fp4_recipe(model_type):
    model = _RecipeBackbone()
    config = Quantizer(
        QuantizationConfig(format=QuantFormat.FP4),
        ModelConfig(model_type=model_type),
        logging.getLogger(__name__),
    ).get_quant_config(n_steps=1, backbone=model)

    mtq.replace_quant_module(model)
    mtq.set_quantizer_by_cfg(model, config["quant_cfg"])

    for quantizer in (model.linear.input_quantizer, model.linear.weight_quantizer):
        assert quantizer.is_enabled
        assert quantizer.is_nvfp4_dynamic
        assert quantizer.block_sizes[-1] == 16
    for projection in (model.attn.to_q, model.attn.to_k, model.attn.to_v):
        assert not projection.input_quantizer.is_enabled
        assert not projection.weight_quantizer.is_enabled
    for quantizer in (model.conv.input_quantizer, model.conv.weight_quantizer):
        assert quantizer.is_enabled
        assert quantizer.is_fp8


@pytest.mark.parametrize(
    ("format_config", "mha_config", "expected_format", "disable_fp8_mha"),
    [
        pytest.param(
            _NVFP4_QUANTIZER_CONFIG,
            _FP8_QUANTIZER_CONFIG,
            QuantFormat.FP4,
            False,
            id="mixed-fp4",
        ),
        pytest.param(
            _FP8_QUANTIZER_CONFIG,
            _FP8_QUANTIZER_CONFIG,
            QuantFormat.FP8,
            False,
            id="fp8",
        ),
        pytest.param(
            {"num_bits": 8},
            {**_FP8_QUANTIZER_CONFIG, "enabled": False},
            QuantFormat.INT8,
            True,
            id="int8-disabled-fp8",
        ),
        pytest.param(
            {"num_bits": 8},
            {"num_bits": 8},
            QuantFormat.INT8,
            True,
            id="int8-mha",
        ),
    ],
)
def test_restored_quantizer_state_drives_format_and_fp8_mha(
    monkeypatch, format_config, mha_config, expected_format, disable_fp8_mha
):
    backbone = nn.Module()
    backbone.quantizer = _quantizer(**format_config)
    backbone.attention = nn.Module()
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
        setattr(backbone.attention, name, _quantizer(**mha_config))
    backbone.attention.bmm2_output_quantizer = lambda output: output

    fp8_sdpa = Mock(return_value=torch.empty(0))
    monkeypatch.setattr(diffusers_plugin.FP8SDPA, "apply", fp8_sdpa)
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)

    assert _infer_restored_quantization_format([("transformer", backbone)]) == expected_format
    diffusers_plugin._quantized_sdpa(backbone.attention, *(torch.empty(1) for _ in range(3)))
    assert fp8_sdpa.call_args.args[-1] is disable_fp8_mha


def test_restore_infers_checkpoint_format_for_export(monkeypatch, tmp_path):
    backbone = nn.Module()
    backbone.quantizer = _quantizer(**_NVFP4_QUANTIZER_CONFIG)

    pipeline_manager = Mock()
    pipeline_manager.create_pipeline.return_value = object()
    pipeline_manager.iter_backbones.return_value = [("transformer", backbone)]
    export_manager = Mock()
    monkeypatch.setattr(_quantize, "PipelineManager", lambda *args: pipeline_manager)
    monkeypatch.setattr(_quantize, "ExportManager", lambda *args: export_manager)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize.py",
            "--model",
            "flux-schnell",
            "--restore-from",
            str(tmp_path),
            "--onnx-dir",
            str(tmp_path / "onnx"),
        ],
    )

    _quantize.main()

    export_manager.restore_checkpoint.assert_called_once_with()
    assert export_manager.export_onnx.call_args.args[-1] == QuantFormat.FP4
    export_manager.export_hf_ckpt.assert_called_once()


def test_flux_fp8_export_saves_converted_rope_graph(monkeypatch, tmp_path):
    original_model = Mock()
    converted_model = Mock()
    monkeypatch.setattr(
        diffusion_export,
        "generate_dummy_kwargs_and_dynamic_axes_and_shapes",
        lambda *args: ({}, {}, None),
    )
    monkeypatch.setattr(diffusion_export, "onnx_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(diffusion_export.onnx, "load", lambda *args, **kwargs: original_model)
    convert_rope_weight_type = Mock(return_value=converted_model)
    monkeypatch.setattr(diffusion_export, "flux_convert_rope_weight_type", convert_rope_weight_type)
    save_onnx = Mock()
    monkeypatch.setattr(diffusion_export, "save_onnx", save_onnx)

    diffusion_export.modelopt_export_sd(nn.Module(), tmp_path, "flux-dev", "fp8")

    convert_rope_weight_type.assert_called_once_with(original_model)
    save_onnx.assert_called_once_with(converted_model, tmp_path / "model.onnx")
