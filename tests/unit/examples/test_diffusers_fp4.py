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

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnx_graphsurgeon")
pytest.importorskip("diffusers")

import modelopt.torch.quantization as mtq
from examples.diffusers.quantization.onnx_utils import export as diffusion_export
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer

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
_restore_quantization_policy = _quantize._restore_quantization_policy


class _RecipeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=False)
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(16, 16, bias=False)
        self.attn.to_k = nn.Linear(16, 16, bias=False)
        self.attn.to_v = nn.Linear(16, 16, bias=False)
        self.conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)


def _quantizer(*, num_bits=(4, 3), enabled=True, calibrated=True, block_sizes=None):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=num_bits, axis=None, block_sizes=block_sizes)
    )
    if calibrated:
        quantizer.amax = torch.tensor(448.0)
    if not enabled:
        quantizer.disable()
    return quantizer


def _add_quantizers(module, **kwargs):
    module.input_quantizer = _quantizer(**kwargs)
    module.weight_quantizer = _quantizer(**kwargs)


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
    ("scenario", "expected_format", "mha_enabled"),
    [
        ("mixed-fp4", QuantFormat.FP4, True),
        ("fp8", QuantFormat.FP8, True),
        ("int8-disabled-fp8-mha", QuantFormat.INT8, False),
    ],
)
def test_restore_policy_uses_enabled_checkpoint_state(scenario, expected_format, mha_enabled):
    backbone = nn.Module()
    backbone.linear = nn.Linear(16, 16)
    if scenario == "mixed-fp4":
        _add_quantizers(
            backbone.linear,
            num_bits=(2, 1),
            block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        )
        backbone.conv = nn.Conv2d(1, 1, 1)
        _add_quantizers(backbone.conv)
    elif scenario == "fp8":
        _add_quantizers(backbone.linear)
    else:
        _add_quantizers(backbone.linear, num_bits=8)

    backbone.attention = nn.Module()
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
        setattr(backbone.attention, name, _quantizer(enabled=mha_enabled))
    quantizers = [module for module in backbone.modules() if isinstance(module, TensorQuantizer)]
    state = {q: (q.is_enabled, q._num_bits, q._amax) for q in quantizers}

    restored_format = _restore_quantization_policy([("transformer", backbone)])

    assert restored_format == expected_format
    assert backbone.attention._disable_fp8_mha is not mha_enabled
    for quantizer, (enabled, num_bits, amax) in state.items():
        assert quantizer.is_enabled is enabled
        assert quantizer._num_bits == num_bits
        assert quantizer._amax is amax


def test_restore_defaults_reach_exports_with_checkpoint_policy(monkeypatch, tmp_path):
    backbone = nn.Module()
    backbone.linear = nn.Linear(16, 16)
    _add_quantizers(
        backbone.linear,
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    )
    backbone.attention = nn.Module()
    mha_quantizers = []
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
        quantizer = _quantizer()
        setattr(backbone.attention, name, quantizer)
        mha_quantizers.append(quantizer)

    pipeline_manager = Mock()
    pipeline_manager.create_pipeline.return_value = object()
    pipeline_manager.iter_backbones.side_effect = lambda: iter([("transformer", backbone)])
    export_manager = Mock()
    monkeypatch.setattr(_quantize, "PipelineManager", lambda *args: pipeline_manager)
    monkeypatch.setattr(_quantize, "ExportManager", lambda *args: export_manager)
    monkeypatch.setattr(torch.nn, "RMSNorm", torch.nn.RMSNorm)
    monkeypatch.setattr(torch.nn.modules.normalization, "RMSNorm", torch.nn.RMSNorm)
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
    assert all(quantizer.is_enabled for quantizer in mha_quantizers)
    assert backbone.attention._disable_fp8_mha is False


def test_temporary_fp8_export_scales_filter_and_restore_on_error():
    model = nn.Module()
    modules = {
        "linear": nn.Linear(1, 1),
        "conv1d": nn.Conv1d(1, 1, 1),
        "conv2d": nn.Conv2d(1, 1, 1),
        "conv3d": nn.Conv3d(1, 1, 1),
        "disabled": nn.Conv2d(1, 1, 1),
        "uncalibrated": nn.Linear(1, 1),
        "int8": nn.Conv2d(1, 1, 1),
    }
    for name, module in modules.items():
        setattr(model, name, module)
        _add_quantizers(
            module,
            enabled=name != "disabled",
            calibrated=name != "uncalibrated",
            num_bits=8 if name == "int8" else (4, 3),
        )
    quantizers = [module for module in model.modules() if isinstance(module, TensorQuantizer)]
    state = {q: (q._num_bits, getattr(q, "_amax", None)) for q in quantizers}

    for conv_only, changed_names in (
        (True, {"conv2d"}),
        (False, {"linear", "conv1d", "conv2d", "conv3d"}),
    ):
        with (
            pytest.raises(RuntimeError, match="export failed"),
            diffusion_export._temporary_fp8_export_scales(model, conv_only=conv_only),
        ):
            for name, module in modules.items():
                for quantizer in (module.input_quantizer, module.weight_quantizer):
                    if name in changed_names:
                        assert quantizer.num_bits == 8
                        assert quantizer.amax == 127.0
                    else:
                        assert (
                            quantizer._num_bits,
                            getattr(quantizer, "_amax", None),
                        ) == state[quantizer]
            raise RuntimeError("export failed")

        for quantizer, (num_bits, amax) in state.items():
            assert quantizer._num_bits == num_bits
            assert getattr(quantizer, "_amax", None) is amax


def test_flux_export_saves_converted_rope_model(monkeypatch, tmp_path):
    original_model = object()
    converted_model = object()
    monkeypatch.setattr(
        diffusion_export,
        "generate_dummy_kwargs_and_dynamic_axes_and_shapes",
        lambda *args: ({}, {}, None),
    )
    monkeypatch.setattr(diffusion_export, "onnx_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(diffusion_export.onnx, "load", lambda *args, **kwargs: original_model)
    monkeypatch.setattr(
        diffusion_export, "flux_convert_rope_weight_type", lambda model: converted_model
    )
    save_onnx = Mock()
    monkeypatch.setattr(diffusion_export, "save_onnx", save_onnx)

    diffusion_export.modelopt_export_sd(nn.Module(), tmp_path, "flux-dev", "fp8")

    save_onnx.assert_called_once_with(converted_model, tmp_path / "model.onnx")


@pytest.mark.parametrize("model_name", ["sdxl-1.0", "sdxl-turbo"])
def test_sdxl_fp4_processing_order_and_opset(monkeypatch, model_name):
    model = onnx.ModelProto()
    model.opset_import.add(domain="", version=20)
    calls = []

    def record(name):
        def process(current_model):
            calls.append(name)
            return current_model

        return process

    monkeypatch.setattr(diffusion_export, "_normalize_fp8_qdq", record("fp8"))
    monkeypatch.setattr(diffusion_export.NVFP4QuantExporter, "process_model", record("nvfp4"))

    assert diffusion_export._process_fp4_onnx_graph(model, model_name) is model
    assert calls == ["fp8", "nvfp4"]
    assert model.opset_import[0].version == 23
