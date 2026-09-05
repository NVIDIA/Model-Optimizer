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

import logging
import sys
from pathlib import Path

import pytest
import torch
from diffusers.models.attention_processor import Attention
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer

_QUANTIZATION_EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples" / "diffusers" / "quantization"
)
sys.path.insert(0, str(_QUANTIZATION_EXAMPLE))

import quantize as quantize_module
from models_utils import ModelType
from quantize import ExportManager, Quantizer, _finalize_backbone_quantization
from quantize_config import ExportConfig, ModelConfig, QuantFormat, QuantizationConfig
from utils import check_conv_and_mha, validate_nvfp4_quantizers

from examples.diffusers.quantization import onnx_utils as diffusion_onnx_utils
from examples.diffusers.quantization.onnx_utils import export as diffusion_export

_MHA_QUANTIZER_NAMES = (
    "q_bmm_quantizer",
    "k_bmm_quantizer",
    "v_bmm_quantizer",
    "softmax_quantizer",
    "bmm2_output_quantizer",
)


def _nvfp4_quantizer(block_size=16, enabled=True, amax=1.0):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: block_size, "type": "dynamic", "scale_bits": (4, 3)},
        )
    )
    if amax is not None:
        quantizer.amax = torch.tensor(amax)
    if not enabled:
        quantizer.disable()
    return quantizer


def _fp8_quantizer(enabled=True, amax=1.0):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=None))
    if amax is not None:
        quantizer.amax = torch.tensor(amax)
    if not enabled:
        quantizer.disable()
    return quantizer


class _QuantizedLinear(nn.Linear):
    def __init__(self, input_quantizer=None, weight_quantizer=None):
        super().__init__(16, 16, bias=False)
        self.input_quantizer = input_quantizer or _nvfp4_quantizer()
        self.weight_quantizer = weight_quantizer or _nvfp4_quantizer()


class _Backbone(nn.Module):
    def __init__(self, linear=None, conv=None, attention=None):
        super().__init__()
        self.linear = linear or _QuantizedLinear()
        if conv is not None:
            self.conv = conv
        if attention is not None:
            self.attention = attention


def _quantized_conv(conv_cls, quantizer_factory):
    conv = conv_cls(4, 4, kernel_size=1, bias=False)
    conv.input_quantizer = quantizer_factory()
    conv.weight_quantizer = quantizer_factory()
    return conv


def _attention(head_size=16, enabled=True, softmax_amax=1.0):
    attention = Attention(query_dim=head_size, heads=1, dim_head=head_size)
    for name in _MHA_QUANTIZER_NAMES:
        quantizer_enabled = enabled and name != "bmm2_output_quantizer"
        amax = softmax_amax if name == "softmax_quantizer" else 1.0
        setattr(attention, name, _fp8_quantizer(enabled=quantizer_enabled, amax=amax))
    return attention


@pytest.mark.parametrize("model_type", [ModelType.SDXL_BASE, ModelType.SDXL_TURBO])
def test_sdxl_fp4_uses_nvfp4_linear_fp8_conv_recipe(model_type):
    config = Quantizer(
        QuantizationConfig(format=QuantFormat.FP4, block_size=32),
        ModelConfig(model_type=model_type),
        logging.getLogger(__name__),
    ).get_quant_config(n_steps=1, backbone=nn.Module())

    linear_entries = {
        entry["quantizer_name"]: entry["cfg"]
        for entry in config["quant_cfg"]
        if entry.get("parent_class") == "nn.Linear"
        and entry.get("quantizer_name") in ("*input_quantizer", "*weight_quantizer")
    }
    conv_entries = {
        entry["quantizer_name"]: entry["cfg"]
        for entry in config["quant_cfg"]
        if entry.get("parent_class") == "nn.Conv2d"
    }

    assert set(linear_entries) == {"*input_quantizer", "*weight_quantizer"}
    assert set(conv_entries) == {"*input_quantizer", "*weight_quantizer"}
    for cfg in linear_entries.values():
        assert cfg["num_bits"] == (2, 1)
        assert cfg["block_sizes"][-1] == 32
        assert cfg["block_sizes"]["scale_bits"] == (4, 3)
    for cfg in conv_entries.values():
        assert cfg["num_bits"] == (4, 3)
        assert cfg["axis"] is None
        assert "block_sizes" not in cfg


def test_sdxl_fp4_recipe_applies_only_to_linear_and_conv():
    model = nn.Module()
    model.linear = nn.Linear(16, 16, bias=False)
    model.attn = nn.Module()
    model.attn.to_q = nn.Linear(16, 16, bias=False)
    model.attn.to_k = nn.Linear(16, 16, bias=False)
    model.attn.to_v = nn.Linear(16, 16, bias=False)
    model.conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)
    model.norm = nn.LayerNorm(16)
    config = Quantizer(
        QuantizationConfig(format=QuantFormat.FP4),
        ModelConfig(model_type=ModelType.SDXL_BASE),
        logging.getLogger(__name__),
    ).get_quant_config(n_steps=1, backbone=model)

    mtq.replace_quant_module(model)
    mtq.set_quantizer_by_cfg(model, config["quant_cfg"])

    assert model.linear.input_quantizer.is_enabled
    assert model.linear.input_quantizer.is_nvfp4_dynamic
    assert model.linear.weight_quantizer.is_enabled
    assert model.linear.weight_quantizer.is_nvfp4_dynamic
    for projection in (model.attn.to_q, model.attn.to_k, model.attn.to_v):
        assert not projection.input_quantizer.is_enabled
        assert not projection.weight_quantizer.is_enabled
    assert model.conv.input_quantizer.is_enabled
    assert model.conv.input_quantizer.is_fp8
    assert model.conv.weight_quantizer.is_enabled
    assert model.conv.weight_quantizer.is_fp8
    assert not model.norm.input_quantizer.is_enabled


def test_fp4_finalization_preserves_calibrated_fp8_conv():
    conv = _quantized_conv(nn.Conv2d, lambda: _fp8_quantizer(amax=123.0))
    backbone = _Backbone(conv=conv)
    before = {
        name: quantizer.amax.clone()
        for name, quantizer in (
            ("input", conv.input_quantizer),
            ("weight", conv.weight_quantizer),
        )
    }

    _finalize_backbone_quantization(
        backbone,
        "unet",
        QuantizationConfig(format=QuantFormat.FP4),
        ModelType.SDXL_BASE,
        restored=False,
    )

    for name, quantizer in (
        ("input", conv.input_quantizer),
        ("weight", conv.weight_quantizer),
    ):
        assert quantizer.is_enabled
        assert quantizer.is_fp8
        assert torch.equal(quantizer.amax, before[name])


@pytest.mark.parametrize("export_fails", [False, True])
def test_onnx_fp8_scale_workaround_restores_state_before_hf_export(
    monkeypatch, tmp_path, export_fails
):
    conv = _quantized_conv(nn.Conv2d, lambda: _fp8_quantizer(amax=448.0))
    backbone = _Backbone(conv=conv)
    original_state = {
        name: (quantizer.num_bits, quantizer.amax)
        for name, quantizer in (
            ("input", conv.input_quantizer),
            ("weight", conv.weight_quantizer),
        )
    }
    manager = ExportManager(
        ExportConfig(onnx_dir=tmp_path / "onnx", hf_ckpt_dir=tmp_path / "hf"),
        logging.getLogger(__name__),
        pipeline_manager=None,
    )

    class _Pipeline:
        def to(self, *args, **kwargs):
            return self

    def fake_onnx_export(*args, **kwargs):
        del args, kwargs
        assert conv.input_quantizer.num_bits == 8
        assert conv.weight_quantizer.num_bits == 8
        assert conv.input_quantizer.amax == 127.0
        assert conv.weight_quantizer.amax == 127.0
        if export_fails:
            raise RuntimeError("export failed")

    def fake_hf_export(*args, **kwargs):
        del args, kwargs
        for name, quantizer in (
            ("input", conv.input_quantizer),
            ("weight", conv.weight_quantizer),
        ):
            num_bits, amax = original_state[name]
            assert quantizer.num_bits == num_bits
            assert quantizer.amax is amax

    monkeypatch.setattr(backbone, "to", lambda *args, **kwargs: backbone)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setitem(sys.modules, "onnx_utils", diffusion_onnx_utils)
    monkeypatch.setitem(sys.modules, "onnx_utils.export", diffusion_export)
    monkeypatch.setattr(diffusion_export, "modelopt_export_sd", fake_onnx_export)
    monkeypatch.setattr(quantize_module, "export_hf_checkpoint", fake_hf_export)

    if export_fails:
        with pytest.raises(RuntimeError, match="export failed"):
            manager.export_onnx(_Pipeline(), backbone, ModelType.SDXL_BASE, QuantFormat.FP4)
    else:
        manager.export_onnx(_Pipeline(), backbone, ModelType.SDXL_BASE, QuantFormat.FP4)
        manager.export_hf_ckpt(_Pipeline(), ModelConfig(model_type=ModelType.SDXL_BASE))

    for name, quantizer in (
        ("input", conv.input_quantizer),
        ("weight", conv.weight_quantizer),
    ):
        num_bits, amax = original_state[name]
        assert quantizer.num_bits == num_bits
        assert quantizer.amax is amax


@pytest.mark.parametrize("conv_cls", [nn.Conv1d, nn.Conv2d])
def test_fp4_finalization_disables_unsupported_nvfp4_conv(conv_cls):
    conv = _quantized_conv(conv_cls, _nvfp4_quantizer)
    backbone = _Backbone(conv=conv)
    backbone.fp8_conv = _quantized_conv(nn.Conv2d, _fp8_quantizer)

    _finalize_backbone_quantization(
        backbone,
        "unet",
        QuantizationConfig(format=QuantFormat.FP4),
        ModelType.SDXL_BASE,
        restored=False,
    )

    assert not conv.input_quantizer.is_enabled
    assert not conv.weight_quantizer.is_enabled


@pytest.mark.parametrize(
    ("input_quantizer", "weight_quantizer", "match"),
    [
        (_nvfp4_quantizer(), _nvfp4_quantizer(enabled=False), "enable input and weight"),
        (_fp8_quantizer(), _nvfp4_quantizer(), "dynamic E2M1"),
        (_nvfp4_quantizer(32), _nvfp4_quantizer(), "requires block size 16"),
        (
            _nvfp4_quantizer(enabled=False),
            _nvfp4_quantizer(enabled=False),
            "at least one enabled Linear",
        ),
    ],
    ids=["unpaired", "wrong-format", "wrong-block-size", "no-enabled-pair"],
)
def test_nvfp4_linear_validation_rejects_invalid_state(input_quantizer, weight_quantizer, match):
    backbone = _Backbone(
        linear=_QuantizedLinear(
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
        )
    )

    with pytest.raises(ValueError, match=match):
        validate_nvfp4_quantizers(backbone, expected_block_size=16, quantize_mha=False)


def test_nvfp4_linear_validation_allows_disabled_exclusion_pair():
    backbone = nn.Module()
    backbone.enabled = _QuantizedLinear()
    backbone.excluded = _QuantizedLinear(
        input_quantizer=_nvfp4_quantizer(enabled=False),
        weight_quantizer=_nvfp4_quantizer(enabled=False),
    )

    validate_nvfp4_quantizers(backbone, expected_block_size=16, quantize_mha=False)


@pytest.mark.parametrize("projection_name", ["to_q", "to_k", "to_v"])
def test_sdxl_fp4_validation_rejects_enabled_attention_projection(projection_name):
    backbone = _Backbone(conv=_quantized_conv(nn.Conv2d, _fp8_quantizer))
    backbone.attn = nn.Module()
    setattr(backbone.attn, projection_name, _QuantizedLinear())

    with pytest.raises(ValueError, match="must keep input and weight quantizers disabled"):
        validate_nvfp4_quantizers(
            backbone,
            expected_block_size=16,
            quantize_mha=False,
            validate_sdxl_mixed_recipe=True,
        )


@pytest.mark.parametrize(
    ("input_enabled", "weight_enabled"),
    [(True, False), (False, True)],
    ids=["input-only", "weight-only"],
)
def test_sdxl_fp4_validation_rejects_partial_attention_projection(input_enabled, weight_enabled):
    backbone = _Backbone(conv=_quantized_conv(nn.Conv2d, _fp8_quantizer))
    backbone.attn = nn.Module()
    backbone.attn.to_q = _QuantizedLinear(
        input_quantizer=_nvfp4_quantizer(enabled=input_enabled),
        weight_quantizer=_nvfp4_quantizer(enabled=weight_enabled),
    )

    with pytest.raises(ValueError, match="must keep input and weight quantizers disabled"):
        validate_nvfp4_quantizers(
            backbone,
            expected_block_size=16,
            quantize_mha=False,
            validate_sdxl_mixed_recipe=True,
        )


def test_sdxl_fp4_validation_allows_disabled_attention_projections():
    backbone = _Backbone(conv=_quantized_conv(nn.Conv2d, _fp8_quantizer))
    backbone.attn = nn.Module()
    for projection_name in ("to_q", "to_k", "to_v"):
        setattr(
            backbone.attn,
            projection_name,
            _QuantizedLinear(
                input_quantizer=_nvfp4_quantizer(enabled=False),
                weight_quantizer=_nvfp4_quantizer(enabled=False),
            ),
        )

    validate_nvfp4_quantizers(
        backbone,
        expected_block_size=16,
        quantize_mha=False,
        validate_sdxl_mixed_recipe=True,
    )


@pytest.mark.parametrize(
    ("quantizer_name", "amax"),
    [
        ("input_quantizer", None),
        ("weight_quantizer", float("nan")),
        ("weight_quantizer", -1.0),
    ],
)
def test_restore_finalization_rejects_uncalibrated_nvfp4_linear(quantizer_name, amax):
    linear = _QuantizedLinear()
    setattr(linear, quantizer_name, _nvfp4_quantizer(amax=amax))
    backbone = _Backbone(linear=linear, conv=_quantized_conv(nn.Conv2d, _fp8_quantizer))

    with pytest.raises(ValueError, match="must have a finite nonnegative calibrated amax"):
        _finalize_backbone_quantization(
            backbone,
            "unet",
            QuantizationConfig(format=QuantFormat.FP4),
            ModelType.SDXL_BASE,
            restored=True,
        )


def test_nvfp4_linear_validation_accepts_zero_amax():
    backbone = _Backbone(
        linear=_QuantizedLinear(
            input_quantizer=_nvfp4_quantizer(amax=0.0),
            weight_quantizer=_nvfp4_quantizer(amax=0.0),
        )
    )

    validate_nvfp4_quantizers(backbone, expected_block_size=16, quantize_mha=False)


@pytest.mark.parametrize(
    ("quantizer_factory", "match"),
    [
        (_nvfp4_quantizer, "only supported on Linear input and weight quantizers"),
        (_fp8_quantizer, "only supported on SDXL Conv2d"),
    ],
    ids=["nvfp4-layernorm", "fp8-layernorm"],
)
def test_sdxl_fp4_validation_rejects_quantized_layernorm(quantizer_factory, match):
    conv = _quantized_conv(nn.Conv2d, _fp8_quantizer)
    backbone = _Backbone(conv=conv)
    backbone.norm = nn.LayerNorm(16)
    backbone.norm.input_quantizer = quantizer_factory()

    with pytest.raises(ValueError, match=match):
        validate_nvfp4_quantizers(
            backbone,
            expected_block_size=16,
            quantize_mha=False,
            validate_sdxl_mixed_recipe=True,
        )


def test_restore_finalization_rejects_enabled_nvfp4_layernorm():
    conv = _quantized_conv(nn.Conv2d, _fp8_quantizer)
    backbone = _Backbone(conv=conv)
    backbone.norm = nn.LayerNorm(16)
    backbone.norm.input_quantizer = _nvfp4_quantizer()

    with pytest.raises(ValueError, match="only supported on Linear input and weight quantizers"):
        _finalize_backbone_quantization(
            backbone,
            "unet",
            QuantizationConfig(format=QuantFormat.FP4),
            ModelType.SDXL_BASE,
            restored=True,
        )


@pytest.mark.parametrize("restored", [False, True])
def test_non_sdxl_fp4_finalization_remains_permissive(restored):
    backbone = _Backbone()
    backbone.norm = nn.LayerNorm(16)
    backbone.norm.input_quantizer = _nvfp4_quantizer()

    _finalize_backbone_quantization(
        backbone,
        "transformer",
        QuantizationConfig(format=QuantFormat.FP4),
        ModelType.FLUX_DEV,
        restored=restored,
    )

    assert backbone.norm.input_quantizer.is_enabled


@pytest.mark.parametrize(
    ("input_factory", "weight_factory", "match"),
    [
        (
            _fp8_quantizer,
            lambda: _fp8_quantizer(enabled=False),
            "enable input and weight quantizers as a pair",
        ),
        (
            _fp8_quantizer,
            _nvfp4_quantizer,
            "must use per-tensor FP8",
        ),
        (
            lambda: _fp8_quantizer(amax=None),
            _fp8_quantizer,
            "finite positive calibrated amax",
        ),
        (
            lambda: _fp8_quantizer(amax=0.0),
            _fp8_quantizer,
            "finite positive calibrated amax",
        ),
        (
            lambda: _fp8_quantizer(amax=float("nan")),
            _fp8_quantizer,
            "finite positive calibrated amax",
        ),
    ],
    ids=["partial", "wrong-format", "missing-amax", "zero-amax", "nonfinite-amax"],
)
def test_sdxl_fp8_conv_validation_rejects_invalid_state(input_factory, weight_factory, match):
    conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)
    conv.input_quantizer = input_factory()
    conv.weight_quantizer = weight_factory()
    backbone = _Backbone(conv=conv)

    with pytest.raises(ValueError, match=match):
        validate_nvfp4_quantizers(
            backbone,
            expected_block_size=16,
            quantize_mha=False,
            validate_sdxl_mixed_recipe=True,
        )


def test_sdxl_fp8_conv_validation_requires_enabled_pair():
    conv = _quantized_conv(nn.Conv2d, lambda: _fp8_quantizer(enabled=False))
    backbone = _Backbone(conv=conv)

    with pytest.raises(ValueError, match="at least one enabled calibrated FP8 Conv2d"):
        validate_nvfp4_quantizers(
            backbone,
            expected_block_size=16,
            quantize_mha=False,
            validate_sdxl_mixed_recipe=True,
        )


@pytest.mark.parametrize(("quantize_mha", "enabled"), [(False, False), (True, True)])
def test_restore_finalization_accepts_matching_mha_state(quantize_mha, enabled):
    attention = _attention(enabled=enabled)
    conv = _quantized_conv(nn.Conv2d, _fp8_quantizer)
    backbone = _Backbone(conv=conv, attention=attention)

    _finalize_backbone_quantization(
        backbone,
        "unet",
        QuantizationConfig(format=QuantFormat.FP4, quantize_mha=quantize_mha),
        ModelType.SDXL_BASE,
        restored=True,
    )

    for name in _MHA_QUANTIZER_NAMES:
        expected_enabled = enabled and name != "bmm2_output_quantizer"
        assert getattr(attention, name).is_enabled is expected_enabled


def test_restore_finalization_accepts_uncalibrated_softmax_quantizer():
    attention = _attention(enabled=True, softmax_amax=None)
    backbone = _Backbone(conv=_quantized_conv(nn.Conv2d, _fp8_quantizer), attention=attention)

    _finalize_backbone_quantization(
        backbone,
        "unet",
        QuantizationConfig(format=QuantFormat.FP4, quantize_mha=True),
        ModelType.SDXL_BASE,
        restored=True,
    )

    assert attention.softmax_quantizer.is_enabled
    assert attention.softmax_quantizer.amax is None


@pytest.mark.parametrize(
    ("quantize_mha", "head_size", "mutate", "match"),
    [
        (False, 16, lambda attention: None, "must be disabled"),
        (
            True,
            16,
            lambda attention: attention.q_bmm_quantizer.disable(),
            "requires 'q_bmm_quantizer' to be enabled",
        ),
        (
            True,
            16,
            lambda attention: attention.softmax_quantizer.disable(),
            "requires 'softmax_quantizer' to be enabled",
        ),
        (
            True,
            16,
            lambda attention: attention.bmm2_output_quantizer.enable(),
            "requires 'bmm2_output_quantizer' to be disabled",
        ),
        (
            True,
            16,
            lambda attention: setattr(attention, "softmax_quantizer", _nvfp4_quantizer()),
            "must use per-tensor FP8",
        ),
        (
            True,
            16,
            lambda attention: setattr(attention, "q_bmm_quantizer", _fp8_quantizer(amax=None)),
            "finite positive calibrated amax",
        ),
        (True, 8, lambda attention: None, "must be disabled because FP8 MHA is unsupported"),
    ],
    ids=[
        "on-to-off",
        "off-to-on",
        "missing-softmax",
        "enabled-bmm2-output",
        "wrong-format",
        "missing-amax",
        "unsupported-head-size",
    ],
)
def test_restore_finalization_rejects_mha_state_mismatch(quantize_mha, head_size, mutate, match):
    attention = _attention(head_size=head_size, enabled=True)
    mutate(attention)
    backbone = _Backbone(attention=attention)

    with pytest.raises(ValueError, match=match):
        _finalize_backbone_quantization(
            backbone,
            "unet",
            QuantizationConfig(format=QuantFormat.FP4, quantize_mha=quantize_mha),
            ModelType.SDXL_BASE,
            restored=True,
        )


@pytest.mark.parametrize("quantize_mha", [False, True])
def test_fresh_finalization_applies_and_validates_mha_policy(quantize_mha):
    attention = _attention(enabled=True)
    conv = _quantized_conv(nn.Conv2d, _fp8_quantizer)
    backbone = _Backbone(conv=conv, attention=attention)

    _finalize_backbone_quantization(
        backbone,
        "unet",
        QuantizationConfig(format=QuantFormat.FP4, quantize_mha=quantize_mha),
        ModelType.SDXL_BASE,
        restored=False,
    )

    for name in _MHA_QUANTIZER_NAMES:
        quantizer = getattr(attention, name)
        expected_enabled = quantize_mha and name != "bmm2_output_quantizer"
        assert quantizer.is_enabled is expected_enabled
        if quantizer.is_enabled:
            assert quantizer.is_fp8


def test_check_conv_does_not_disable_non_nvfp4_quantizers():
    conv = _quantized_conv(nn.Conv2d, _fp8_quantizer)
    backbone = _Backbone(conv=conv)

    check_conv_and_mha(backbone, if_fp4=True, quantize_mha=False)

    assert conv.input_quantizer.is_enabled
    assert conv.weight_quantizer.is_enabled
