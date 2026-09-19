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

import fnmatch
from typing import cast

import pytest
import torch
from _test_utils.torch.export.utils import (
    ToyModel,
    partial_fp8_config,
    partial_nvfp4_config,
    partial_w4a8_config,
)

import modelopt.torch.export.unified_export_megatron as unified_export_megatron
import modelopt.torch.quantization as mtq
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.quant_format import (
    KV_CACHE_FP8,
    KV_CACHE_FP8_K_NVFP4_V,
    KV_CACHE_NVFP4,
    QUANTIZATION_FP8,
    QUANTIZATION_IQ1_S,
    QUANTIZATION_IQ2_XS,
    QUANTIZATION_NVFP4,
    QUANTIZATION_W4A8_AWQ,
)
from modelopt.torch.export.quant_utils import (
    _get_carried_over_module_names,
    _has_large_fp8_scale,
    get_kv_cache_scaling_factor,
    get_quant_config,
    get_quantization_format,
    postprocess_state_dict,
    process_layer_quant_config,
    seed_carried_over_exclusions,
    uses_iq_quantization,
)
from modelopt.torch.quantization.nn import (
    NVFP4StaticQuantizer,
    SequentialQuantizer,
    TensorQuantizer,
)


class _FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()


@pytest.mark.parametrize(
    ("num_bits", "quantization_format", "payload_bytes", "effective_bits"),
    [
        ("iq1_s", QUANTIZATION_IQ1_S, 50, 1.5625),
        ("iq2_xs", QUANTIZATION_IQ2_XS, 74, 2.3125),
    ],
)
def test_iq_quantization_config(num_bits, quantization_format, payload_bytes, effective_bits):
    model = torch.nn.Sequential(torch.nn.Linear(256, 256, bias=False))
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {
                        "num_bits": num_bits,
                        "block_sizes": {-1: 256},
                        "backend": "ggml",
                    },
                },
            ],
            "algorithm": None,
        },
    )

    assert get_quantization_format(model) == quantization_format
    config = get_quant_config(model)
    assert config["quantization"]["quant_algo"] == num_bits.upper()
    assert config["quantization"]["block_payload_bytes"] == payload_bytes
    assert config["quantization"]["effective_bits"] == effective_bits
    hf_config = convert_hf_quant_config_format(config)
    assert "config_groups" not in hf_config
    assert hf_config["group_size"] == 256
    assert hf_config["effective_bits"] == effective_bits
    assert hf_config["packing"] == "ggml"
    assert hf_config["block_payload_bytes"] == payload_bytes


def _quantize_sequential(layer_cfgs):
    """Quantize a two-Linear model, one quantizer config per layer."""
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256, bias=False), torch.nn.Linear(256, 256, bias=False)
    )
    mtq.quantize(
        model,
        {
            "quant_cfg": [{"quantizer_name": "*", "enable": False}, *layer_cfgs],
            "algorithm": None,
        },
    )
    return model


_IQ_WEIGHT_CFG = {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"}


def test_uses_iq_quantization_sees_iq_behind_another_format():
    """get_quantization_format stops at the first format, so the TP guard cannot rely on it."""
    model = _quantize_sequential(
        [
            {"quantizer_name": "0.weight_quantizer", "cfg": {"num_bits": (4, 3)}},
            {"quantizer_name": "1.weight_quantizer", "cfg": _IQ_WEIGHT_CFG},
        ]
    )

    assert get_quantization_format(model) == QUANTIZATION_FP8
    assert uses_iq_quantization(model)


def test_uses_iq_quantization_false_without_iq_layers():
    model = _quantize_sequential(
        [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3)}}]
    )

    assert not uses_iq_quantization(model)


def test_uses_iq_quantization_tolerates_sequential_quantizer():
    """A SequentialQuantizer has is_enabled but no num_bits, and is never IQ.

    save_pretrained calls this on every Megatron export, so reading num_bits directly would
    raise AttributeError on a W4A8_AWQ model before any format dispatch.
    """
    layer = torch.nn.Linear(256, 256, bias=False)
    layer.weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
    assert not hasattr(layer.weight_quantizer, "num_bits")

    assert not uses_iq_quantization(torch.nn.Sequential(layer))


def test_iq_export_rejects_enabled_input_quantizer():
    """IQ payloads carry no activation scale, so W-IQ + A-FP8 must not export as weight-only."""
    model = _quantize_sequential(
        [
            {"quantizer_name": "*weight_quantizer", "cfg": _IQ_WEIGHT_CFG},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (4, 3)}},
        ]
    )

    with pytest.raises(NotImplementedError, match="weight-only"):
        get_quantization_format(model)


def test_iq_hf_config_rejects_mismatched_group_size():
    """A uniformly-IQ config must validate group_size, not silently rewrite it to the block size.

    The MIXED_PRECISION branch already forwards the per-layer group size; this covers the
    top-level branch, which did not.
    """
    with pytest.raises(ValueError, match="IQ2_XS requires group size 256, got 128"):
        convert_hf_quant_config_format(
            {
                "quantization": {
                    "quant_algo": "IQ2_XS",
                    "group_size": 128,
                    "effective_bits": 2.3125,
                    "packing": "ggml",
                    "block_payload_bytes": 74,
                }
            }
        )


def test_mixed_iq_config_group_does_not_claim_integer_weight_schema():
    converted = convert_hf_quant_config_format(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.mlp.down_proj": {
                        "quant_algo": "IQ2_XS",
                        "group_size": 256,
                        "effective_bits": 2.3125,
                        "packing": "ggml",
                        "block_payload_bytes": 74,
                    }
                },
            }
        }
    )

    group = converted["config_groups"]["group_0"]
    assert "weights" not in group
    assert group["quant_algo"] == "IQ2_XS"
    assert group["packing"] == "ggml"


def test_iq_quantization_config_rejects_mismatched_block_size():
    with pytest.raises(ValueError, match="IQ2_XS requires block size 256, got 128"):
        process_layer_quant_config(
            {
                "model.layers.0.mlp.down_proj.quantization": "iq2_xs",
                "model.layers.0.mlp.down_proj.awq_block_size": 128,
            }
        )


class _FakeKVCacheQuantizer(torch.nn.Module):
    """Minimal FP8 KV cache quantizer for scaling-factor tests."""

    is_enabled = True
    num_bits = (4, 3)
    maxbound = 448.0

    def export_amax(self):
        return torch.tensor([224.0])


def test_get_kv_cache_scaling_factor_can_disable_fp8_clamping():
    """FP8 KV cache scales below 1.0 are retained when explicitly requested."""
    attention = torch.nn.Module()
    attention.k_bmm_quantizer = _FakeKVCacheQuantizer()
    attention.v_bmm_quantizer = _FakeKVCacheQuantizer()

    clamped_scales = get_kv_cache_scaling_factor(attention)
    unclamped_scales = get_kv_cache_scaling_factor(attention, clamp_fp8_scales=False)

    assert all(torch.equal(scale, torch.tensor([1.0])) for scale in clamped_scales)
    assert all(torch.equal(scale, torch.tensor([0.5])) for scale in unclamped_scales)


def test_large_fp8_scale_check_is_device_agnostic():
    class CudaLikeScale:
        device = torch.device("cuda")

        def __gt__(self, _threshold):
            return torch.tensor([True])

    assert _has_large_fp8_scale(cast("torch.Tensor", CudaLikeScale()))


@pytest.mark.parametrize("clamp_kv_cache_scales", [True, False])
def test_export_mcore_gpt_to_hf_passes_kv_cache_clamping_option(
    monkeypatch, tmp_path, clamp_kv_cache_scales
):
    """The public Megatron export API forwards the KV cache clamping option."""
    exporter_args = {}

    class FakeExporter:
        def __init__(self, *args, **kwargs):
            exporter_args.update(kwargs)
            self.export_extra_modules = False

        def save_pretrained(self, *args):
            pass

    monkeypatch.setattr(unified_export_megatron, "GPTModelExporter", FakeExporter)

    unified_export_megatron.export_mcore_gpt_to_hf(
        object(),
        tmp_path,
        export_dir=tmp_path,
        clamp_kv_cache_scales=clamp_kv_cache_scales,
    )

    assert exporter_args["clamp_kv_cache_scales"] is clamp_kv_cache_scales


@pytest.mark.parametrize(
    ("config", "expected"),
    [(partial_fp8_config, QUANTIZATION_FP8), (partial_w4a8_config, QUANTIZATION_W4A8_AWQ)],
)
def test_get_quantization_format(config, expected):
    model = ToyModel()
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10)))
    assert get_quantization_format(model) == expected


def test_nvfp4_static_quantizer_export():
    """NVFP4StaticQuantizer: get_quantization_format returns NVFP4 and get_quant_config returns export config."""
    model = ToyModel()
    mtq.quantize(model, partial_nvfp4_config, lambda x: x(torch.randn(1, 4, 10)))

    # Convert all weight quantizers to NVFP4StaticQuantizer
    for module in model.modules():
        tq = getattr(module, "weight_quantizer", None)
        if tq is not None and hasattr(tq, "_amax") and not isinstance(tq, NVFP4StaticQuantizer):
            global_amax = tq._amax.max() if tq._amax.dim() > 0 else tq._amax
            NVFP4StaticQuantizer.from_tensor_quantizer(tq, global_amax=global_amax)

    assert get_quantization_format(model) == QUANTIZATION_NVFP4

    quant_config = get_quant_config(model)
    assert quant_config["quantization"]["quant_algo"] == "NVFP4"
    assert quant_config["quantization"]["group_size"] == 16


def test_projection_output_quantizers_are_not_exported_as_kv_cache():
    model = ToyModel()
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*.weight_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
                "enable": True,
            },
            {
                "quantizer_name": "*.input_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
                "enable": True,
            },
            {
                "quantizer_name": "*.output_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10)))

    quantization = get_quant_config(model)["quantization"]

    assert quantization["quant_algo"] == "FP8"
    assert quantization["kv_cache_quant_algo"] is None
    assert "kv_cache_quantized_layers" not in quantization


def test_uniform_vlm_export_ignores_disabled_vision_attention():
    model = ToyModel()
    weight_config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*.weight_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
                "enable": True,
            },
            {
                "quantizer_name": "*.input_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(
        model,
        weight_config,
        lambda quantized_model: quantized_model(torch.randn(1, 4, 10)),
    )
    model.language_model = torch.nn.Module()
    model.language_model.attention = _FakeAttention()
    model.vision_attention = _FakeAttention()
    mtq.set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*[kv]_bmm_quantizer", "enable": False},
            {
                "quantizer_name": "language_model.*[kv]_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
                "enable": True,
            },
        ],
    )

    quantization = get_quant_config(model)["quantization"]

    assert quantization["quant_algo"] == "FP8"
    assert quantization["kv_cache_quant_algo"] == "FP8"
    assert "kv_cache_quantized_layers" not in quantization


def test_quant_config_tolerates_ambiguous_language_model_roots():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.model.language_model.attention = _FakeAttention()
    mtq.set_quantizer_by_cfg(model, [{"quantizer_name": "*", "enable": False}])

    quantization = get_quant_config(model)["quantization"]

    assert quantization["kv_cache_quant_algo"] is None


@pytest.mark.parametrize("enabled_quantizer", ["k_bmm_quantizer", "output_quantizer"])
def test_legacy_single_sided_kv_detection_is_preserved(enabled_quantizer):
    model = torch.nn.Module()
    model.attention = _FakeAttention()
    model.attention.output_quantizer = TensorQuantizer()
    mtq.set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": f"*.{enabled_quantizer}",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
                "enable": True,
            },
        ],
    )

    quantization = get_quant_config(model)["quantization"]

    assert quantization["kv_cache_quant_algo"] == "FP8"
    assert "kv_cache_quantized_layers" not in quantization


@pytest.mark.parametrize(
    ("quantizer_cfg", "expected_format"),
    [
        ({"num_bits": (4, 3), "constant_amax": 1.0}, "FP8"),
        (
            {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "constant_amax": 1.0,
            },
            "NVFP4",
        ),
    ],
)
def test_uniform_kv_only_export_preserves_bf16_weight_metadata(quantizer_cfg, expected_format):
    model = torch.nn.Module()
    model.attention = _FakeAttention()
    mtq.set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": quantizer_cfg,
                "enable": True,
            },
        ],
    )

    hf_quant_config = get_quant_config(model)
    quantization = hf_quant_config["quantization"]
    assert quantization["quant_algo"] is None
    assert quantization["kv_cache_quant_algo"] == expected_format
    assert quantization["quantized_layers"] == {}
    assert quantization["kv_cache_quantized_layers"] == {
        "attention": {"quant_algo": expected_format}
    }

    converted = convert_hf_quant_config_format(hf_quant_config)
    assert "quant_algo" not in converted
    assert "config_groups" not in converted
    assert converted["kv_cache_scheme"] == (
        {"dynamic": False, "num_bits": 8, "type": "float"}
        if expected_format == "FP8"
        else expected_format
    )
    assert converted["kv_cache_quantized_layers"] == {"attention": {"quant_algo": expected_format}}


def test_uniform_kv_export_retains_scheme_and_layer_map_without_search_marker():
    model = torch.nn.Module()
    model.attention = _FakeAttention()
    mtq.set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
                "enable": True,
            },
        ],
    )
    quantization = get_quant_config(model)["quantization"]

    assert quantization["quant_algo"] is None
    assert quantization["quantized_layers"] == {}
    assert quantization["kv_cache_quant_algo"] == "FP8"
    assert quantization["kv_cache_quantized_layers"] == {"attention": {"quant_algo": "FP8"}}

    converted = convert_hf_quant_config_format({"quantization": quantization})
    assert "quant_algo" not in converted
    assert "config_groups" not in converted
    assert converted["kv_cache_scheme"] == {
        "dynamic": False,
        "num_bits": 8,
        "type": "float",
    }
    assert converted["kv_cache_quantized_layers"] == {"attention": {"quant_algo": "FP8"}}


def test_partial_uniform_kv_with_uniform_weights_preserves_legacy_schema():
    model = ToyModel()
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*.weight_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                    "enable": True,
                },
                {
                    "quantizer_name": "*.input_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                    "enable": True,
                },
            ],
            "algorithm": "max",
        },
        lambda quantized_model: quantized_model(torch.randn(1, 4, 10)),
    )
    model.attn0 = _FakeAttention()
    model.attn1 = _FakeAttention()
    mtq.set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*[kv]_bmm_quantizer", "enable": False},
            {
                "quantizer_name": "attn0.*[kv]_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
                "enable": True,
            },
        ],
    )

    quantization = get_quant_config(model)["quantization"]

    assert quantization["quant_algo"] == "FP8"
    assert quantization["kv_cache_quant_algo"] == "FP8"
    assert "kv_cache_quantized_layers" not in quantization


def test_mixed_kv_cache_quantization_exports_per_layer_map():
    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_bmm_quantizer = TensorQuantizer()
            self.v_bmm_quantizer = TensorQuantizer()

    model = torch.nn.Module()
    model.attn0 = FakeAttention()
    model.attn1 = FakeAttention()
    model.attn2 = FakeAttention()
    mtq.set_quantizer_by_cfg(
        model.attn0,
        [
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "use_constant_amax": True},
            }
        ],
    )
    mtq.set_quantizer_by_cfg(
        model.attn1,
        [
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    "use_constant_amax": True,
                },
            }
        ],
    )
    mtq.set_quantizer_by_cfg(
        model.attn2,
        [
            {
                "quantizer_name": "*k_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "use_constant_amax": True},
            },
            {
                "quantizer_name": "*v_bmm_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    "use_constant_amax": True,
                },
            },
        ],
    )

    quantization = get_quant_config(model)["quantization"]
    assert quantization["quant_algo"] is None
    assert quantization["kv_cache_quant_algo"] == "MIXED_PRECISION"
    assert quantization["quantized_layers"] == {}
    assert quantization["kv_cache_quantized_layers"] == {
        "attn0": {"quant_algo": "FP8"},
        "attn1": {"quant_algo": "NVFP4"},
        "attn2": {"quant_algo": "FP8_K_NVFP4_V"},
    }


def test_unsupported_asymmetric_kv_cache_pair_fails_export():
    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_bmm_quantizer = TensorQuantizer()
            self.v_bmm_quantizer = TensorQuantizer()

    model = FakeAttention()
    mtq.set_quantizer_by_cfg(
        model,
        [
            {
                "quantizer_name": "*k_bmm_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    "use_constant_amax": True,
                },
            },
            {
                "quantizer_name": "*v_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "use_constant_amax": True},
            },
        ],
    )

    with pytest.raises(NotImplementedError, match="Unsupported mixed K/V cache"):
        get_quant_config(model)


def test_mixed_kv_cache_postprocess_uses_each_layers_format():
    state_dict = {
        "attn0.k_bmm_quantizer._amax": torch.tensor([448.0]),
        "attn0.v_bmm_quantizer._amax": torch.tensor([224.0]),
        "attn1.k_bmm_quantizer._amax": torch.tensor([112.0]),
        "attn1.v_bmm_quantizer._amax": torch.tensor([56.0]),
    }
    layer_formats = {
        "attn0": {"quant_algo": KV_CACHE_FP8},
        "attn1": {"quant_algo": KV_CACHE_NVFP4},
        "attn2": {"quant_algo": KV_CACHE_FP8_K_NVFP4_V},
    }
    state_dict.update(
        {
            "attn2.k_bmm_quantizer._amax": torch.tensor([448.0]),
            "attn2.v_bmm_quantizer._amax": torch.tensor([112.0]),
        }
    )

    processed = postprocess_state_dict(state_dict, 448.0, layer_formats)

    assert processed == {
        "attn0.k_proj.k_scale": torch.tensor([1.0]),
        "attn0.v_proj.v_scale": torch.tensor([0.5]),
        "attn1.k_proj.k_scale": torch.tensor([0.25]),
        "attn1.v_proj.v_scale": torch.tensor([0.125]),
        "attn2.k_proj.k_scale": torch.tensor([1.0]),
        "attn2.v_proj.v_scale": torch.tensor([0.25]),
    }


class _FakeTopKRouter(torch.nn.Module):
    """Mimics a transformers>=5.0 MoE router: owns a ``weight`` but is NOT an ``nn.Linear``.

    ``mtq.quantize`` only attaches quantizers to registered modules (e.g. ``nn.Linear``), so a
    router like this never receives one -- reproducing the condition behind NVBug 5718750.
    """

    def __init__(self, hidden: int, num_experts: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_experts, hidden))
        self.top_k = 2
        self.num_experts = num_experts

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class _FakeMoEBlock(torch.nn.Module):
    def __init__(self, hidden: int = 16, num_experts: int = 4):
        super().__init__()
        self.gate = _FakeTopKRouter(hidden, num_experts)
        self.experts = torch.nn.ModuleList(
            torch.nn.Linear(hidden, hidden, bias=False) for _ in range(num_experts)
        )

    def forward(self, x):
        self.gate(x)  # exercise the router so it is reachable
        out = x
        for expert in self.experts:
            out = expert(out)
        return out


class _FakeMoEModel(torch.nn.Module):
    def __init__(self, hidden: int = 16, num_experts: int = 4):
        super().__init__()
        self.block = _FakeMoEBlock(hidden, num_experts)

    def forward(self, x):
        return self.block(x)


_nvfp4_all_linears_config = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
    ],
    "algorithm": "max",
}


def test_moe_router_excluded_when_not_quantized():
    """NVBug 5718750: a non-Linear MoE router (transformers>=5.0 TopKRouter) gets no quantizer.

    Its BF16 weight is still exported, so it must be listed in ``exclude_modules``; otherwise
    deployment frameworks treat it as a quantized weight and fail to load the checkpoint.
    """
    hidden = 16
    model = _FakeMoEModel(hidden=hidden)
    mtq.quantize(model, _nvfp4_all_linears_config, lambda m: m(torch.randn(2, hidden)))

    # The router is not an nn.Linear, so quantize attached no quantizer to it.
    assert not hasattr(model.block.gate, "weight_quantizer")
    # The experts are quantized to NVFP4.
    assert get_quantization_format(model.block.experts[0]) == QUANTIZATION_NVFP4

    quant_config = get_quant_config(model)
    assert quant_config["quantization"]["quant_algo"] == "NVFP4"

    exclude_modules = quant_config["quantization"]["exclude_modules"]
    assert any(fnmatch.fnmatch("block.gate", pattern) for pattern in exclude_modules), (
        f"MoE router 'block.gate' missing from exclude_modules: {exclude_modules}"
    )
    # The quantized experts must NOT be excluded.
    assert not any(fnmatch.fnmatch("block.experts.0", pattern) for pattern in exclude_modules), (
        f"Quantized expert wrongly excluded: {exclude_modules}"
    )


def test_moe_router_names_handle_root_module():
    """When the MoE block itself is the root module, router names have no leading dot."""
    from modelopt.torch.export.quant_utils import _get_unquantized_moe_router_names

    block = _FakeMoEBlock(hidden=16)
    # name == "" for the root module; the router must be "gate", not ".gate".
    assert _get_unquantized_moe_router_names(block) == ["gate"]


def test_carried_over_weights_are_excluded_from_quantization():
    """A carried MTP head has no module, so nothing in the quantizer walk can see it.

    Its original-precision weight is copied into the export verbatim, so it must still reach
    ``exclude_modules`` -- otherwise a deployment framework reads the top-level ``quant_algo``
    and tries to load ``eh_proj`` as an NVFP4 weight. Same failure as the MoE router above,
    different cause: no quantizer there, no module at all here.
    """
    hidden = 16
    model = _FakeMoEModel(hidden=hidden)
    mtq.quantize(model, _nvfp4_all_linears_config, lambda m: m(torch.randn(2, hidden)))

    # The loader could not place these; the exporter copies them straight from the checkpoint.
    model._modelopt_unplaced_source_keys = [
        "model.mtp.eh_proj.weight",
        "model.mtp.eh_proj.bias",
        "model.mtp.embed_tokens.weight",
    ]

    quant_config = get_quant_config(model)
    assert quant_config["quantization"]["quant_algo"] == "NVFP4"

    exclude_modules = quant_config["quantization"]["exclude_modules"]
    for carried in ("model.mtp.eh_proj", "model.mtp.embed_tokens"):
        assert any(fnmatch.fnmatch(carried, pattern) for pattern in exclude_modules), (
            f"carried weight {carried!r} missing from exclude_modules: {exclude_modules}"
        )
    # The quantized experts must NOT be excluded.
    assert not any(fnmatch.fnmatch("block.experts.0", pattern) for pattern in exclude_modules), (
        f"Quantized expert wrongly excluded: {exclude_modules}"
    )


def test_carried_over_module_names_strip_parameter_and_dedup():
    """Keys are ``<module>.<param>``; two params of one module yield one module name."""
    model = torch.nn.Module()
    # No attribute at all -- the common case, and every non-carry-over caller.
    assert _get_carried_over_module_names(model) == []

    model._modelopt_unplaced_source_keys = [
        "a.b.weight",
        "a.b.bias",  # same module as above
        "c.weight",
        "toplevel",  # no dot: no owning module, skipped
    ]
    assert _get_carried_over_module_names(model) == ["a.b", "c"]


def test_carried_over_names_prefer_what_the_export_actually_wrote():
    """Off-index sidecars never appear in unexpected_keys, so the unplaced list alone misses them.

    GLM-4.7 ships its MTP head in a standalone ``mtp.safetensors`` that the loader never opens.
    The export copies it verbatim, so its tensors are in the checkpoint in original precision and
    must reach ``exclude_modules`` -- the same requirement as a carried weight, via the other
    mechanism. The export records both under ``_modelopt_carried_over_names``.
    """
    model = torch.nn.Module()
    model._modelopt_unplaced_source_keys = ["model.mtp.eh_proj.weight"]
    # What the export actually wrote: the carried weight plus the copied sidecar's tensors.
    model._modelopt_carried_over_names = [
        "model.mtp.eh_proj.weight",
        "model.mtp.embed_tokens.weight",  # only in the sidecar
    ]
    assert _get_carried_over_module_names(model) == [
        "model.mtp.eh_proj",
        "model.mtp.embed_tokens",
    ]


def test_carried_over_names_fall_back_before_the_export_records():
    """Callers that never ran the export still get the wider unplaced answer."""
    model = torch.nn.Module()
    model._modelopt_unplaced_source_keys = ["model.mtp.eh_proj.weight"]
    assert _get_carried_over_module_names(model) == ["model.mtp.eh_proj"]

    # An export that wrote nothing is an answer, not a missing one -- do not fall back to the
    # wider list and claim exclusions for weights the checkpoint does not contain.
    model._modelopt_carried_over_names = []
    assert _get_carried_over_module_names(model) == []


def test_seed_carried_over_exclusions_adds_missing_names():
    """The layerwise exporter snapshots its config before the carried set exists, so it re-seeds."""
    model = torch.nn.Module()
    model._modelopt_carried_over_names = [
        "model.mtp.eh_proj.weight",
        "model.mtp.embed_tokens.weight",
    ]
    cfg = {"quantization": {"quant_algo": "NVFP4", "exclude_modules": ["lm_head"]}}
    added = seed_carried_over_exclusions(model, cfg)

    assert added == ["model.mtp.eh_proj", "model.mtp.embed_tokens"]
    assert cfg["quantization"]["exclude_modules"] == [
        "lm_head",
        "model.mtp.eh_proj",
        "model.mtp.embed_tokens",
    ]


def test_seed_carried_over_exclusions_respects_existing_wildcards():
    """A recipe that already excluded mtp* must not gain redundant per-module entries."""
    model = torch.nn.Module()
    model._modelopt_carried_over_names = ["model.mtp.eh_proj.weight"]
    cfg = {"quantization": {"quant_algo": "NVFP4", "exclude_modules": ["model.mtp*"]}}

    assert seed_carried_over_exclusions(model, cfg) == []
    assert cfg["quantization"]["exclude_modules"] == ["model.mtp*"]


def test_seed_carried_over_exclusions_noop_without_a_uniform_format():
    """No single quant_algo means nothing for a deployment framework to misapply."""
    model = torch.nn.Module()
    model._modelopt_carried_over_names = ["model.mtp.eh_proj.weight"]
    for algo in (None, "MIXED_PRECISION"):
        cfg = {"quantization": {"quant_algo": algo}}
        assert seed_carried_over_exclusions(model, cfg) == []
        assert "exclude_modules" not in cfg["quantization"]


def test_both_export_paths_exclude_carried_weights_identically():
    """The unified and layerwise exporters must emit the same exclusions for the same model.

    They reach exclude_modules at different times -- get_quant_config after its per-layer pass,
    the layerwise exporter from finalize() because bind() snapshotted its config during
    calibration -- so it is easy for them to drift into different formats (wildcards one side,
    literals the other) for identical inputs. Both go through seed_carried_over_exclusions now;
    this pins that they agree.
    """
    hidden = 16
    model = _FakeMoEModel(hidden=hidden)
    mtq.quantize(model, _nvfp4_all_linears_config, lambda m: m(torch.randn(2, hidden)))
    model._modelopt_carried_over_names = [
        "model.mtp.eh_proj.weight",
        "model.mtp.embed_tokens.weight",
    ]

    # Unified: seeded inside get_quant_config.
    unified = get_quant_config(model)["quantization"]["exclude_modules"]

    # Layerwise: the same config minus the carried names, re-seeded at finalize() time.
    layerwise_cfg = {
        "quantization": {
            "quant_algo": "NVFP4",
            "exclude_modules": [e for e in unified if not e.startswith("model.mtp")],
        }
    }
    seed_carried_over_exclusions(model, layerwise_cfg)

    assert sorted(layerwise_cfg["quantization"]["exclude_modules"]) == sorted(unified)


def test_seeded_exclusions_are_literal_module_names():
    """Exact names, not prefix wildcards: a literal cannot over-match a quantized module."""
    model = torch.nn.Module()
    model._modelopt_carried_over_names = ["model.mtp.eh_proj.weight"]
    cfg = {"quantization": {"quant_algo": "NVFP4", "exclude_modules": []}}

    assert seed_carried_over_exclusions(model, cfg) == ["model.mtp.eh_proj"]
    assert cfg["quantization"]["exclude_modules"] == ["model.mtp.eh_proj"]
    assert not any("*" in e for e in cfg["quantization"]["exclude_modules"])
