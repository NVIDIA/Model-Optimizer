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

import modelopt.torch.export.quant_format as quant_format
import modelopt.torch.quantization.ggml as ggml
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.quant_format import IQ_FORMATS
from modelopt.torch.export.unified_export_hf import _revert_hf_quant_config_names
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY


def _geometry(fmt):
    """Block geometry from the codec's own constants, independent of the registry under test."""
    upper = fmt.upper()
    return (
        getattr(ggml, f"{upper}_BLOCK_SIZE"),
        getattr(ggml, f"{upper}_BLOCK_BYTES"),
        getattr(ggml, f"{upper}_EFFECTIVE_BITS"),
    )


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


@pytest.mark.parametrize("fmt", sorted(IQ_FORMATS))
def test_iq_config_carries_block_metadata(fmt):
    """Every IQ format must describe its packed block, not just name itself.

    A consumer reads group_size and block_payload_bytes to walk the payload, so a format
    that falls through to the generic branch produces a checkpoint that cannot be decoded.
    """
    block_size, payload_bytes, effective_bits = _geometry(fmt)
    converted = convert_hf_quant_config_format(
        {
            "producer": {"name": "modelopt", "version": "test"},
            "quantization": {"quant_algo": fmt.upper()},
        }
    )

    assert converted["quant_algo"] == fmt.upper()
    assert converted["group_size"] == block_size
    assert converted["block_payload_bytes"] == payload_bytes
    assert converted["effective_bits"] == pytest.approx(effective_bits)
    assert converted["packing"] == "ggml"
    # IQ payloads are self-contained blocks, not compressed-tensors integer groups.
    assert "config_groups" not in converted


@pytest.mark.parametrize("fmt", sorted(IQ_FORMATS))
def test_iq_config_rejects_mismatched_group_size(fmt):
    """A caller's group size is rejected rather than silently rewritten to the block size."""
    block_size, _, _ = _geometry(fmt)
    with pytest.raises(ValueError, match=f"requires group size {block_size}"):
        convert_hf_quant_config_format(
            {
                "producer": {"name": "modelopt", "version": "test"},
                "quantization": {"quant_algo": fmt.upper(), "group_size": block_size // 2},
            }
        )


def test_export_formats_are_the_registered_formats():
    """Export and backend dispatch agree on which IQ formats exist, name constants included."""
    assert frozenset(IQ_FORMAT_REGISTRY) == IQ_FORMATS
    constants = {v for k, v in vars(quant_format).items() if k.startswith("QUANTIZATION_IQ")}
    assert constants == IQ_FORMATS


@pytest.mark.parametrize("fmt", sorted(IQ_FORMATS))
def test_iq_mixed_precision_config_group_carries_block_metadata(fmt):
    """A per-layer IQ config must describe its block too, not only a uniform one.

    Mixed exports route each distinct layer config through the same helper, so a format
    missing there loses its geometry for exactly the layers that use it.
    """
    block_size, payload_bytes, effective_bits = _geometry(fmt)
    converted = convert_hf_quant_config_format(
        {
            "producer": {"name": "modelopt", "version": "test"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.mlp.gate_proj": {"quant_algo": fmt.upper()},
                    "model.layers.1.mlp.up_proj": {"quant_algo": "FP8"},
                },
            },
        }
    )

    groups = converted["config_groups"].values()
    iq_group = next(g for g in groups if g.get("quant_algo") == fmt.upper())
    assert iq_group["group_size"] == block_size
    assert iq_group["block_payload_bytes"] == payload_bytes
    assert iq_group["effective_bits"] == pytest.approx(effective_bits)
    assert iq_group["packing"] == "ggml"
    assert iq_group["targets"] == ["model.layers.0.mlp.gate_proj"]
    # The FP8 layer keeps its own compressed-tensors scheme.
    assert any("weights" in g for g in groups)


@pytest.mark.parametrize("fmt", sorted(IQ_FORMATS))
def test_iq_mixed_precision_rejects_bad_per_layer_group_size(fmt):
    """A per-layer group size is validated, not silently rewritten to the block size."""
    block_size, _, _ = _geometry(fmt)
    with pytest.raises(ValueError, match=f"requires group size {block_size}"):
        convert_hf_quant_config_format(
            {
                "producer": {"name": "modelopt", "version": "test"},
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "model.layers.0.mlp.gate_proj": {
                            "quant_algo": fmt.upper(),
                            "group_size": block_size // 2,
                        }
                    },
                },
            }
        )
