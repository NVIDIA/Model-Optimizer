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

"""Convert modelopt quantization export config to align with llm-compressor config format."""

from collections import defaultdict
from typing import Any


def _quant_algo_to_group_config(quant_algo: str, group_size: int | None = None) -> dict[str, Any]:
    """Map a per-layer quant_algo string to compressed-tensors config group details.

    Args:
        quant_algo: The quantization algorithm name (e.g. "FP8", "NVFP4").
        group_size: Optional group size for block-wise quantization algorithms.

    Returns:
        Dictionary with ``input_activations`` and ``weights`` entries suitable for
        a compressed-tensors ``config_groups`` entry.
    """
    if quant_algo == "FP8":
        return {
            "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
            "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
        }
    elif quant_algo == "FP8_PER_CHANNEL_PER_TOKEN":
        return {
            "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
            "weights": {"dynamic": False, "num_bits": 8, "type": "float", "strategy": "channel"},
        }
    elif quant_algo == "NVFP4":
        gs = group_size if group_size else 16
        return {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": gs,
            },
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": gs},
        }
    elif quant_algo in ("NVFP4_AWQ", "W4A16_AWQ", "W4A8_AWQ"):
        gs = group_size if group_size else 128
        weight_bits = 4
        act_bits = 8 if "A8" in quant_algo else 4
        return {
            "input_activations": {
                "dynamic": False,
                "num_bits": act_bits,
                "type": "float",
                "group_size": gs,
            },
            "weights": {
                "dynamic": False,
                "num_bits": weight_bits,
                "type": "float",
                "group_size": gs,
            },
        }
    elif quant_algo == "W8A16":
        return {
            "weights": {"dynamic": False, "num_bits": 8, "type": "int"},
        }
    elif quant_algo == "W8A8_SQ_PER_CHANNEL":
        return {
            "input_activations": {"dynamic": False, "num_bits": 8, "type": "int"},
            "weights": {
                "dynamic": False,
                "num_bits": 8,
                "type": "int",
                "strategy": "channel",
            },
        }
    elif quant_algo in ("W4A8_NVFP4_FP8", "W4A8_MXFP4_FP8"):
        gs = group_size if group_size else 16
        return {
            "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": gs},
        }
    elif quant_algo == "MXFP8":
        gs = group_size if group_size else 32
        return {
            "input_activations": {
                "dynamic": False,
                "num_bits": 8,
                "type": "float",
                "group_size": gs,
            },
            "weights": {"dynamic": False, "num_bits": 8, "type": "float", "group_size": gs},
        }
    else:
        # Fallback: store the raw algo name so downstream consumers can still inspect it.
        return {"quant_algo": quant_algo}


def convert_hf_quant_config_format(input_config: dict[str, Any]) -> dict[str, Any]:
    """Converts modelopt quantization config dictionary to align with llm-compressor config format.

    Args:
        input_config: The original quantization config dictionary.

    Note:
        The "targets" field specifies which PyTorch module types to quantize. Compressed-tensors
        works with any PyTorch module type and uses dynamic matching against module.__class__.__name__.
        Typically this includes "Linear" modules, but can also include "Embedding" and other types.

        See: https://github.com/neuralmagic/compressed-tensors/blob/fa6a48f1da6b47106912bcd25eba7171ba7cfec7/src/sparsetensors/quantization/quant_scheme.py#L29
        Example usage: https://github.com/neuralmagic/compressed-tensors/blob/9938a6ec6e10498d39a3071dfd1c40e3939ee80b/tests/test_quantization/lifecycle/test_apply.py#L118

    Example:

        .. code-block:: python

            {
                "producer": {"name": "modelopt", "version": "0.19.0"},
                "quantization": {
                    "quant_algo": "FP8",
                    "kv_cache_quant_algo": "FP8",
                    "exclude_modules": ["lm_head"],
                },
            }

    Returns:
        A new dictionary in the target format.

        Example (for FP8 input):

        .. code-block:: python

            {
                "config_groups": {
                    "group_0": {
                        "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                        "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    }
                },
                "ignore": ["lm_head"],
                "quant_algo": "FP8",
                "kv_cache_scheme": "FP8",
                "producer": {"name": "modelopt", "version": "0.29.0"},
            }
    """
    new_config: dict[str, Any] = {}

    original_quantization_details = input_config.get("quantization", {})
    quant_algo_value = original_quantization_details.get("quant_algo")

    # This structure is derived based on the example for "FP8" and "NVFP4"
    # TODO: Handle other quantization algorithms
    if quant_algo_value == "FP8":
        config_group_details = {
            "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
            "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
            "targets": ["Linear"],
        }
        new_config["config_groups"] = {"group_0": config_group_details}
    elif quant_algo_value == "NVFP4":
        group_size = original_quantization_details.get("group_size", 16)
        config_group_details = {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": group_size,
            },
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": group_size},
            "targets": ["Linear"],
        }
        new_config["config_groups"] = {"group_0": config_group_details}
    elif quant_algo_value == "MIXED_PRECISION":
        quantized_layers = original_quantization_details.get("quantized_layers", {})

        # Group layers by their unique quantization config so each distinct
        # (quant_algo, group_size, ...) combination becomes one config_group.
        algo_to_layers: dict[tuple, list[str]] = defaultdict(list)
        for layer_name, layer_cfg in quantized_layers.items():
            # Create a hashable key from the layer config
            key = tuple(sorted(layer_cfg.items()))
            algo_to_layers[key].append(layer_name)

        config_groups: dict[str, Any] = {}
        for idx, (config_key, layer_names) in enumerate(algo_to_layers.items()):
            layer_cfg = dict(config_key)
            algo = layer_cfg.get("quant_algo", "")
            layer_group_size = layer_cfg.get("group_size")

            group_config = _quant_algo_to_group_config(algo, layer_group_size)
            group_config["targets"] = sorted(layer_names)
            config_groups[f"group_{idx}"] = group_config

        new_config["config_groups"] = config_groups
        # Preserve the full per-layer detail for consumers that need it.
        new_config["quantized_layers"] = quantized_layers

    exclude_modules = original_quantization_details.get("exclude_modules")

    new_config["ignore"] = exclude_modules if exclude_modules is not None else []

    if quant_algo_value:
        new_config["quant_algo"] = quant_algo_value

    kv_cache_quant_algo = original_quantization_details.get("kv_cache_quant_algo")
    if kv_cache_quant_algo:
        if kv_cache_quant_algo == "FP8":
            new_config["kv_cache_scheme"] = {"dynamic": False, "num_bits": 8, "type": "float"}
        else:
            # TODO: Handle other kv cache quantization algorithms
            new_config["kv_cache_scheme"] = kv_cache_quant_algo

    producer_info = input_config.get("producer")
    if producer_info:
        new_config["producer"] = producer_info

    new_config["quant_method"] = "modelopt"

    return new_config
