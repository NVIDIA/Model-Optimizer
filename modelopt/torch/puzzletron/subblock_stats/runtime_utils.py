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
"""Utilities for runtime benchmarking and model saving in ModelOpt NAS.

This module provides classes and utility functions used for empirical runtime
estimation of Transformer subblocks and for saving models and tokenizers in
formats suitable for benchmarking (e.g., vLLM latency benchmark) or the
AnyModel subblock-safetensors format. It defines the configuration dataclass
used to parameterize runtime benchmarks, as well as model checkpointing helpers
to ensure compatibility with downstream evaluation pipelines.
"""

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

from ..anymodel.converter import Converter
from ..anymodel.model_descriptor import ModelDescriptor
from ..anymodel.models.llama import LlamaModelDescriptor
from ..tools.logger import mprint
from ..utils.vllm_adapter import convert_block_configs_to_per_layer_config

# Hybrid Nemotron-H benchmarks are saved as native HF checkpoints for vLLM.
# AnyModel subblock safetensors are not loadable for these architectures yet.
_NATIVE_VLLM_HYBRID_DESCRIPTOR_NAMES = frozenset(
    {"NemotronHModelDescriptor", "NemotronHV2ModelDescriptor"}
)

# vLLM requires an explicit mamba cache block size for hybrid Mamba models.
# Must be a multiple of 8; 16 matches vLLM CacheConfig.DEFAULT_BLOCK_SIZE.
DEFAULT_MAMBA_BLOCK_SIZE = 16


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for a vLLM latency benchmark run."""

    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    tokenizer_path: str
    repeat_block_n_times: int
    prefill_seq_len: int
    generation_seq_len: int
    batch_size: int
    num_iters: int
    num_warmup_iters: int
    # Fraction of total GPU memory vLLM may use. Kept well below the default
    # (~0.9) because the parent puzzletron process is co-resident on the same
    # GPU during benchmarking; requesting too much fails vLLM's startup
    # free-memory check.
    gpu_memory_utilization: float = 0.5
    benchmark_model_key: str = "llama"
    benchmark_model_config: PretrainedConfig | None = field(default=None, compare=False, hash=False)
    benchmark_model_descriptor: type[ModelDescriptor] = field(
        default=LlamaModelDescriptor, compare=False, hash=False
    )
    # Required by vLLM v1 for hybrid Mamba models (e.g. Nemotron-H). Set automatically
    # when benchmarking Mamba subblocks; override via runtime_stats_config.
    mamba_block_size: int | None = None


def save_model(
    model: PreTrainedModel,
    tokenizer_path: Path,
    output_path: Path,
    descriptor: type[ModelDescriptor] = LlamaModelDescriptor,
) -> None:
    """Save a benchmark checkpoint and copy the tokenizer to ``output_path``."""
    model = model.to(dtype=torch.bfloat16)
    if descriptor.__name__ in _NATIVE_VLLM_HYBRID_DESCRIPTOR_NAMES:
        _save_model_as_native_hf(model, output_path)
    else:
        _save_model_as_anymodel(model, output_path, descriptor)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=descriptor.requires_trust_remote_code()
    )
    tokenizer.save_pretrained(output_path)


def _nemotron_h_weight_key_for_vllm(key: str) -> str:
    """Normalize Nemotron-H state-dict keys to the HF layout vLLM's loader expects.

    ``save_pretrained`` may emit the on-disk "original" format (``embedding.*`` /
    ``backbone.embedding.*``). vLLM's Nemotron-H weight mapper only rewrites
    ``backbone.embeddings`` to ``model.embed_tokens``.
    """
    if key == "embedding.weight":
        return "backbone.embeddings.weight"
    if key.startswith("embedding."):
        return "backbone.embeddings." + key.removeprefix("embedding.")

    key = key.replace("backbone.embedding.", "backbone.embeddings.")

    if key.startswith("model.embed_tokens"):
        return key.replace("model.embed_tokens", "backbone.embeddings", 1)
    if key.startswith("model.layers."):
        return "backbone." + key.removeprefix("model.")
    if key.startswith("model.norm_f"):
        return key.replace("model.norm_f", "backbone.norm_f", 1)
    return key


def _save_model_as_native_hf(model: PreTrainedModel, output_dir: Path) -> None:
    """Save a homogeneous hybrid benchmark model in native HF format for vLLM."""
    from safetensors.torch import save_file

    base_architecture = getattr(model.config, "base_architecture", None)
    if not base_architecture:
        architectures = getattr(model.config, "architectures", None) or []
        base_architecture = architectures[0] if architectures else "NemotronHForCausalLM"

    lm_config = model.config
    if hasattr(model.config, "get_text_config"):
        lm_config = model.config.get_text_config()

    if hasattr(lm_config, "block_configs"):
        lm_config.block_configs = None

    model.config.architectures = [base_architecture]
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(output_dir)

    # Write weights directly from the in-memory state dict. Using
    # ``save_pretrained`` for weights can revert Nemotron-H keys to the original
    # on-disk format (``embedding.weight``), which vLLM cannot load.
    state_dict = {
        _nemotron_h_weight_key_for_vllm(key): tensor.contiguous().cpu()
        for key, tensor in model.state_dict().items()
    }
    save_file(state_dict, output_dir / "model.safetensors")

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        config_data["architectures"] = [base_architecture]
        config_data.pop("block_configs", None)
        config_data.pop("base_architecture", None)
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)


def _save_model_as_anymodel(
    model: PreTrainedModel, output_dir: Path, descriptor: type[ModelDescriptor]
) -> None:
    """Save a model checkpoint in AnyModel subblock-safetensors format."""
    # Save standard model checkpoint (as safetensors, HF format)
    save_pretrained_kwargs = {"safe_serialization": True}
    if "save_original_format" in inspect.signature(model.save_pretrained).parameters:
        save_pretrained_kwargs["save_original_format"] = False
    model.save_pretrained(output_dir, **save_pretrained_kwargs)

    # Convert/slice weights into AnyModel subblock_safetensors format
    Converter.convert_model_weights(
        input_dir=output_dir,
        output_dir=output_dir,
        descriptor=descriptor,
        num_hidden_layers=model.config.num_hidden_layers,
    )
    # Load the model config.json, update "architectures" to ["AnyModel"], and write back to disk.

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        config_data["architectures"] = ["AnyModel"]
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)


def convert_config_to_vllm_anymodel(input_dir: Path, output_dir: Path):
    """Convert a model to vLLM AnyModel format."""
    # Load the model config.json, update "architectures" to ["AnyModel"], and write back to disk.
    input_config_path = Path(input_dir) / "config.json"
    if not input_config_path.exists():
        raise FileNotFoundError(f"Config file not found at {input_config_path}")
    try:
        with open(input_config_path) as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error loading config file: {e}") from e

    config = SimpleNamespace(**config_data)
    config.architectures = ["AnyModel"]
    if not getattr(config, "base_architecture", None):
        config.base_architecture = "LlamaForCausalLM"

    if convert_block_configs_to_per_layer_config(config):
        mprint("Converted block configs to per-layer config")
    else:
        mprint("No block configs to convert")
    with open(Path(output_dir) / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)
