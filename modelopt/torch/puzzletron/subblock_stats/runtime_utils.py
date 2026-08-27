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
"""Utilities for runtime benchmarking and model saving in Puzzletron.

This module provides classes and utility functions used for empirical runtime
estimation of Transformer subblocks and for saving models and tokenizers in
formats suitable for benchmarking with vLLM.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedModel

from ..export.vllm import prepare_vllm_config
from .topology import RuntimeTopology


def _thaw_config_value(value: Any) -> Any:
    """Restore mappings frozen into tuples for hashable runtime identities."""
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw_config_value(item) for key, item in value}
        return tuple(_thaw_config_value(item) for item in value)
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for a vLLM latency benchmark run."""

    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    descriptor: type
    model_config_fields: tuple[tuple[str, Any], ...]
    tokenizer_path: str
    repeat_block_n_times: int
    prefill_seq_len: int
    generation_seq_len: int
    batch_size: int
    num_iters: int
    num_warmup_iters: int
    extra_vllm_args: tuple[str, ...] = ()
    # vLLM benchmark concurrency (``--max-num-seqs``). ``None`` keeps the
    # historical single-stream behavior (max-num-seqs=1), in which the
    # ``batch_size`` prompts run one at a time. Set this to ``batch_size`` (or
    # higher) so the prompts run concurrently and the measured latency reflects
    # true batched throughput. Note the KV cache for ``max_num_seqs`` concurrent
    # sequences at ``prefill_seq_len + generation_seq_len`` must fit in GPU
    # memory, which bounds the usable value at long context.
    max_num_seqs: int | None = None
    topology: RuntimeTopology = RuntimeTopology()
    estimator_schema: str = "candidate_slope_v1"
    estimator_mode: str = "homogeneous"
    effective_repeat_count: int | None = None
    scaffold_policy: str = "none"
    vllm_env: tuple[tuple[str, str], ...] = ()

    def model_config_value(self, key: str, default: Any = None) -> Any:
        """Return a descriptor-specific benchmark config value."""
        return _thaw_config_value(dict(self.model_config_fields).get(key, default))


def save_model(
    model: PreTrainedModel, tokenizer_path: Path, output_path: Path, descriptor: type
) -> None:
    """Save model weights as AnyModel and copy the tokenizer to ``output_path``."""
    model = model.to(dtype=torch.bfloat16)
    save_model_as_anymodel(
        model,
        output_path,
        descriptor.runtime_benchmark_export_descriptor(),
        runtime_descriptor=descriptor,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_path)


def save_model_as_anymodel(model, output_dir: Path, descriptor, runtime_descriptor=None):
    """Save a temporary vLLM-compatible AnyModel benchmark checkpoint."""
    model.save_pretrained(output_dir, safe_serialization=True)
    descriptor_for_config = runtime_descriptor or descriptor
    descriptor_for_config.postprocess_runtime_benchmark_checkpoint(output_dir)

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        arch_info = dict(descriptor_for_config.anymodel_arch_info())
        arch_info.update(dict(config_data.get("anymodel_arch_info") or {}))
        config_data["anymodel_arch_info"] = arch_info
        descriptor_for_config.update_runtime_benchmark_config(config_data)
        prepare_vllm_config(config_data, descriptor_name=descriptor_for_config.__name__)
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)


def convert_config_to_vllm_anymodel(config_dir: Path):
    """Convert a model to vLLM AnyModel format."""
    # Load the model config.json, update "architectures" to ["AnyModel"], and write back to disk.
    config_path = Path(config_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    backup_config_path = config_path.with_suffix(".bak")
    if backup_config_path.exists():
        raise FileExistsError(f"Backup config file already exists at {backup_config_path}")

    shutil.copy(config_path, backup_config_path)

    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error loading config file: {e}") from e

    config = SimpleNamespace(**config_data)
    config.architectures = ["AnyModel"]
    config.base_architecture = "LlamaForCausalLM"  # TODO: extend support to other models

    if convert_block_configs_to_per_layer_config(config):
        mprint("Converted block configs to per-layer config")
    else:
        mprint("No block configs to convert")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)


if __name__ == "__main__":
    import fire

    fire.Fire()
