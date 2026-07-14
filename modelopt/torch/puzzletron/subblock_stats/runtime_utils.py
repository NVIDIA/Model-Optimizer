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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedModel

from ..anymodel.converter import Converter


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
    # vLLM benchmark concurrency (``--max-num-seqs``). ``None`` keeps the
    # historical single-stream behavior (max-num-seqs=1), in which the
    # ``batch_size`` prompts run one at a time. Set this to ``batch_size`` (or
    # higher) so the prompts run concurrently and the measured latency reflects
    # true batched throughput. Note the KV cache for ``max_num_seqs`` concurrent
    # sequences at ``prefill_seq_len + generation_seq_len`` must fit in GPU
    # memory, which bounds the usable value at long context.
    max_num_seqs: int | None = None

    def model_config_value(self, key: str, default: Any = None) -> Any:
        """Return a descriptor-specific benchmark config value."""
        return dict(self.model_config_fields).get(key, default)


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
    """Save a model checkpoint in AnyModel subblock-safetensors format."""
    # Save standard model checkpoint (as safetensors, HF format)
    model.save_pretrained(output_dir, safe_serialization=True)

    # Convert/slice weights into AnyModel subblock_safetensors format
    Converter.convert_model_weights(
        input_dir=output_dir,
        output_dir=output_dir,
        descriptor=descriptor,
        num_hidden_layers=descriptor.get_language_model_config(model.config).num_hidden_layers,
    )
    # Load the model config.json, update "architectures" to ["AnyModel"], and write back to disk.

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        config_data["architectures"] = ["AnyModel"]
        (runtime_descriptor or descriptor).update_runtime_benchmark_config(config_data)
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
