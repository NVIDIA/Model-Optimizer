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

"""Matching random-weight text prefixes for the local-distillation performance study."""

import copy
import importlib
from dataclasses import dataclass

import torch
from transformers import AutoConfig

__all__ = ["build_unpartitioned_models"]


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    model_class: str
    model_id: str
    revision: str


_MODEL_SPECS = {
    "qwen3": ModelSpec(
        "qwen3_moe",
        "Qwen3MoeForCausalLM",
        "Qwen/Qwen3-30B-A3B",
        "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    ),
    "moonlight": ModelSpec(
        "deepseek_v3",
        "DeepseekV3ForCausalLM",
        "moonshotai/Moonlight-16B-A3B",
        "476b36a473d4467f94469414bef6cee75c9c8172",
    ),
}


def build_unpartitioned_models(args, device):
    """Build identical frozen BF16 Transformers prefixes without downloading weights."""
    spec = _MODEL_SPECS[args.family]
    args.model = args.model or spec.model_id
    if args.revision is None and args.model == spec.model_id:
        args.revision = spec.revision
    source = AutoConfig.from_pretrained(args.model, revision=args.revision)
    config = copy.deepcopy(getattr(source, "text_config", source))
    if config.model_type != spec.model_type:
        raise ValueError(
            f"Family {args.family!r} requires {spec.model_type!r}, got {config.model_type!r}."
        )
    if not 1 <= args.layers <= config.num_hidden_layers:
        raise ValueError("Prefix depth must be positive and no larger than the source depth.")
    args.original_context_limit = config.max_position_embeddings
    if args.seq_len > args.original_context_limit and not args.extend_context:
        raise ValueError(
            "Context exceeds the source config; explicitly opt in with --extend-context."
        )
    config.max_position_embeddings = max(args.original_context_limit, args.seq_len)
    args.effective_context_limit = config.max_position_embeddings
    args.source_model = getattr(config, "_benchmark_source_model", args.model)
    args.source_revision = getattr(config, "_benchmark_source_revision", args.revision)
    config.num_hidden_layers = args.layers
    config.dtype = torch.bfloat16
    config.use_cache = False
    config.tie_word_embeddings = False
    config._attn_implementation = "sdpa"
    config._experts_implementation = args.hf_experts
    torch.manual_seed(42)
    # Sample in FP32 before casting, preserving the original Qwen benchmark's init.
    hf_module = importlib.import_module(
        f"transformers.models.{spec.model_type}.modeling_{spec.model_type}"
    )
    init_device = device if getattr(args, "init_device", "cpu") == "cuda" else torch.device("cpu")
    with torch.device(init_device):
        canonical = getattr(hf_module, spec.model_class)(copy.deepcopy(config))
    canonical = canonical.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    canonical.config.dtype = torch.bfloat16
    teacher = canonical.to(device)
    del canonical
    teacher.eval().requires_grad_(False)
    return copy.deepcopy(teacher), teacher, config
