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

"""Translate typed Puzzletron block configs to Megatron-Core's heterogeneous schema."""

from collections.abc import Iterable
from typing import Any

from ..block_config import AttentionConfig, BlockConfig, FFNConfig

__all__ = ["build_mcore_heterogeneous_config"]


def _coerce_block_config(block_config: BlockConfig | dict[str, Any]) -> BlockConfig:
    return block_config if isinstance(block_config, BlockConfig) else BlockConfig(**block_config)


def _require_dense_subblocks(block_config: BlockConfig) -> tuple[AttentionConfig, FFNConfig]:
    subblocks: dict[str, Any] = {}
    for subblock in block_config.subblock_configs:
        if subblock.kind not in {"attention", "ffn"}:
            raise ValueError(
                "Megatron-Bridge heterogeneous configs support only attention and ffn "
                f"subblocks, got {subblock.kind!r}"
            )
        if subblock.kind in subblocks:
            raise ValueError(
                "Megatron-Bridge heterogeneous configs require exactly one subblock of each "
                f"supported kind, got duplicate {subblock.kind!r} subblocks"
            )
        subblocks[subblock.kind] = subblock

    missing = {"attention", "ffn"} - subblocks.keys()
    if missing:
        raise ValueError(
            "Megatron-Bridge heterogeneous configs require attention and ffn subblocks, "
            f"missing {sorted(missing)}"
        )

    attention = subblocks["attention"]
    ffn = subblocks["ffn"]
    if not isinstance(attention, AttentionConfig) or not isinstance(ffn, FFNConfig):
        raise TypeError("attention and ffn subblocks must use their typed Puzzletron configs")
    return attention, ffn


def _convert_attention_config(
    attention: AttentionConfig, *, num_attention_heads: int
) -> dict[str, Any]:
    unsupported_fields = {
        field
        for field in (
            "qk_head_dim",
            "v_head_dim",
            "sliding_window_size",
            "k_eq_v",
            "kv_source_layer",
            "llama4",
        )
        if getattr(attention, field) is not None
    }
    if unsupported_fields:
        raise ValueError(
            "Megatron-Bridge heterogeneous configs cannot represent attention fields "
            f"{sorted(unsupported_fields)}"
        )
    if attention.num_query_heads not in {None, num_attention_heads}:
        raise ValueError(
            "Megatron-Bridge heterogeneous configs cannot vary num_query_heads by layer; "
            f"expected {num_attention_heads}, got {attention.num_query_heads}"
        )
    if attention.num_kv_heads is not None and (
        attention.num_kv_heads <= 0 or num_attention_heads % attention.num_kv_heads
    ):
        raise ValueError(
            f"num_kv_heads ({attention.num_kv_heads}) must be a positive divisor of "
            f"num_attention_heads ({num_attention_heads})"
        )
    return {
        "no_op": attention.no_op,
        "num_query_groups": attention.num_kv_heads,
    }


def _convert_ffn_config(ffn: FFNConfig) -> dict[str, Any]:
    if ffn.intermediate_size is not None and ffn.intermediate_size <= 0:
        raise ValueError(f"intermediate_size must be positive, got {ffn.intermediate_size}")
    return {
        "no_op": ffn.no_op,
        "ffn_hidden_size": ffn.intermediate_size,
    }


def build_mcore_heterogeneous_config(
    block_configs: Iterable[BlockConfig | dict[str, Any]], *, num_attention_heads: int
) -> dict[str, list[dict[str, Any]]]:
    """Build the JSON-compatible block schema consumed by Megatron-Core.

    Megatron-Core's heterogeneous transformer currently supports per-layer KV
    groups, FFN width, and no-op attention or FFN blocks. Other Puzzletron axes
    are rejected here so the bridge never silently drops an admitted axis.
    """

    if num_attention_heads <= 0:
        raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")

    converted = []
    for index, raw_block_config in enumerate(block_configs):
        try:
            attention, ffn = _require_dense_subblocks(_coerce_block_config(raw_block_config))
            converted.append(
                {
                    "attention": _convert_attention_config(
                        attention, num_attention_heads=num_attention_heads
                    ),
                    "ffn": _convert_ffn_config(ffn),
                }
            )
        except (TypeError, ValueError) as error:
            raise type(error)(f"block_configs[{index}]: {error}") from error

    if not converted:
        raise ValueError("block_configs must contain at least one layer")
    return {"block_configs": converted}
