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

"""Checkpoint contract for the initial Qwen 3.5 VLM backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["verify_checkpoint"]

_MODEL_TYPE = "qwen3_5"
_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_TEXT_GEOMETRY = {
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "vocab_size": 248320,
}


def verify_checkpoint(checkpoint: Path, *, profile: str) -> None:
    """Verify the exact Qwen 3.5 0.8B VLM architecture and text geometry."""
    config = _checkpoint_config(checkpoint)
    if config.get("model_type") != _MODEL_TYPE:
        raise ValueError(f"{profile} checkpoint model_type must be qwen3_5")
    architectures = config.get("architectures")
    native_checkpoint = architectures == [_ARCHITECTURE]
    realized_checkpoint = (
        architectures == ["AnyModel"] and config.get("base_architecture") == _ARCHITECTURE
    )
    if not native_checkpoint and not realized_checkpoint:
        raise ValueError(
            f"{profile} checkpoint must identify {_ARCHITECTURE} directly or as the "
            "AnyModel base_architecture"
        )
    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen3_5_text":
        raise ValueError(f"{profile} checkpoint text_config.model_type must be qwen3_5_text")
    mismatches = {
        key: (text_config.get(key), expected)
        for key, expected in _TEXT_GEOMETRY.items()
        if text_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint geometry differs from Qwen 3.5 0.8B: {mismatches}")


def _checkpoint_config(checkpoint: Path) -> dict[str, object]:
    try:
        config = json.loads((checkpoint / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"checkpoint config.json is unreadable: {checkpoint}") from error
    if not isinstance(config, dict):
        raise RuntimeError(f"checkpoint config.json must contain an object: {checkpoint}")
    return config
