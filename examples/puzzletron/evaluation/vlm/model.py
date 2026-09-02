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

"""Checkpoint contract for the Qwen 3.5 VLM backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["no_think_chat_template", "verify_checkpoint"]

_MODEL_TYPE = "qwen3_5"
_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_PROCESSOR_ASSETS = ("preprocessor_config.json", "video_preprocessor_config.json")
_TEXT_GEOMETRY_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "vocab_size",
)
_NO_THINK_TEMPLATE_PREFIX = "{%- set enable_thinking = false %}\n"
_NO_THINK_GENERATION_PREFIX = "<think>\n\n</think>\n\n"


def no_think_chat_template(checkpoint: Path, output_directory: Path) -> Path:
    """Copy a checkpoint chat template and verify that thinking is disabled."""
    source = checkpoint / "chat_template.jinja"
    try:
        content = source.read_text()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Qwen 3.5 chat template is unreadable: {source}") from error
    candidate = _NO_THINK_TEMPLATE_PREFIX + content
    if not _render_template(candidate, source=source).endswith(_NO_THINK_GENERATION_PREFIX):
        raise ValueError(f"Qwen 3.5 chat template cannot disable thinking: {source}")
    target = output_directory / "modelopt_qwen35_no_think.jinja"
    target.write_text(candidate)
    return target


def _render_template(content: str, *, source: Path) -> str:
    try:
        return (
            ImmutableSandboxedEnvironment()
            .from_string(content)
            .render(
                add_generation_prompt=True,
                messages=[{"content": "test", "role": "user"}],
            )
        )
    except TemplateError as error:
        raise ValueError(f"Qwen 3.5 chat template is invalid: {source}") from error


def verify_checkpoint(checkpoint: Path, *, profile: str) -> None:
    """Verify a Qwen 3.5 VLM-family checkpoint and its local processor assets."""
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
    invalid_geometry = {
        key: text_config.get(key)
        for key in _TEXT_GEOMETRY_FIELDS
        if not isinstance(text_config.get(key), int)
        or isinstance(text_config.get(key), bool)
        or text_config.get(key) <= 0
    }
    if invalid_geometry:
        raise ValueError(f"{profile} checkpoint has invalid Qwen 3.5 geometry: {invalid_geometry}")
    processor_assets = []
    for name in _PROCESSOR_ASSETS:
        path = checkpoint / name
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{profile} checkpoint processor asset is invalid: {name}") from error
        if not isinstance(config, dict):
            raise ValueError(f"{profile} checkpoint processor asset must be an object: {name}")
        processor_assets.append(name)
    if not processor_assets:
        raise ValueError(f"{profile} checkpoint requires local multimodal processor assets")


def _checkpoint_config(checkpoint: Path) -> dict[str, object]:
    try:
        config = json.loads((checkpoint / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"checkpoint config.json is unreadable: {checkpoint}") from error
    if not isinstance(config, dict):
        raise RuntimeError(f"checkpoint config.json must contain an object: {checkpoint}")
    return config
