# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Shared conversation manipulation and augmentation utilities for dataset preparation.

Imported by make_nemotron_ptv2_dataset.py and make_nemotron_ptv3_dataset.py.

These scripts produce *input conversations* for synthetic data generation: the
conversations are fed to a target model which generates responses, producing
training data for the speculative-decoding draft model.

Conversation format
-------------------
Each conversation is stripped down to a skeleton of system + user turns only — all
assistant turns are removed.  The downstream generation pipeline (query.py) feeds this
skeleton to the target model turn-by-turn, appending each generated response before
sending the next user turn, so the model produces coherent multi-turn continuations.

Augmentations (``user_suffix``) are applied to *all* user messages so that the
language or style instruction is present at every turn — important for multi-turn
synthetic generation where the model must maintain the requested style throughout.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AugmentationSpec:
    """One augmentation variant.

    type: "user_suffix"   — appends ``text`` to the last user message.
          "system_prompt" — prepends a system message with ``content``.
    """

    type: str
    text: str = ""
    content: str = ""
    enabled: bool = True

    def __post_init__(self):
        if self.type not in ("user_suffix", "system_prompt"):
            raise ValueError(
                f"Unknown augmentation type '{self.type}'. "
                "Expected 'user_suffix' or 'system_prompt'."
            )
        if self.type == "user_suffix" and not self.text:
            raise ValueError("user_suffix augmentation requires a non-empty 'text' field.")
        if self.type == "system_prompt" and not self.content:
            raise ValueError("system_prompt augmentation requires a non-empty 'content' field.")


def load_augmentations(config_path: Path) -> list[AugmentationSpec]:
    """Load and validate augmentation specs from a YAML file, returning only enabled ones."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    specs = [AugmentationSpec(**entry) for entry in data.get("augmentations", [])]
    enabled = [s for s in specs if s.enabled]

    if not enabled:
        raise ValueError(
            f"No enabled augmentations found in {config_path}. "
            "Enable at least one entry or pass --no-augmentation."
        )

    logger.info(
        "Loaded %d augmentation spec(s) from %s (%d disabled).",
        len(enabled),
        config_path,
        len(specs) - len(enabled),
    )
    for spec in enabled:
        if spec.type == "user_suffix":
            logger.info("  [user_suffix]   %r", spec.text.strip())
        else:
            logger.info("  [system_prompt] %r", spec.content.strip())

    return enabled


def make_augment_fn(specs: list[AugmentationSpec]):
    """Return a datasets.map-compatible function that cycles through *specs* by row index.

    user_suffix: appended to ALL user messages so the language/style instruction is
    present at every turn — important for multi-turn synthetic generation where the
    model must maintain the requested language or style throughout the conversation.

    system_prompt: prepended only when the conversation has no existing system
    message.  If the dataset already provides a system prompt it is kept as-is and
    this augmentation variant is skipped for that row (returning it unchanged).
    """

    def _augment(example: dict[str, Any], idx: int) -> dict[str, Any]:
        spec = specs[idx % len(specs)]
        messages = [dict(m) for m in example["messages"]]  # shallow copy per row

        if spec.type == "user_suffix":
            for msg in messages:
                if msg["role"] == "user":
                    msg["content"] = msg["content"] + spec.text
        else:  # system_prompt
            has_system = messages and messages[0]["role"] == "system"
            if has_system:
                # Conflict: dataset already has a system prompt — skip this augmentation.
                pass
            else:
                messages = [{"role": "system", "content": spec.content}, *messages]

        return {"messages": messages}

    return _augment


def has_tool_turns(example: dict[str, Any]) -> bool:
    """Return True if the conversation contains any ``tool`` role message."""
    return any(m.get("role") == "tool" for m in example["messages"])


def strip_assistant_turns(example: dict[str, Any], idx: int) -> dict[str, Any]:
    """Keep only the system prompt and user turns; remove all assistant turns.

    This produces a conversation skeleton for synthetic data generation.
    The downstream generation pipeline feeds this skeleton to the target model
    turn-by-turn, appending each generated response before sending the next user
    turn:

        dataset:  [system, user1, user2, user3]

        step 1:   feed [system, user1]                               → gen_asst1
        step 2:   feed [system, user1, gen_asst1, user2]             → gen_asst2
        step 3:   feed [system, user1, gen_asst1, user2, gen_asst2, user3] → gen_asst3

    Rows with no user turns are returned empty and filtered out by the caller.
    """
    messages = [m for m in example["messages"] if m["role"] in ("system", "user")]
    if not any(m["role"] == "user" for m in messages):
        return {"messages": []}
    return {"messages": messages}


def normalize_messages(example: dict[str, Any], idx: int) -> dict[str, Any]:
    """Normalize to clean OpenAI message format for SFT training.

    Drops dataset-specific extra fields (``reasoning_content``, etc.) while
    preserving the fields required by the OpenAI chat format for each role:

      system / user   → {role, content}
      assistant       → {role, content} + tool_calls if present
      tool            → {role, content, tool_call_id}

    ``tool`` turns are kept because dropping them breaks agentic conversations:
    an assistant message that issued a tool_call must be followed by the tool
    result before the next assistant message, or the training signal is corrupted.

    Prompt-only rows (no assistant turn) are returned with their messages intact;
    callers should filter them out for training use.
    """
    normalized = []
    for m in example["messages"]:
        role = m.get("role")
        if role in ("system", "user"):
            normalized.append({"role": role, "content": m.get("content") or ""})
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            normalized.append(msg)
        elif role == "tool":
            normalized.append(
                {
                    "role": "tool",
                    "content": m.get("content") or "",
                    "tool_call_id": m.get("tool_call_id", ""),
                }
            )
        elif role == "developer":
            # Map developer-role messages to system per OpenAI schema conventions.
            normalized.append({"role": "system", "content": m.get("content") or ""})
        # other roles (e.g. function, unknown) are dropped
    return {"messages": normalized}
