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

"""
Parsing and formatting utilities for configuration handling in model compression.

This module provides utilities for:
- Parsing command-line arguments and configuration strings
- Formatting and displaying model configurations (block configs, attention, FFN)
- Formatting loss metrics for logging and visualization
"""
# mypy: ignore-errors

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    if args_string is None:
        return {}
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]}
    return args_dict


def parse_json(s: str | None) -> Any:
    if s is None:
        return None
    return json.loads(s)


def parse_path(s: str | None) -> Path | None:
    if s is None or s == "":
        return None
    return Path(s)


def parse_dtype(dtype_name: str) -> torch.dtype:
    dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
    }[dtype_name]
    return dtype


def get_nested_key(dictionary: dict[str, Any], nested_key: str) -> Any:
    """
    If nested_key is "a.b.c" returns dictionary["a"]["b"]["c"]
    """
    value = dictionary
    for key in nested_key.split("."):
        value = value[key]
    return value


def format_block_configs(config) -> str:
    """
    Formats block_configs from a model configuration into a beautiful, readable string.

    Each line represents a layer with attention and FFN configuration.

    Args:
        config: PretrainedConfig object containing block_configs

    Returns:
        Formatted string with layer configurations

    Example output:
        ╭─────────────────────── Model Architecture ────────────────────────╮
        │  Layer  1  │  Attention: no_op                │  FFN: mult = 4.95   │
        │  Layer  2  │  Attention: 4 heads in group     │  FFN: mult = 4.95   │
        │  Layer  3  │  Attention: 4 heads in group     │  FFN: no_op         │
        ╰────────────────────────────────────────────────────────────────────╯
    """
    if not hasattr(config, "block_configs") or not config.block_configs:
        return "❌ No block configs found"

    lines = []

    # Header
    header = "╭─────────────────────────────────────── Model Architecture ────────────────────────────────────────╮"
    lines.append(header)

    # Format each layer
    for i, block in enumerate(config.block_configs, 1):
        attention_info = _format_attention_config(block.attention)
        ffn_info = _format_ffn_config(block.ffn)

        # Create formatted line with proper padding
        layer_str = f"Layer {i:2d}"
        attention_str = f"Attention: {attention_info}"
        ffn_str = f"FFN: {ffn_info}"

        line = f"│  {layer_str:8s} │  {attention_str:30s} │  {ffn_str:18s} │"
        lines.append(line)

    # Footer
    footer = "╰────────────────────────────────────────────────────────────────────────────────────────────────────╯"
    lines.append(footer)

    return "\n".join(lines)


def _format_attention_config(attention_config) -> str:
    """Format attention configuration for display with visual indicators."""
    if not attention_config:
        return "default"

    if attention_config.no_op:
        return "❌ no_op"

    num_kv_heads = attention_config.num_key_value_heads
    if num_kv_heads is not None:
        return f"{num_kv_heads} kv heads"

    if attention_config.replace_with_linear:
        return "linear replacement"

    # Check for other attention types
    if attention_config.mamba:
        return "🐍 mamba"
    if attention_config.llama4:
        return "🦙 llama4"

    window_length = attention_config.window_length
    if window_length is not None:
        return f"windowed ({window_length})"

    if attention_config.sparsify:
        return "sparse"

    return "default"


def _format_ffn_config(ffn_config) -> str:
    """Format FFN configuration for display with visual indicators."""
    if not ffn_config:
        return "default"

    if ffn_config.no_op:
        return "❌ no_op"

    if ffn_config.replace_with_linear:
        return "linear"

    ffn_intermediate = ffn_config.intermediate_size
    if ffn_intermediate is not None:
        return f"ffn_intermediate = {ffn_intermediate}"

    # Check for MoE configuration
    moe_config = ffn_config.moe
    if moe_config:
        return "MoE"

    if ffn_config.sparsify:
        return "sparse"

    return "default"


def format_global_config(config: DictConfig, title: str = "Global Configuration") -> str:
    """
    Pretty prints a global DictConfig with nice formatting and visual indicators.

    Args:
        config: DictConfig object to format
        title: Title to display at the top of the formatted output

    Returns:
        Formatted string with configuration details

    Example output:
        ╭─────────────────── Global Configuration ────────────────────╮
        │  Training                                                    │
        │    • learning_rate: 1e-4                                     │
        │    • batch_size: 32                                          │
        │    • epochs: 100                                             │
        │  Model                                                       │
        │    • hidden_dim: 512                                         │
        │    • num_layers: 6                                           │
        │  Data                                                        │
        │    • dataset_path: /path/to/data                             │
        │    • block_size: 2048                                        │
        ╰──────────────────────────────────────────────────────────────╯
    """
    if not config:
        return "❌ No configuration found"

    lines = []

    # Calculate box width based on title
    box_width = max(60, len(title) + 10)
    title_padding = (box_width - len(title) - 2) // 2

    # Header
    header = f"\n╭{'─' * (box_width - 2)}╮"
    title_line = (
        f"│{' ' * title_padding}{title}{' ' * (box_width - 2 - title_padding - len(title))}│"
    )
    lines.extend([header, title_line])

    def _format_value(value: Any, indent: int = 0) -> str:
        """Format a value with appropriate type indicators."""
        prefix = "  " * indent

        if isinstance(value, (bool, int, float)):
            return f"{prefix} {value}"
        elif isinstance(value, str):
            # Show truncated long strings
            if len(value) > 50:
                return f"{prefix} {value[:47]}..."
            return f"{prefix} {value}"
        elif isinstance(value, (list, tuple)):
            if not value:
                return f"{prefix} []"
            elif len(value) <= 3:
                return f"{prefix} {list(value)}"
            else:
                return f"{prefix} [{len(value)} items]"
        elif value is None:
            return f"{prefix} None"
        else:
            return f"{prefix} {value!s}"

    def _add_config_section(cfg: DictConfig, section_name: str = "", indent: int = 0):
        """Recursively add configuration sections."""
        if section_name:
            indent_str = "  " * indent
            section_line = f"│  {indent_str}{section_name}"
            # Pad to box width
            padding_needed = box_width - len(section_line) - 1
            section_line += " " * padding_needed + "│"
            lines.append(section_line)

        for key, value in cfg.items():
            if isinstance(value, DictConfig):
                # Nested configuration section
                _add_config_section(value, f"{key}", indent + 1)
            else:
                # Regular key-value pair
                indent_str = "  " * (indent + 1)
                value_str = _format_value(value).replace("  " * 0, "").strip()
                line = f"│  {indent_str} {key}: {value_str}"
                # Pad to box width
                if len(line) >= box_width - 1:
                    # Truncate long lines
                    line = line[: box_width - 4] + "..."
                padding_needed = box_width - len(line) - 1
                line += " " * padding_needed + "│"
                lines.append(line)

    # Add configuration sections
    _add_config_section(config)

    # Footer
    footer = f"╰{'─' * (box_width - 2)}╯"
    lines.append(footer)

    return "\n".join(lines)


def format_stitched_losses(
    losses_dict: dict[str, float],
    best_steps_dict: dict[str, int] | None = None,
    best_values_dict: dict[str, float] | None = None,
    step_number: int | None = None,
    title: str = "Stitched Module Losses",
) -> str:
    """
    Pretty prints stitched module losses with comprehensive tracking and visual indicators.

    Args:
        losses_dict: Dictionary with block names as keys and current loss values as floats
        best_steps_dict: Optional dictionary with block names as keys and best step numbers as values
        best_values_dict: Optional dictionary with block names as keys and best loss values as floats
        step_number: Optional current step number to include in summary
        title: Title to display at the top of the formatted output

    Returns:
        Formatted string with loss values in a comprehensive table format

    Example output:
        ╭─────────────────── Stitched Module Losses ──────────────────╮
        │ Block │ Loss Value │ Best Step │ Best Value │ Change from avg  │
        │───────┼────────────┼───────────┼────────────┼──────────────────│
        │  00   │ 6.21e-03   │   Step 5  │ 5.95e-03   │ ↑ +2.6e-04       │
        │  01   │ 5.14e-04   │   Step 12 │ 5.14e-04   │ ↓ -1.2e-04       │
        │  02   │ 9.84e-05   │   Step 15 │ 9.84e-05   │ ↓ -3.1e-04       │
        ╰──────────────────────────────────────────────────────────────╯
    """
    if not losses_dict:
        return "❌ No losses found"

    lines = []

    # Calculate statistics
    loss_values = list(losses_dict.values())
    max_loss = max(loss_values)
    min_loss = min(loss_values)
    avg_loss = sum(loss_values) / len(loss_values)

    # Calculate box width for new layout (removed Bar column)
    box_width = 74
    title_padding = (box_width - len(title) - 2) // 2

    # Header
    header = f"╭{'─' * (box_width - 2)}╮"
    title_line = (
        f"│{' ' * title_padding}{title}{' ' * (box_width - 2 - title_padding - len(title))}│"
    )
    separator = (
        f"│ {'Block':<5} │ {'Loss Value':<12} │ {'Best Step':<10} │ "
        f"{'Best Value':<12} │ {'Change from avg':<18} │"
    )
    divider = f"│{'─' * 7}┼{'─' * 14}┼{'─' * 12}┼{'─' * 14}┼{'─' * 20}│"

    lines.extend([header, title_line, separator, divider])

    # Format each loss
    for block_name, loss_value in losses_dict.items():
        # Format current loss value
        loss_str = f"{loss_value:.2e}"

        # Format best step
        if best_steps_dict and block_name in best_steps_dict:
            best_step_str = f"Step {best_steps_dict[block_name]}"
        else:
            best_step_str = "   --"

        # Format best value
        if best_values_dict and block_name in best_values_dict:
            best_value = best_values_dict[block_name]
            best_value_str = f"{best_value:.2e}"
        else:
            best_value = loss_value  # Assume current is best if no history
            best_value_str = f"{best_value:.2e}"

        # Calculate change from average
        change_from_avg = loss_value - avg_loss
        if abs(change_from_avg) > 1e-8:  # Only show if meaningful
            change_str = f"{abs(change_from_avg):.1e}"
            if change_from_avg > 0:
                # Current is above average (worse for loss)
                change_display = f"↑ +{change_str}"
            else:
                # Current is below average (better for loss)
                change_display = f"↓ -{change_str}"
        else:
            # At average value
            change_display = "↔ 0.0e+00"

        # Format the line
        block_display = block_name.replace("block_", "").zfill(2)

        line = (
            f"│ {block_display:<5} │ {loss_str:<12} │ {best_step_str:<10} │ "
            f"{best_value_str:<12} │ {change_display:<18} │"
        )
        lines.append(line)

    # Add summary statistics
    lines.append(divider)

    # Build summary string with optional step number
    summary_parts = []
    if step_number is not None:
        summary_parts.append(f"Step {step_number}")
    summary_parts.extend([f"Avg={avg_loss:.2e}", f"Max={max_loss:.2e}", f"Min={min_loss:.2e}"])

    summary_text = ", ".join(summary_parts)
    summary = f"│ Summary: {summary_text}"

    # Pad summary to box width
    padding_needed = box_width - len(summary) - 1
    summary += " " * padding_needed + "│"
    lines.append(summary)

    # Add best step summary if we have best step data
    if best_steps_dict and best_values_dict:
        # Find the most common best step (modal step)
        step_counts = {}
        for step in best_steps_dict.values():
            step_counts[step] = step_counts.get(step, 0) + 1

        if step_counts:
            modal_best_step = max(step_counts, key=step_counts.get)

            # Get values at the modal best step for blocks that have it as their best
            best_step_values = []
            for block_name, best_step in best_steps_dict.items():
                if best_step == modal_best_step and block_name in best_values_dict:
                    best_step_values.append(best_values_dict[block_name])

            if best_step_values:
                best_step_avg = sum(best_step_values) / len(best_step_values)
                best_step_max = max(best_step_values)
                best_step_min = min(best_step_values)

                best_step_summary_text = (
                    f"Best:   Step {modal_best_step}, Avg={best_step_avg:.2e}, "
                    f"Max={best_step_max:.2e}, Min={best_step_min:.2e}"
                )
                best_step_summary = f"│ {best_step_summary_text}"

                # Pad best step summary to box width
                padding_needed = box_width - len(best_step_summary) - 1
                best_step_summary += " " * padding_needed + "│"
                lines.append(best_step_summary)

    # Footer
    footer = f"╰{'─' * (box_width - 2)}╯"
    lines.append(footer)

    return "\n".join(lines)
