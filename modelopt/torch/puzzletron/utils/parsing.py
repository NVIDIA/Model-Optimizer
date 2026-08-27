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
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

__all__ = [
    "handle_arg_string",
    "simple_parse_args_string",
    "parse_json",
    "parse_path",
    "get_nested_key",
    "format_global_config",
    "format_stitched_losses",
]


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
    """Parse ``args1=val1,arg2=val2`` into a dictionary."""
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

    Each line represents a layer with typed subblock configuration.

    Args:
        config: PretrainedConfig object containing block_configs

    Returns:
        Formatted string with layer configurations

    Example output:
        Layer  1 | attention: no_op | ffn: intermediate_size=8192
    """
    if not hasattr(config, "block_configs") or not config.block_configs:
        return "No block configs found"

    lines = []
    for i, block in enumerate(config.block_configs, 1):
        parts = []
        for ref in block.subblocks():
            parts.append(f"{ref.name}: {_format_subblock_config(ref.config)}")
        lines.append(f"Layer {i:2d} | " + " | ".join(parts))

    return "\n".join(lines)


def _format_subblock_config(subblock_config) -> str:
    if not subblock_config:
        return "default"
    if subblock_config.kind == "attention":
        return _format_attention_config(subblock_config)
    if subblock_config.kind == "ffn":
        return _format_ffn_config(subblock_config)
    if subblock_config.kind == "moe":
        return _format_moe_config(subblock_config)
    if subblock_config.kind == "mamba":
        return _format_mamba_config(subblock_config)
    return str(subblock_config)


def _format_attention_config(attention_config) -> str:
    """Format attention configuration for display with visual indicators."""
    if not attention_config:
        return "default"

    if attention_config.no_op:
        return "no_op"

    num_kv_heads = attention_config.num_kv_heads
    if num_kv_heads is not None:
        return f"num_kv_heads={num_kv_heads}"

    num_query_heads = attention_config.num_query_heads
    if num_query_heads is not None:
        return f"num_query_heads={num_query_heads}"
    if attention_config.llama4:
        return "llama4"

    sliding_window_size = attention_config.sliding_window_size
    if sliding_window_size is not None:
        return f"sliding_window_size={sliding_window_size}"

    return "default"


def _format_ffn_config(ffn_config) -> str:
    """Format FFN configuration for display with visual indicators."""
    if not ffn_config:
        return "default"

    if ffn_config.no_op:
        return "no_op"

    ffn_intermediate = ffn_config.intermediate_size
    if ffn_intermediate is not None:
        return f"intermediate_size={ffn_intermediate}"

    return "default"


def _format_moe_config(moe_config) -> str:
    if moe_config.no_op:
        return "no_op"
    parts = []
    if moe_config.num_experts is not None:
        parts.append(f"num_experts={moe_config.num_experts}")
    if moe_config.top_k is not None:
        parts.append(f"top_k={moe_config.top_k}")
    if moe_config.expert_intermediate_size is not None:
        parts.append(f"expert_intermediate_size={moe_config.expert_intermediate_size}")
    if moe_config.shared_expert_intermediate_size is not None:
        parts.append(
            f"shared_expert_intermediate_size={moe_config.shared_expert_intermediate_size}"
        )
    if moe_config.latent_dim is not None:
        parts.append(f"latent_dim={moe_config.latent_dim}")
    return ", ".join(parts) if parts else "default"


def _format_mamba_config(mamba_config) -> str:
    if mamba_config.no_op:
        return "no_op"
    parts = []
    if mamba_config.num_heads is not None:
        parts.append(f"num_heads={mamba_config.num_heads}")
    if mamba_config.head_dim is not None:
        parts.append(f"head_dim={mamba_config.head_dim}")
    if mamba_config.state_dim is not None:
        parts.append(f"state_dim={mamba_config.state_dim}")
    return ", ".join(parts) if parts else "default"


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
                line = f"│  {indent_str} • {key}: {value_str}"
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
    initial_values_dict: dict[str, float] | None = None,
    not_trainable_names: set[str] | None = None,
    step_number: int | None = None,
    title: str = "Stitched Module Losses",
) -> str:
    """
    Pretty prints stitched module losses with comprehensive tracking and visual indicators.

    Args:
        losses_dict: Dictionary with block names as keys and current loss values as floats
        best_steps_dict: Optional dictionary with block names as keys and best step numbers as values
        best_values_dict: Optional dictionary with block names as keys and best loss values as floats
        initial_values_dict: Optional dictionary with block names as keys and initial loss values
            (from the first log chunk) as floats. Used to render the "Δ from initial" column as
            a per-block training-progress signal.
        step_number: Optional current step number to include in summary
        title: Title to display at the top of the formatted output

    Returns:
        Formatted string with loss values in a comprehensive table format

    Example output:
        ╭─────────────────── Stitched Module Losses ──────────────────╮
        │ Block │ Loss Value │ Δ from initial   │ Best Value │ Best Step │
        │───────┼────────────┼──────────────────┼────────────┼───────────│
        │  00   │ 6.21e-03   │ ↓ -3.2e-04 (-5%) │ 5.95e-03   │   Step 5  │
        │  01   │ 5.14e-04   │ ↓ -1.8e-03 (-78%)│ 5.14e-04   │   Step 12 │
        │  02   │ 9.84e-05   │ ↓ -4.1e-04 (-81%)│ 9.84e-05   │   Step 15 │
        ╰──────────────────────────────────────────────────────────────╯
    """
    if not losses_dict:
        if not_trainable_names:
            return (
                "No trainable losses found; "
                f"skipped {len(not_trainable_names)} non-trainable blocks"
            )
        return "❌ No losses found"

    if best_steps_dict:
        best_steps_dict = {k: v for k, v in best_steps_dict.items() if k in losses_dict}
    if best_values_dict:
        best_values_dict = {k: v for k, v in best_values_dict.items() if k in losses_dict}
    if initial_values_dict:
        initial_values_dict = {k: v for k, v in initial_values_dict.items() if k in losses_dict}

    lines = []

    # Calculate statistics
    loss_values = list(losses_dict.values())
    finite_loss_values = [value for value in loss_values if math.isfinite(value)]
    if finite_loss_values:
        max_loss = max(finite_loss_values)
        min_loss = min(finite_loss_values)
        avg_loss = sum(finite_loss_values) / len(finite_loss_values)
    else:
        max_loss = min_loss = avg_loss = float("nan")

    # Calculate box width for new layout (removed Bar column)
    box_width = 74
    title_padding = (box_width - len(title) - 2) // 2

    # Header
    header = f"╭{'─' * (box_width - 2)}╮"
    title_line = (
        f"│{' ' * title_padding}{title}{' ' * (box_width - 2 - title_padding - len(title))}│"
    )
    separator = (
        f"│ {'Block':<5} │ {'Loss Value':<12} │ {'Δ from initial':<18} │ "
        f"{'Best Value':<12} │ {'Best Step':<10} │"
    )
    divider = f"│{'─' * 7}┼{'─' * 14}┼{'─' * 20}┼{'─' * 14}┼{'─' * 12}│"

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

        # Calculate change from initial: current loss minus the block's loss in the
        # first log chunk we saw. Per-block training-progress signal — answers "is
        # whether training is actually reducing this block's loss?" and stays
        # apples-to-apples even when blocks have very different intrinsic loss scales.
        if not initial_values_dict or block_name not in initial_values_dict:
            # No baseline supplied (callers may omit initial_values_dict).
            change_display = "  --"
        elif not math.isfinite(loss_value) or not math.isfinite(initial_values_dict[block_name]):
            change_display = "non-finite"
        else:
            initial_value = initial_values_dict[block_name]
            delta = loss_value - initial_value
            if abs(delta) > 1e-8:
                pct = (delta / initial_value * 100.0) if initial_value != 0.0 else 0.0
                # Clamp percentage display to keep the cell within the 18-char column
                # even on pathological divergence (e.g. a block whose loss 10x'd).
                pct_clamped = max(-999.0, min(999.0, pct))
                arrow = "↓" if delta < 0 else "↑"
                sign = "-" if delta < 0 else "+"
                change_display = f"{arrow} {sign}{abs(delta):.1e} ({pct_clamped:+.0f}%)"
            else:
                change_display = "↔ 0.0e+00"

        # Format the line
        block_display = block_name.replace("block_", "").zfill(2)

        line = (
            f"│ {block_display:<5} │ {loss_str:<12} │ {change_display:<18} │ "
            f"{best_value_str:<12} │ {best_step_str:<10} │"
        )
        lines.append(line)

    # Add summary statistics
    lines.append(divider)

    # Build summary string with optional step number
    summary_parts = []
    if step_number is not None:
        summary_parts.append(f"Step {step_number}")
    summary_parts.extend([f"Avg={avg_loss:.2e}", f"Max={max_loss:.2e}", f"Min={min_loss:.2e}"])
    if not_trainable_names:
        summary_parts.append(f"Skipped={len(not_trainable_names)}")

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
                    best_value = best_values_dict[block_name]
                    if math.isfinite(best_value):
                        best_step_values.append(best_value)

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
