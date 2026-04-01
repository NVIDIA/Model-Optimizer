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

"""Checkpoint save/resume utilities for sequential calibration.

Provides:

* A pluggable **save registry** — plugins (e.g. huggingface.py) register a
  ``(predicate, save_fn)`` pair at import time so that
  :func:`get_checkpoint_saver` can find the right saver for any model.

* **Resume detection** — :func:`detect_sequential_resume_layer` reads progress
  metadata previously attached to the model and returns the layer index to
  resume from.

* **Checkpoint saving** — :func:`save_sequential_checkpoint` collects layer
  output metadata, attaches progress to the model, and delegates to the
  registered saver.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from modelopt.torch.utils import print_rank_0

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

#: Model attribute name used to store sequential calibration progress.
SEQ_CALIB_PROGRESS_ATTR = "_seq_calib_progress"

# ---------------------------------------------------------------------------
# Save registry
# ---------------------------------------------------------------------------
_CHECKPOINT_SAVE_SUPPORT: list[
    tuple[Callable[[nn.Module], bool], Callable[[nn.Module, str], None]]
] = []


def register_seq_calib_checkpoint_saver(
    is_supported: Callable[[nn.Module], bool],
    save_fn: Callable[[nn.Module, str], None],
) -> None:
    """Register a ``(predicate, saver)`` pair for sequential calibration checkpointing."""
    entry = (is_supported, save_fn)
    if entry not in _CHECKPOINT_SAVE_SUPPORT:
        _CHECKPOINT_SAVE_SUPPORT.append(entry)


def get_checkpoint_saver(
    model: nn.Module,
) -> Callable[[nn.Module, str], None] | None:
    """Return the registered save function for *model*, or *None*."""
    for is_supported, save_fn in _CHECKPOINT_SAVE_SUPPORT:
        if is_supported(model):
            return save_fn
    return None


def detect_sequential_resume_layer(model: nn.Module, num_layers: int) -> tuple[int, dict | None]:
    """Read checkpoint progress from the model and return ``(resume_layer_idx, layer_output_metas)``.

    Returns ``(0, None)`` for a fresh run with no checkpoint present.
    The attribute is **not** deleted here — cleanup is owned by
    :func:`sequential_calibrate`'s ``finally`` block.
    """
    progress = getattr(model, SEQ_CALIB_PROGRESS_ATTR, None)
    if progress is None:
        return 0, None

    if not isinstance(progress, dict):
        raise ValueError(
            f"Expected seq_calib_progress to be a dict, got {type(progress).__name__}."
        )
    for key in ("completed_layer_idx", "total_layers"):
        if key not in progress:
            raise ValueError(f"Checkpoint progress is missing required key {key!r}.")

    completed_layer = progress["completed_layer_idx"]
    saved_total = progress["total_layers"]

    if not isinstance(completed_layer, int) or not isinstance(saved_total, int):
        raise ValueError(
            f"Checkpoint progress values must be ints, got "
            f"completed_layer_idx={completed_layer!r}, total_layers={saved_total!r}."
        )

    if saved_total != num_layers:
        raise ValueError(
            f"Checkpoint was saved with {saved_total} layers but model has "
            f"{num_layers} layers. Cannot resume."
        )

    if not (0 <= completed_layer < num_layers):
        raise ValueError(
            f"completed_layer_idx={completed_layer} is out of range for "
            f"{num_layers} layers (expected 0..{num_layers - 1})."
        )

    resume_from = completed_layer + 1
    print_rank_0(
        f"Resuming sequential calibration from layer {resume_from} "
        f"(layers 0..{completed_layer} already calibrated)"
    )
    return resume_from, progress.get("layer_output_metas", {})


def should_save_seq_calib_checkpoint(
    layer_idx: int, num_layers: int, checkpoint_dir: str | None, checkpoint_interval: int | None
) -> bool:
    """Return *True* when a checkpoint should be saved after calibrating *layer_idx*."""
    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError(
            f"checkpoint_interval must be a positive integer, got {checkpoint_interval}."
        )
    return (
        checkpoint_dir is not None
        and checkpoint_interval is not None
        and (layer_idx + 1) % checkpoint_interval == 0
        and layer_idx < num_layers - 1  # never save after the final layer
    )


def save_sequential_checkpoint(
    model: nn.Module,
    completed_layer_idx: int,
    total_layers: int,
    checkpoint_dir: str,
    layer_output_metas: dict,
) -> None:
    """Save a rolling checkpoint during sequential calibration.

    Temporarily attaches progress to the model so that ``update_quantize_metadata``
    can serialize it during ``save_pretrained``.  The attribute is **not** deleted
    here — cleanup is owned by :func:`sequential_calibrate`'s ``finally`` block.
    """
    saver = get_checkpoint_saver(model)
    if saver is None:
        print_rank_0(
            "Warning: checkpoint_dir is set but no checkpoint saver is registered "
            "for this model type. Skipping checkpoint save."
        )
        return

    model._seq_calib_progress = {
        "completed_layer_idx": completed_layer_idx,
        "total_layers": total_layers,
        "layer_output_metas": layer_output_metas,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    saver(model, checkpoint_dir)
    print_rank_0(
        f"Saved sequential calibration checkpoint at layer "
        f"{completed_layer_idx + 1}/{total_layers} to {checkpoint_dir}"
    )
