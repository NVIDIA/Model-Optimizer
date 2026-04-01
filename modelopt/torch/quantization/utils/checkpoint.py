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

* **Resume detection** — :func:`detect_resume_point` reads progress metadata
  previously attached to the model and returns the layer index to resume from.

* **Checkpoint saving** — :func:`save_sequential_checkpoint` collects layer
  output metadata, attaches progress to the model, and delegates to the
  registered saver.

This follows the same registry pattern as
:attr:`LayerActivationCollector._decoder_layer_support` in
``activation_collector.py``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from modelopt.torch.utils import print_rank_0

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

    from .activation_collector import LayerActivationCollector

#: Model attribute name used to store sequential calibration progress.
SEQ_CALIB_PROGRESS_ATTR = "_seq_calib_progress"

# ---------------------------------------------------------------------------
# Save registry
# ---------------------------------------------------------------------------

# Global registry of (predicate, save_fn) pairs.  Populated at import time
# by plugins.  Order matters: the first matching entry wins.
_CHECKPOINT_SAVE_SUPPORT: list[
    tuple[Callable[[nn.Module], bool], Callable[[nn.Module, str], None]]
] = []


def register_seq_calib_checkpoint_saver(
    is_supported: Callable[[nn.Module], bool],
    save_fn: Callable[[nn.Module, str], None],
) -> None:
    """Register a (predicate, saver) pair for sequential calibration checkpointing.

    Args:
        is_supported: ``Callable(model) -> bool`` — returns *True* if *save_fn*
            can handle this model.
        save_fn: ``Callable(model, checkpoint_dir) -> None`` — saves the model
            (weights + modelopt state) to *checkpoint_dir*.
    """
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


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------


def detect_seq_calib_resume_point(model: nn.Module, num_layers: int) -> tuple[int, dict | None]:
    """Read checkpoint progress from the model and return where to resume.

    Returns:
        ``(resume_layer_idx, saved_output_metas)`` — ``(0, None)`` for a fresh
        run (no checkpoint present).  Removes the progress attribute from the
        model after reading it.

    Raises:
        ValueError: If the checkpoint's layer count doesn't match *num_layers*.
    """
    progress = getattr(model, SEQ_CALIB_PROGRESS_ATTR, None)
    if progress is None:
        return 0, None

    completed_layer = progress["completed_layer_idx"]
    saved_total = progress["total_layers"]

    if saved_total != num_layers:
        raise ValueError(
            f"Checkpoint was saved with {saved_total} layers but model has "
            f"{num_layers} layers. Cannot resume."
        )

    delattr(model, SEQ_CALIB_PROGRESS_ATTR)
    resume_from = completed_layer + 1
    print_rank_0(
        f"Resuming sequential calibration from layer {resume_from} "
        f"(layers 0..{completed_layer} already calibrated)"
    )
    return resume_from, progress.get("layer_output_metas", {})


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------


def should_save_seq_calib_checkpoint(
    layer_idx: int, num_layers: int, checkpoint_dir: str | None, checkpoint_interval: int | None
) -> bool:
    """Return *True* when a checkpoint should be saved after calibrating *layer_idx*."""
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
    input_getter: LayerActivationCollector,
) -> None:
    """Save a rolling checkpoint during sequential calibration.

    Collects layer output metadata from *input_getter*, attaches progress to
    the model (so ``update_quantize_metadata`` serialises it), calls the
    registered saver, then cleans up the progress attribute.
    """
    saver = get_checkpoint_saver(model)
    if saver is None:
        print_rank_0(
            "Warning: checkpoint_dir is set but no checkpoint saver is registered "
            "for this model type. Skipping checkpoint save."
        )
        return

    layer_output_metas = input_getter.get_layer_output_metas(completed_layer_idx)

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

    del model._seq_calib_progress
