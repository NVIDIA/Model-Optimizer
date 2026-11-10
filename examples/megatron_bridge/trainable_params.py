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

"""Parameter selection for the Megatron-Bridge distillation example."""

import fnmatch
from pathlib import Path

from torch import nn

from modelopt.torch.opt import ModeloptStateManager
from modelopt.torch.quantization import (
    enable_quant_input_gradients,
    has_quant_lora,
    initialize_qdq_weights,
)
from modelopt.torch.quantization.qtensor import QTensorWrapper

__all__ = ["configure_trainable_parameters", "initialize_student_weights"]


def initialize_student_weights(
    model: nn.Module, qdq_student_weight: bool, checkpoint_dir: str | Path
) -> nn.Module:
    """Initialize selected student weights on a fresh run, preserving native resumes."""
    if (
        not qdq_student_weight
        or (Path(checkpoint_dir) / "latest_checkpointed_iteration.txt").is_file()
    ):
        return model
    return initialize_qdq_weights(model, trainable_only=True)


def configure_trainable_parameters(model: nn.Module, target_parameters: list[str]) -> nn.Module:
    """Train only parameters matching shell-style patterns, retaining fake quantizers.

    Restore student ModelOpt state first, then select before DDP/optimizer construction.
    Match names relative to the student root.
    Every pattern must match locally; packed weights and unmerged adapters are rejected.
    Other parameters are frozen, including any learned quantization scales.
    """
    if not target_parameters:
        raise ValueError("target_parameters must contain at least one pattern.")
    if has_quant_lora(model):
        raise ValueError("Full-rank training requires an adapter-free student checkpoint.")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    selected = set()
    for pattern in target_parameters:
        matches = {name for name in parameters if fnmatch.fnmatchcase(name, pattern)}
        if not matches:
            raise ValueError(f"No student parameters match {pattern!r} on this model chunk.")
        selected.update(matches)
    selected_ids = {id(parameters[name]) for name in selected}
    for name, parameter in parameters.items():
        if id(parameter) not in selected_ids:
            continue
        if name not in selected:
            raise ValueError(f"Selected parameter is shared with unselected parameter {name!r}.")
        if isinstance(parameter, QTensorWrapper) or not parameter.is_floating_point():
            raise ValueError(
                f"Full-rank training requires an uncompressed floating weight: {name}."
            )
    # Creating mode state here would make native checkpoint loading skip saved quantizers.
    if not ModeloptStateManager.is_converted(model):
        raise ValueError(
            "Restore student ModelOpt state with --student_megatron_path before selecting "
            "trainable parameters, including on native resume."
        )
    model = enable_quant_input_gradients(model)
    model.requires_grad_(False)
    for name in selected:
        parameters[name].requires_grad_(True)
    return model
