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
import re
from pathlib import Path

import torch
from torch import nn

from modelopt.torch.opt import ModeloptStateManager
from modelopt.torch.quantization import (
    enable_quant_input_gradients,
    has_quant_lora,
    initialize_qdq_weights,
)
from modelopt.torch.quantization.qtensor import QTensorWrapper

__all__ = [
    "configure_trainable_parameters",
    "freeze_router_bias_updates",
    "initialize_student_weights",
    "keep_dense_optimizer_non_empty",
    "should_initialize_student_weights",
]

# The decoder's own final norm; an MTP head has a ``final_layernorm`` of its own.
_DECODER_FINAL_NORM = re.compile(r"(^|\.)decoder\.final_(layernorm|norm)\.")


def should_initialize_student_weights(
    qdq_student_weight: bool,
    checkpoint_dir: str | Path,
    student_checkpoint: str | Path | None = None,
) -> bool:
    """Skip native resumes and refuse QDQ reinitialization of a trained source."""
    if not qdq_student_weight or any(
        (Path(checkpoint_dir) / name).is_file()
        for name in ("latest_checkpointed_iteration.txt", "latest_train_state.pt")
    ):
        return False
    if student_checkpoint is not None:
        source = Path(student_checkpoint)
        iteration = re.fullmatch(r"iter_(\d+)", source.name)
        step = int(iteration[1]) if iteration else 0
        train_state = source / "train_state.pt"
        if train_state.is_file():
            step = torch.load(train_state, map_location="cpu", weights_only=True)["step"]
        if step > 0:
            raise ValueError(
                "--qdq_student_weight would reinitialize a trained student checkpoint. "
                "Remove the flag to retain its trained latents; use the original output directory "
                "for a native resume that also restores optimizer state."
            )
    return True


def initialize_student_weights(
    model: nn.Module,
    qdq_student_weight: bool,
    checkpoint_dir: str | Path,
    student_checkpoint: str | Path | None = None,
) -> nn.Module:
    """Initialize selected student weights on a fresh run, preserving native resumes."""
    if not should_initialize_student_weights(
        qdq_student_weight, checkpoint_dir, student_checkpoint
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


def freeze_router_bias_updates(model: nn.Module) -> int:
    """Stop the load-balancing update of ``expert_bias`` in routers whose weight is frozen.

    ``expert_bias`` is a buffer that Megatron nudges after every step from the expert token
    counts, whatever ``requires_grad`` says, so a frozen router would still drift. Returns the
    number of routers held fixed.
    """
    frozen = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if getattr(module, "expert_bias", None) is not None and not (
            isinstance(weight, nn.Parameter) and weight.requires_grad
        ):
            module.frozen_expert_bias = True
            frozen += 1
    return frozen


def keep_dense_optimizer_non_empty(model: nn.Module) -> list[str]:
    """Keep the decoder final norm trainable when only routed-expert weights are selected.

    Megatron builds an empty dense optimizer when every non-expert parameter is frozen, and
    the first checkpoint save then crashes in ``DistributedOptimizer.state_dict``. Selecting the
    final norm gives it one parameter. Returns the names re-enabled; raises when none can be.
    """
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(".mlp.experts." not in f".{n}" for n in trainable):
        return []
    norms = [n for n, _ in model.named_parameters() if _DECODER_FINAL_NORM.search(n)]
    if not norms:
        raise ValueError(
            "Only routed-expert weights are trainable and this model chunk has no decoder final "
            "norm to keep the dense optimizer non-empty, so the first checkpoint save would "
            "crash. Select a dense parameter too, or use --pp_size 1."
        )
    parameters = dict(model.named_parameters())
    for name in norms:
        parameters[name].requires_grad_(True)
    return norms
