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

"""Meta-model wrapper to support layerwise-enabled knowledge-distillation learning."""

import warnings
from typing import Any

import torch.nn as nn

from .distillation_model import DistillationModel, student_output_capture_fwd_hook

__all__ = ["LayerwiseDistillationModel"]


class LayerwiseDistillationModel(DistillationModel):
    """Meta-model wrapper to support layerwise-enabled knowledge-distillation learning.

    The LayerwiseDistillationModel is a subclass of the DistillationModel that injects teacher inputs
    into the corresponding student layers. This accomodates the case where the student model is the
    teacher with specific submodules replaced, which now need to be trained to mimic the original
    submodule in the teacher.
    """

    def modify(self, *args, **kwargs):
        """Modify the distillation model."""
        super().modify(*args, **kwargs)

        # Freeze student layers except those in criterion.
        self.requires_grad_(False)
        for student_layer, _ in self._layers_to_loss:
            student_layer.requires_grad_(True)

        # Make lm heads (if we have them) no-ops to save compute.
        if hasattr(self, "lm_head"):
            self._lm_head = self.lm_head
            self.lm_head = nn.Identity()
        if hasattr(self._teacher_model, "lm_head"):
            self._teacher_model._lm_head = self._teacher_model.lm_head
            self._teacher_model.lm_head = nn.Identity()

        return self

    def _register_hooks(self):
        """Register hooks for intermediate tensors from teacher models and the student model."""
        for student_layer, teacher_layer in self._layers_to_loss:
            setattr(student_layer, "_teacher_layer", [teacher_layer])
            handle_s1 = student_layer.register_forward_pre_hook(student_input_bypass_fwd_hook)
            setattr(student_layer, "_intermediate_output", None)
            handle_s2 = student_layer.register_forward_hook(student_output_capture_fwd_hook)
            setattr(teacher_layer, "_intermediate_input", None)
            setattr(teacher_layer, "_intermediate_output", None)
            handle_t = teacher_layer.register_forward_hook(teacher_input_output_capture_fwd_hook)
            self._hook_handles.update([handle_s1, handle_s2, handle_t])

    def export(self):
        """Export the distillation model."""
        for student_layer, _ in self._layers_to_loss:
            delattr(student_layer, "_teacher_layer")

        if hasattr(self, "_lm_head"):
            self.lm_head = self._lm_head
        if hasattr(self._teacher_model, "_lm_head"):
            self._teacher_model.lm_head = self._teacher_model._lm_head

        return super().export()


def student_input_bypass_fwd_hook(module: nn.Module, input: Any):
    """A hook to inject teacher input into corresponding student layer."""
    # NOTE: Defined externally to allow pickling during DDP initialization.

    if getattr(module, "_only_teacher_fwd", False):
        return input  # Might be hooked on entire model fwd

    teacher_layer = module._teacher_layer[0]
    teacher_input = teacher_layer._intermediate_input
    if teacher_input is None:
        warnings.warn(
            f"Teacher's Module `{type(teacher_layer).__name__}` has no intermediate input stored."
            " This is expected when the `only_student_forward` context manager is in use."
        )
        return input

    teacher_layer._intermediate_input = None  # reset
    return teacher_input


def teacher_input_output_capture_fwd_hook(module: nn.Module, input: Any, output: Any):
    """A hook to capture layer input and output."""
    # NOTE: Defined externally to allow pickling during DDP initialization.

    if module._intermediate_output is not None:
        # NOTE: cannot tell if train or eval since teacher is always eval
        warnings.warn(
            f"Teacher's Module `{type(module).__name__}` already has an intermediate output stored."
            " This is expected when `DistillationModel.compute_kd_loss` is not called in eval mode."
        )

    module._intermediate_input = input
    module._intermediate_output = output
