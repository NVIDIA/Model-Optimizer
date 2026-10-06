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

"""API for converting a model into a `modelopt.torch.distill.DistillationModel` to be used directly in training."""

import warnings

import torch.nn as nn

from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.opt.mode import ModeLike
from modelopt.torch.utils import unwrap_model

from .distillation_model import DistillationModel
from .mode import DistillModeRegistry

__all__ = ["convert", "export"]


def convert(model: nn.Module, mode: ModeLike) -> nn.Module:
    """Main conversion function to turn a student model into a distillation-ready model.

    Args:
        model: The base model to be used as the student.
        mode: A (list of) string(s) or Mode(s) or a list of tuples containing the mode and its
            config indicating the desired mode(s) (and configurations) for the convert
            process. Modes set up the model for different algorithms for model optimization. The
            following modes are available:

            *   :class:`"kd_loss"<modelopt.torch.distill.mode.KnowledgeDistillationModeDescriptor>`: The
                ``model`` will be converted into meta-model encapsulating both teacher and student.
                The mode's config is described in
                :class:`KDLossConfig<modelopt.torch.distill.config.KDLossConfig>`.

            If the mode argument is specified as a dictionary, the keys should indicate the mode and
            the values specify the per-mode configuration.

    Returns:
        An instance of :class:`DistillationModel <modelopt.torch.distill.DistillationModel`.

    """
    return apply_mode(model, mode, registry=DistillModeRegistry)


def export(model: nn.Module) -> nn.Module:
    """Export a distillation meta-model to the original student model.

    Args:
        model: Model to be exported out of a distillation mode into the student only.

    Returns:
        The inner student model.

    """
    # Unwrap a DP/DDP model.
    model = unwrap_model(
        model,
        warn=True,
        msg=(
            f"Unwrapping a {type(model).__name__} model for export! Note that the export is"
            " in-place and the model wrapper should be re-created after export since the wrapper"
            " might not support changing parameters after initialization."
        ),
    )

    # Check if the model is a distillation model.
    if not isinstance(model, DistillationModel):
        warnings.warn("Export of model failed: Model is not a `DistillationModel`")
        return model

    # apply export mode
    return apply_mode(model, "export_student", registry=DistillModeRegistry, init_state=False)
