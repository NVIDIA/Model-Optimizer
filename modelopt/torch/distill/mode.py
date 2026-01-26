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

"""Module implementing and describing modes that can be used during the NAS convert process.

Check out :meth:`mtn.convert <modelopt.torch.nas.conversion.convert>` to learn more about modes.
"""

import warnings

import torch.nn as nn

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    ModeDescriptor,
    RestoreEntrypoint,
    _ModeRegistryCls,
)
from modelopt.torch.utils import init_model_from_model_like, unwrap_model

from .config import ExportStudentConfig, KDLossConfig, LayerwiseKDConfig
from .distillation_model import DistillationModel
from .layerwise_distillation_model import LayerwiseDistillationModel
from .registry import DistillationDMRegistry, LayerwiseDistillationDMRegistry

DistillModeRegistry = _ModeRegistryCls("distill")


@DistillModeRegistry.register_mode
class KnowledgeDistillationModeDescriptor(ModeDescriptor):
    """Class to describe the Knowledge-Distillation mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "kd_loss"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return KDLossConfig

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode of this mode."""
        return "export_student"

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return _convert_for_kd

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        raise NotImplementedError(f"{self.name} mode does not support restore.")

    @property
    def save_mode_in_state(self) -> bool:
        """Whether the mode should be saved into the modelopt state."""
        return False


@DistillModeRegistry.register_mode
class LayerwiseKDModeDescriptor(KnowledgeDistillationModeDescriptor):
    """Class to describe the Layerwise Knowledge-Distillation mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "layerwise_kd"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return LayerwiseKDConfig

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return _convert_for_layerwise


@DistillModeRegistry.register_mode
class ExportStudentModeDescriptor(ModeDescriptor):
    """Class to describe the specific Export mode to be used with Knowledge Distillation.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "export_student"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return ExportStudentConfig

    @property
    def is_export_mode(self) -> bool:
        """Specifies whether the mode is an export mode."""
        return True

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return _export_student

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        raise NotImplementedError(f"{self.name} mode does not support restore.")

    @property
    def save_mode_in_state(self) -> bool:
        """Whether the mode should be saved into the modelopt state."""
        return False


def _convert_for_kd(
    model: nn.Module,
    config: KDLossConfig,
    model_cls: type[nn.Module] = DistillationModel,
    model_registry: _DMRegistryCls = DistillationDMRegistry,
) -> ConvertReturnType:
    """Function for converting a model to a distillation meta-model.

    This is the only utility needed to use the ``modelopt.torch.distill`` API directly.

    Args:
        model: The base model to be used as the student.
        config: A KDLossConfig instance defining the configuration options.

    Returns:
        A ``DistillationModel`` encapsulating the teacher and students, to be used in place
        of the original model.
        Metadata dictionary containing the config arguments needed to recreate the model
        from a checkpoint.
    """
    # we need to do a "strict" validation here to ensure that all required fields are present.
    # we don't do the strict validation by default since this is okay during restore or when other
    # modes are added on top.
    config._strict_validate()

    teacher = init_model_from_model_like(config.teacher_model)
    student = model

    # hide teacher's state if it exists within teacher (will be reverted during export)
    if ModeloptStateManager.is_converted(teacher):
        teacher._modelopt_state_storage = [nn.Module()]
        ModeloptStateManager.transfer_state_dict(teacher, teacher._modelopt_state_storage[0])
        warnings.warn(
            "Distillation mode in progress. ModelOpt operations on teacher disabled until the"
            " 'export_student' mode is used."
        )

    # initialize distillation model
    original_cls = type(student)
    if original_cls not in model_registry:
        model_registry.register({original_cls: "student_class"})(model_cls)
    # TODO (lucasl): look into ways to avoid registering every class manually
    # (e.g. by just registering nn.Module and disable the "forward" check for the inherited class check

    distillation_model = model_registry.convert(student)
    distillation_model.modify(
        **{**config, "teacher_model": teacher}  # overwrite with instantiated teacher
    )

    # no metadata, all specified via config.
    metadata = {}

    return distillation_model, metadata


def _convert_for_layerwise(model: nn.Module, config: LayerwiseKDConfig) -> ConvertReturnType:
    """Function for converting a model to a layerwise distillation meta-model."""
    return _convert_for_kd(
        model,
        config,
        model_cls=LayerwiseDistillationModel,
        model_registry=LayerwiseDistillationDMRegistry,
    )


def _export_student(model: nn.Module, config: ExportStudentConfig) -> ConvertReturnType:
    """Export a ``DistillationModel`` to its inner student model, including modelopt state transfer."""
    if not isinstance(model, DistillationModel):
        raise RuntimeError("`model` must be of type `mtd.DistillationModel`")

    # restore previously-renamed state in teacher just in case being used after export.
    teacher = model.teacher_model
    if hasattr(teacher, "_modelopt_state_storage"):
        mod_storage = teacher._modelopt_state_storage[0]
        delattr(teacher, "_modelopt_state_storage")
        ModeloptStateManager.transfer_state_dict(mod_storage, teacher)

    # retrieve student model
    student_model = model.export()
    # just in case; This way all our APIs input and output unwrapped models.
    student_model = unwrap_model(
        student_model,
        warn=True,
        msg=(
            f"The student model is wrapped into {type(student_model).__name__}. Unwrapping and exporting it ..."
        ),
    )

    return student_model, {}
