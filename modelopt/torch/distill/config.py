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

"""Configurations for distillation modes."""

import warnings
from typing import Any, Literal, TypeAlias

import pydantic
from torch.nn.modules.loss import _Loss as Loss

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.utils.network import ModelLike

from .loss_balancers import DistillationLossBalancer

__all__ = ["ExportStudentConfig", "KDLossConfig", "LayerwiseKDConfig", "LocalDistillationConfig"]


class LocalDistillationConfig(ModeloptBaseConfig):
    """Configuration for independent local repair of an existing quantized model.

    Learning rates for scales refer to positive log-scale parameters. Context
    length and global batch are explicit: microbatching must not change them.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    units: list[str] | None = None
    unit_types: tuple[str, ...] = ("mlp", "moe")
    mode: Literal["scale_only", "weight_only", "joint"] = "scale_only"
    requantize: bool = False
    tune_activations: bool = False
    optimizer: Literal["adagrad", "adamw"] = "adagrad"
    scale_lr: float = pydantic.Field(default=0.01, gt=0)
    weight_lr: float = pydantic.Field(default=0.00001, gt=0)
    activation_lr: float | None = pydantic.Field(default=None, gt=0)
    scheduler: Literal["constant", "cosine", "linear"] = "cosine"
    warmup_fraction: float = pydantic.Field(default=0.05, ge=0, lt=1)
    updates: int = pydantic.Field(default=512, ge=0)
    global_batch_size: int = pydantic.Field(default=8, gt=0)
    micro_batch_size: int | None = pydantic.Field(default=None, gt=0)
    sequence_length: int = pydantic.Field(default=4096, gt=0)
    gradient_clip_norm: float = pydantic.Field(default=1.0, gt=0)
    loss: Literal["mse", "normalized_mse"] = "mse"
    validation_interval: int | None = pydantic.Field(default=None, gt=0)
    report_interval: int = pydantic.Field(default=16, gt=0)
    selection: Literal["final", "best_validation_per_unit"] = "final"
    checkpoint_interval: int = pydantic.Field(default=128, gt=0)
    checkpoint_dir: str | None = None
    resume_from: str | None = None
    output_dir: str | None = None
    seed: int = 42

    @pydantic.model_validator(mode="after")
    def _validate_repair_settings(self):
        if self.tune_activations and self.activation_lr is None:
            raise ValueError("Activation tuning requires an explicit activation_lr.")
        if self.units is not None and len(self.units) != len(set(self.units)):
            raise ValueError("Repair units must be unique.")
        return self


Criterion: TypeAlias = Loss | dict[tuple[str, str], Loss]


class KDLossConfig(ModeloptBaseConfig):
    """Configuration for the Knowledge-Distillation mode.

    This mode is used to distill knowledge from a teacher model to a student model.
    """

    # TODO: we should really think about a better to configure KDLossConfig
    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    teacher_model: ModelLike | None = ModeloptField(
        default=None,
        title="Teacher model",
        description=(
            "The module, class, callable, or tuple to initialize the teacher model using"
            " :meth:`init_model_from_model_like"
            " <modelopt.torch.utils.network.init_model_from_model_like>`."
        ),
    )
    criterion: Criterion | None = ModeloptField(
        default=None,
        title="Output-only or layer-wise loss criterion",
        description=(
            "If an instance of Loss class, a distillation loss will only be computed "
            "between outputs of a student and teacher; if a dictionary in the format "
            "{(student_layer_name, teacher_layer_name): loss_module}, a distillation loss will be "
            "computed for each specified student-teacher pair of layers using the corresponding "
            "``loss_module``."
        ),
    )
    loss_balancer: Any | None = ModeloptField(
        default=None,
        title="Loss balancer",
        description=(
            "A balancer to reduce distillation and non-distillation losses into a single "
            "value using some weighing scheme."
        ),
    )
    expose_minimal_state_dict: bool = ModeloptField(
        default=True,
        title="Expose student state dict only",
        description=(
            "Hide teacher model's state_dict in the returned wrapped model. This reduces the "
            "checkpoint size by not re-storing the teacher unnecessarily again. "
            ".. note: Set to False if using `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_"
        ),
    )

    @pydantic.field_validator("criterion")
    @classmethod
    def format_criterion(cls, criterion: Criterion | None) -> dict[tuple[str, str], Loss]:
        """Ensure criterion is a mapping from layer names to loss (potentially entire module)."""
        if not isinstance(criterion, dict):
            # Output-only distillation.
            criterion = {("", ""): criterion}
        return criterion

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Dump the config to a dictionary but avoid serializing teacher model to dict.

        This avoids issues when the teacher is a tuple with callable and args.
        If any of the args are Dataclasses, they are dumped as a dict and cannot be restored with their class.
        """
        return {**super().model_dump(*args, **kwargs), "teacher_model": self.teacher_model}

    def _strict_validate(self) -> None:
        """Strictly validate the configuration."""
        for k, v in self.items():
            if k == "loss_balancer":
                pydantic.TypeAdapter(
                    DistillationLossBalancer | None, config=self.model_config
                ).validate_python(v)
                continue
            assert v is not None, f"Missing required field: {k}."

        # Cannot have multiple loss layers without LossBalancer.
        if self.loss_balancer is not None:
            assert len(tuple(self.loss_balancer.parameters())) == 0, (
                "Loss Balancer cannot have parameters."
            )
        elif len(self.criterion) == 0:  # type: ignore[arg-type]
            warnings.warn(
                "No distillation loss criterion provided. Ignore if in pipeline-parallel setup."
            )
        elif len(self.criterion) > 1:  # type: ignore[arg-type]
            raise ValueError(
                "Cannot have multiple layer-loss pairs without a `DistillationLossBalancer`"
            )


class LayerwiseKDConfig(KDLossConfig):
    """Configuration for the Layerwise Knowledge-Distillation mode.

    This mode is used to distill knowledge from a teacher model to a student model using layerwise distillation.
    """

    @pydantic.field_validator("criterion")
    @classmethod
    def format_criterion(cls, criterion: Criterion | None) -> dict[tuple[str, str], Loss]:
        """Ensure criterion is a mapping from layer names to loss (potentially entire module)."""
        if not isinstance(criterion, dict):
            raise ValueError("Layerwise Distillation mode requires explicit criterion pairs.")
        if any(key == ("", "") for key in criterion):
            raise ValueError(
                "Layerwise Distillation mode does not support output-only distillation."
            )
        return criterion


class ExportStudentConfig(ModeloptBaseConfig):
    """Configuration for the export_student mode.

    This mode is used to export a student model after distillation training/fine-tuning.
    """
