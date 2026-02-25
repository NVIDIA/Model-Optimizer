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

# TODO: Upstream this fix to Megatron-Bridge and remove this local copy.

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Optional

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core.models.gpt import GPTModel as MCoreGPTModel

import modelopt.torch.distill as mtd
import modelopt.torch.distill.plugins.megatron as mtd_mcore

if TYPE_CHECKING:
    from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig


logger = logging.getLogger(__name__)


@dataclass
class DistillationProvider(TransformerConfig):
    """Provider for Megatron Core GPT models in distillation mode.

    Please use `convert_to_distillation_provider()` to create an instance of this class.
    """

    teacher: Optional[GPTModelProvider | MambaModelProvider] = None
    kd_config: Optional["ModelOptDistillConfig"] = None

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Use `convert_to_distillation_provider()` to create an instance of this class."
        )

    def __post_init__(self):
        assert getattr(self, "teacher", None) is not None, "Teacher model must be provided."

        shared_attrs = [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "seq_length",
            "pipeline_dtype",
        ]
        for attr in shared_attrs:
            if getattr(self, attr) != getattr(self.teacher, attr):
                raise ValueError(f"Student and teacher providers must have the same {attr}.")

        # Logits are overwritten in-place when TE cross-entropy loss is enabled, so switch it back to native version.
        self.cross_entropy_fusion_impl = "native"

        # Hack to dynamically subclass other providers and still use their methods
        self._super_class = self.__class__.__bases__[0]

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Configure and instantiate a ModelOpt DistillationModel based on this configuration.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreGPTModel: Configured ModelOpt DistillationModel instance
        """
        if vp_stage is not None:
            raise ValueError("ModelOpt KD currently does not support virtual-pipeline parallel.")

        assert self.teacher is not None, "Teacher model must be provided."
        student_model = self._super_class.provide(self, pre_process, post_process, vp_stage)  # type: ignore[attr-defined]

        # Finalize teacher provider before creating model (required for heterogeneous models).
        #
        # per_block_parameters is an attribute of HeterogeneousTransformerConfig (defined in
        # MCoreHeterogeneousTransformerConfig, heterogeneous_config.py:197). It's created during
        # provider creation (bridge.to_megatron_provider()), but finalize() ensures they're consistent
        # with current parallelism settings and distributed context. Student model creation (above)
        # initializes parallel_state (process groups, TP/PP config), which weight loading/scatter
        # requires. During teacher model creation, get_config_for_layer() is called (transformer_block.py:341)
        # for each layer, which uses per_block_parameters and current tensor_model_parallel_size to
        # determine layer architecture. Without finalize() in this context, architecture expectations
        # don't match checkpoint weights, causing:
        #   ValueError: ProcessGroupNCCL::scatter: invalid tensor size at index 0
        #   (expected (2880, 4096), got (3584, 4096))
        #
        # Note: This explanation needs to be confirmed yet.
        self.teacher.finalize()

        # Hack to get teacher's pre-wrap hooks called to potentially load HF weights
        teacher_model = self.teacher.provide_distributed_model(
            wrap_with_ddp=False, mixed_precision_wrapper=None
        )[0]

        kd_cfg = mtd_mcore.setup_distillation_config(
            self.kd_config, student_model.config, teacher_model.config
        )
        modelopt_cfg = {
            "teacher_model": teacher_model,
            "criterion": kd_cfg.criterion,
            "loss_balancer": kd_cfg.loss_balancer,
        }
        kd_model = mtd.convert(student_model, mode=[("kd_loss", modelopt_cfg)])
        mtd_mcore.adjust_distillation_model_for_mcore(kd_model, kd_cfg)

        return kd_model

    def to_cfg_dict(self) -> dict[str, Any]:
        """Custom method to save equivalent to the original provider class.

        Used by `_ConfigContainerBase` to serialize the main `ConfigContainer` to YAML.
        There is no need to restore a `DistillationProvider` from the run config file, as
        it can always be re-converted using the original student provider.

        Returns:
            Dictionary representation of this provider class
        """
        from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

        result = {"_target_": f"{self._super_class.__module__}.{self._super_class.__qualname__}"}

        # Include all fields from the original provider class (self._super_class), not just DistillationProvider
        # This ensures fields like heterogeneous_layers_config_encoded_json are preserved
        excluded_fields = {"teacher", "kd_config"}
        for field in fields(self._super_class):
            if field.name.startswith("_") or field.name in excluded_fields:
                continue
            # Only include if the field exists on this instance (it should, since we converted from the original provider)
            if hasattr(self, field.name):
                result[field.name] = _ConfigContainerBase._convert_value_to_dict(
                    getattr(self, field.name)
                )

        # Also include any additional fields from DistillationProvider itself (if any)
        for field in fields(self):
            if field.name.startswith("_") or field.name in excluded_fields:
                continue
            # Skip if already included from _super_class
            if field.name not in result:
                result[field.name] = _ConfigContainerBase._convert_value_to_dict(
                    getattr(self, field.name)
                )

        return result

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # Mirror to teacher if it has that attribute
        if hasattr(self.teacher, name):
            setattr(self.teacher, name, value)


def convert_to_distillation_provider(
    student_provider: GPTModelProvider | MambaModelProvider,
    teacher_provider: GPTModelProvider | MambaModelProvider,
    kd_config: Optional["ModelOptDistillConfig"] = None,
) -> "DistillationProvider":
    """Convert a given model provider to a DistillationProvider."""

    assert isinstance(student_provider, (GPTModelProvider, MambaModelProvider)), (
        "Student provider must be a subclass of GPTModelProvider or MambaModelProvider."
    )
    assert isinstance(teacher_provider, (GPTModelProvider, MambaModelProvider)), (
        "Teacher provider must be a subclass of GPTModelProvider or MambaModelProvider."
    )

    DistillationProvider.__bases__ = (type(student_provider),)
    student_provider.__class__ = DistillationProvider

    student_provider.teacher = teacher_provider
    student_provider.kd_config = kd_config
    student_provider.__post_init__()

    return student_provider
