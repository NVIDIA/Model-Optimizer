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
"""Minimal extension of Megatron-Bridge's ``DistillationProvider`` for nemo:26.06 and older containers.

Adds two things over the stock provider: (1) the KD conversion runs in a post-weight-load pre-wrap hook
instead of in ``provide()``, and (2) a ``distill_submodule`` option to distill only a submodule (e.g. a
VLM ``language_model``, leaving the vision tower / projector untouched). This is the same behavior as the
upstream change in NVIDIA-NeMo/Megatron-Bridge and is implemented here as a small delta (via a dynamic
subclass, without mutating the stock class) so the example works on the current container.

TODO: Remove this module and import ``convert_to_distillation_provider`` directly from
``megatron.bridge.models.distillation_provider`` once we require the nemo:26.08 container (Megatron-Bridge#4707).
"""

import inspect

from megatron.bridge.models.distillation_provider import (
    convert_to_distillation_provider as _base_convert_to_distillation_provider,
)
from megatron.core.utils import unwrap_model

import modelopt.torch.distill as mtd
import modelopt.torch.distill.plugins.megatron as mtd_mcore


def _provide(self, pre_process=None, post_process=None, vp_stage=None):
    """Build the un-converted student; the KD conversion is deferred to ``_convert_hook``."""
    if vp_stage is not None:
        raise ValueError("ModelOpt KD currently does not support virtual-pipeline parallel.")
    return self._super_class.provide(self, pre_process, post_process, vp_stage)


def _convert_hook(self, model_chunks):
    """Pre-wrap hook (runs after weight-load): distill the whole model or ``distill_submodule``."""
    assert len(model_chunks) == 1, "ModelOpt KD does not support virtual pipeline (>1 model chunk)."
    student = unwrap_model(model_chunks[0])
    # Hack to get teacher's pre-wrap hooks called to potentially load HF weights
    teacher = unwrap_model(
        self.teacher.provide_distributed_model(wrap_with_ddp=False, mixed_precision_wrapper=None)[0]
    )
    if self.distill_submodule is not None:
        # Retain the full model so the (in-place) distilled submodule can be exported back within it.
        self.full_model = student
        student = getattr(student, self.distill_submodule)
        teacher = getattr(teacher, self.distill_submodule)

    kd_cfg = mtd_mcore.setup_distillation_config(self.kd_config, student.config, teacher.config)
    modelopt_cfg = {
        "teacher_model": teacher,
        "criterion": kd_cfg.criterion,
        "loss_balancer": kd_cfg.loss_balancer,
    }
    kd_model = mtd.convert(student, mode=[("kd_loss", modelopt_cfg)])
    mtd_mcore.adjust_distillation_model_for_mcore(kd_model, kd_cfg)
    return [kd_model]


def _shim_convert_to_distillation_provider(
    student_provider, teacher_provider, kd_config=None, *, distill_submodule=None
):
    """Like ``megatron.bridge``'s ``convert_to_distillation_provider`` but defers the KD conversion to a
    pre-wrap hook (so the student is weight-loaded first) and can target a submodule. See module docstring.
    """
    provider = _base_convert_to_distillation_provider(student_provider, teacher_provider, kd_config)
    # Dynamically subclass the (already rebased) provider class to add the deferred-convert behavior
    # without mutating Megatron-Bridge's DistillationProvider. isinstance(provider, DistillationProvider)
    # stays True, so megatron.bridge.training.distill.distill() still accepts it.
    submodule_cls = type(
        "SubmoduleDistillationProvider",
        (type(provider),),
        {
            "provide": _provide,
            "_convert_hook": _convert_hook,
            "distill_submodule": distill_submodule,
        },
    )
    # Use object.__setattr__ to bypass DistillationProvider.__setattr__, which mirrors every attribute
    # set onto the teacher -- assigning ``__class__`` normally would also switch the teacher's class.
    object.__setattr__(provider, "__class__", submodule_cls)
    # Append the convert hook after the bridge's weight-load hook so the student is fully weight-loaded
    # before conversion. Set _pre_wrap_hooks via object.__setattr__ (not register_pre_wrap_hook) to
    # bypass the teacher-mirroring __setattr__: when the student starts with no hooks (QAD builds it
    # with load_weights=False), the mirror would share the hook list with the teacher, so building the
    # teacher inside _convert_hook would re-run _convert_hook -> infinite recursion.
    hooks = [*getattr(provider, "_pre_wrap_hooks", []), provider._convert_hook]
    object.__setattr__(provider, "_pre_wrap_hooks", hooks)
    return provider


# Prefer Megatron-Bridge's native implementation when it supports submodule distillation; otherwise
# fall back to the local back-port for older containers.
convert_to_distillation_provider = (
    _base_convert_to_distillation_provider
    if "distill_submodule" in inspect.signature(_base_convert_to_distillation_provider).parameters
    else _shim_convert_to_distillation_provider
)
